"""
LLM-powered Broadcast Chain topology.

Speakers take a fixed order; each broadcasts to all agents that follow it.

Decision Flow:
    Agent 0: send → plan (first speaker)
    Agent 1+: wait → send → plan

    Each agent:
        1. wait_for: [previous_agent] (empty for agent 0)
        2. send_to: [all following agents]
        3. generate plan via LLM

Interrupt (an earlier speaker's broadcast arrives while waiting or executing):
    The LLM decides whether to RESUME the committed plan or REPLAN
    (BaseLLMAgent.decide_interrupt). Only a revised plan is broadcast to the
    following speakers.
"""

import time
from typing import Dict, List, Optional

from cognitive.agent import LLMClient, InterruptDecision
from cognitive.agent.base_llm_agent import BaseLLMAgent
from cognitive.plan import SymbolicPlan


def get_broadcast_chain_role(speaker_order: int, n_agents: int) -> str:
    """Describe an agent's position in the Broadcast Chain."""
    if speaker_order == 0:
        return """
## Your Role: FIRST SPEAKER
You speak first in the Broadcast Chain. Your responsibilities:
- Set the initial direction and strategy for the team
- Broadcast your analysis to all other agents
- Your message will be heard by everyone
"""
    elif speaker_order == n_agents - 1:
        return """
## Your Role: FINAL SPEAKER
You speak last in the Broadcast Chain. Your responsibilities:
- Consider all previous speakers' messages
- Make a final decision incorporating all viewpoints
- You have heard from everyone else
"""
    else:
        return f"""
## Your Role: SPEAKER {speaker_order + 1} of {n_agents}
You speak in the middle of the Broadcast Chain. Your responsibilities:
- Listen to the previous speaker
- Add your perspective and broadcast to remaining agents
- Build on the discussion so far
"""


class LLMBroadcastChainAgent(BaseLLMAgent):
    """Speaks in a fixed order and broadcasts its proposal to all later speakers."""

    role_name = "BROADCAST CHAIN"

    def __init__(self, agent_id: str, llm_client: LLMClient, speaker_order: int,
                 following_agent_ids: List[str], previous_agent_id: str,
                 n_agents: int, temperature: float = 0.7, verbose: bool = True):
        super().__init__(agent_id, llm_client, temperature=temperature, verbose=verbose)
        self.speaker_order = speaker_order
        self.n_agents = n_agents
        self.role_prompt = get_broadcast_chain_role(speaker_order, n_agents)
        self.wait_for = [previous_agent_id] if previous_agent_id else []
        self.send_to = following_agent_ids
        self._all_agents: Dict[str, 'LLMBroadcastChainAgent'] = {}
        self.broadcast_history: List[str] = []

    def _plan_agent_names(self) -> List[str]:
        return [f"agent_{i}" for i in range(self.n_agents)]

    def _plan_context_prefix(self) -> str:
        if not self.broadcast_history:
            return ""
        return "## Earlier Proposals\n" + "\n".join(self.broadcast_history) + "\n\n"

    def _earlier_proposals_context(self) -> Optional[str]:
        """Earlier proposals as an extra prompt section for interrupt decisions."""
        if not self.broadcast_history:
            return None
        return "## Earlier Proposals\n" + "\n".join(self.broadcast_history)

    def _generate_message(self, context: str) -> str:
        """Generate a Broadcast Chain proposal via LLM."""
        messages = [
            {"role": "system", "content": f"You are {self.agent_id}, speaker {self.speaker_order + 1} in a Broadcast Chain. Generate a brief proposal (2-3 sentences) for all following agents."},
            {"role": "user", "content": context}
        ]
        return self._generate_text_from_messages(messages, fallback=f"Broadcast Chain contribution from {self.agent_id}")

    def _wait_for_previous_speaker(self):
        """Wait for the previous speaker's broadcast while that speaker is still planning."""
        if self.wait_for and self._any_waiting_agent_not_ready(self._all_agents):
            while not self.wait_for_messages_from(self.wait_for):
                if not self._any_waiting_agent_not_ready(self._all_agents):
                    break
                time.sleep(0.05)

    def _collect_broadcasts(self) -> List[Dict]:
        """Move buffered messages from earlier speakers into the broadcast history."""
        messages = self.get_messages(clear_buffer=True)
        for msg in messages:
            self.broadcast_history.append(f"[{msg['sender']}]: {msg['content']}")
        return messages

    def _broadcast_to_following(self):
        """Broadcast the current plan proposal to all following speakers."""
        if self.send_to and self.message_broker is not None:
            plan_hint = self.plan.specification if self.plan else "exploring"
            history_summary = f"Earlier proposals: {len(self.broadcast_history)}" if self.broadcast_history else "First proposal"
            content = self._generate_message(f"My plan: {plan_hint}\n{history_summary}")
            self.send_message(recipients=self.send_to, content=content, metadata={'type': 'broadcast_chain', 'speaker_order': self.speaker_order})
            if self.verbose:
                print(f"  [{self.agent_id}] Broadcast to {self.send_to}: {content[:50]}...")

    def _execute_flow(self) -> SymbolicPlan:
        """
        Broadcast Chain decision flow:
        1. Wait for the previous speaker (if any) while it is still planning
        2. Collect earlier speakers' broadcasts
        3. Broadcast to all following speakers
        4. Generate plan via LLM
        """
        self._wait_for_previous_speaker()
        self._collect_broadcasts()
        self._broadcast_to_following()
        return self.generate_plan()

    def handle_reasoning(self):
        """Run the Broadcast Chain flow."""
        self._execute_flow()

    def handle_interrupt(self):
        """
        Interrupted during W/X by an earlier speaker's broadcast: let the LLM
        decide whether to resume the committed plan or replan. Only a revised
        plan is broadcast to the following speakers. COOP2 repair requests
        always trigger replanning.
        """
        self._wait_for_previous_speaker()
        earlier_proposals = self._earlier_proposals_context()
        messages = self._collect_broadcasts()
        if not messages:
            return
        if self._handle_coop2_repair_interrupt(messages=messages):
            self._broadcast_to_following()
            return
        decision = self.decide_interrupt(messages, extra_context=earlier_proposals)
        if decision == InterruptDecision.REPLAN:
            self._broadcast_to_following()

    def reset(self):
        """Reset agent state."""
        super().reset()
        self.broadcast_history = []


def create_llm_broadcast_chain_topology(
    n_agents: int,
    llm_client: LLMClient,
    temperature: float = 0.7,
    verbose: bool = True
) -> Dict[str, LLMBroadcastChainAgent]:
    """
    Create LLM-powered agents for Broadcast Chain.
    
    Broadcast order: Agent 0 speaks first, then Agent 1, ..., Agent n-1 speaks last.
    Each agent broadcasts to ALL agents that come after them.
    """
    agents = {}
    
    for i in range(n_agents):
        agent_id = f"agent_{i}"
        following_ids = [f"agent_{j}" for j in range(i + 1, n_agents)]
        previous_id = f"agent_{i - 1}" if i > 0 else None
        
        agents[agent_id] = LLMBroadcastChainAgent(
            agent_id, llm_client, i, following_ids, previous_id, n_agents, temperature, verbose
        )
    
    # Set references to all agents
    for agent in agents.values():
        agent._all_agents = agents
    
    return agents
