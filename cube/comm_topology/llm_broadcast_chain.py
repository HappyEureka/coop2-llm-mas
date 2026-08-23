"""
LLM-powered Broadcast Chain topology.

The communication pattern is a fixed order in which each agent broadcasts to
all agents that follow it.
Each agent broadcasts to ALL following agents.

Decision Flow:
    Agent 0: send → plan (first speaker)
    Agent 1+: wait → send → plan
    
    Each agent:
        1. wait_for: [previous_agent] (empty for agent 0)
        2. send_to: [all following agents]
        3. generate plan via LLM
"""

import time
from typing import List, Dict

from cognitive.agent import LLMClient
from cognitive.agent.base_llm_agent import BaseLLMAgent
from cognitive.agent.prompts import build_system_prompt, build_observation_prompt
from cognitive.agent.cognitive_agent import (
    extract_status, extract_position, extract_facing, extract_visible_area, parse_plan_response,
)
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
    """
    LLM-powered agent for Broadcast Chain.
    
    Decision Flow:
        wait_for: [previous_agent_id] (empty for first speaker)
        send_to: [all following_agent_ids]
        then: generate plan via LLM
    """
    
    def __init__(self, agent_id: str, llm_client: LLMClient, speaker_order: int,
                 following_agent_ids: List[str], previous_agent_id: str,
                 n_agents: int, temperature: float = 0.7, verbose: bool = True):
        super().__init__(agent_id, llm_client, temperature=temperature, verbose=verbose)
        self.speaker_order = speaker_order
        self.n_agents = n_agents
        
        # Decision flow configuration.
        self.wait_for = [previous_agent_id] if previous_agent_id else []
        self.send_to = following_agent_ids
        
        self._last_flow_step = None
        self._all_agents: Dict[str, 'LLMBroadcastChainAgent'] = {}
        self._system_prompt = None
        self.broadcast_history: List[str] = []
    
    def _any_previous_agent_not_ready(self) -> bool:
        """Check if any agent we're waiting for is not ready."""
        if not self.wait_for or not self._all_agents:
            return False
        return any(
            self._all_agents.get(aid) and not self._all_agents[aid].ready
            for aid in self.wait_for
        )
    
    def _get_system_prompt(self) -> str:
        """Build the system prompt for this position in the chain."""
        if self._system_prompt is None:
            base = build_system_prompt(self.agent_id, max_actions=6, include_env_description=True)
            self._system_prompt = base + get_broadcast_chain_role(self.speaker_order, self.n_agents)
        return self._system_prompt
    
    def _generate_message(self, context: str) -> str:
        """Generate a Broadcast Chain message via the LLM."""
        messages = [
            {"role": "system", "content": f"You are {self.agent_id}, speaker {self.speaker_order + 1} in a Broadcast Chain. Generate a brief proposal (2-3 sentences) for all following agents."},
            {"role": "user", "content": context}
        ]
        try:
            response, usage = self.llm_client.generate(messages, response_format=None, temperature=self.temperature)
            self.api_calls += 1
            self.total_tokens_used += usage['total_tokens']
            self.prompt_tokens += usage['prompt_tokens']
            self.completion_tokens += usage['completion_tokens']
            return response
        except Exception:
            return f"Broadcast Chain contribution from {self.agent_id}"
    
    def _generate_plan_with_role(self) -> SymbolicPlan:
        """Generate a plan with the earlier Broadcast Chain messages."""
        broadcast_context = ""
        if self.broadcast_history:
            broadcast_context = "## Earlier Proposals\n" + "\n".join(self.broadcast_history) + "\n\n"
        
        obs_prompt = build_observation_prompt(
            env_step=self.env_step, agent_id=self.agent_id,
            agent_names=[f"agent_{i}" for i in range(self.n_agents)],
            status=extract_status(self.observation), position=extract_position(self.observation),
            facing=extract_facing(self.observation), visible_area=extract_visible_area(self.observation),
            memory=self.memory.get_events(), coop_config=self.coop_config, symbolic_view=self.symbolic_view,
        )
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": broadcast_context + obs_prompt}
        ]
        
        if self.verbose:
            print(f"  [{self.agent_id}] BROADCAST CHAIN calling LLM for plan...")
        
        try:
            response, usage = self.llm_client.generate_plan(messages, self.temperature)
            self.api_calls += 1
            self.total_tokens_used += usage['total_tokens']
            self.prompt_tokens += usage['prompt_tokens']
            self.completion_tokens += usage['completion_tokens']
            
            plan = parse_plan_response(response, self.agent_id, self.env_step, self.plan_count + 1)
            if self.verbose:
                print(f"  [{self.agent_id}] Plan: {plan.specification}")
            return plan
        except Exception as e:
            print(f"  [{self.agent_id}] LLM error: {e}")
            return self._generate_fallback_plan()
    
    def _execute_flow(self) -> SymbolicPlan:
        """
        Execute the Broadcast Chain decision flow:
        1. Wait for message from previous agent (if not first)
        2. Broadcast to all following agents
        3. Generate plan via LLM
        """
        # Step 1: Wait for previous agent (only if they're not ready)
        if self.wait_for and self._any_previous_agent_not_ready():
            while not self.wait_for_messages_from(self.wait_for):
                if not self._any_previous_agent_not_ready():
                    break
                time.sleep(0.05)
        
        # Step 2: Collect messages from earlier speakers.
        for msg in self.get_messages(clear_buffer=True):
            self.broadcast_history.append(f"[{msg['sender']}]: {msg['content']}")
        
        # Step 3: Broadcast to all following agents
        if self.send_to and self.message_broker is not None:
            plan_hint = self.plan.specification if self.plan else "exploring"
            history_summary = f"Earlier proposals: {len(self.broadcast_history)}" if self.broadcast_history else "First proposal"
            content = self._generate_message(f"My plan: {plan_hint}\n{history_summary}")
            self.send_message(recipients=self.send_to, content=content, metadata={'type': 'broadcast_chain', 'speaker_order': self.speaker_order})
            if self.verbose:
                print(f"  [{self.agent_id}] Broadcast to {self.send_to}: {content[:50]}...")
        
        # Step 4: Generate plan via LLM
        self._last_flow_step = self.env_step
        self.plan = self._generate_plan_with_role()
        self.plan_count += 1
        return self.plan
    
    def handle_reasoning(self):
        """Execute the decision flow."""
        self._execute_flow()
    
    def handle_interrupt(self):
        """Replan on interrupt."""
        self._execute_flow()
    
    def reset(self):
        """Reset agent state."""
        super().reset()
        self._last_flow_step = None
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
