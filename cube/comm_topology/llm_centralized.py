"""
LLM-powered Centralized topology: one leader, n-1 followers.

Decision Flow:
    Leader:
        1. wait_for: []
        2. send_to: [all followers] (LLM-generated directive)
        3. wait_for_response: [all followers]
        4. generate plan (LLM)

    Follower:
        1. wait_for: [leader]
        2. send_to: [leader] (LLM-generated acknowledgment)
        3. generate plan (LLM)

Interrupt (message arrives while waiting or executing):
    Leader: re-runs its flow, i.e. always replans (nothing interrupts a leader
        except a COOP2 repair request).
    Follower: acknowledges the leader, then the LLM decides whether to RESUME
        the committed plan or REPLAN (BaseLLMAgent.decide_interrupt).
"""

import time
from typing import Dict, List, Optional

from cognitive.agent import LLMClient
from cognitive.agent.base_llm_agent import BaseLLMAgent
from cognitive.plan import SymbolicPlan


LEADER_ROLE = """
## Your Role: LEADER
You are the leader of a team. Your responsibilities:
- Coordinate team activities by broadcasting directives to followers
- Wait for acknowledgments from all followers before finalizing your plan
- Make strategic decisions that benefit the whole team
"""

FOLLOWER_ROLE = """
## Your Role: FOLLOWER
You are a follower in a team. Your responsibilities:
- Wait for and follow the leader's directives
- Acknowledge the leader's messages
- Execute tasks that support the team's goals
"""


class LLMLeaderAgent(BaseLLMAgent):
    """Leader: broadcasts a directive, waits for acknowledgments, then plans."""

    role_prompt = LEADER_ROLE
    role_name = "LEADER"

    def __init__(self, agent_id: str, llm_client: LLMClient, follower_ids: List[str],
                 temperature: float = 0.7, verbose: bool = True):
        super().__init__(agent_id, llm_client, temperature=temperature, verbose=verbose)
        self.wait_for = []
        self.send_to = follower_ids
        self.wait_for_response = follower_ids
        self.expected_responses = set()

    def _plan_agent_names(self) -> List[str]:
        return [self.agent_id] + list(self.send_to)

    def _generate_message(self, context: str) -> str:
        """Generate the directive to followers via LLM."""
        messages = [
            {"role": "system", "content": f"You are {self.agent_id}, the LEADER. Generate a brief directive (2-3 sentences)."},
            {"role": "user", "content": context}
        ]
        return self._generate_text_from_messages(messages, fallback=f"Leader directive from {self.agent_id}")

    def _execute_flow(self) -> SymbolicPlan:
        """
        Leader decision flow:
        1. Send a directive to all followers
        2. Wait (bounded) for acknowledgments from all followers
        3. Generate plan via LLM
        """
        if self.send_to and self.message_broker is not None:
            plan_hint = self.plan.specification if self.plan else "exploring"
            content = self._generate_message(f"Plan: {plan_hint}\nFollowers: {list(self.send_to)}")
            self.send_message(recipients=self.send_to, content=content, metadata={'type': 'leader_broadcast'})
            if self.verbose:
                print(f"  [{self.agent_id}] Sent: {content[:60]}...")
            self.expected_responses = set(self.wait_for_response)

        max_wait = 10.0
        start = time.time()
        while self.expected_responses and (time.time() - start) < max_wait:
            for msg in self.get_messages(clear_buffer=True):
                if msg['metadata'].get('type') == 'follower_response':
                    self.expected_responses.discard(msg['sender'])
                    if self.verbose:
                        print(f"  [{self.agent_id}] Got ack from {msg['sender']}")
            if self.expected_responses:
                time.sleep(0.05)

        if self.expected_responses:
            print(f"  [{self.agent_id}] Warning: timeout waiting for {self.expected_responses}")
        elif self.verbose:
            print(f"  [{self.agent_id}] All acks received")

        return self.generate_plan()

    def handle_reasoning(self):
        """Run the leader flow."""
        self._execute_flow()

    def handle_interrupt(self):
        """The leader replans on interrupt by re-running its round."""
        self._execute_flow()


class LLMFollowerAgent(BaseLLMAgent):
    """Follower: waits for the leader's directive, acknowledges it, then plans."""

    role_prompt = FOLLOWER_ROLE
    role_name = "FOLLOWER"

    def __init__(self, agent_id: str, llm_client: LLMClient, leader_id: str,
                 temperature: float = 0.7, verbose: bool = True):
        super().__init__(agent_id, llm_client, temperature=temperature, verbose=verbose)
        self.wait_for = [leader_id]
        self.send_to = [leader_id]
        self.leader_id = leader_id
        self.leader_agent: Optional[LLMLeaderAgent] = None
        self.last_leader_directive: Optional[str] = None

    def _plan_agent_names(self) -> List[str]:
        return [self.leader_id, self.agent_id]

    def _plan_context_prefix(self) -> str:
        if not self.last_leader_directive:
            return ""
        return f"## Leader's Directive\n{self.last_leader_directive}\n\n"

    def _leader_is_not_ready(self) -> bool:
        """True while the leader is still reasoning (and so waiting for acks)."""
        return self.leader_agent is not None and not self.leader_agent.ready

    def _generate_message(self, directive: str) -> str:
        """Generate the acknowledgment to the leader via LLM."""
        messages = [
            {"role": "system", "content": f"You are {self.agent_id}, a FOLLOWER. Generate a brief acknowledgment (1-2 sentences)."},
            {"role": "user", "content": f"Leader's directive: {directive}"}
        ]
        return self._generate_text_from_messages(messages, fallback=f"Acknowledged from {self.agent_id}")

    def _store_leader_directive(self, messages: List[Dict]) -> bool:
        """Remember the latest leader directive; return True if one arrived."""
        received = False
        for msg in messages:
            if msg['sender'] == self.leader_id:
                self.last_leader_directive = msg['content']
                received = True
        return received

    def _acknowledge_leader(self):
        """Send an acknowledgment while the leader is waiting for follower responses."""
        if self.send_to and self.message_broker is not None and self._leader_is_not_ready():
            content = self._generate_message(self.last_leader_directive or "")
            self.send_message(recipients=self.send_to, content=content, metadata={'type': 'follower_response'})
            if self.verbose:
                print(f"  [{self.agent_id}] Sent ack: {content[:60]}...")

    def _execute_flow(self) -> SymbolicPlan:
        """
        Follower decision flow:
        1. Wait for the leader's directive while the leader is still planning
        2. Acknowledge it (only while the leader is waiting for acks)
        3. Generate plan via LLM
        """
        if self.wait_for and self._leader_is_not_ready():
            while not self.wait_for_messages_from(self.wait_for):
                if not self._leader_is_not_ready():
                    break
                time.sleep(0.05)

        self._store_leader_directive(self.get_messages(clear_buffer=True))
        self._acknowledge_leader()
        return self.generate_plan()

    def handle_reasoning(self):
        """Run the follower flow."""
        self._execute_flow()

    def handle_interrupt(self):
        """
        Interrupted during W/X by a leader message: acknowledge the leader,
        then let the LLM decide whether to resume the committed plan or replan.
        COOP2 repair requests always trigger replanning.
        """
        messages = self.get_messages(clear_buffer=True)
        if not messages:
            return
        if self._store_leader_directive(messages):
            self._acknowledge_leader()
        if self._handle_coop2_repair_interrupt(messages=messages):
            return
        self.decide_interrupt(messages)


def create_llm_centralized_topology(
    n_agents: int,
    llm_client: LLMClient,
    temperature: float = 0.7,
    verbose: bool = True
) -> Dict[str, BaseLLMAgent]:
    """
    Create LLM-powered agents for centralized topology (1 leader, n-1 followers).
    """
    if n_agents < 2:
        raise ValueError("Centralized topology requires at least 2 agents")
    
    leader_id = "agent_0"
    follower_ids = [f"agent_{i}" for i in range(1, n_agents)]
    
    agents = {}
    leader = LLMLeaderAgent(leader_id, llm_client, follower_ids, temperature, verbose)
    agents[leader_id] = leader
    
    for fid in follower_ids:
        follower = LLMFollowerAgent(fid, llm_client, leader_id, temperature, verbose)
        follower.leader_agent = leader
        agents[fid] = follower
    
    return agents
