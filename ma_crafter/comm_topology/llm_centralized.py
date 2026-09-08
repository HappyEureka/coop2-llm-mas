"""
LLM-powered Centralized topology: one leader, n-1 followers.

Decision Flow:
    Leader:
        1. wait_for: []
        2. send_to: [all followers] with planning request
        3. wait_for_response: [all followers]
        4. generate plan (LLM)

    Follower:
        1. wait_for: [leader]
        2. send_to: [leader] (status/proposal response)
        3. generate plan (LLM)

Interrupt (message arrives while waiting or executing):
    Leader: re-runs its flow, i.e. always replans (nothing interrupts a leader
        except a COOP2 repair request).
    Follower: the LLM decides whether to RESUME the committed plan or REPLAN
        (BaseLLMAgent.decide_interrupt), then the follower replies to the
        leader and reports that decision.

The shared protocol lives in centralized_flow.py; this module adds the LLM
role prompts, message text, and planning hooks.
"""

import json
from typing import List, Dict, Optional

from cognitive.agent import LLMClient, InterruptDecision
from cognitive.agent.base_llm_agent import BaseLLMAgent
from cognitive.agent.cognitive_agent import extract_position, extract_status
from cognitive.plan import SymbolicPlan
from .centralized_flow import CentralizedFollowerFlow, CentralizedLeaderFlow


LEADER_ROLE = """
## Your Role: LEADER
Ask followers for local status/proposals, wait for their responses, then commit
a team plan using your observation plus those responses.
"""

FOLLOWER_ROLE = """
## Your Role: FOLLOWER
Reply to the leader with concise local status and one feasible proposal, then
execute a local plan that supports the team objective from your observation.
"""


class LLMLeaderAgent(CentralizedLeaderFlow, BaseLLMAgent):
    """Leader: requests follower status, waits for responses, then plans."""

    role_prompt = LEADER_ROLE
    role_name = "LEADER"

    def __init__(self, agent_id: str, llm_client: LLMClient, follower_ids: List[str],
                 temperature: float = 0.7, verbose: bool = True):
        super().__init__(agent_id, llm_client, temperature=temperature, verbose=verbose)
        self.wait_for = []
        self.send_to = follower_ids
        self.wait_for_response = follower_ids
        self.expected_responses = set()
        self.follower_responses: List[str] = []
        self.centralized_response_timeout_seconds = 30.0

    def _plan_agent_names(self) -> List[str]:
        return [self.agent_id] + list(self.send_to)

    def _plan_context_prefix(self) -> str:
        return self._follower_response_context()

    def _format_planning_request(self) -> str:
        """Request follower status before the leader commits a plan."""
        return (
            f"Leader planning request from {self.agent_id}: report current "
            "position/inventory, one useful visible target or missing prerequisite, "
            "and your next action proposal. Keep it concise."
        )

    def _follower_response_context(self) -> str:
        """Format collected follower responses for the leader planning prompt."""
        if not self.follower_responses:
            return ""
        return "## Follower Status Responses\n" + "\n".join(self.follower_responses) + "\n\n"

    def _build_leader_broadcast(self):
        return self._format_planning_request(), {
            'type': 'leader_broadcast',
            'interrupts_execution': True,
        }

    def _generate_leader_plan_after_responses(self) -> SymbolicPlan:
        return self.generate_plan()

    def _handle_leader_response_timeout(self, expected_responses: set[str]) -> None:
        print(f"  [{self.agent_id}] Warning: timeout waiting for {expected_responses}")

    def _on_leader_broadcast_sent(self, content):
        if self.verbose:
            print(f"  [{self.agent_id}] Sent: {str(content)[:60]}...")

    def _on_follower_response_received(self, sender: str) -> None:
        if self.verbose:
            print(f"  [{self.agent_id}] Got response from {sender}")

    def _on_all_follower_responses_received(self) -> None:
        if self.verbose:
            print(f"  [{self.agent_id}] All follower responses received")

    def handle_reasoning(self):
        """Run the leader flow."""
        self._execute_flow()

    def handle_interrupt(self):
        """The leader replans on interrupt: COOP2 repair directly, otherwise a new round."""
        if self._handle_coop2_repair_interrupt():
            return
        self._execute_flow()


class LLMFollowerAgent(CentralizedFollowerFlow, BaseLLMAgent):
    """Follower: waits for the leader's request, replies with status, then plans."""

    role_prompt = FOLLOWER_ROLE
    role_name = "FOLLOWER"

    def __init__(self, agent_id: str, llm_client: LLMClient, leader_id: str,
                 temperature: float = 0.7, verbose: bool = True):
        super().__init__(agent_id, llm_client, temperature=temperature, verbose=verbose)
        self.wait_for = [leader_id]
        self.send_to = [leader_id]
        self.leader_id = leader_id
        self.leader_agent: Optional['LLMLeaderAgent'] = None
        self.team_agent_ids: List[str] = [leader_id, agent_id]
        self.last_leader_request = None

    def _plan_agent_names(self) -> List[str]:
        return self.team_agent_ids

    def _plan_context_prefix(self) -> str:
        if not self.last_leader_request:
            return ""
        return f"## Leader Planning Request\n{self.last_leader_request}\n\n"

    def _current_plan_summary(self) -> str:
        """Summarize the follower's committed plan for centralized status replies."""
        if self.plan is None:
            return "Current plan: none."
        remaining = self.plan.actions[self.plan.current_action_index:]
        action_text = "; ".join(str(action) for action in remaining[:4])
        if len(remaining) > 4:
            action_text += f"; ... +{len(remaining) - 4} more"
        if not action_text:
            action_text = "no remaining actions"
        return (
            f"Current plan #{self.plan.plan_id}: {self.plan.specification}; "
            f"status={self.plan.status.value}; remaining={action_text}."
        )

    def _generate_message(self, request: str, decision: Optional[InterruptDecision] = None) -> str:
        """Generate a concise status/proposal response for the leader, reporting an interrupt decision if one was made."""
        status = extract_status(self.observation)
        position = extract_position(self.observation)
        if decision is None:
            intention_line = (
                "Include current position/status, current plan, one useful target or prerequisite, "
                "and whether you intend to resume or revise."
            )
        else:
            decided = "revised your plan" if decision == InterruptDecision.REPLAN else "resumed your current plan"
            intention_line = (
                "Include current position/status, your current plan, one useful target or prerequisite, "
                f"and state that you {decided} in response to the leader's message."
            )
        user_prompt = "\n".join([
            "The leader is collecting follower status before committing a centralized plan.",
            "Reply with 2-4 concise sentences, not JSON.",
            intention_line,
            "",
            "Leader request:",
            request or "Report local status and a feasible next task.",
            "",
            self._current_plan_summary(),
            "",
            "Your current position:",
            json.dumps(position, default=str),
            "",
            "Your current status/inventory:",
            json.dumps(status, default=str),
            "",
            "Current reachable targets:",
            self.target_hints or "unknown",
            "",
            "Recent memory:",
            json.dumps(self.memory.get_events()[-3:], default=str),
        ])
        messages = [
            {
                "role": "system",
                "content": (
                    self._get_system_prompt()
                    + "\n\nFor this response, write only the message to send back to the leader."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        return self._generate_text_from_messages(
            messages=messages,
            fallback=(
                f"Status from {self.agent_id}: I will choose a feasible local "
                "prerequisite or reachable target from my current observation."
            ),
        )

    def _build_follower_response(self, decision: Optional[InterruptDecision] = None):
        return self._generate_message(self.last_leader_request or "", decision), {'type': 'follower_response'}

    def _handle_leader_messages(self, messages: List[Dict]) -> None:
        for msg in messages:
            if msg['sender'] == self.leader_id:
                self.last_leader_request = msg['content']

    def _generate_follower_plan_after_response(self) -> SymbolicPlan:
        return self.generate_plan()

    def _send_follower_response(self, decision: Optional[InterruptDecision] = None) -> None:
        """Reply to the leader while it is still waiting for follower responses."""
        if self.send_to and self.message_broker is not None and self._leader_is_not_ready():
            content, metadata = self._build_follower_response(decision)
            self.send_message(
                recipients=self.send_to,
                content=content,
                metadata=metadata,
            )
            self._on_follower_response_sent(content)

    def _on_follower_response_sent(self, content) -> None:
        if self.verbose:
            print(f"  [{self.agent_id}] Sent response: {str(content)[:60]}...")

    def handle_reasoning(self):
        """Run the follower flow."""
        self._execute_flow()

    def handle_interrupt(self):
        """
        Interrupted during W/X by a leader message: the LLM decides whether to
        resume the committed plan or revise it, then the follower replies to
        the leader and reports that decision. COOP2 repair requests always
        trigger replanning.
        """
        if self._handle_coop2_repair_interrupt():
            return
        messages = self.get_messages(clear_buffer=True)
        if not messages:
            return
        leader_messages = [msg for msg in messages if msg.get('sender') == self.leader_id]
        self._handle_leader_messages(leader_messages)
        decision = self.decide_interrupt(messages)
        if leader_messages:
            self._send_follower_response(decision)


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
        follower.team_agent_ids = [leader_id] + follower_ids
        agents[fid] = follower
    
    return agents
