"""
LLM-powered Centralized Topology.

Same structure as centralized.py but extends BaseLLMAgent directly.
Role information is injected into prompts.

Decision Flow:
    Leader:
        1. wait_for: []
        2. send_to: [all followers] (LLM-generated message)
        3. wait_for_response: [all followers]
        4. generate plan (LLM)
    
    Follower:
        1. wait_for: [leader]
        2. send_to: [leader] (LLM-generated response)
        3. generate plan (LLM)
"""

import time
from typing import List, Dict, Optional

from cognitive.agent import LLMClient
from cognitive.agent.base_llm_agent import BaseLLMAgent
from cognitive.agent.prompts import build_system_prompt, build_observation_prompt
from cognitive.agent.cognitive_agent import (
    extract_status, extract_position, extract_facing, extract_visible_area, parse_plan_response,
)
from cognitive.plan import SymbolicPlan


# Role descriptions injected into prompts
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
    """
    LLM-powered Leader agent for centralized topology.
    
    Decision Flow:
        wait_for: []
        send_to: [all followers]
        wait_for_response: [all followers]
        then: generate plan via LLM
    """
    
    def __init__(self, agent_id: str, llm_client: LLMClient, follower_ids: List[str],
                 temperature: float = 0.7, verbose: bool = True):
        super().__init__(agent_id, llm_client, temperature=temperature, verbose=verbose)
        
        # Decision Flow Configuration (same as centralized.py)
        self.wait_for = []
        self.send_to = follower_ids
        self.wait_for_response = follower_ids
        
        self.expected_responses = set()
        self._system_prompt = None
    
    def _get_system_prompt(self) -> str:
        """Build system prompt with leader role injected."""
        if self._system_prompt is None:
            base = build_system_prompt(self.agent_id, max_actions=6, include_env_description=True)
            self._system_prompt = base + LEADER_ROLE
        return self._system_prompt
    
    def _generate_message(self, context: str) -> str:
        """Generate directive message via LLM."""
        messages = [
            {"role": "system", "content": f"You are {self.agent_id}, the LEADER. Generate a brief directive (2-3 sentences)."},
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
            return f"Leader directive from {self.agent_id}"
    
    def _generate_plan_with_role(self) -> SymbolicPlan:
        """Generate plan via LLM with role context."""
        obs_prompt = build_observation_prompt(
            env_step=self.env_step, agent_id=self.agent_id,
            agent_names=[self.agent_id] + list(self.send_to),
            status=extract_status(self.observation), position=extract_position(self.observation),
            facing=extract_facing(self.observation), visible_area=extract_visible_area(self.observation),
            memory=self.memory.get_events(), coop_config=self.coop_config, symbolic_view=self.symbolic_view,
        )
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": obs_prompt}
        ]
        
        if self.verbose:
            print(f"  [{self.agent_id}] LEADER calling LLM for plan...")
        
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
        Execute leader decision flow (same structure as centralized.py):
        1. Send directive to all followers
        2. Wait for responses from all followers
        3. Generate plan via LLM
        """
        # Step 1: Send to all followers (LLM-generated message)
        if self.send_to and self.message_broker is not None:
            plan_hint = self.plan.specification if self.plan else "exploring"
            content = self._generate_message(f"Plan: {plan_hint}\nFollowers: {list(self.send_to)}")
            self.send_message(recipients=self.send_to, content=content, metadata={'type': 'leader_broadcast'})
            if self.verbose:
                print(f"  [{self.agent_id}] Sent: {content[:60]}...")
            self.expected_responses = set(self.wait_for_response)
        
        # Step 2: Wait for responses from all followers
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
        
        # Step 3: Generate plan via LLM
        self.plan = self._generate_plan_with_role()
        self.plan_count += 1
        return self.plan
    
    def handle_reasoning(self):
        """Execute the decision flow."""
        self._execute_flow()
    
    def handle_interrupt(self):
        """Leader replans on interrupt."""
        self._execute_flow()


class LLMFollowerAgent(BaseLLMAgent):
    """
    LLM-powered Follower agent for centralized topology.
    
    Decision Flow:
        wait_for: [leader]
        send_to: [leader] (response)
        then: generate plan via LLM
    """
    
    def __init__(self, agent_id: str, llm_client: LLMClient, leader_id: str,
                 temperature: float = 0.7, verbose: bool = True):
        super().__init__(agent_id, llm_client, temperature=temperature, verbose=verbose)
        
        # Decision Flow Configuration (same as centralized.py)
        self.wait_for = [leader_id]
        self.send_to = [leader_id]
        
        self.leader_id = leader_id
        self.leader_agent: Optional['LLMLeaderAgent'] = None
        self.last_leader_directive = None
        self._system_prompt = None
    
    def _leader_is_not_ready(self) -> bool:
        """Check if leader is not ready (same as centralized.py)."""
        if self.leader_agent is None:
            return False
        return not self.leader_agent.ready
    
    def _get_system_prompt(self) -> str:
        """Build system prompt with follower role injected."""
        if self._system_prompt is None:
            base = build_system_prompt(self.agent_id, max_actions=6, include_env_description=True)
            self._system_prompt = base + FOLLOWER_ROLE
        return self._system_prompt
    
    def _generate_message(self, directive: str) -> str:
        """Generate acknowledgment via LLM."""
        messages = [
            {"role": "system", "content": f"You are {self.agent_id}, a FOLLOWER. Generate a brief acknowledgment (1-2 sentences)."},
            {"role": "user", "content": f"Leader's directive: {directive}"}
        ]
        try:
            response, usage = self.llm_client.generate(messages, response_format=None, temperature=self.temperature)
            self.api_calls += 1
            self.total_tokens_used += usage['total_tokens']
            self.prompt_tokens += usage['prompt_tokens']
            self.completion_tokens += usage['completion_tokens']
            return response
        except Exception:
            return f"Acknowledged from {self.agent_id}"
    
    def _generate_plan_with_role(self) -> SymbolicPlan:
        """Generate plan via LLM with role context and leader's directive."""
        directive_ctx = ""
        if self.last_leader_directive:
            directive_ctx = f"## Leader's Directive\n{self.last_leader_directive}\n\n"
        
        obs_prompt = build_observation_prompt(
            env_step=self.env_step, agent_id=self.agent_id,
            agent_names=[self.leader_id, self.agent_id],
            status=extract_status(self.observation), position=extract_position(self.observation),
            facing=extract_facing(self.observation), visible_area=extract_visible_area(self.observation),
            memory=self.memory.get_events(), coop_config=self.coop_config, symbolic_view=self.symbolic_view,
        )
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": directive_ctx + obs_prompt}
        ]
        
        if self.verbose:
            print(f"  [{self.agent_id}] FOLLOWER calling LLM for plan...")
        
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
        Execute follower decision flow (same structure as centralized.py):
        1. Wait for leader message (if leader not ready)
        2. Send response to leader (if leader waiting)
        3. Generate plan via LLM
        """
        # Step 1: Wait for leader message (only if leader is not ready)
        if self.wait_for and self._leader_is_not_ready():
            while not self.wait_for_messages_from(self.wait_for):
                if not self._leader_is_not_ready():
                    break
                time.sleep(0.05)
        
        # Step 2: Get leader message and store directive
        for msg in self.get_messages(clear_buffer=True):
            if msg['sender'] == self.leader_id:
                self.last_leader_directive = msg['content']
        
        # Step 3: Send ack (only if leader is waiting for us)
        if self.send_to and self.message_broker is not None and self._leader_is_not_ready():
            content = self._generate_message(self.last_leader_directive or "")
            self.send_message(recipients=self.send_to, content=content, metadata={'type': 'follower_response'})
            if self.verbose:
                print(f"  [{self.agent_id}] Sent ack: {content[:60]}...")
        
        # Step 4: Generate plan via LLM
        self.plan = self._generate_plan_with_role()
        self.plan_count += 1
        return self.plan
    
    def handle_reasoning(self):
        """Execute the decision flow."""
        self._execute_flow()
    
    def handle_interrupt(self):
        """Follower replans on interrupt."""
        self._execute_flow()


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
