"""
LLM-powered Individual Topology.

Same structure as individual.py but extends BaseLLMAgent directly.
No communication, independent decision-making via LLM.

Decision Flow:
    Each agent independently:
        1. wait_for: [] (no waiting)
        2. send_to: [] (no sending)
        3. generate plan via LLM
"""

from typing import Dict

from cognitive.agent import LLMClient
from cognitive.agent.base_llm_agent import BaseLLMAgent
from cognitive.agent.prompts import build_system_prompt, build_observation_prompt
from cognitive.agent.cognitive_agent import (
    extract_status, extract_position, extract_facing, extract_visible_area, parse_plan_response,
)
from cognitive.plan import SymbolicPlan


# Role description
INDIVIDUAL_ROLE = """
## Your Role: INDEPENDENT AGENT
You operate independently without communication with other agents.
Make decisions based solely on your own observations and goals.
"""


class LLMIndividualAgent(BaseLLMAgent):
    """
    LLM-powered agent for individual topology - makes decisions independently.
    
    Decision Flow:
        wait_for: []
        send_to: []
        then: generate plan via LLM
    """
    
    def __init__(self, agent_id: str, llm_client: LLMClient,
                 temperature: float = 0.7, verbose: bool = True):
        super().__init__(agent_id, llm_client, temperature=temperature, verbose=verbose)
        
        # Decision Flow Configuration (same as individual.py)
        self.wait_for = []
        self.send_to = []
        self._system_prompt = None
    
    def _get_system_prompt(self) -> str:
        """Build system prompt with individual role."""
        if self._system_prompt is None:
            base = build_system_prompt(self.agent_id, max_actions=6, include_env_description=True)
            self._system_prompt = base + INDIVIDUAL_ROLE
        return self._system_prompt
    
    def _generate_plan_with_role(self) -> SymbolicPlan:
        """Generate plan via LLM."""
        obs_prompt = build_observation_prompt(
            env_step=self.env_step, agent_id=self.agent_id,
            agent_names=[self.agent_id],
            status=extract_status(self.observation), position=extract_position(self.observation),
            facing=extract_facing(self.observation), visible_area=extract_visible_area(self.observation),
            memory=self.memory.get_events(), coop_config=self.coop_config, symbolic_view=self.symbolic_view,
        )
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": obs_prompt}
        ]
        
        if self.verbose:
            print(f"  [{self.agent_id}] Calling LLM for plan...")
        
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
        Execute decision flow (same structure as individual.py):
        1. No waiting
        2. Clear any stray messages
        3. No sending
        4. Generate plan via LLM
        """
        # Clear any stray messages
        self.get_messages(clear_buffer=True)
        
        # Generate plan via LLM
        self.plan = self._generate_plan_with_role()
        self.plan_count += 1
        return self.plan
    
    def handle_reasoning(self):
        """Execute the decision flow."""
        self._execute_flow()
    
    def handle_interrupt(self):
        """Individual agents resume on interrupt (no replanning needed)."""
        self.get_messages(clear_buffer=True)


def create_llm_individual_topology(
    n_agents: int,
    llm_client: LLMClient,
    temperature: float = 0.7,
    verbose: bool = True
) -> Dict[str, LLMIndividualAgent]:
    """
    Create LLM-powered agents for individual topology (no communication).
    """
    agents = {}
    for i in range(n_agents):
        agent_id = f"agent_{i}"
        agents[agent_id] = LLMIndividualAgent(agent_id, llm_client, temperature, verbose)
    return agents
