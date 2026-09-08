"""
LLM-powered Individual topology: no communication.

Decision Flow:
    Each agent independently:
        1. wait_for: [] (no waiting)
        2. send_to: [] (no sending)
        3. generate plan via LLM

Interrupts: individual agents exchange no messages, so only a COOP2 repair
request can interrupt them, and the base agent handles it by replanning.
"""

from typing import Dict, List, Optional

from cognitive.agent import LLMClient
from cognitive.agent.base_llm_agent import BaseLLMAgent


INDIVIDUAL_ROLE = """
## Your Role: INDEPENDENT AGENT
Plan from your own observation during normal execution. Use COOP2 repair context
only when it is provided.
"""


class LLMIndividualAgent(BaseLLMAgent):
    """Plans from its own observation; reasoning and interrupt handling are the base defaults."""

    role_prompt = INDIVIDUAL_ROLE
    role_name = "INDIVIDUAL"

    def __init__(
        self,
        agent_id: str,
        llm_client: LLMClient,
        temperature: float = 0.7,
        verbose: bool = True,
        goal_instruction: str = "",
    ):
        super().__init__(agent_id, llm_client, temperature=temperature, verbose=verbose)
        self.wait_for = []
        self.send_to = []
        self.goal_instruction = goal_instruction
        self.team_agent_ids: List[str] = [agent_id]

    def _plan_agent_names(self) -> List[str]:
        return self.team_agent_ids

    def _plan_coop_config(self) -> Optional[str]:
        """Prepend the global objective, when one is configured, to the cooperative config."""
        coop_config = self.coop_config
        if self.goal_instruction:
            goal_text = f"GLOBAL OBJECTIVE: {self.goal_instruction}"
            coop_config = f"{goal_text}\n\n{coop_config}" if coop_config else goal_text
        return coop_config


def create_llm_individual_topology(
    n_agents: int,
    llm_client: LLMClient,
    temperature: float = 0.7,
    verbose: bool = True,
    goal_instruction: str = "",
) -> Dict[str, LLMIndividualAgent]:
    """
    Create LLM-powered agents for individual topology (no communication).
    """
    agents = {}
    team_agent_ids = [f"agent_{i}" for i in range(n_agents)]
    for i in range(n_agents):
        agent_id = team_agent_ids[i]
        agents[agent_id] = LLMIndividualAgent(
            agent_id,
            llm_client,
            temperature,
            verbose,
            goal_instruction=goal_instruction,
        )
        agents[agent_id].team_agent_ids = team_agent_ids
    return agents
