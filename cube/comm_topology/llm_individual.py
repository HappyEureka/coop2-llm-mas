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

from typing import Dict

from cognitive.agent import LLMClient
from cognitive.agent.base_llm_agent import BaseLLMAgent


INDIVIDUAL_ROLE = """
## Your Role: INDEPENDENT AGENT
You operate independently without communication with other agents.
Make decisions based solely on your own observations and goals.
"""


class LLMIndividualAgent(BaseLLMAgent):
    """Plans from its own observation; reasoning and interrupt handling are the base defaults."""

    role_prompt = INDIVIDUAL_ROLE
    role_name = "INDIVIDUAL"

    def __init__(self, agent_id: str, llm_client: LLMClient,
                 temperature: float = 0.7, verbose: bool = True):
        super().__init__(agent_id, llm_client, temperature=temperature, verbose=verbose)
        self.wait_for = []
        self.send_to = []


def create_llm_individual_topology(
    n_agents: int,
    llm_client: LLMClient,
    temperature: float = 0.7,
    verbose: bool = True
) -> Dict[str, LLMIndividualAgent]:
    """Create LLM-powered agents for the individual topology (no communication)."""
    return {
        f"agent_{i}": LLMIndividualAgent(f"agent_{i}", llm_client, temperature, verbose)
        for i in range(n_agents)
    }
