"""Planning-wrapper delivery path for COOP2 repair requests.

Before a repair context is delivered, the affected agents hold one ordered
repair round: each agent states its intention in ascending agent-id order and
sees the earlier statements. The transcript is attached to the context, which
is then delivered through the message broker so every affected agent is
interrupted and revises its plan with the same evidence.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, Optional

from coop2_repair.message_protocol import COOP2_REPAIR_SENDER_ID, coop2_repair_metadata


class Coop2RepairDispatcher:
    """Run the ordered repair round, then interrupt the affected agents."""

    def __init__(
        self,
        agents: Dict[str, Any],
        message_broker_getter: Callable[[], Optional[Any]],
    ):
        self.agents = agents
        self._message_broker_getter = message_broker_getter

    def dispatch(
        self,
        affected_agents: Iterable[str],
        repair_context: Dict[str, Any],
        env_step: int,
    ) -> None:
        """Run the repair round and deliver its complete context."""
        affected_agents = list(affected_agents)
        message_broker = self._message_broker_getter()
        if message_broker is None:
            raise RuntimeError("COOP2 repair requires a configured message broker")
        self._attach_repair_channel(affected_agents, repair_context, env_step)
        message_broker.send_message(
            sender_id=COOP2_REPAIR_SENDER_ID,
            recipients=affected_agents,
            content=repair_context,
            metadata=coop2_repair_metadata(),
            env_step=env_step,
        )

    def _attach_repair_channel(
        self,
        affected_agents: Iterable[str],
        repair_context: Dict[str, Any],
        env_step: int,
    ) -> None:
        """Run one ordered intention round before all agents revise."""
        order = sorted(affected_agents, key=self._agent_id_sort_key)
        statements = []
        for agent_id in order:
            agent = self.agents.get(agent_id)
            if agent is None:
                raise RuntimeError(f"COOP2 repair agent is not configured: {agent_id}")
            statement = agent.describe_repair_intention(
                repair_context=repair_context,
                previous_statements=statements,
            )
            statements.append({
                "turn_index": len(statements),
                "agent_id": agent_id,
                "env_step": env_step,
                "statement": str(statement),
            })

        repair_context["repair_channel"] = {
            "protocol": "ordered_one_round_intention_then_global_revision",
            "order": order,
            "statements": statements,
            "revision_instruction": (
                "Revise your plan using the repair failures, all committed plan "
                "views, and the full ordered repair-channel transcript."
            ),
        }

    @staticmethod
    def _agent_id_sort_key(agent_id: Any):
        text = str(agent_id)
        match = re.search(r"(\d+)$", text)
        if match:
            return (text[:match.start()], int(match.group(1)), text)
        return (text, -1, text)
