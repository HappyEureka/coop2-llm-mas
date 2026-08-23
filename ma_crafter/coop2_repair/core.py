"""Core COOP2 contracts shared across PettingZoo-style environments.

These types intentionally avoid MA-Crafter concepts. Environment-specific code
should translate native state, actions, and task metadata into these contracts.
"""

from dataclasses import dataclass, field
from enum import Enum
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple


class ConstraintType(str, Enum):
    """General cooperative constraint classes from COOP2."""

    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    DEPENDENCY = "dependency"


COOP2_CONSTRAINT_TYPES = (
    ConstraintType.SPATIAL,
    ConstraintType.TEMPORAL,
    ConstraintType.DEPENDENCY,
)


@dataclass
class TaskSpec:
    """Environment-neutral cooperative task specification."""

    task_id: str
    task_type: str
    target_id: Optional[str] = None
    target_type: Optional[str] = None
    required_agents: int = 1
    required_capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "required_agents": self.required_agents,
            "required_capabilities": self.required_capabilities,
            "metadata": self.metadata,
        }


@dataclass
class ConstraintResult:
    """Observed or predicted satisfaction of one cooperative constraint."""

    task_id: str
    constraint_type: ConstraintType
    satisfied: bool
    score: float
    agents: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "constraint_type": self.constraint_type.value,
            "satisfied": self.satisfied,
            "score": self.score,
            "agents": self.agents,
            "reason": self.reason,
            "metadata": self.metadata,
        }


@dataclass
class ActionOutcome:
    """Grounded result of one agent action after an environment step."""

    agent_id: str
    action_type: str
    status: str
    reason: Optional[str] = None
    task_id: Optional[str] = None
    effects: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "status": self.status,
            "reason": self.reason,
            "task_id": self.task_id,
            "effects": self.effects,
            "metadata": self.metadata,
        }


@dataclass
class AgentPlanView:
    """Small, serializable view of an agent's committed plan."""

    agent_id: str
    plan_id: int
    task_id: Optional[str]
    specification: str
    remaining_actions: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "specification": self.specification,
            "remaining_actions": self.remaining_actions,
            "metadata": self.metadata,
        }


@dataclass
class Coop2TraceEvent:
    """Single event in a COOP2 trace."""

    event_type: str
    env_step: int
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "env_step": self.env_step,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "payload": self.payload,
        }


class Coop2TraceLogger:
    """Append-only JSON-serializable COOP2 trace."""

    def __init__(self):
        self.events: List[Coop2TraceEvent] = []

    def log(
        self,
        event_type: str,
        env_step: int,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **payload: Any,
    ) -> Coop2TraceEvent:
        event = Coop2TraceEvent(
            event_type=event_type,
            env_step=env_step,
            agent_id=agent_id,
            task_id=task_id,
            payload=payload,
        )
        self.events.append(event)
        return event

    def to_dict(self) -> List[Dict[str, Any]]:
        return [self._json_safe(event.to_dict()) for event in self.events]

    def save(self, output_path: str):
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def _json_safe(self, value: Any) -> Any:
        """Convert common scientific/runtime values into plain JSON values."""
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, set):
            return sorted(self._json_safe(item) for item in value)
        if hasattr(value, "item") and callable(value.item):
            try:
                return value.item()
            except (TypeError, ValueError):
                pass
        return value


class ParallelEnvAdapter(Protocol):
    """Adapter surface expected from a PettingZoo parallel-style environment."""

    @property
    def agent_ids(self) -> Sequence[str]:
        ...

    def reset(self, *args, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...

    def step(
        self,
        actions: Dict[str, Any],
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        ...

    def get_task_specs(self) -> Iterable[TaskSpec]:
        ...

    def get_action_outcomes(self, infos: Dict[str, Any]) -> Dict[str, ActionOutcome]:
        ...

    def get_constraint_results(self, infos: Dict[str, Any]) -> Iterable[ConstraintResult]:
        ...


class PettingZooParallelAdapter:
    """Thin default adapter for existing PettingZoo parallel environments."""

    def __init__(self, env: Any):
        self.env = env

    @property
    def agent_ids(self) -> Sequence[str]:
        return list(getattr(self.env, "possible_agents", getattr(self.env, "agents", [])))

    def reset(self, *args, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self.env.reset(*args, **kwargs)

    def step(
        self,
        actions: Dict[str, Any],
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        return self.env.step(actions)

    def get_task_specs(self) -> Iterable[TaskSpec]:
        return []

    def get_action_outcomes(self, infos: Dict[str, Any]) -> Dict[str, ActionOutcome]:
        outcomes: Dict[str, ActionOutcome] = {}
        for agent_id, info in infos.items():
            raw = info.get("action_outcome") if isinstance(info, dict) else None
            if raw:
                outcomes[agent_id] = ActionOutcome(
                    agent_id=raw.get("agent_id", agent_id),
                    action_type=raw.get("action_type", raw.get("primitive_action", "unknown")),
                    status=raw.get("status", "success"),
                    reason=raw.get("reason"),
                    task_id=raw.get("task_id"),
                    effects=raw.get("effects", {}),
                    metadata=raw.get("metadata", {}),
                )
        return outcomes

    def get_constraint_results(self, infos: Dict[str, Any]) -> Iterable[ConstraintResult]:
        return []
