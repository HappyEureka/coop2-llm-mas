"""COOP2 adapter helpers for MA-Crafter."""

from typing import Any, Dict, Iterable, List, Optional

from .core import (
    ActionOutcome,
    AgentPlanView,
    ConstraintResult,
    ConstraintType,
    PettingZooParallelAdapter,
    TaskSpec,
)
from .macrafter_prediction import MacrafterPredictionMixin
from .macrafter_repair import MacrafterRepairGuidanceMixin
from .prediction import PredictedAgentState


class MacrafterCoopAdapter(
    MacrafterRepairGuidanceMixin,
    MacrafterPredictionMixin,
    PettingZooParallelAdapter,
):
    """Expose CooperativeEnv task tracking through the generic COOP2 contracts."""

    def __init__(
        self,
        env: Any,
        spatial_tolerance: Optional[int] = None,
        temporal_tolerance: Optional[int] = None,
    ):
        super().__init__(env)
        self.spatial_tolerance = spatial_tolerance
        self.temporal_tolerance = temporal_tolerance

    def _base_env(self):
        return getattr(self.env, "env", self.env)

    @property
    def _user_to_env_agent(self) -> Dict[str, str]:
        return {str(k): str(v) for k, v in getattr(self.env, "name_map", {}).items()}

    @property
    def _env_to_user_agent(self) -> Dict[str, str]:
        reverse = getattr(self.env, "reverse_name_map", {})
        return {str(k): str(v) for k, v in reverse.items()}

    def _to_env_agent_id(self, agent_id: str) -> str:
        return self._user_to_env_agent.get(str(agent_id), str(agent_id))

    def _to_user_agent_id(self, agent_id: str) -> str:
        return self._env_to_user_agent.get(str(agent_id), str(agent_id))

    def get_task_specs(self) -> Iterable[TaskSpec]:
        tracker = getattr(self._base_env(), "task_tracker", None)
        if tracker is None:
            return []

        specs = []
        for task in tracker.get_all_task_states().values():
            specs.append(
                TaskSpec(
                    task_id=str(task.resource_id),
                    task_type="collect",
                    target_id=str(task.resource_id),
                    target_type=task.resource_type,
                    required_agents=task.required_agents,
                    required_capabilities=(
                        list(task.required_tool)
                        if isinstance(task.required_tool, (list, tuple))
                        else ([task.required_tool] if task.required_tool else [])
                    ),
                    metadata={
                        "position": task.position,
                        "distance_threshold": task.distance_threshold,
                        "required_tool_mode": task.required_tool_mode,
                    },
                )
            )
        return specs

    def resolve_task_spec(self, task_id: str) -> Optional[TaskSpec]:
        for spec in self.get_task_specs():
            if str(spec.task_id) == str(task_id):
                return spec
        return None

    def get_prediction_initial_state(self, agent_id: str) -> Dict[str, Any]:
        """Return current MA-Crafter state used by heuristic timeline prediction."""
        base_env = self._base_env()
        env_agent_id = self._to_env_agent_id(agent_id)
        try:
            player = base_env._players[int(env_agent_id)]
        except (AttributeError, IndexError, TypeError, ValueError):
            return {}
        return {
            "position": tuple(player.pos),
            "inventory": player.inventory.copy(),
            "metadata": {
                "nearby_utilities": self._nearby_materials(
                    tuple(player.pos),
                    {"table", "furnace"},
                ),
            },
        }

    def resolve_prediction_target(
        self,
        plan_view: AgentPlanView,
        action: Dict[str, Any],
        task_specs_by_id: Dict[str, TaskSpec],
    ) -> Dict[str, Any]:
        """Resolve symbolic action targets to MA-Crafter task/position metadata."""
        args = action.get("args", {})
        explicit_target = (
            args.get("task_id")
            or args.get("resource_id")
            or args.get("target_id")
            or args.get("item_id")
        )
        task_spec = None
        if explicit_target is not None:
            task_spec = task_specs_by_id.get(str(explicit_target))
        if task_spec is None and plan_view.task_id is not None:
            task_spec = task_specs_by_id.get(str(plan_view.task_id))

        target_type = self._normalize_resource_type(
            args.get("resource_type")
            or args.get("object_type")
            or args.get("target")
        )
        if task_spec is None and target_type is not None:
            matching = [
                spec
                for spec in task_specs_by_id.values()
                if spec.target_type == target_type
            ]
            if len(matching) == 1:
                task_spec = matching[0]

        if task_spec is not None:
            return {
                "task_id": str(task_spec.task_id),
                "target_id": str(task_spec.target_id or task_spec.task_id),
                "target_type": task_spec.target_type,
                "position": task_spec.metadata.get("position"),
                "distance_threshold": task_spec.metadata.get("distance_threshold", 1),
            }

        position = None
        symbolic_world = getattr(self._base_env(), "symbolic_world", None)
        if explicit_target is not None and symbolic_world is not None:
            try:
                from cognitive.action.navigation import find_object_position
                position = find_object_position(
                    symbolic_world,
                    int(explicit_target),
                    object_type=target_type,
                )
            except (TypeError, ValueError):
                position = None

        return {
            "task_id": str(explicit_target) if explicit_target is not None else None,
            "target_id": str(explicit_target) if explicit_target is not None else None,
            "target_type": target_type,
            "position": position,
            "distance_threshold": 1,
        }

    def estimate_prediction_duration(
        self,
        state: PredictedAgentState,
        action: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Optional[int]:
        """Estimate MA-Crafter symbolic action duration in primitive env steps."""
        if action.get("action_type") != "navigate":
            return None
        if state.position is None or target.get("position") is None:
            return None
        target_pos = tuple(target["position"])
        threshold = int(target.get("distance_threshold", 1))
        distance = abs(state.position[0] - target_pos[0]) + abs(state.position[1] - target_pos[1])
        return max(1, distance - threshold)

    def get_prediction_spatial_tolerance(self) -> int:
        """Configured slack for predicted spatial checks."""
        if self.spatial_tolerance is not None:
            return max(0, int(self.spatial_tolerance))
        repair_config = self.get_repair_config()
        return max(0, int(repair_config.get("spatial_tolerance", 0)))

    def get_prediction_temporal_tolerance(self) -> int:
        """Configured slack for predicted synchronization checks."""
        if self.temporal_tolerance is not None:
            return max(0, int(self.temporal_tolerance))
        repair_config = self.get_repair_config()
        return max(0, int(repair_config.get("temporal_tolerance", 0)))

    def get_repair_config(self) -> Dict[str, Any]:
        """Return MA-Crafter COOP2 repair configuration."""
        base_env = self._base_env()
        return dict(getattr(base_env, "coop_config", {}).get("coop2_repair", {}))

    def get_action_outcomes(self, infos: Dict[str, Any]) -> Dict[str, ActionOutcome]:
        return super().get_action_outcomes(infos)

    def get_constraint_results(self, infos: Dict[str, Any]) -> Iterable[ConstraintResult]:
        first_info = next(iter(infos.values()), {}) if infos else {}
        task_states = first_info.get("task_states", {}) if isinstance(first_info, dict) else {}
        results = []

        for task in task_states.values():
            required = max(task.required_agents, 1)
            checks = [
                (
                    ConstraintType.SPATIAL,
                    task.spatial_count >= task.required_agents,
                    task.spatial_count / required,
                    task.agents_nearby,
                ),
                (
                    ConstraintType.TEMPORAL,
                    task.temporal_count >= task.required_agents,
                    task.temporal_count / required,
                    task.collect_actions,
                ),
                (
                    ConstraintType.DEPENDENCY,
                    task.dependency_met,
                    task.dependency_count / required,
                    task.agents_with_tools,
                ),
            ]

            for constraint_type, satisfied, score, agents in checks:
                results.append(
                    ConstraintResult(
                        task_id=str(task.resource_id),
                        constraint_type=constraint_type,
                        satisfied=satisfied,
                        score=float(score),
                        agents=list(agents),
                        metadata={
                            "resource_type": task.resource_type,
                            "required_agents": task.required_agents,
                            "required_tool": task.required_tool,
                            "required_tool_mode": task.required_tool_mode,
                            "distance_threshold": task.distance_threshold,
                        },
                    )
                )

        return results
