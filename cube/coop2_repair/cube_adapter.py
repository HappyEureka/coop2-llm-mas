"""CUBE adapter for environment-neutral COOP2 traces and checks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .core import (
    ActionOutcome,
    AgentPlanView,
    ConstraintResult,
    ConstraintType,
    PettingZooParallelAdapter,
    TaskSpec,
)


FACE_TO_PUSH_ACTION = {
    "left": 4,
    "right": 3,
    "up": 2,
    "down": 1,
}

FACE_TO_PUSH_DIRECTION = {
    "left": "right",
    "right": "left",
    "up": "down",
    "down": "up",
}

ACTION_TO_NAME = {
    0: "noop",
    1: "move_up",
    2: "move_down",
    3: "move_left",
    4: "move_right",
}


class CubeCoopAdapter(PettingZooParallelAdapter):
    """Translate CUBE block-face tasks into COOP2 task/constraint contracts."""

    def __init__(
        self,
        env: Any,
        spatial_tolerance: int = 0,
        temporal_tolerance: int = 1,
        task_cooldown_steps: int = 5,
    ):
        super().__init__(env)
        self.spatial_tolerance = max(0, int(spatial_tolerance))
        self.temporal_tolerance = max(0, int(temporal_tolerance))
        self.task_cooldown_steps = max(0, int(task_cooldown_steps))

    @property
    def agent_ids(self) -> Sequence[str]:
        symbolic_env = self._symbolic_env()
        if hasattr(symbolic_env, "possible_agents"):
            return list(symbolic_env.possible_agents)
        return list(getattr(self._base_env(), "possible_agents", []))

    def get_task_specs(self) -> Iterable[TaskSpec]:
        tracker = getattr(self._base_env(), "task_tracker", None)
        if tracker is None:
            return []

        specs = []
        for task in tracker.get_current_state().values():
            if self._enum_value(getattr(task, "status", None)) == "completed":
                continue
            block = self._block_by_id(getattr(task, "block_id", None))
            face_cells = self._face_cells(block, task.face) if block is not None else []
            required = max(1, int(task.weight))
            target_type = f"weight_{required}"
            specs.append(
                TaskSpec(
                    task_id=str(task.task_id),
                    task_type="push",
                    target_id=str(task.block_id),
                    target_type=target_type,
                    required_agents=required,
                    required_capabilities=[],
                    metadata={
                        "block_id": task.block_id,
                        "face": task.face,
                        "weight": required,
                        "block_position": tuple(task.block_position),
                        "position": self._representative_cell(face_cells),
                        "face_cells": face_cells,
                        "push_action": FACE_TO_PUSH_ACTION.get(task.face),
                        "push_direction": FACE_TO_PUSH_DIRECTION.get(task.face),
                        "resource_type": target_type,
                    },
                )
            )
        return specs

    def get_constraint_results(self, infos: Dict[str, Any]) -> Iterable[ConstraintResult]:
        results: List[ConstraintResult] = []
        tracker = getattr(self._base_env(), "task_tracker", None)
        if tracker is None:
            return results

        for task in tracker.get_current_state().values():
            if self._enum_value(getattr(task, "status", None)) == "completed":
                continue
            required = max(1, int(task.weight))
            adjacent_agents = sorted(self._to_user_agent(agent) for agent in task.adjacent_agents)
            pushing_agents = sorted(self._to_user_agent(agent) for agent in task.pushing_agents)
            metadata = {
                "required_agents": required,
                "block_id": task.block_id,
                "face": task.face,
                "weight": task.weight,
                "spatial_count": len(adjacent_agents),
                "temporal_count": len(pushing_agents),
                "target_type": f"weight_{required}",
                "resource_type": f"weight_{required}",
            }
            results.extend(
                [
                    ConstraintResult(
                        task_id=str(task.task_id),
                        constraint_type=ConstraintType.SPATIAL,
                        satisfied=len(adjacent_agents) >= required,
                        score=min(1.0, len(adjacent_agents) / required),
                        agents=adjacent_agents,
                        reason=None if len(adjacent_agents) >= required else "Not enough agents adjacent to block face",
                        metadata=metadata,
                    ),
                    ConstraintResult(
                        task_id=str(task.task_id),
                        constraint_type=ConstraintType.TEMPORAL,
                        satisfied=len(pushing_agents) >= required,
                        score=min(1.0, len(pushing_agents) / required),
                        agents=pushing_agents,
                        reason=None if len(pushing_agents) >= required else "Not enough agents pushing this face together",
                        metadata=metadata,
                    ),
                    ConstraintResult(
                        task_id=str(task.task_id),
                        constraint_type=ConstraintType.DEPENDENCY,
                        satisfied=True,
                        score=1.0,
                        agents=list(self.agent_ids),
                        reason=None,
                        metadata={**metadata, "dependency_count": len(self.agent_ids)},
                    ),
                ]
            )
        return results

    def get_action_outcomes(self, infos: Dict[str, Any]) -> Dict[str, ActionOutcome]:
        info = self._first_info(infos)
        last_actions = dict(info.get("last_actions") or {})
        last_events = dict(info.get("last_step_events") or {})
        push_event_by_agent = self._push_event_by_agent(last_events)
        pushed_task_by_agent = self._pushed_task_by_agent()

        outcomes: Dict[str, ActionOutcome] = {}
        for env_agent, raw_action in last_actions.items():
            agent_id = self._to_user_agent(env_agent)
            action_value = int(raw_action)
            push_event = push_event_by_agent.get(agent_id)
            if push_event is not None:
                status = "success" if push_event["kind"] == "success" else "blocked"
                outcomes[agent_id] = ActionOutcome(
                    agent_id=agent_id,
                    action_type="push",
                    status=status,
                    reason=None if status == "success" else push_event.get("reason", "push_failed"),
                    task_id=push_event.get("task_id"),
                    effects={
                        "moved_block": status == "success",
                        "delivered_block": bool(push_event.get("delivered")),
                    },
                    metadata={
                        "primitive_action": action_value,
                        "block_id": push_event.get("block_id"),
                        "face": push_event.get("face"),
                        "required_agents": push_event.get("required"),
                        "target_type": f"weight_{push_event.get('required')}",
                        "resource_type": f"weight_{push_event.get('required')}",
                        "agents": push_event.get("agents", []),
                    },
                )
                continue

            task = pushed_task_by_agent.get(agent_id)
            if task is None:
                outcomes[agent_id] = ActionOutcome(
                    agent_id=agent_id,
                    action_type=ACTION_TO_NAME.get(action_value, str(action_value)),
                    status="success",
                    metadata={"primitive_action": action_value},
                )
                continue

            block_id = int(task.block_id)
            status = "blocked"
            outcomes[agent_id] = ActionOutcome(
                agent_id=agent_id,
                action_type="push",
                status=status,
                reason=None if status == "success" else "insufficient_force_or_blocked_destination",
                task_id=str(task.task_id),
                effects={
                    "moved_block": False,
                    "delivered_block": False,
                },
                metadata={
                    "primitive_action": action_value,
                    "block_id": block_id,
                    "face": task.face,
                    "required_agents": task.weight,
                },
            )
        return outcomes

    def infer_plan_task_id(self, plan_view: AgentPlanView) -> Optional[str]:
        for action in plan_view.remaining_actions:
            args = self._action_args(action)
            explicit = args.get("task_id")
            if explicit is not None:
                return str(explicit)
            block_id = args.get("block_id")
            if block_id is None:
                block_id = args.get("target_id")
            if block_id is None:
                continue
            face = args.get("face") or self._infer_face_from_agent_block(plan_view.agent_id, int(block_id))
            return self._task_id(int(block_id), str(face or "left"))
        return None

    def get_pre_execution_constraint_results_for_plans(
        self,
        plan_views: Sequence[AgentPlanView],
        task_specs: Sequence[TaskSpec],
        env_step: int = 0,
        infos: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        specs_by_id = {str(spec.task_id): spec for spec in task_specs}
        views_by_task: Dict[str, List[AgentPlanView]] = defaultdict(list)
        for view in plan_views:
            task_id = view.task_id or self.infer_plan_task_id(view)
            if task_id is None or not self._plan_has_push(view):
                continue
            view.task_id = task_id
            views_by_task[str(task_id)].append(view)

        results: List[ConstraintResult] = []
        for task_id, views in views_by_task.items():
            spec = specs_by_id.get(task_id) or self.resolve_task_spec(task_id)
            if spec is None:
                continue
            required = max(1, int(spec.required_agents))
            agents = sorted({view.agent_id for view in views})
            adjacent = [
                agent_id
                for agent_id in agents
                if self._agent_near_task_face(agent_id, spec, tolerance=self.spatial_tolerance)
            ]
            results.extend(
                [
                    ConstraintResult(
                        task_id=task_id,
                        constraint_type=ConstraintType.SPATIAL,
                        satisfied=len(adjacent) >= required,
                        score=min(1.0, len(adjacent) / required),
                        agents=agents,
                        reason=None if len(adjacent) >= required else f"Requires {required} agents near the push face; found {len(adjacent)}",
                        metadata={"required_agents": required, "adjacent_agents": adjacent, **spec.metadata},
                    ),
                    ConstraintResult(
                        task_id=task_id,
                        constraint_type=ConstraintType.TEMPORAL,
                        satisfied=len(agents) >= required,
                        score=min(1.0, len(agents) / required),
                        agents=agents,
                        reason=None if len(agents) >= required else f"Requires {required} agents planning push; found {len(agents)}",
                        metadata={"required_agents": required, "pushing_agents": agents, **spec.metadata},
                    ),
                    ConstraintResult(
                        task_id=task_id,
                        constraint_type=ConstraintType.DEPENDENCY,
                        satisfied=True,
                        score=1.0,
                        agents=agents,
                        metadata={"required_agents": required, **spec.metadata},
                    ),
                ]
            )

        return {
            "results": results,
            "metadata": {
                "tasks_evaluated": sorted(views_by_task),
                "unresolved_agents": [
                    view.agent_id
                    for view in plan_views
                    if view.task_id is None and self._plan_has_push(view)
                ],
            },
        }

    def resolve_task_spec(self, task_id: str) -> Optional[TaskSpec]:
        return {str(spec.task_id): spec for spec in self.get_task_specs()}.get(str(task_id))

    def get_repair_config(self) -> Dict[str, Any]:
        return {
            "spatial_tolerance": self.spatial_tolerance,
            "temporal_tolerance": self.temporal_tolerance,
            "task_cooldown_steps": self.task_cooldown_steps,
            "cooldown_scope": "task",
        }

    def build_repair_guidance(self, evaluation: Any, affected_agents: List[str], env_step: int) -> Dict[str, Any]:
        return {
            "strategy": "align_push",
            "instructions": [
                "Choose one block face and assign enough agents equal to the block weight.",
                "Agents assigned to the same block should move to adjacent cells on the same face.",
                "Assigned agents should push in the same direction during the same short window.",
                "If a block is too far or blocked, switch to a closer feasible block face.",
            ],
            "task_hints": [
                {
                    "task_id": failure.task_id,
                    "constraint": failure.constraint_type.value,
                    "reason": failure.reason,
                    "required_agents": failure.metadata.get("required_agents"),
                    "face": failure.metadata.get("face"),
                    "block_id": failure.metadata.get("block_id"),
                }
                for failure in evaluation.failures
            ],
        }

    def _symbolic_env(self) -> Any:
        return self.env

    def _base_env(self) -> Any:
        return getattr(self.env, "env", self.env)

    def _block_by_id(self, block_id: Any) -> Optional[Any]:
        try:
            block_id = int(block_id)
        except (TypeError, ValueError):
            return None
        for block in getattr(self._base_env(), "_blocks", []):
            if int(block.id) == block_id:
                return block
        return None

    def _face_cells(self, block: Any, face: str) -> List[Tuple[int, int]]:
        r, c, w = int(block.r), int(block.c), int(block.weight)
        if face == "up":
            return [(r - 1, c + dc) for dc in range(w)]
        if face == "down":
            return [(r + w, c + dc) for dc in range(w)]
        if face == "left":
            return [(r + dr, c - 1) for dr in range(w)]
        if face == "right":
            return [(r + dr, c + w) for dr in range(w)]
        return []

    def _representative_cell(self, cells: Sequence[Tuple[int, int]]) -> Optional[Tuple[float, float]]:
        if not cells:
            return None
        return (
            sum(cell[0] for cell in cells) / len(cells),
            sum(cell[1] for cell in cells) / len(cells),
        )

    def _task_id(self, block_id: int, face: str) -> str:
        return f"block_{block_id}_{face}"

    def _infer_face_from_agent_block(self, agent_id: str, block_id: int) -> str:
        block = self._block_by_id(block_id)
        env_agent = self._to_env_agent(agent_id)
        agent_pos = getattr(self._base_env(), "_agent_positions", {}).get(env_agent)
        if block is None or agent_pos is None:
            return "left"
        for face in ("left", "right", "up", "down"):
            if tuple(agent_pos) in self._face_cells(block, face):
                return face
        block_center = (block.r + (block.weight - 1) / 2, block.c + (block.weight - 1) / 2)
        dr = block_center[0] - agent_pos[0]
        dc = block_center[1] - agent_pos[1]
        if abs(dc) >= abs(dr):
            return "left" if dc > 0 else "right"
        return "up" if dr > 0 else "down"

    def _agent_near_task_face(self, agent_id: str, spec: TaskSpec, tolerance: int = 0) -> bool:
        env_agent = self._to_env_agent(agent_id)
        pos = getattr(self._base_env(), "_agent_positions", {}).get(env_agent)
        face_cells = spec.metadata.get("face_cells") or []
        if pos is None:
            return False
        if tuple(pos) in {tuple(cell) for cell in face_cells}:
            return True
        if tolerance <= 0:
            return False
        return any(abs(pos[0] - cell[0]) + abs(pos[1] - cell[1]) <= tolerance for cell in face_cells)

    def _pushed_task_by_agent(self) -> Dict[str, Any]:
        tracker = getattr(self._base_env(), "task_tracker", None)
        if tracker is None:
            return {}
        by_agent = {}
        for task in tracker.get_current_state().values():
            for env_agent in task.pushing_agents:
                by_agent[self._to_user_agent(env_agent)] = task
        return by_agent

    def _push_event_by_agent(self, last_events: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        by_agent: Dict[str, Dict[str, Any]] = {}
        delivered_task_ids = {
            event.get("task_id")
            for event in last_events.get("delivered_blocks") or []
            if isinstance(event, dict)
        }
        for event in last_events.get("successful_pushes") or []:
            if not isinstance(event, dict):
                continue
            normalized = dict(event)
            normalized["kind"] = "success"
            normalized["delivered"] = normalized.get("task_id") in delivered_task_ids
            normalized["agents"] = [self._to_user_agent(agent) for agent in normalized.get("agents", [])]
            for agent_id in normalized["agents"]:
                by_agent[agent_id] = normalized
        for event in last_events.get("failed_pushes") or []:
            if not isinstance(event, dict):
                continue
            normalized = dict(event)
            normalized["kind"] = "failed"
            normalized["agents"] = [self._to_user_agent(agent) for agent in normalized.get("agents", [])]
            face = {
                1: "down",
                2: "up",
                3: "right",
                4: "left",
            }.get(normalized.get("direction"))
            normalized["face"] = face
            normalized["task_id"] = f"block_{normalized.get('block_id')}_{face}" if face else None
            for agent_id in normalized["agents"]:
                by_agent.setdefault(agent_id, normalized)
        return by_agent

    def _to_user_agent(self, env_agent: str) -> str:
        reverse = getattr(self._symbolic_env(), "reverse_name_map", {})
        return reverse.get(env_agent, env_agent)

    def _to_env_agent(self, user_agent: str) -> str:
        name_map = getattr(self._symbolic_env(), "name_map", {})
        return name_map.get(user_agent, user_agent)

    def _first_info(self, infos: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(infos, dict):
            return {}
        for value in infos.values():
            if isinstance(value, dict):
                return value
        return {}

    def _plan_has_push(self, view: AgentPlanView) -> bool:
        return any(action.get("action_type") == "push" for action in view.remaining_actions)

    @staticmethod
    def _action_args(action: Dict[str, Any]) -> Dict[str, Any]:
        args = action.get("args")
        if isinstance(args, dict):
            return args
        return {
            key: value
            for key, value in action.items()
            if key not in {"action_type", "status", "start_step", "end_step"}
        }

    @staticmethod
    def _enum_value(value: Any) -> Any:
        return getattr(value, "value", value)
