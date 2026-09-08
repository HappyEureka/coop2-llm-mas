"""COOP2-Repair adapter for CUBE.

Connects the block-pushing environment to the shared repair core:

- task specs and grounded constraint results from the block-face task tracker
- plan-effect prediction hooks used by ``HeuristicPlanTimelinePredictor``
  (initial positions, navigate targets on a block face, action durations)
- pre-execution prediction of the spatial and temporal constraints from the
  predicted push intervals
- repair guidance that assigns distinct push-face cells and a synchronized
  push step to a minimum-travel cohort
"""

from __future__ import annotations

from collections import deque
from itertools import combinations, permutations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .core import (
    ActionOutcome,
    AgentPlanView,
    ConstraintResult,
    ConstraintType,
    PettingZooParallelAdapter,
    TaskSpec,
)
from .prediction import (
    HeuristicPlanTimelinePredictor,
    PredictedActionEvent,
    PredictedAgentState,
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
    """Translate CUBE block-face tasks into COOP2 task and constraint contracts."""

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

    # ----------------------------------------------------------------------
    # Task specs and grounded outcomes
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # Plan-effect prediction: hooks used by HeuristicPlanTimelinePredictor
    # ----------------------------------------------------------------------
    def infer_plan_task_id(self, plan_view: AgentPlanView) -> Optional[str]:
        for action in plan_view.remaining_actions:
            args = self._action_args(action)
            explicit = args.get("task_id")
            if explicit is not None:
                return str(explicit)
            block_id = args.get("block_id")
            if block_id is None:
                block_id = args.get("target_id")
            if (
                block_id is None
                and action.get("action_type") == "navigate"
                and str(args.get("entity_type", "")).lower() == "block"
            ):
                block_id = args.get("entity_id")
            if block_id is None:
                continue
            # CUBE's block-navigation controller targets the left face so that
            # the subsequent push moves the block toward the goal.
            face = args.get("face") or "left"
            return self._task_id(int(block_id), str(face))
        return None

    def get_prediction_initial_state(self, agent_id: str) -> Dict[str, Any]:
        """Return the current CUBE state used by plan-timeline prediction."""
        env_agent = self._to_env_agent(agent_id)
        position = getattr(self._base_env(), "_agent_positions", {}).get(env_agent)
        return {
            "position": tuple(position) if position is not None else None,
            "inventory": {},
        }

    def resolve_prediction_target(
        self,
        plan_view: AgentPlanView,
        action: Dict[str, Any],
        task_specs_by_id: Dict[str, TaskSpec],
    ) -> Dict[str, Any]:
        """Resolve navigate/push actions to the block's goal-facing task."""
        args = action.get("args", {})
        block_id = args.get("block_id")
        if block_id is None:
            block_id = args.get("target_id")
        if (
            block_id is None
            and action.get("action_type") == "navigate"
            and str(args.get("entity_type", "")).lower() == "block"
        ):
            block_id = args.get("entity_id")

        if block_id is None and plan_view.task_id:
            task_spec = task_specs_by_id.get(str(plan_view.task_id))
            if task_spec is not None:
                block_id = task_spec.target_id
        try:
            block_id = int(block_id)
        except (TypeError, ValueError):
            return {}

        task_id = self._task_id(block_id, "left")
        task_spec = task_specs_by_id.get(task_id)
        block = self._block_by_id(block_id)
        face_cells = (
            list(task_spec.metadata.get("face_cells") or [])
            if task_spec is not None
            else self._face_cells(block, "left") if block is not None else []
        )
        current_position = self.get_prediction_initial_state(plan_view.agent_id).get("position")
        target_position = None
        if face_cells:
            if current_position is None:
                target_position = tuple(face_cells[0])
            else:
                target_position = min(
                    (tuple(cell) for cell in face_cells),
                    key=lambda cell: self._manhattan(current_position, cell),
                )

        return {
            "task_id": task_id,
            "target_id": str(block_id),
            "target_type": (
                task_spec.target_type
                if task_spec is not None
                else f"weight_{getattr(block, 'weight', 1)}"
            ),
            "position": target_position,
            "distance_threshold": 0,
            "face_cells": face_cells,
        }

    def estimate_prediction_duration(
        self,
        state: PredictedAgentState,
        action: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Optional[int]:
        """Estimate CUBE symbolic-action duration in primitive steps."""
        action_type = action.get("action_type")
        args = action.get("args", {})
        if action_type == "navigate":
            target_position = target.get("position")
            if state.position is not None and target_position is not None:
                return max(1, self._manhattan(state.position, tuple(target_position)))
            return max(1, int(args.get("timeout", 30)))
        if action_type == "push":
            return max(1, int(args.get("num_steps", 1)))
        if action_type == "wait":
            return max(1, int(args.get("num_steps", 1)))
        return None

    # ----------------------------------------------------------------------
    # Pre-execution constraint prediction
    # ----------------------------------------------------------------------
    def get_pre_execution_constraint_results_for_plans(
        self,
        plan_views: Sequence[AgentPlanView],
        task_specs: Sequence[TaskSpec],
        env_step: int = 0,
        infos: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Predict spatial and temporal constraint satisfaction per push task.

        Every plan is unrolled into a predicted timeline. For each task with a
        predicted push, the largest set of push intervals that overlap within
        ``temporal_tolerance`` steps is the synchronized cohort; the temporal
        constraint holds when that cohort is large enough, and the spatial
        constraint holds when enough of those agents are predicted to stand on
        the push face (within ``spatial_tolerance``) when they push.
        """
        specs_by_id = {str(spec.task_id): spec for spec in task_specs}
        for view in plan_views:
            inferred_task_id = self.infer_plan_task_id(view)
            if inferred_task_id is not None:
                view.task_id = inferred_task_id

        predictor = HeuristicPlanTimelinePredictor(self)
        prediction = predictor.predict(
            plan_views,
            env_step=env_step,
            task_specs=task_specs,
        )
        results: List[ConstraintResult] = []
        task_event_summary: Dict[str, Any] = {}
        tasks_evaluated = []
        for task_id in sorted(
            {
                str(event.task_id)
                for event in prediction.events
                if event.action_type == "push" and event.task_id is not None
            }
        ):
            spec = specs_by_id.get(task_id) or self.resolve_task_spec(task_id)
            if spec is None:
                continue
            push_events = prediction.events_for(task_id=task_id, action_type="push")
            if not push_events:
                continue

            tasks_evaluated.append(task_id)
            required = max(1, int(spec.required_agents))
            anchor_step, synchronized_events = self._select_temporal_cluster(
                push_events,
                self.temporal_tolerance,
            )
            planned_agents = sorted({event.agent_id for event in push_events})
            synchronized_agents = sorted(
                {event.agent_id for event in synchronized_events}
            )
            face_cells = {
                tuple(cell)
                for cell in spec.metadata.get("face_cells") or []
            }
            spatial_agents = sorted(
                {
                    event.agent_id
                    for event in synchronized_events
                    if self._position_near_cells(
                        event.position,
                        face_cells,
                        self.spatial_tolerance,
                    )
                }
            )
            task_event_summary[task_id] = {
                "planned_agents": planned_agents,
                "synchronized_agents": synchronized_agents,
                "spatial_agents": spatial_agents,
                "selected_step": anchor_step,
                "temporal_tolerance": self.temporal_tolerance,
                "push_intervals": {
                    event.agent_id: [event.start_step, event.end_step]
                    for event in push_events
                },
            }
            results.extend(
                [
                    ConstraintResult(
                        task_id=task_id,
                        constraint_type=ConstraintType.SPATIAL,
                        satisfied=len(spatial_agents) >= required,
                        score=min(1.0, len(spatial_agents) / required),
                        agents=spatial_agents or synchronized_agents or planned_agents,
                        reason=(
                            None
                            if len(spatial_agents) >= required
                            else (
                                f"Requires {required} agents at the predicted push face; "
                                f"found {len(spatial_agents)}"
                            )
                        ),
                        metadata={
                            "required_agents": required,
                            "spatial_agents": spatial_agents,
                            "selected_step": anchor_step,
                            **spec.metadata,
                        },
                    ),
                    ConstraintResult(
                        task_id=task_id,
                        constraint_type=ConstraintType.TEMPORAL,
                        satisfied=len(synchronized_agents) >= required,
                        score=min(1.0, len(synchronized_agents) / required),
                        agents=synchronized_agents or planned_agents,
                        reason=(
                            None
                            if len(synchronized_agents) >= required
                            else (
                                f"Requires {required} overlapping push intervals within "
                                f"{self.temporal_tolerance} step(s); found "
                                f"{len(synchronized_agents)}"
                            )
                        ),
                        metadata={
                            "required_agents": required,
                            "pushing_agents": synchronized_agents,
                            "selected_step": anchor_step,
                            "temporal_tolerance": self.temporal_tolerance,
                            **spec.metadata,
                        },
                    ),
                    ConstraintResult(
                        task_id=task_id,
                        constraint_type=ConstraintType.DEPENDENCY,
                        satisfied=True,
                        score=1.0,
                        agents=synchronized_agents or planned_agents,
                        metadata={"required_agents": required, **spec.metadata},
                    ),
                ]
            )

        return {
            "results": results,
            "metadata": {
                "tasks_evaluated": tasks_evaluated,
                "prediction": prediction.to_dict(),
                "task_event_summary": task_event_summary,
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

    # ----------------------------------------------------------------------
    # Repair guidance
    # ----------------------------------------------------------------------
    def build_repair_guidance(self, evaluation: Any, affected_agents: List[str], env_step: int) -> Dict[str, Any]:
        """
        Recommend one block face and a synchronized push for the affected agents.

        Candidate tasks are left-face pushes that the affected agents can staff,
        preferring unblocked blocks, then failed tasks, heavier blocks, and
        blocks closer to the goal. For the first candidate that admits distinct
        reachable face cells, the cohort and cells minimizing total travel are
        chosen, and each participant gets a plan skeleton: the move path, a wait
        so everyone arrives together, then pushes to the goal. The plans are
        advisory; ``recommendation_policy`` says so to the agents.
        """
        failed_task_ids = {
            str(failure.task_id)
            for failure in evaluation.failures
            if failure.task_id is not None
        }
        candidates = [
            spec
            for spec in self.get_task_specs()
            if spec.metadata.get("face") == "left"
            and int(spec.required_agents) <= len(affected_agents)
        ]
        candidates = [
            spec for spec in candidates
            if not self._blockers_to_goal(spec.metadata.get("block_id"))
        ] or candidates
        candidates = sorted(
            candidates,
            key=lambda spec: (
                str(spec.task_id) in failed_task_ids,
                int(spec.required_agents),
                -self._distance_to_goal(spec.metadata.get("block_id")),
            ),
            reverse=True,
        )

        recommended_plans: Dict[str, Dict[str, Any]] = {}
        recommended_target = None
        target = None
        cell_assignments: Dict[str, Tuple[int, int]] = {}
        for candidate in candidates:
            required = max(1, int(candidate.required_agents))
            face_cells = {
                tuple(cell)
                for cell in candidate.metadata.get("face_cells") or []
            }
            try:
                assignments = self._select_face_assignments(
                    affected_agents,
                    sorted(face_cells),
                    required,
                )
            except ValueError:
                continue
            target = candidate
            cell_assignments = assignments
            break

        if target is not None:
            required = max(1, int(target.required_agents))
            participants = sorted(cell_assignments)
            paths = {
                agent_id: self._repair_path_directions(
                    agent_id,
                    cell_assignments[agent_id],
                )
                for agent_id in participants
            }
            travel_steps = {
                agent_id: len(paths[agent_id])
                for agent_id in participants
            }
            synchronize_at = max(travel_steps.values(), default=0)
            synchronize_at_step = env_step + synchronize_at
            block_id = int(target.metadata["block_id"])
            push_steps = max(1, self._distance_to_goal(block_id))
            for agent_id in participants:
                target_row, target_col = cell_assignments[agent_id]
                actions = self._move_actions(paths[agent_id])
                wait_steps = synchronize_at - travel_steps[agent_id]
                if wait_steps > 0:
                    actions.extend(self._repeated_action("wait", "num_steps", wait_steps, 10))
                actions.extend(
                    self._repeated_action(
                        "push",
                        "num_steps",
                        push_steps,
                        10,
                        block_id=block_id,
                    )
                )
                recommended_plans[agent_id] = {
                    "task": (
                        f"Coordinate {required} agent(s) to deliver "
                        f"weight-{required} block {block_id}"
                    ),
                    "assigned_face_cell": [target_row, target_col],
                    "synchronize_push_at_step": synchronize_at_step,
                    "actions": actions,
                }
            recommended_target = {
                "task_id": str(target.task_id),
                "target_id": str(target.target_id),
                "target_type": target.target_type,
                "required_agents": required,
                "participants": participants,
                "synchronize_push_at_step": synchronize_at_step,
                "cell_assignments": {
                    agent_id: list(cell_assignments[agent_id])
                    for agent_id in participants
                },
            }

        return {
            "strategy": "align_push",
            "instructions": [
                "Choose one block face and assign enough agents equal to the block weight.",
                "Keep each participant's distinct assigned_face_cell and recommended movement path.",
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
            "recommended_target": recommended_target,
            "recommended_plans": recommended_plans,
            "recommendation_policy": (
                "Treat each recommended plan as advisory. Generate and commit your own "
                "plan. When feasible, preserve its target block, assigned_face_cell, "
                "movement path, and synchronization wait exactly; otherwise repair the "
                "same predicted failure from the current observation."
            ),
        }

    def _select_face_assignments(
        self,
        candidate_agents: Sequence[str],
        face_cells: Sequence[Tuple[int, int]],
        required: int,
    ) -> Dict[str, Tuple[int, int]]:
        """Select a minimum-path repair cohort and distinct push-face cells."""
        ordered_agents = sorted(set(candidate_agents))
        ordered_cells = [tuple(cell) for cell in face_cells]
        if len(ordered_agents) < required or len(ordered_cells) < required:
            raise ValueError(
                "CUBE repair requires at least one distinct face cell per participant"
            )
        if required <= 0:
            return {}

        assignments = []
        for agents in combinations(ordered_agents, required):
            for cells in permutations(ordered_cells, required):
                paths = [
                    self._repair_path_directions(agent_id, cell)
                    for agent_id, cell in zip(agents, cells)
                ]
                if any(path is None for path in paths):
                    continue
                assignments.append(
                    (
                        sum(len(path) for path in paths),
                        agents,
                        cells,
                    )
                )
        if not assignments:
            raise ValueError("CUBE repair could not find reachable distinct push cells")
        _, selected_agents, selected_cells = min(assignments)
        return dict(zip(selected_agents, selected_cells))

    def _repair_path_directions(
        self,
        agent_id: str,
        target: Tuple[int, int],
    ) -> Optional[List[str]]:
        """Find a primitive path used only to construct repair guidance."""
        base_env = self._base_env()
        start = getattr(base_env, "_agent_positions", {}).get(
            self._to_env_agent(agent_id)
        )
        if start is None:
            return None
        start = tuple(start)
        target = tuple(target)
        if start == target:
            return []

        blocked = {
            tuple(cell)
            for block in getattr(base_env, "_blocks", [])
            for cell in block.cells()
        }
        blocked.update(
            tuple(position)
            for env_agent, position in getattr(base_env, "_agent_positions", {}).items()
            if env_agent != self._to_env_agent(agent_id)
        )
        blocked.discard(target)
        grid_size = int(getattr(base_env, "K", 0))
        moves = [
            (0, 1, "right"),
            (1, 0, "down"),
            (-1, 0, "up"),
            (0, -1, "left"),
        ]
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            (row, col), path = queue.popleft()
            for dr, dc, direction in moves:
                next_position = (row + dr, col + dc)
                if next_position in visited or next_position in blocked:
                    continue
                if not (
                    0 <= next_position[0] < grid_size
                    and 0 <= next_position[1] < grid_size
                ):
                    continue
                next_path = path + [direction]
                if next_position == target:
                    return next_path
                visited.add(next_position)
                queue.append((next_position, next_path))
        return None

    @staticmethod
    def _move_actions(directions: Sequence[str]) -> List[Dict[str, Any]]:
        """Compress a primitive repair path into existing move operations."""
        actions: List[Dict[str, Any]] = []
        for direction in directions:
            if (
                actions
                and actions[-1]["args"]["direction"] == direction
                and actions[-1]["args"]["num_steps"] < 5
            ):
                actions[-1]["args"]["num_steps"] += 1
            else:
                actions.append(
                    {
                        "action_type": "move",
                        "args": {
                            "direction": direction,
                            "num_steps": 1,
                        },
                    }
                )
        return actions

    @staticmethod
    def _repeated_action(
        action_type: str,
        count_key: str,
        count: int,
        maximum: int,
        **args: Any,
    ) -> List[Dict[str, Any]]:
        """Split repair operations to respect the existing action schema."""
        actions = []
        remaining = max(0, int(count))
        while remaining:
            chunk = min(remaining, maximum)
            actions.append(
                {
                    "action_type": action_type,
                    "args": {
                        **args,
                        count_key: chunk,
                    },
                }
            )
            remaining -= chunk
        return actions

    # ----------------------------------------------------------------------
    # Geometry, lookup, and name-mapping helpers
    # ----------------------------------------------------------------------
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

    def _distance_from_agent_to_cells(
        self,
        agent_id: str,
        cells: Sequence[Tuple[int, int]],
    ) -> int:
        position = self.get_prediction_initial_state(agent_id).get("position")
        if position is None or not cells:
            return 10**6
        return min(self._manhattan(tuple(position), tuple(cell)) for cell in cells)

    def _distance_to_goal(self, block_id: Any) -> int:
        block = self._block_by_id(block_id)
        if block is None:
            return 10**6
        base_env = self._base_env()
        calculator = getattr(base_env, "_block_distance_to_goal", None)
        if callable(calculator):
            return int(calculator(block))
        grid_size = int(getattr(base_env, "K", 1))
        return max(0, grid_size - 1 - (int(block.c) + int(block.weight) - 1))

    def _blockers_to_goal(self, block_id: Any) -> List[int]:
        block = self._block_by_id(block_id)
        if block is None:
            return []
        calculator = getattr(self._base_env(), "_blocking_blocks_to_goal", None)
        return list(calculator(block)) if callable(calculator) else []

    @staticmethod
    def _position_near_cells(
        position: Optional[Tuple[int, int]],
        cells: Sequence[Tuple[int, int]],
        tolerance: int,
    ) -> bool:
        if position is None or not cells:
            return False
        return min(
            abs(position[0] - cell[0]) + abs(position[1] - cell[1])
            for cell in cells
        ) <= tolerance

    @staticmethod
    def _manhattan(first: Tuple[int, int], second: Tuple[int, int]) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    @staticmethod
    def _select_temporal_cluster(
        push_events: Sequence[PredictedActionEvent],
        temporal_tolerance: int,
    ) -> Tuple[int, List[PredictedActionEvent]]:
        """Select the largest set of predicted push intervals that overlap."""
        if not push_events:
            return (0, [])
        anchors = sorted(
            {
                step
                for event in push_events
                for step in (
                    event.start_step,
                    max(event.start_step, event.end_step - 1),
                )
            }
        )
        best_anchor = anchors[0]
        best_events: List[PredictedActionEvent] = []
        for anchor in anchors:
            overlapping = [
                event
                for event in push_events
                if (
                    event.start_step - temporal_tolerance
                    <= anchor
                    <= max(event.start_step, event.end_step - 1)
                    + temporal_tolerance
                )
            ]
            if (
                len({event.agent_id for event in overlapping})
                > len({event.agent_id for event in best_events})
            ):
                best_anchor = anchor
                best_events = overlapping
        return (best_anchor, best_events)

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
