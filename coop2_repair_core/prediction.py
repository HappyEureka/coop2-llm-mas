"""Heuristic plan timeline prediction for COOP2 repair checks."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .core import AgentPlanView, TaskSpec


Position = Optional[Tuple[int, int]]


@dataclass
class PredictedAgentState:
    """Predicted state for one agent at a point in a plan timeline."""

    agent_id: str
    env_step: int
    position: Position = None
    inventory: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "PredictedAgentState":
        return PredictedAgentState(
            agent_id=self.agent_id,
            env_step=self.env_step,
            position=self.position,
            inventory=self.inventory.copy(),
            metadata=self.metadata.copy(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "env_step": self.env_step,
            "position": list(self.position) if self.position is not None else None,
            "inventory": self.inventory,
            "metadata": self.metadata,
        }


@dataclass
class PredictedActionEvent:
    """Predicted timing, location, and inventory state for a symbolic action."""

    agent_id: str
    action_type: str
    start_step: int
    end_step: int
    args: Dict[str, Any] = field(default_factory=dict)
    position: Position = None
    position_before: Position = None
    position_after: Position = None
    task_id: Optional[str] = None
    target_id: Optional[str] = None
    target_type: Optional[str] = None
    inventory_before: Dict[str, int] = field(default_factory=dict)
    inventory_after: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "args": self.args,
            "position": list(self.position) if self.position is not None else None,
            "position_before": (
                list(self.position_before) if self.position_before is not None else None
            ),
            "position_after": (
                list(self.position_after) if self.position_after is not None else None
            ),
            "task_id": self.task_id,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "inventory_before": self.inventory_before,
            "inventory_after": self.inventory_after,
            "metadata": self.metadata,
        }


@dataclass
class PlanTimelinePrediction:
    """Predicted timelines for a set of committed plans."""

    env_step: int
    initial_states: Dict[str, PredictedAgentState]
    final_states: Dict[str, PredictedAgentState]
    events: List[PredictedActionEvent]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def events_for(
        self,
        task_id: Optional[str] = None,
        action_type: Optional[str] = None,
    ) -> List[PredictedActionEvent]:
        events = self.events
        if task_id is not None:
            events = [event for event in events if event.task_id == str(task_id)]
        if action_type is not None:
            events = [event for event in events if event.action_type == action_type]
        return events

    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_step": self.env_step,
            "initial_states": {
                agent_id: state.to_dict()
                for agent_id, state in self.initial_states.items()
            },
            "final_states": {
                agent_id: state.to_dict()
                for agent_id, state in self.final_states.items()
            },
            "events": [event.to_dict() for event in self.events],
            "metadata": self.metadata,
        }


class HeuristicPlanTimelinePredictor:
    """Predict action timing and approximate state changes from symbolic plans."""

    def __init__(self, adapter: Any = None):
        self.adapter = adapter

    def predict(
        self,
        plan_views: Sequence[AgentPlanView],
        env_step: int = 0,
        task_specs: Optional[Sequence[TaskSpec]] = None,
    ) -> PlanTimelinePrediction:
        task_specs_by_id = {
            str(spec.task_id): spec
            for spec in (task_specs or [])
        }
        initial_states = {
            view.agent_id: self._initial_state(view.agent_id, env_step)
            for view in plan_views
        }
        scheduling_states = {
            agent_id: state.copy()
            for agent_id, state in initial_states.items()
        }

        events: List[PredictedActionEvent] = []
        for view in plan_views:
            state = scheduling_states[view.agent_id]
            clock = env_step
            plan_task_id = view.task_id
            for raw_action in view.remaining_actions:
                action = self._normalize_action(raw_action)
                target = self._resolve_target(view, action, plan_task_id, task_specs_by_id)
                duration = self._estimate_duration(state, action, target)
                start_step = clock
                end_step = clock + duration
                position_before = state.position
                position_after = self._apply_position_change(state.position, action, target)

                event_position = (
                    position_after
                    if action["action_type"] in {"navigate", "move"}
                    else position_before
                )
                event = PredictedActionEvent(
                    agent_id=view.agent_id,
                    action_type=action["action_type"],
                    start_step=start_step,
                    end_step=end_step,
                    args=action["args"],
                    position=event_position,
                    position_before=position_before,
                    position_after=position_after,
                    task_id=target.get("task_id") or plan_task_id,
                    target_id=target.get("target_id"),
                    target_type=target.get("target_type"),
                    metadata={
                        "duration": duration,
                        "target_position": (
                            list(target["position"])
                            if target.get("position") is not None
                            else None
                        ),
                    },
                )
                if event.task_id is not None:
                    plan_task_id = event.task_id
                events.append(event)
                state.position = position_after
                state.env_step = end_step
                clock = end_step

        inventory_states = {
            agent_id: state.copy()
            for agent_id, state in initial_states.items()
        }
        for event in sorted(events, key=lambda e: (e.end_step, e.start_step, e.agent_id)):
            state = inventory_states[event.agent_id]
            state.env_step = event.end_step
            event.inventory_before = state.inventory.copy()
            self._apply_inventory_effect(event, inventory_states)
            event.inventory_after = inventory_states[event.agent_id].inventory.copy()

        final_states = {
            agent_id: scheduling_states[agent_id].copy()
            for agent_id in scheduling_states
        }
        for agent_id, final_state in final_states.items():
            final_state.inventory = inventory_states[agent_id].inventory.copy()

        return PlanTimelinePrediction(
            env_step=env_step,
            initial_states=initial_states,
            final_states=final_states,
            events=events,
            metadata={"task_ids": sorted(task_specs_by_id.keys())},
        )

    def _initial_state(self, agent_id: str, env_step: int) -> PredictedAgentState:
        getter = getattr(self.adapter, "get_prediction_initial_state", None)
        raw = getter(agent_id) if getter is not None else {}
        return PredictedAgentState(
            agent_id=agent_id,
            env_step=env_step,
            position=self._as_position(raw.get("position")),
            inventory=dict(raw.get("inventory", {})),
            metadata=dict(raw.get("metadata", {})),
        )

    def _resolve_target(
        self,
        plan_view: AgentPlanView,
        action: Dict[str, Any],
        plan_task_id: Optional[str],
        task_specs_by_id: Dict[str, TaskSpec],
    ) -> Dict[str, Any]:
        resolver = getattr(self.adapter, "resolve_prediction_target", None)
        if resolver is not None:
            target = resolver(plan_view, action, task_specs_by_id)
            if target:
                return target

        args = action["args"]
        target_id = (
            args.get("task_id")
            or args.get("resource_id")
            or args.get("target_id")
            or args.get("item_id")
            or plan_task_id
        )
        task_spec = task_specs_by_id.get(str(target_id)) if target_id is not None else None
        return {
            "task_id": str(task_spec.task_id) if task_spec else (str(target_id) if target_id is not None else None),
            "target_id": str(task_spec.target_id) if task_spec and task_spec.target_id else (str(target_id) if target_id is not None else None),
            "target_type": task_spec.target_type if task_spec else args.get("target"),
            "position": self._as_position(task_spec.metadata.get("position")) if task_spec else None,
            "distance_threshold": task_spec.metadata.get("distance_threshold", 1) if task_spec else 1,
        }

    def _estimate_duration(
        self,
        state: PredictedAgentState,
        action: Dict[str, Any],
        target: Dict[str, Any],
    ) -> int:
        estimator = getattr(self.adapter, "estimate_prediction_duration", None)
        if estimator is not None:
            duration = estimator(state, action, target)
            if duration is not None:
                return max(1, int(duration))

        action_type = action["action_type"]
        args = action["args"]
        if action_type in {"noop", "wait", "sleep"}:
            return max(1, int(args.get("duration", args.get("steps", 1))))
        if action_type == "move":
            return max(1, int(args.get("num_steps", args.get("steps", 1))))
        if action_type == "collect":
            return max(1, int(args.get("duration", args.get("steps", 3))))
        if action_type == "navigate":
            target_pos = self._as_position(target.get("position"))
            threshold = int(target.get("distance_threshold", 1))
            if state.position is not None and target_pos is not None:
                distance = self._manhattan(state.position, target_pos)
                return max(1, distance - threshold)
            return max(1, int(args.get("estimated_steps", args.get("duration", 1))))
        return 1

    def _apply_position_change(
        self,
        position: Position,
        action: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Position:
        if position is None:
            return None
        action_type = action["action_type"]
        args = action["args"]
        if action_type == "move":
            direction = str(args.get("direction", "")).lower()
            steps = max(1, int(args.get("num_steps", args.get("steps", 1))))
            dx, dy = {
                "left": (-1, 0),
                "right": (1, 0),
                "up": (0, -1),
                "down": (0, 1),
            }.get(direction, (0, 0))
            return (position[0] + dx * steps, position[1] + dy * steps)
        if action_type == "navigate":
            target_pos = self._as_position(target.get("position"))
            if target_pos is None:
                return position
            threshold = int(target.get("distance_threshold", 1))
            return self._approach_position(position, target_pos, threshold)
        return position

    def _apply_inventory_effect(
        self,
        event: PredictedActionEvent,
        states: Dict[str, PredictedAgentState],
    ):
        applier = getattr(self.adapter, "apply_prediction_event_effect", None)
        if applier is not None:
            applier(event, states)

    def _normalize_action(self, raw_action: Dict[str, Any]) -> Dict[str, Any]:
        action_type = raw_action.get("action_type", "noop")
        args = raw_action.get("args")
        if not isinstance(args, dict):
            args = {
                key: value
                for key, value in raw_action.items()
                if key not in {"action_type", "status", "start_step", "end_step"}
            }
        return {"action_type": action_type, "args": args}

    @staticmethod
    def _as_position(value: Any) -> Position:
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (int(value[0]), int(value[1]))
        return None

    @staticmethod
    def _manhattan(first: Tuple[int, int], second: Tuple[int, int]) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    def _approach_position(
        self,
        start: Tuple[int, int],
        target: Tuple[int, int],
        threshold: int,
    ) -> Tuple[int, int]:
        if self._manhattan(start, target) <= threshold:
            return start
        x, y = target
        if start[0] < target[0]:
            x = target[0] - threshold
        elif start[0] > target[0]:
            x = target[0] + threshold
        elif start[1] < target[1]:
            y = target[1] - threshold
        elif start[1] > target[1]:
            y = target[1] + threshold
        return (x, y)
