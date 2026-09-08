"""MA-Crafter-specific COOP2 prediction and precheck helpers."""

from typing import Any, Dict, Iterable, List, Optional, Sequence

from .core import AgentPlanView, ConstraintResult, ConstraintType, TaskSpec
from .prediction import (
    HeuristicPlanTimelinePredictor,
    PredictedActionEvent,
    PredictedAgentState,
)


class MacrafterPredictionMixin:
    """Predict MA-Crafter plan effects and cooperative constraint satisfaction."""

    def apply_prediction_event_effect(
        self,
        event: PredictedActionEvent,
        states: Dict[str, PredictedAgentState],
    ):
        """Apply optimistic MA-Crafter inventory effects for predicted actions."""
        from macrafter import constants

        state = states[event.agent_id]
        inventory = state.inventory
        action_type = event.action_type
        args = event.args
        event.metadata.setdefault("heuristic_success", True)

        if action_type == "collect":
            resource_type = self._normalize_resource_type(
                event.target_type or args.get("target")
            )
            if resource_type == "cow":
                inventory["food"] = inventory.get("food", 0) + 6
                return
            collect_info = constants.collect.get(resource_type)
            if collect_info is None:
                return
            for item_name, amount in collect_info.get("receive", {}).items():
                inventory[item_name] = inventory.get(item_name, 0) + amount
            return

        if action_type == "craft":
            item_name = args.get("object_type") or args.get("item")
            make_info = constants.make.get(item_name)
            if make_info is None:
                event.metadata["heuristic_success"] = False
                event.metadata["failure_reason"] = f"Unknown craft target: {item_name}"
                return
            required_utilities = make_info.get("nearby", [])
            if not self._state_has_nearby_utilities(state, required_utilities):
                event.metadata["heuristic_success"] = False
                event.metadata["failure_reason"] = (
                    "Missing nearby craft utility: "
                    + ", ".join(required_utilities)
                )
                return
            if not self._has_inventory(inventory, make_info.get("uses", {})):
                event.metadata["heuristic_success"] = False
                event.metadata["failure_reason"] = "Missing craft ingredients"
                return
            self._consume_inventory(inventory, make_info.get("uses", {}))
            inventory[item_name] = inventory.get(item_name, 0) + make_info.get("gives", 1)
            return

        if action_type == "place":
            item_name = args.get("object_type") or args.get("item")
            place_info = constants.place.get(item_name)
            if place_info is None:
                event.metadata["heuristic_success"] = False
                event.metadata["failure_reason"] = f"Unknown place target: {item_name}"
                return
            if not self._has_inventory(inventory, place_info.get("uses", {})):
                event.metadata["heuristic_success"] = False
                event.metadata["failure_reason"] = "Missing place resources"
                return
            self._consume_inventory(inventory, place_info.get("uses", {}))
            if place_info.get("type") == "material":
                self._mark_predicted_nearby_utility(state, item_name)
            return

        if action_type == "share":
            recipient_id = str(args.get("recipient_agent_id"))
            resource_type = args.get("resource_type")
            quantity = int(args.get("quantity", 1))
            recipient_state = states.get(recipient_id)
            if recipient_state is None or not resource_type:
                event.metadata["heuristic_success"] = False
                event.metadata["failure_reason"] = "Invalid share recipient or resource"
                return
            available = inventory.get(resource_type, 0)
            if available <= 0:
                event.metadata["heuristic_success"] = False
                event.metadata["failure_reason"] = f"No {resource_type} available to share"
                return
            quantity = min(quantity, available)
            inventory[resource_type] = available - quantity
            recipient_state.inventory[resource_type] = (
                recipient_state.inventory.get(resource_type, 0) + quantity
            )

    def get_pre_execution_constraint_results_for_plans(
        self,
        plan_views: Sequence[AgentPlanView],
        task_specs: Sequence[TaskSpec],
        env_step: int = 0,
        infos: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate predicted MA-Crafter task constraints from plan timelines."""
        from macrafter.cooperative_tasks import satisfies_tool_requirement

        predictor = HeuristicPlanTimelinePredictor(self)
        prediction = predictor.predict(plan_views, env_step=env_step, task_specs=task_specs)
        results: List[ConstraintResult] = []
        task_event_summary = {}

        for task_spec in task_specs:
            task_id = str(task_spec.task_id)
            collect_events = prediction.events_for(task_id=task_id, action_type="collect")
            if not collect_events:
                continue

            required_agents = max(int(task_spec.required_agents), 1)
            planned_agents = sorted({event.agent_id for event in collect_events})
            events_by_step: Dict[int, List[PredictedActionEvent]] = {}
            for event in collect_events:
                events_by_step.setdefault(event.start_step, []).append(event)
            temporal_tolerance = self.get_prediction_temporal_tolerance()
            selected_step, selected_events = self._select_temporal_cluster(
                collect_events,
                temporal_tolerance,
            )
            selected_agents = sorted({event.agent_id for event in selected_events})
            selected_agent_count = len(selected_agents)
            task_event_summary[task_id] = {
                "planned_agents": planned_agents,
                "selected_step": selected_step,
                "temporal_tolerance": temporal_tolerance,
                "collect_steps": {
                    str(step): sorted(event.agent_id for event in events)
                    for step, events in events_by_step.items()
                },
                "selected_window": [
                    min(event.start_step for event in selected_events),
                    max(event.start_step for event in selected_events),
                ] if selected_events else None,
            }

            results.append(
                ConstraintResult(
                    task_id=task_id,
                    constraint_type=ConstraintType.TEMPORAL,
                    satisfied=selected_agent_count >= required_agents,
                    score=selected_agent_count / required_agents,
                    agents=selected_agents or planned_agents,
                    reason=(
                        None
                        if selected_agent_count >= required_agents
                        else (
                            f"Requires {required_agents} collect actions within "
                            f"{temporal_tolerance} primitive step(s), best predicted "
                            f"window around step {selected_step} has "
                            f"{selected_agent_count} distinct agent(s)"
                        )
                    ),
                    metadata={
                        **task_event_summary[task_id],
                        "resource_type": task_spec.target_type,
                    },
                )
            )

            task_position = self._as_position(task_spec.metadata.get("position"))
            distance_threshold = int(task_spec.metadata.get("distance_threshold", 1))
            spatial_tolerance = self.get_prediction_spatial_tolerance()
            effective_distance_threshold = distance_threshold + spatial_tolerance
            spatial_agents = set()
            for event in selected_events:
                if task_position is None or event.position is None:
                    continue
                distance = self._manhattan(event.position, task_position)
                event.metadata["distance_to_task"] = distance
                if distance <= effective_distance_threshold:
                    spatial_agents.add(event.agent_id)
            sorted_spatial_agents = sorted(spatial_agents)
            results.append(
                ConstraintResult(
                    task_id=task_id,
                    constraint_type=ConstraintType.SPATIAL,
                    satisfied=len(spatial_agents) >= required_agents,
                    score=len(spatial_agents) / required_agents,
                    agents=sorted_spatial_agents or selected_agents or planned_agents,
                    reason=(
                        None
                        if len(spatial_agents) >= required_agents
                        else (
                            f"Requires {required_agents} agents within predicted distance "
                            f"{effective_distance_threshold} at collect step, "
                            f"found {len(spatial_agents)}"
                        )
                    ),
                    metadata={
                        "resource_type": task_spec.target_type,
                        "position": task_position,
                        "distance_threshold": distance_threshold,
                        "spatial_tolerance": spatial_tolerance,
                        "effective_distance_threshold": effective_distance_threshold,
                        "selected_step": selected_step,
                    },
                )
            )

            task_state = self._get_task_state(task_id)
            required_tool = getattr(task_state, "required_tool", None)
            required_tool_mode = getattr(task_state, "required_tool_mode", "all")
            if required_tool is None:
                required_tool = list(task_spec.required_capabilities)
            capable_agents = {
                event.agent_id
                for event in selected_events
                if satisfies_tool_requirement(
                    event.inventory_before,
                    required_tool,
                    required_tool_mode,
                )
            }
            sorted_capable_agents = sorted(capable_agents)
            dependency_satisfied = (
                not required_tool
                or len(capable_agents) >= required_agents
            )
            results.append(
                ConstraintResult(
                    task_id=task_id,
                    constraint_type=ConstraintType.DEPENDENCY,
                    satisfied=dependency_satisfied,
                    score=1.0 if not required_tool else len(capable_agents) / required_agents,
                    agents=sorted_capable_agents or selected_agents or planned_agents,
                    reason=(
                        None
                        if dependency_satisfied
                        else (
                            f"Requires {required_agents} agents with {required_tool} "
                            f"at collect step, found {len(capable_agents)}"
                        )
                    ),
                    metadata={
                        "resource_type": task_spec.target_type,
                        "required_tool": required_tool,
                        "required_tool_mode": required_tool_mode,
                        "selected_step": selected_step,
                    },
                )
            )

        return {
            "results": results,
            "metadata": {
                "prediction": prediction.to_dict(),
                "task_event_summary": task_event_summary,
            },
        }

    def infer_plan_task_id(self, plan_view: AgentPlanView) -> Optional[str]:
        """Infer the intended MA-Crafter resource task from a symbolic plan."""
        task_specs = list(self.get_task_specs())
        task_ids = {str(spec.task_id) for spec in task_specs}

        for action in plan_view.remaining_actions:
            args = self._action_args(action)
            for key in ("task_id", "resource_id", "target_id", "item_id"):
                value = args.get(key)
                if value is not None and str(value) in task_ids:
                    return str(value)

        target_type = None
        for action in plan_view.remaining_actions:
            args = self._action_args(action)
            if action.get("action_type") == "collect":
                target_type = (
                    args.get("resource_type")
                    or args.get("object_type")
                    or args.get("target")
                )
                break

        resource_type = self._normalize_resource_type(target_type)
        if resource_type is None:
            return None

        matching = [
            spec
            for spec in task_specs
            if spec.target_type == resource_type
        ]
        if len(matching) == 1:
            return str(matching[0].task_id)
        return None

    def get_pre_execution_constraint_results(
        self,
        task_spec: TaskSpec,
        plan_views: Sequence[AgentPlanView],
        env_step: int = 0,
        infos: Optional[Dict[str, Any]] = None,
    ) -> Iterable[ConstraintResult]:
        """Predict MA-Crafter collection constraint satisfaction before stepping."""
        tracker = getattr(self._base_env(), "task_tracker", None)
        task_state = None
        if tracker is not None:
            try:
                task_state = tracker.get_task_state(int(task_spec.task_id))
            except (TypeError, ValueError):
                task_state = None

        required_agents = max(int(task_spec.required_agents), 1)
        collectors = [
            view
            for view in plan_views
            if self._plan_has_action(view, "collect")
        ]
        collector_agents = [view.agent_id for view in collectors]
        participant_agents = [view.agent_id for view in plan_views]

        results: List[ConstraintResult] = [
            ConstraintResult(
                task_id=str(task_spec.task_id),
                constraint_type=ConstraintType.TEMPORAL,
                satisfied=len(collector_agents) >= required_agents,
                score=len(collector_agents) / required_agents,
                agents=collector_agents or participant_agents,
                reason=(
                    None
                    if len(collector_agents) >= required_agents
                    else (
                        f"Requires {required_agents} planned collect actions, "
                        f"found {len(collector_agents)}"
                    )
                ),
                metadata={
                    "resource_type": task_spec.target_type,
                    "required_agents": required_agents,
                },
            ),
        ]

        if task_state is None:
            return results

        spatial_agents = self._spatially_planned_agents(task_state, plan_views)
        results.append(
            ConstraintResult(
                task_id=str(task_spec.task_id),
                constraint_type=ConstraintType.SPATIAL,
                satisfied=len(spatial_agents) >= required_agents,
                score=len(spatial_agents) / required_agents,
                agents=spatial_agents or participant_agents,
                reason=(
                    None
                    if len(spatial_agents) >= required_agents
                    else (
                        f"Requires {required_agents} agents near or navigating to target, "
                        f"found {len(spatial_agents)}"
                    )
                ),
                metadata={
                    "resource_type": task_spec.target_type,
                    "position": task_state.position,
                    "nearby_agents": [
                        self._to_user_agent_id(agent_id)
                        for agent_id in task_state.agents_nearby
                    ],
                },
            )
        )

        required_tool = task_state.required_tool
        required_tool_mode = getattr(task_state, "required_tool_mode", "all")
        capability_agents = self._capable_or_planned_agents(
            task_state,
            collectors or plan_views,
        )
        tools_required = bool(task_spec.required_capabilities)
        results.append(
            ConstraintResult(
                task_id=str(task_spec.task_id),
                constraint_type=ConstraintType.DEPENDENCY,
                satisfied=(not tools_required) or len(capability_agents) >= required_agents,
                score=(
                    1.0
                    if not tools_required
                    else len(capability_agents) / required_agents
                ),
                agents=capability_agents or collector_agents or participant_agents,
                reason=(
                    None
                    if (not tools_required) or len(capability_agents) >= required_agents
                    else (
                        f"Requires {required_agents} agents with {required_tool}, "
                        f"found {len(capability_agents)}"
                    )
                ),
                metadata={
                    "resource_type": task_spec.target_type,
                    "required_tool": required_tool,
                    "required_tool_mode": required_tool_mode,
                    "required_capabilities": task_spec.required_capabilities,
                },
            )
        )

        return results

    def _spatially_planned_agents(self, task_state: Any, plan_views: Sequence[AgentPlanView]) -> List[str]:
        task_id = str(task_state.resource_id)
        nearby_user_ids = {
            self._to_user_agent_id(agent_id)
            for agent_id in task_state.agents_nearby
        }
        agents = []
        for view in plan_views:
            if view.agent_id in nearby_user_ids or self._plan_navigates_to_task(view, task_id):
                agents.append(view.agent_id)
        return agents

    def _capable_or_planned_agents(
        self,
        task_state: Any,
        plan_views: Sequence[AgentPlanView],
    ) -> List[str]:
        from macrafter.cooperative_tasks import (
            normalize_tool_requirement,
            satisfies_tool_requirement,
        )

        required_tools = normalize_tool_requirement(task_state.required_tool)
        if not required_tools:
            return [view.agent_id for view in plan_views]

        agents = []
        base_env = self._base_env()
        for view in plan_views:
            env_agent_id = self._to_env_agent_id(view.agent_id)
            inventory = {}
            try:
                inventory = base_env._players[int(env_agent_id)].inventory
            except (AttributeError, IndexError, TypeError, ValueError):
                inventory = {}

            if satisfies_tool_requirement(
                inventory,
                task_state.required_tool,
                getattr(task_state, "required_tool_mode", "all"),
            ) or self._plan_acquires_tools_before_collect(view, required_tools):
                agents.append(view.agent_id)
        return agents

    def _plan_navigates_to_task(self, plan_view: AgentPlanView, task_id: str) -> bool:
        for action in plan_view.remaining_actions:
            if action.get("action_type") == "collect":
                return False
            args = self._action_args(action)
            if action.get("action_type") == "navigate" and str(args.get("item_id")) == task_id:
                return True
        return False

    def _plan_acquires_tools_before_collect(
        self,
        plan_view: AgentPlanView,
        required_tools: Sequence[str],
    ) -> bool:
        if not required_tools:
            return True
        acquired = set()
        for action in plan_view.remaining_actions:
            action_type = action.get("action_type")
            args = self._action_args(action)
            if action_type == "collect":
                break
            if action_type == "craft" and args.get("object_type") in required_tools:
                acquired.add(args["object_type"])
            if action_type == "share" and args.get("resource_type") in required_tools:
                recipient = args.get("recipient_agent_id")
                if recipient is None or str(recipient) == plan_view.agent_id:
                    acquired.add(args["resource_type"])
        return set(required_tools).issubset(acquired)

    def _get_task_state(self, task_id: str) -> Optional[Any]:
        tracker = getattr(self._base_env(), "task_tracker", None)
        if tracker is None:
            return None
        try:
            return tracker.get_task_state(int(task_id))
        except (TypeError, ValueError):
            return None

    def _nearby_materials(self, position: Optional[tuple], materials: set) -> List[str]:
        """Return configured material utilities near a position in the current env."""
        if position is None:
            return []
        world = getattr(self._base_env(), "_world", None)
        if world is None:
            return []
        try:
            nearby, _objects = world.nearby(position, 1)
        except Exception:
            return []
        return sorted(str(material) for material in nearby if str(material) in materials)

    def _state_inventory(self, state: Any) -> Dict[str, int]:
        if isinstance(state, dict):
            inventory = state.get("inventory", {})
        else:
            inventory = getattr(state, "inventory", {})
        return dict(inventory or {})

    def _state_metadata(self, state: Any) -> Dict[str, Any]:
        if isinstance(state, dict):
            return dict(state.get("metadata", {}) or {})
        return dict(getattr(state, "metadata", {}) or {})

    def _state_nearby_utilities(self, state: Any) -> List[str]:
        metadata = self._state_metadata(state)
        utilities = metadata.get("nearby_utilities") or []
        return sorted({str(utility) for utility in utilities})

    def _state_has_nearby_utilities(self, state: PredictedAgentState, utilities: Sequence[str]) -> bool:
        if not utilities:
            return True
        nearby = set(self._state_nearby_utilities(state))
        return all(str(utility) in nearby for utility in utilities)

    def _mark_predicted_nearby_utility(self, state: PredictedAgentState, utility: Optional[str]):
        if not utility:
            return
        nearby = set(self._state_nearby_utilities(state))
        nearby.add(str(utility))
        state.metadata["nearby_utilities"] = sorted(nearby)

    @staticmethod
    def _compact_prerequisite_inventory(inventory: Dict[str, int]) -> Dict[str, int]:
        relevant = [
            "wood",
            "stone",
            "coal",
            "iron",
            "wood_pickaxe",
            "stone_pickaxe",
            "iron_pickaxe",
        ]
        return {
            item: int(inventory.get(item, 0))
            for item in relevant
            if int(inventory.get(item, 0)) > 0
        }

    @staticmethod
    def _has_inventory(inventory: Dict[str, int], required: Dict[str, int]) -> bool:
        return all(inventory.get(item, 0) >= amount for item, amount in required.items())

    @staticmethod
    def _consume_inventory(inventory: Dict[str, int], required: Dict[str, int]):
        for item, amount in required.items():
            inventory[item] = inventory.get(item, 0) - amount

    @staticmethod
    def _as_position(value: Any) -> Optional[tuple]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (int(value[0]), int(value[1]))
        return None

    @staticmethod
    def _manhattan(first: tuple, second: tuple) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    @staticmethod
    def _select_temporal_cluster(
        collect_events: Sequence[PredictedActionEvent],
        temporal_tolerance: int,
    ) -> tuple:
        """Select the densest predicted collect cluster using collect windows."""
        if not collect_events:
            return (0, [])
        sorted_events = sorted(collect_events, key=lambda event: (event.start_step, event.agent_id))
        best_anchor = sorted_events[0].start_step
        best_events = [sorted_events[0]]
        anchors = sorted(
            {
                step
                for event in sorted_events
                for step in (event.start_step, max(event.start_step, event.end_step - 1))
            }
        )
        for anchor in anchors:
            window_events = [
                event
                for event in sorted_events
                if (
                    event.start_step - temporal_tolerance
                    <= anchor
                    <= max(event.start_step, event.end_step - 1) + temporal_tolerance
                )
            ]
            if (
                len(window_events) > len(best_events)
                or (
                    len(window_events) == len(best_events)
                    and anchor < best_anchor
                )
            ):
                best_anchor = anchor
                best_events = window_events
        return (best_anchor, best_events)

    @staticmethod
    def _plan_has_action(plan_view: AgentPlanView, action_type: str) -> bool:
        return any(action.get("action_type") == action_type for action in plan_view.remaining_actions)

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
    def _normalize_resource_type(target: Any) -> Optional[str]:
        if target is None:
            return None
        target = str(target).lower().replace(" ", "_")
        received_to_material = {
            "wood": "tree",
            "food": "cow",
            "drink": "water",
        }
        return received_to_material.get(target, target)

    @staticmethod
    def _collect_target_for_repair(target_type: str) -> str:
        material_to_received = {
            "tree": "wood",
            "cow": "food",
            "water": "drink",
        }
        return material_to_received.get(str(target_type).lower(), str(target_type).lower())
