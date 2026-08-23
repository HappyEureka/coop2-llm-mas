"""MA-Crafter-specific COOP2 repair guidance helpers."""

from typing import Any, Dict, List, Optional, Sequence

from .core import AgentPlanView, ConstraintType, TaskSpec


class MacrafterRepairGuidanceMixin:
    """Build soft repair guidance and prerequisite hints for MA-Crafter."""

    def build_repair_guidance(
        self,
        evaluation: Any,
        affected_agents: Sequence[str],
        env_step: int,
    ) -> Dict[str, Any]:
        """Build soft, actionable MA-Crafter repair guidance for LLM agents."""
        prediction = (evaluation.metadata or {}).get("prediction") or {}
        initial_states = prediction.get("initial_states") or {}
        task_specs = list(self.get_task_specs())
        task_specs_by_id = {str(spec.task_id): spec for spec in task_specs}

        failed_task_ids = sorted({
            str(failure.task_id)
            for failure in evaluation.failures
            if failure.task_id is not None
        })
        failed_target_types = sorted({
            task_specs_by_id[task_id].target_type
            for task_id in failed_task_ids
            if task_id in task_specs_by_id
        })

        guidance = {
            "strategy": "soft_sync",
            "instructions": [
                "Pick one reachable shared target, then all required agents should navigate directly to that same item_id.",
                "Do not collect alone. Nearby agents only count if they also issue collect in the same tolerated window.",
                "If you arrive early and a partner is still navigating, moving, or nooping, add another noop before collect.",
                "Prefer waiting slightly too long over collecting while a required partner is still on a non-collect action.",
                "If the predicted arrival gap is large, switch to a closer shared target instead of trying to hit an exact primitive step.",
                "Prefer a good-enough collect window where all required agents are nearby within the configured tolerance.",
            ],
            "task_hints": self._repair_task_hints(evaluation),
        }
        dependency_hints = self._repair_dependency_hints(evaluation, initial_states, task_specs_by_id)
        has_dependency_failure = bool(dependency_hints)
        if dependency_hints:
            guidance["strategy"] = "dependency_first"
            guidance["instructions"] = [
                "A dependency failure means at least one needed participant lacks a prerequisite.",
                "Ready agents may navigate toward the shared target and wait; missing agents should repair the prerequisite first.",
                "Do not repeat the exact failed collect-only plan until the missing tool/resource prerequisite is addressed.",
            ] + guidance["instructions"]
            guidance["dependency_prerequisites"] = dependency_hints

        required_tools_by_task = self._repair_required_tools_by_task(evaluation, task_specs_by_id)
        plan_views_by_agent = {
            str(view.agent_id): view
            for view in getattr(evaluation, "plan_views", []) or []
        }
        candidates = self._repair_shared_target_candidates(
            task_specs=task_specs,
            affected_agents=affected_agents,
            initial_states=initial_states,
            target_types=failed_target_types,
            plan_views_by_agent=plan_views_by_agent,
            required_tools_by_task=required_tools_by_task,
            limit=5,
        )
        if candidates:
            guidance["recommended_target"] = candidates[0]
            guidance["recommended_plans"] = self._repair_recommended_plan_skeletons(candidates[0])
            guidance["recommendation_policy"] = (
                "Use the recommended plan skeleton unless current observation or memory "
                "shows it is infeasible; if deviating, keep the same repair goal and explain why."
            )
        if candidates and not has_dependency_failure:
            guidance["reachable_shared_targets"] = candidates
        elif candidates:
            guidance["blocked_targets_after_prerequisites"] = candidates[:3]

        return guidance

    def _repair_recommended_plan_skeletons(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Return per-agent suggested symbolic actions without enforcing them."""
        target_type = str(candidate.get("target_type") or "")
        target_id = candidate.get("target_id") or candidate.get("task_id")
        if target_id is None:
            return {}
        try:
            action_item_id = int(target_id)
        except (TypeError, ValueError):
            action_item_id = str(target_id)
        try:
            recommended_timeout = int(candidate.get("recommended_timeout") or 10)
        except (TypeError, ValueError):
            recommended_timeout = 10
        try:
            distance_threshold = int(candidate.get("distance_threshold", 1))
        except (TypeError, ValueError):
            distance_threshold = 1
        steps_by_agent = candidate.get("estimated_steps_by_recommended_agent") or {}
        try:
            latest_arrival = max(int(value) for value in steps_by_agent.values())
        except (TypeError, ValueError):
            latest_arrival = 0

        readiness_by_agent = candidate.get("participant_readiness") or {}
        all_collect_ready = bool(
            candidate.get("all_recommended_participants_ready_after_stage", True)
        )
        required_tools = candidate.get("required_tool") or []
        collect_ready_now = bool(candidate.get("collect_ready_now", all_collect_ready))
        plans = {}
        for agent_id in candidate.get("recommended_participants") or []:
            readiness = readiness_by_agent.get(str(agent_id), {})
            prerequisite_actions = list(readiness.get("prerequisite_actions") or [])
            has_required_now = bool(readiness.get("has_required_tool", False) or not required_tools)
            try:
                own_arrival = int(steps_by_agent.get(agent_id, latest_arrival))
            except (TypeError, ValueError):
                own_arrival = latest_arrival
            wait_steps = max(1, min(4, latest_arrival - own_arrival + 1))
            navigate_and_wait = [
                {
                    "action_type": "navigate",
                    "args": {
                        "object_type": target_type,
                        "item_id": action_item_id,
                        "timeout": max(recommended_timeout, own_arrival + 5, 5),
                        "radius": max(1, distance_threshold),
                    },
                },
            ]
            navigate_and_wait.extend(
                {"action_type": "noop", "args": {}}
                for _ in range(wait_steps)
            )

            if collect_ready_now and has_required_now:
                actions = prerequisite_actions + navigate_and_wait
                collect_args = {
                    "target": self._collect_target_for_repair(target_type),
                    "resource_type": target_type,
                    "task_id": target_id,
                    "target_id": target_id,
                    "item_id": action_item_id,
                    "steps": max(3, self.get_prediction_temporal_tolerance() + 1),
                }
                actions.append({
                    "action_type": "collect",
                    "args": collect_args,
                })
                task = (
                    f"collect_{self._collect_target_for_repair(target_type)}"
                    f"({target_type}#{target_id})"
                )
                rationale = (
                    f"Predicted reachable shared {target_type} target for selected "
                    "participants that already satisfy the required tool dependency."
                )
            elif has_required_now:
                actions = prerequisite_actions + navigate_and_wait
                task = f"wait_for_{target_type}_partner({target_type}#{target_id})"
                rationale = (
                    f"Get ready near shared {target_type} target, but do not collect "
                    "until enough required partners already have the required tool."
                )
            else:
                actions = prerequisite_actions or [{"action_type": "noop", "args": {}}]
                missing_tools = readiness.get("missing_required_tools") or required_tools
                tool_label = "_".join(str(tool) for tool in missing_tools) or "prerequisite"
                task = (
                    f"prepare_{tool_label}_for_"
                    f"{self._collect_target_for_repair(target_type)}({target_type}#{target_id})"
                )
                rationale = (
                    "Repair the missing prerequisite first; this skeleton intentionally "
                    "does not include the cooperative collect yet."
                )

            plans[str(agent_id)] = {
                "task": task,
                "actions": actions,
                "rationale": rationale,
            }
        return plans

    def _repair_task_hints(self, evaluation: Any) -> List[Dict[str, Any]]:
        hints_by_task: Dict[str, Dict[str, Any]] = {}
        for failure in evaluation.failures:
            task_id = str(failure.task_id)
            hint = hints_by_task.setdefault(
                task_id,
                {
                    "task_id": task_id,
                    "constraints": set(),
                    "agents": set(),
                    "collect_steps_by_agent": {},
                    "advice": [],
                },
            )
            hint["constraints"].add(failure.constraint_type.value)
            hint["agents"].update(failure.agents)
            collect_steps = failure.metadata.get("collect_steps")
            if isinstance(collect_steps, dict):
                for step, agents in collect_steps.items():
                    for agent_id in agents:
                        hint["collect_steps_by_agent"][agent_id] = int(step)

            if failure.constraint_type == ConstraintType.TEMPORAL:
                hint["advice"].append("early collectors should wait; late collectors should navigate directly")
            elif failure.constraint_type == ConstraintType.SPATIAL:
                hint["advice"].append("all collectors should be near the same object before collect")
            elif failure.constraint_type == ConstraintType.DEPENDENCY:
                hint["advice"].append("repair missing tools/resources before retrying collection")

        task_hints = []
        for hint in hints_by_task.values():
            collect_steps = hint["collect_steps_by_agent"]
            if collect_steps:
                steps = list(collect_steps.values())
                hint["predicted_collect_spread"] = max(steps) - min(steps)
                if hint["predicted_collect_spread"] > self.get_prediction_temporal_tolerance():
                    hint["advice"].append(
                        "arrival gap is large; choose a closer shared target or wait before collecting"
                    )
            hint["constraints"] = sorted(hint["constraints"])
            hint["agents"] = sorted(hint["agents"])
            hint["advice"] = sorted(set(hint["advice"]))
            task_hints.append(hint)
        return task_hints

    def _repair_dependency_hints(
        self,
        evaluation: Any,
        initial_states: Dict[str, Any],
        task_specs_by_id: Dict[str, TaskSpec],
    ) -> List[Dict[str, Any]]:
        """Describe missing prerequisite steps for dependency failures."""
        from macrafter.cooperative_tasks import (
            normalize_tool_requirement,
            satisfies_tool_requirement,
        )

        hints = []
        for failure in evaluation.failures:
            if failure.constraint_type != ConstraintType.DEPENDENCY:
                continue
            task_id = str(failure.task_id)
            task_spec = task_specs_by_id.get(task_id)
            required_tool = (
                failure.metadata.get("required_tool")
                if isinstance(failure.metadata, dict)
                else None
            )
            if required_tool is None and task_spec is not None:
                required_tool = list(task_spec.required_capabilities)
            required_tools = normalize_tool_requirement(required_tool)
            required_tool_mode = (
                failure.metadata.get("required_tool_mode", "all")
                if isinstance(failure.metadata, dict)
                else "all"
            )
            required_agents = int(
                (task_spec.required_agents if task_spec is not None else 1) or 1
            )

            plan_views_by_agent = {
                str(view.agent_id): view
                for view in getattr(evaluation, "plan_views", []) or []
            }
            candidate_agent_ids = sorted(
                set(plan_views_by_agent) | {str(agent_id) for agent_id in failure.agents}
            )

            agent_hints = []
            for agent_id in candidate_agent_ids:
                state = initial_states.get(agent_id) or self.get_prediction_initial_state(agent_id)
                inventory = self._state_inventory(state)
                has_tool = satisfies_tool_requirement(
                    inventory,
                    required_tool,
                    required_tool_mode,
                )
                planned_tool = self._plan_acquires_tools_before_collect(
                    plan_views_by_agent[agent_id],
                    required_tools,
                ) if agent_id in plan_views_by_agent else False
                agent_hints.append({
                    "agent_id": agent_id,
                    "has_required_tool": bool(has_tool),
                    "planned_acquires_required_tool": bool(planned_tool),
                    "can_participate_after_current_plan": bool(has_tool or planned_tool),
                    "inventory": self._compact_prerequisite_inventory(inventory),
                    "nearby_utilities": self._state_nearby_utilities(state),
                    "suggested_next_step": (
                        "navigate to the shared target and collect with partners"
                        if has_tool
                        else "finish current prerequisite plan, then join the shared target"
                        if planned_tool
                        else self._suggest_dependency_next_step(required_tools, inventory, state)
                    ),
                })

            hints.append({
                "task_id": task_id,
                "target_type": task_spec.target_type if task_spec is not None else None,
                "required_agents": required_agents,
                "required_tool": required_tools,
                "required_tool_mode": required_tool_mode,
                "summary": (
                    f"Need {required_agents} agent(s) with {required_tools} before "
                    "retrying the coordinated collect."
                ),
                "agent_prerequisites": agent_hints,
            })
        return hints

    def _repair_required_tools_by_task(
        self,
        evaluation: Any,
        task_specs_by_id: Dict[str, TaskSpec],
    ) -> Dict[str, Dict[str, Any]]:
        """Return task-level tool requirements observed in failed constraints."""
        from macrafter.cooperative_tasks import normalize_tool_requirement

        requirements: Dict[str, Dict[str, Any]] = {}
        for failure in getattr(evaluation, "failures", []) or []:
            if failure.constraint_type != ConstraintType.DEPENDENCY:
                continue
            task_id = str(failure.task_id)
            task_spec = task_specs_by_id.get(task_id)
            required_tool = (
                failure.metadata.get("required_tool")
                if isinstance(failure.metadata, dict)
                else None
            )
            if required_tool is None and task_spec is not None:
                required_tool = list(task_spec.required_capabilities)
            required_tool_mode = (
                failure.metadata.get("required_tool_mode", "all")
                if isinstance(failure.metadata, dict)
                else "all"
            )
            requirements[task_id] = {
                "required_tool": normalize_tool_requirement(required_tool),
                "required_tool_mode": required_tool_mode,
            }
        return requirements

    def _suggest_dependency_next_step(
        self,
        required_tools: Sequence[str],
        inventory: Dict[str, int],
        state: Any,
    ) -> str:
        """Return a short prerequisite action recommendation for one agent."""
        missing = [tool for tool in required_tools if inventory.get(tool, 0) <= 0]
        if not missing:
            return "navigate to the shared target and collect with partners"
        tool = missing[0]
        nearby = set(self._state_nearby_utilities(state))

        if tool == "wood_pickaxe":
            if "table" in nearby:
                if inventory.get("wood", 0) >= 1:
                    return "craft wood_pickaxe near the table"
                return "collect wood, then return to the table and craft wood_pickaxe"
            if inventory.get("wood", 0) >= 2:
                return "place_table, then craft wood_pickaxe after collecting one extra wood if needed"
            return "collect wood until you have enough to place/use a table and craft wood_pickaxe"

        if tool == "stone_pickaxe":
            if inventory.get("wood_pickaxe", 0) <= 0:
                return "first complete the wood_pickaxe prerequisite"
            if inventory.get("stone", 0) < 1:
                return "coordinate stone collection with wood_pickaxe partners"
            if inventory.get("wood", 0) < 1:
                return "collect one wood for the stone_pickaxe recipe"
            if "table" not in nearby:
                if inventory.get("wood", 0) >= 2:
                    return "place_table, then craft stone_pickaxe"
                return "navigate to a known table or collect enough wood to place one"
            return "craft stone_pickaxe near the table"

        if tool == "iron_pickaxe":
            if inventory.get("stone_pickaxe", 0) <= 0:
                return "first complete the stone_pickaxe prerequisite"
            if inventory.get("iron", 0) < 1:
                return "coordinate iron collection with stone_pickaxe partners"
            if inventory.get("coal", 0) < 1:
                return "coordinate coal collection with wood_pickaxe partners"
            if inventory.get("wood", 0) < 1:
                return "collect one wood for the iron_pickaxe recipe"
            if "furnace" not in nearby:
                if inventory.get("stone", 0) >= 4:
                    return "place_furnace, then craft iron_pickaxe"
                return "collect enough stone to place a furnace"
            return "craft iron_pickaxe near the furnace"

        return f"acquire {tool} before retrying the collect"

    def _repair_shared_target_candidates(
        self,
        task_specs: Sequence[TaskSpec],
        affected_agents: Sequence[str],
        initial_states: Dict[str, Any],
        target_types: Sequence[str],
        plan_views_by_agent: Optional[Dict[str, AgentPlanView]] = None,
        required_tools_by_task: Optional[Dict[str, Dict[str, Any]]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        from macrafter.cooperative_tasks import normalize_tool_requirement

        plan_views_by_agent = plan_views_by_agent or {}
        required_tools_by_task = required_tools_by_task or {}
        target_type_set = set(target_types)
        candidates = []
        for spec in task_specs:
            if target_type_set and spec.target_type not in target_type_set:
                continue
            if int(spec.required_agents) > len(affected_agents):
                continue
            required_agents = max(1, int(spec.required_agents))
            position = self._as_position(spec.metadata.get("position"))
            if position is None:
                continue
            distance_threshold = int(spec.metadata.get("distance_threshold", 1))
            requirement = required_tools_by_task.get(str(spec.task_id), {})
            required_tool = requirement.get("required_tool", list(spec.required_capabilities))
            required_tools = normalize_tool_requirement(required_tool)
            required_tool_mode = requirement.get(
                "required_tool_mode",
                spec.metadata.get("required_tool_mode", "all"),
            )
            steps_by_agent = {}
            readiness_by_agent = {}
            for agent_id in affected_agents:
                state = initial_states.get(agent_id) or self.get_prediction_initial_state(agent_id)
                agent_pos = self._as_position(state.get("position") if isinstance(state, dict) else None)
                if agent_pos is None:
                    continue
                navigation_steps = self._navigation_steps_to_target(
                    agent_id,
                    position,
                    distance_threshold,
                )
                if navigation_steps is None and self._navigation_reachability_available():
                    continue
                steps_by_agent[agent_id] = (
                    navigation_steps
                    if navigation_steps is not None
                    else max(0, self._manhattan(agent_pos, position) - distance_threshold)
                )
                readiness = self._repair_agent_tool_readiness(
                    agent_id=str(agent_id),
                    required_tools=required_tools,
                    required_tool_mode=required_tool_mode,
                    state=state,
                    plan_view=plan_views_by_agent.get(str(agent_id)),
                )
                readiness["navigation_steps"] = steps_by_agent[agent_id]
                readiness["estimated_steps_to_collect"] = (
                    steps_by_agent[agent_id]
                    + int(readiness.get("prerequisite_step_count", 0))
                )
                readiness_by_agent[str(agent_id)] = readiness
            if not steps_by_agent:
                continue
            ready_options = [
                item
                for item in readiness_by_agent.items()
                if item[1].get("has_required_tool") or not required_tools
            ]
            ready_participants = sorted(agent_id for agent_id, _readiness in ready_options)
            collect_ready_now = len(ready_options) >= required_agents
            if collect_ready_now:
                participant_options = sorted(
                    ready_options,
                    key=lambda item: (
                        int(item[1].get("navigation_steps", 9999)),
                        item[0],
                    ),
                )
                dependency_stage = "collect_ready"
            else:
                participant_options = sorted(
                    readiness_by_agent.items(),
                    key=lambda item: (
                        int(item[1].get("readiness_rank", 2)),
                        int(item[1].get(
                            "estimated_steps_to_collect",
                            item[1].get("navigation_steps", 9999),
                        )),
                        item[0],
                    ),
                )
                dependency_stage = "prepare_tools" if required_tools else "collect_ready"
            participant_steps = {
                agent_id: int(readiness.get("estimated_steps_to_collect", 0))
                for agent_id, readiness in participant_options[:required_agents]
            }
            if len(participant_steps) < required_agents:
                continue
            max_steps = max(participant_steps.values())
            min_steps = min(participant_steps.values())
            spread = max_steps - min_steps
            participant_readiness = {
                agent_id: readiness_by_agent[agent_id]
                for agent_id in participant_steps
            }
            all_ready_after_stage = all(
                readiness.get("has_required_tool")
                or readiness.get("will_acquire_required_tool")
                or not required_tools
                for readiness in participant_readiness.values()
            )
            candidates.append({
                "task_id": str(spec.task_id),
                "target_id": str(spec.target_id or spec.task_id),
                "target_type": spec.target_type,
                "position": list(position),
                "distance_threshold": distance_threshold,
                "required_agents": required_agents,
                "required_tool": required_tools,
                "required_tool_mode": required_tool_mode,
                "recommended_participants": sorted(participant_steps),
                "estimated_steps_by_recommended_agent": participant_steps,
                "estimated_steps_by_agent": steps_by_agent,
                "max_estimated_steps": max_steps,
                "recommended_timeout": max_steps + 5,
                "arrival_spread": spread,
                "reachability": self._repair_reachability_label(max_steps, spread),
                "participant_readiness": participant_readiness,
                "all_recommended_participants_ready_after_stage": all_ready_after_stage,
                "collect_ready_now": collect_ready_now,
                "dependency_stage": dependency_stage,
                "ready_participants": ready_participants,
                "tool_ready_participant_count": len(ready_participants),
                "missing_prerequisite_agents": sorted(
                    agent_id
                    for agent_id, readiness in participant_readiness.items()
                    if not (
                        readiness.get("has_required_tool")
                        or readiness.get("will_acquire_required_tool")
                        or not required_tools
                    )
                ),
            })

        candidates.sort(
            key=lambda candidate: (
                0 if candidate.get("collect_ready_now") else 1,
                candidate["max_estimated_steps"],
                candidate["arrival_spread"],
                candidate["task_id"],
            )
        )
        return candidates[:limit]

    def _repair_agent_tool_readiness(
        self,
        agent_id: str,
        required_tools: Sequence[str],
        required_tool_mode: str,
        state: Any,
        plan_view: Optional[AgentPlanView] = None,
    ) -> Dict[str, Any]:
        from macrafter.cooperative_tasks import satisfies_tool_requirement

        inventory = self._state_inventory(state)
        required_tools = list(required_tools or [])
        has_tool = (
            True
            if not required_tools
            else satisfies_tool_requirement(inventory, required_tools, required_tool_mode)
        )
        planned_tool = (
            bool(self._plan_acquires_tools_before_collect(plan_view, required_tools))
            if plan_view is not None
            else False
        )
        if has_tool:
            prerequisite_actions = []
            will_acquire = False
        else:
            prerequisite_actions, will_acquire = self._repair_prerequisite_action_skeleton(
                required_tools,
                required_tool_mode,
                inventory,
                state,
            )
        missing_tools = [
            tool
            for tool in required_tools
            if int(inventory.get(tool, 0)) <= 0
        ]
        readiness_rank = 0 if has_tool else 1 if will_acquire or planned_tool else 2
        return {
            "agent_id": agent_id,
            "has_required_tool": bool(has_tool),
            "planned_acquires_required_tool": bool(planned_tool),
            "will_acquire_required_tool": bool(will_acquire),
            "missing_required_tools": missing_tools,
            "prerequisite_actions": prerequisite_actions,
            "prerequisite_step_count": len(prerequisite_actions),
            "readiness_rank": readiness_rank,
            "inventory": self._compact_prerequisite_inventory(inventory),
            "nearby_utilities": self._state_nearby_utilities(state),
        }

    def _repair_prerequisite_action_skeleton(
        self,
        required_tools: Sequence[str],
        required_tool_mode: str,
        inventory: Dict[str, int],
        state: Any,
    ) -> tuple[List[Dict[str, Any]], bool]:
        """Build prerequisite-only actions and report whether they acquire the tool."""
        if not required_tools:
            return ([], True)
        if required_tool_mode == "any" and any(inventory.get(tool, 0) > 0 for tool in required_tools):
            return ([], True)
        if required_tool_mode != "any" and all(inventory.get(tool, 0) > 0 for tool in required_tools):
            return ([], True)

        working_inventory = dict(inventory or {})
        nearby_utilities = set(self._state_nearby_utilities(state))
        actions: List[Dict[str, Any]] = []
        acquired = set()

        for tool in required_tools:
            if working_inventory.get(tool, 0) > 0:
                acquired.add(tool)
                continue
            tool_actions, acquired_tool = self._repair_actions_to_acquire_tool(
                tool,
                working_inventory,
                nearby_utilities,
            )
            actions.extend(tool_actions)
            if acquired_tool:
                acquired.add(tool)

        if required_tool_mode == "any":
            return (actions, bool(acquired))
        return (actions, set(required_tools).issubset(acquired))

    def _repair_actions_to_acquire_tool(
        self,
        tool: str,
        inventory: Dict[str, int],
        nearby_utilities: set,
    ) -> tuple[List[Dict[str, Any]], bool]:
        from macrafter import constants

        make_info = constants.make.get(tool)
        if make_info is None:
            return ([], False)

        actions: List[Dict[str, Any]] = []
        uses = dict(make_info.get("uses", {}) or {})
        required_utilities = [str(item) for item in make_info.get("nearby", []) or []]
        missing_utilities = [
            utility
            for utility in required_utilities
            if utility not in nearby_utilities
        ]

        if missing_utilities:
            utility = missing_utilities[0]
            place_info = constants.place.get(utility)
            place_uses = dict((place_info or {}).get("uses", {}) or {})
            combined_uses = dict(uses)
            for item, amount in place_uses.items():
                combined_uses[item] = combined_uses.get(item, 0) + amount

            if place_info is not None and self._has_inventory(inventory, combined_uses):
                actions.append({"action_type": "place", "args": {"object_type": utility}})
                self._consume_inventory(inventory, place_uses)
                nearby_utilities.add(utility)
            else:
                missing_resource = self._first_missing_resource(inventory, combined_uses or place_uses)
                return ([self._repair_collect_prerequisite_action(missing_resource)], False)

        if not self._has_inventory(inventory, uses):
            missing_resource = self._first_missing_resource(inventory, uses)
            return (actions + [self._repair_collect_prerequisite_action(missing_resource)], False)

        actions.append({"action_type": "craft", "args": {"object_type": tool}})
        self._consume_inventory(inventory, uses)
        inventory[tool] = inventory.get(tool, 0) + int(make_info.get("gives", 1))
        return (actions, True)

    @staticmethod
    def _first_missing_resource(inventory: Dict[str, int], required: Dict[str, int]) -> str:
        for item, amount in required.items():
            if inventory.get(item, 0) < amount:
                return str(item)
        return "wood"

    @staticmethod
    def _repair_collect_prerequisite_action(resource: str) -> Dict[str, Any]:
        return {
            "action_type": "collect",
            "args": {"target": str(resource or "wood")},
        }

    def _repair_reachability_label(self, max_steps: int, spread: int) -> str:
        if max_steps <= 6 and spread <= self.get_prediction_temporal_tolerance() + 2:
            return "good"
        if max_steps <= 12:
            return "usable"
        return "far"

    def _navigation_reachability_available(self) -> bool:
        symbolic_world = getattr(self._base_env(), "symbolic_world", None)
        return getattr(symbolic_world, "symbolic_matrix", None) is not None

    def _navigation_steps_to_target(
        self,
        agent_id: str,
        target_pos: tuple,
        distance_threshold: int,
    ) -> Optional[int]:
        symbolic_world = getattr(self._base_env(), "symbolic_world", None)
        if getattr(symbolic_world, "symbolic_matrix", None) is None:
            return None
        try:
            from cognitive.action.navigation import shortest_path_to_target_zone
            numeric_id = str(agent_id).split("_")[-1]
            agent_pos = None
            for props in symbolic_world.agent_properties.values():
                if str(props.get("name")) == numeric_id:
                    agent_pos = tuple(props.get("position"))
                    break
            if agent_pos is None:
                return None
            path = shortest_path_to_target_zone(
                symbolic_world,
                agent_pos,
                target_pos,
                radius=max(0, int(distance_threshold)),
            )
        except Exception:
            return None
        if path is None:
            return None
        return max(0, len(path) - 1)

