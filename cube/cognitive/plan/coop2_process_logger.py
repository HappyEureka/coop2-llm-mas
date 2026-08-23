"""Compact COOP2 process trace builder for CUBE."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from coop2_repair import AgentPlanView


class Coop2ProcessLogger:
    """Record step-aligned plan focus, constraints, actions, and score."""

    def __init__(
        self,
        symbolic_env: Any,
        agent_names: List[str],
        agents: Dict[str, Any],
        coop2_adapter: Any,
    ):
        self.symbolic_env = symbolic_env
        self.agent_names = agent_names
        self.agents = agents
        self.coop2_adapter = coop2_adapter
        self.records: List[Dict[str, Any]] = []
        self._last_score_event_count = 0

    def reset(self, info: Optional[Dict[str, Any]] = None) -> None:
        self.records = []
        self._last_score_event_count = 0

    def build_agent_views(self, actions: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        views = []
        specs_by_id = {
            str(spec.task_id): spec
            for spec in self.coop2_adapter.get_task_specs()
        }
        for agent_id in self.agent_names:
            agent = self.agents.get(agent_id)
            plan = getattr(agent, "plan", None) if agent is not None else None
            action = dict(actions.get(agent_id) or {"action_type": "noop"})
            current_action = plan.get_current_action() if plan is not None else None
            plan_view = self._plan_view(agent_id, plan)
            task_id = self.coop2_adapter.infer_plan_task_id(plan_view) if plan_view else None
            spec = specs_by_id.get(str(task_id)) if task_id is not None else None
            focus = spec.target_type if spec is not None else self._focus_from_action(action)
            views.append(
                {
                    "agent_id": agent_id,
                    "plan_id": getattr(plan, "plan_id", None) if plan is not None else None,
                    "specification": getattr(plan, "specification", "") if plan is not None else "",
                    "plan_status": self._enum_value(getattr(plan, "status", None)) if plan is not None else None,
                    "current_action_index": getattr(plan, "current_action_index", None) if plan is not None else None,
                    "action": action,
                    "symbolic_action": current_action.to_dict() if current_action is not None else None,
                    "task_id": task_id,
                    "resource_focus": focus,
                    "active_target_ids": [task_id] if task_id is not None else [],
                }
            )
        return views

    def record_step(
        self,
        env_step: int,
        info: Dict[str, Any],
        agent_views: List[Dict[str, Any]],
    ) -> None:
        action_outcomes = self._action_outcomes(info)
        observed_constraints = self._observed_constraints(info)
        score_summary, score_events = self._score_events()

        for view in agent_views:
            agent_id = view["agent_id"]
            view["action_outcome"] = action_outcomes.get(agent_id)
            agent = self.agents.get(agent_id)
            plan = getattr(agent, "plan", None) if agent is not None else None
            if plan is not None and getattr(plan, "plan_id", None) == view.get("plan_id"):
                view["plan_status_after_step"] = self._enum_value(getattr(plan, "status", None))
                view["current_action_index_after_step"] = getattr(plan, "current_action_index", None)
            else:
                view["plan_status_after_step"] = view.get("plan_status")
                view["current_action_index_after_step"] = view.get("current_action_index")

        record = {
            "env_step": env_step,
            "agents": agent_views,
            "active_tasks": self._active_tasks(agent_views),
            "observed_constraints": observed_constraints,
            "action_outcomes": action_outcomes,
            "score": score_summary,
            "score_events": score_events,
        }
        record["constraint_attempts"] = self._constraint_attempts(
            observed_constraints=observed_constraints,
            action_outcomes=action_outcomes,
        )
        self.records.append(record)

    def save(self, output_path: str) -> None:
        if not self.records:
            return
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self._json_safe(self.records), f, indent=2)
        print(f"Saved {len(self.records)} COOP2 process records to {output_path}")

    def _plan_view(self, agent_id: str, plan: Any) -> Optional[AgentPlanView]:
        if plan is None:
            return None
        current_index = getattr(plan, "current_action_index", 0)
        return AgentPlanView(
            agent_id=agent_id,
            plan_id=getattr(plan, "plan_id", None),
            task_id=getattr(plan, "task_id", None),
            specification=getattr(plan, "specification", ""),
            remaining_actions=[action.to_dict() for action in plan.actions[current_index:]],
            metadata={
                "plan_status": self._enum_value(getattr(plan, "status", None)),
                "current_action_index": current_index,
            },
        )

    def _active_tasks(self, agent_views: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        specs_by_id = {
            str(spec.task_id): spec
            for spec in self.coop2_adapter.get_task_specs()
        }
        agents_by_task: Dict[str, List[str]] = {}
        plan_ids_by_task: Dict[str, List[int]] = {}
        for view in agent_views:
            task_id = view.get("task_id")
            if task_id is None:
                continue
            task_id = str(task_id)
            agents_by_task.setdefault(task_id, []).append(view["agent_id"])
            if view.get("plan_id") is not None:
                plan_ids_by_task.setdefault(task_id, []).append(view["plan_id"])

        active_tasks = []
        for task_id, active_agents in sorted(agents_by_task.items()):
            spec = specs_by_id.get(task_id)
            if spec is None:
                continue
            active_tasks.append(
                {
                    "task_id": task_id,
                    "task_type": spec.task_type,
                    "resource_type": spec.target_type,
                    "target_type": spec.target_type,
                    "target_id": spec.target_id,
                    "required_agents": spec.required_agents,
                    "position": spec.metadata.get("position"),
                    "face": spec.metadata.get("face"),
                    "block_id": spec.metadata.get("block_id"),
                    "active_agents": sorted(set(active_agents)),
                    "plan_ids": sorted(set(plan_ids_by_task.get(task_id, []))),
                }
            )
        return active_tasks

    def _action_outcomes(self, info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        outcomes = {}
        for agent_id, outcome in (self.coop2_adapter.get_action_outcomes(info) or {}).items():
            outcomes[str(agent_id)] = outcome.to_dict() if hasattr(outcome, "to_dict") else dict(outcome)
        return outcomes

    def _observed_constraints(self, info: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            result.to_dict() if hasattr(result, "to_dict") else dict(result)
            for result in (self.coop2_adapter.get_constraint_results(info) or [])
        ]

    def _constraint_attempts(
        self,
        observed_constraints: List[Dict[str, Any]],
        action_outcomes: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Summarize constraints only for task ids that agents actually pushed."""
        attempted_tasks = {
            str(outcome.get("task_id"))
            for outcome in action_outcomes.values()
            if outcome.get("action_type") == "push" and outcome.get("task_id") is not None
        }
        if not attempted_tasks:
            return []

        constraints_by_task: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for result in observed_constraints:
            task_id = str(result.get("task_id"))
            if task_id not in attempted_tasks:
                continue
            constraint = str(result.get("constraint_type"))
            constraints_by_task.setdefault(task_id, {})[constraint] = result

        for outcome in action_outcomes.values():
            task_id = str(outcome.get("task_id")) if outcome.get("task_id") is not None else None
            if task_id is None or task_id in constraints_by_task:
                continue
            metadata = outcome.get("metadata") if isinstance(outcome.get("metadata"), dict) else {}
            agents = metadata.get("agents") or [outcome.get("agent_id")]
            required = max(1, int(metadata.get("required_agents") or len(agents) or 1))
            count = len([agent for agent in agents if agent is not None])
            score = min(1.0, count / required)
            constraints_by_task[task_id] = {
                "spatial": {
                    "task_id": task_id,
                    "constraint_type": "spatial",
                    "score": score,
                    "agents": agents,
                    "metadata": metadata,
                },
                "temporal": {
                    "task_id": task_id,
                    "constraint_type": "temporal",
                    "score": score,
                    "agents": agents,
                    "metadata": metadata,
                },
                "dependency": {
                    "task_id": task_id,
                    "constraint_type": "dependency",
                    "score": 1.0,
                    "agents": agents,
                    "metadata": metadata,
                },
            }

        attempts = []
        for task_id, constraints in sorted(constraints_by_task.items()):
            metadata = {}
            for result in constraints.values():
                if isinstance(result.get("metadata"), dict):
                    metadata.update(result["metadata"])
            attempts.append(
                {
                    "task_id": task_id,
                    "task_type": "push",
                    "target_type": metadata.get("target_type"),
                    "resource_type": metadata.get("resource_type"),
                    "required_agents": metadata.get("required_agents", 1),
                    "block_id": metadata.get("block_id"),
                    "face": metadata.get("face"),
                    "scores": {
                        name: float(result.get("score", 0.0))
                        for name, result in constraints.items()
                    },
                    "violations": {
                        name: 1.0 - min(1.0, float(result.get("score", 0.0)))
                        for name, result in constraints.items()
                    },
                    "agents": sorted(
                        {
                            agent
                            for result in constraints.values()
                            for agent in result.get("agents", [])
                        }
                    ),
                }
            )
        return attempts

    def _score_events(self):
        base_env = getattr(self.symbolic_env, "env", self.symbolic_env)
        score_summary = {}
        events: List[Dict[str, Any]] = []
        if hasattr(base_env, "get_team_score_breakdown"):
            score_summary = base_env.get_team_score_breakdown(include_events=True) or {}
            all_events = score_summary.get("events") or []
            events = [
                dict(event)
                for event in all_events[self._last_score_event_count:]
                if isinstance(event, dict)
            ]
            self._last_score_event_count = len(all_events)
            score_summary = dict(score_summary)
            score_summary.pop("events", None)
        return score_summary, events

    @staticmethod
    def _focus_from_action(action: Dict[str, Any]) -> str:
        if action.get("action_type") == "push":
            return "block"
        return str(action.get("action_type") or "other")

    @staticmethod
    def _enum_value(value: Any) -> Any:
        return getattr(value, "value", value)

    def _json_safe(self, value: Any) -> Any:
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
