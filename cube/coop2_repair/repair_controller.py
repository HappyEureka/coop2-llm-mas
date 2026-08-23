"""COOP2 pre-execution repair controller.

The controller owns the paper-method repair gate: it snapshots committed plans,
predicts cooperative constraint failures before primitive execution, and sends
structured repair requests when plans need coordination. It deliberately keeps
environment-specific reasoning inside adapters.
"""

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .core import AgentPlanView, Coop2TraceLogger
from .evaluator import PreExecutionConstraintEvaluator, PreExecutionEvaluation
from .message_protocol import (
    COOP2_REPAIR_CONTENT_TYPE,
    COOP2_REPAIR_SENDER_ID,
    coop2_repair_metadata,
)


class Coop2RepairController:
    """Run optional pre-execution checks and dispatch repair requests."""

    def __init__(
        self,
        adapter: Any,
        trace_logger: Optional[Coop2TraceLogger] = None,
        enabled: bool = False,
        task_cooldown_steps: Optional[int] = None,
    ):
        self.adapter = adapter
        self.trace_logger = trace_logger
        self.enabled = bool(enabled)
        self.task_cooldown_steps = task_cooldown_steps
        self.evaluator = (
            PreExecutionConstraintEvaluator(adapter)
            if adapter is not None
            else None
        )
        self._last_precheck_signature = None
        self._last_evaluation: Optional[PreExecutionEvaluation] = None
        self._last_repair_step_by_task: Dict[str, int] = {}

    def set_enabled(self, enabled: bool):
        """Enable or disable pre-execution repair checks."""
        self.enabled = bool(enabled)
        self.reset()

    def enable(self):
        """Enable pre-execution repair checks."""
        self.set_enabled(True)

    def disable(self):
        """Disable pre-execution repair checks."""
        self.set_enabled(False)

    def set_adapter(self, adapter: Any):
        """Swap the adapter and rebuild the evaluator."""
        self.adapter = adapter
        self.evaluator = (
            PreExecutionConstraintEvaluator(adapter)
            if adapter is not None
            else None
        )
        self.reset()

    def reset(self):
        """Clear cached evaluation state."""
        self._last_precheck_signature = None
        self._last_evaluation = None
        self._last_repair_step_by_task = {}

    def before_execution(
        self,
        env_step: int,
        agents: Optional[Dict[str, Any]] = None,
        current_info: Optional[Dict[str, Any]] = None,
        message_broker: Optional[Any] = None,
        plan_views: Optional[Sequence[AgentPlanView]] = None,
        repair_dispatcher: Optional[Any] = None,
    ) -> Optional[PreExecutionEvaluation]:
        """Evaluate committed plans and interrupt agents if repair is needed."""
        if not self.enabled or self.evaluator is None:
            return None

        agents = agents or {}
        views = list(plan_views) if plan_views is not None else self.build_plan_views(agents)
        if not views:
            return None
        if not any(self._plan_status(view) == "pending" for view in views):
            return None

        signature = self._precheck_signature(views)
        if signature == self._last_precheck_signature:
            if self._last_evaluation is not None and self._last_evaluation.should_repair:
                if self._task_cooldown_active(env_step, self._last_evaluation):
                    return None
                self._begin_repair(
                    evaluation=self._last_evaluation,
                    env_step=env_step,
                    message_broker=message_broker,
                    repair_dispatcher=repair_dispatcher,
                )
            return self._last_evaluation

        evaluation = self.evaluator.evaluate(
            views,
            env_step=env_step,
            infos=current_info,
        )
        self._last_precheck_signature = signature
        self._last_evaluation = evaluation
        self._log_pre_execution_prediction(evaluation, env_step)

        if not evaluation.should_repair:
            if evaluation.results:
                self._log(
                    "pre_execution_check_passed",
                    env_step=env_step,
                    evaluated_tasks=evaluation.metadata.get("tasks_evaluated", []),
                )
            return evaluation

        for failure in evaluation.failures:
            self._log(
                "constraint_predicted_failure",
                env_step=env_step,
                task_id=failure.task_id,
                constraint=failure.to_dict(),
            )
        if self._task_cooldown_active(env_step, evaluation):
            return None
        self._begin_repair(
            evaluation=evaluation,
            env_step=env_step,
            message_broker=message_broker,
            repair_dispatcher=repair_dispatcher,
        )
        return evaluation

    def build_plan_views(self, agents: Dict[str, Any]) -> List[AgentPlanView]:
        """Build serializable committed-plan views for COOP2 evaluation."""
        plan_views = []
        for agent_id, agent in agents.items():
            plan = getattr(agent, "plan", None)
            if agent is None or not getattr(agent, "ready", False) or plan is None:
                continue
            plan_status = self._enum_value(getattr(plan, "status", None))
            if plan_status not in {"pending", "executing"}:
                continue
            current_action_index = getattr(plan, "current_action_index", 0)
            remaining_actions = [
                action.to_dict()
                for action in plan.actions[current_action_index:]
            ]
            agent_state = self._enum_value(getattr(agent, "state", None))
            plan_views.append(
                AgentPlanView(
                    agent_id=agent_id,
                    plan_id=getattr(plan, "plan_id", None),
                    task_id=getattr(plan, "task_id", None),
                    specification=getattr(plan, "specification", ""),
                    remaining_actions=remaining_actions,
                    metadata={
                        "plan_status": plan_status,
                        "current_action_index": current_action_index,
                        "agent_state": agent_state,
                    },
                )
            )
        return plan_views

    def _enum_value(self, value: Any) -> Any:
        return getattr(value, "value", value)

    def _plan_status(self, view: AgentPlanView) -> str:
        return view.metadata.get("plan_status", "pending")

    def _precheck_signature(
        self,
        plan_views: Sequence[AgentPlanView],
    ) -> Tuple[Any, ...]:
        return tuple(
            sorted(
                (
                    view.agent_id,
                    view.plan_id,
                    view.task_id,
                    view.metadata.get("plan_status"),
                    view.metadata.get("current_action_index"),
                    tuple(
                        (
                            action.get("action_type"),
                            self._normalized_args(action.get("args") or {}),
                        )
                        for action in view.remaining_actions
                    ),
                )
                for view in plan_views
            )
        )

    def _configured_task_cooldown_steps(self) -> int:
        if self.task_cooldown_steps is not None:
            return max(0, int(self.task_cooldown_steps))
        get_repair_config = getattr(self.adapter, "get_repair_config", None)
        if get_repair_config is None:
            return 0
        config = get_repair_config() or {}
        return max(0, int(config.get("task_cooldown_steps", 0)))

    def _configured_cooldown_scope(self) -> str:
        get_repair_config = getattr(self.adapter, "get_repair_config", None)
        if get_repair_config is None:
            return "task"
        config = get_repair_config() or {}
        scope = str(config.get("cooldown_scope", "task")).strip().lower()
        if scope in {"resource", "resource_type", "target"}:
            return "target_type"
        if scope in {"task_type", "target_type", "task"}:
            return scope
        return "task"

    def _task_cooldown_active(
        self,
        env_step: int,
        evaluation: PreExecutionEvaluation,
    ) -> bool:
        """Return True when every failed task is still in repair cooldown."""
        cooldown_steps = self._configured_task_cooldown_steps()
        cooldown_scope = self._configured_cooldown_scope()
        cooldown_keys = self._failure_cooldown_keys(evaluation, cooldown_scope)
        if cooldown_steps <= 0 or not cooldown_keys:
            return False

        cooled_keys = {}
        eligible_keys = []
        for cooldown_key in cooldown_keys:
            last_step = self._last_repair_step_by_task.get(cooldown_key)
            if last_step is None or env_step - last_step >= cooldown_steps:
                eligible_keys.append(cooldown_key)
            else:
                cooled_keys[cooldown_key] = {
                    "last_repair_step": last_step,
                    "remaining_steps": cooldown_steps - (env_step - last_step),
                }

        if not eligible_keys:
            self._log(
                "repair_task_cooldown",
                env_step=env_step,
                cooldown_scope=cooldown_scope,
                cooldown_keys=cooldown_keys,
                cooldown_steps=cooldown_steps,
                cooled_keys=cooled_keys,
                evaluation=evaluation.to_dict(),
            )
            return True

        for cooldown_key in eligible_keys:
            self._last_repair_step_by_task[cooldown_key] = env_step
        return False

    def _failure_cooldown_keys(
        self,
        evaluation: PreExecutionEvaluation,
        cooldown_scope: str,
    ) -> List[str]:
        task_specs_by_id = self._task_specs_by_id()
        cooldown_keys = []
        for failure in evaluation.failures:
            task_id = str(failure.task_id) if failure.task_id is not None else "__unknown_task__"
            task_spec = task_specs_by_id.get(task_id)
            metadata = failure.metadata if isinstance(failure.metadata, dict) else {}
            if cooldown_scope == "target_type":
                target_type = (
                    metadata.get("resource_type")
                    or metadata.get("target_type")
                    or getattr(task_spec, "target_type", None)
                )
                cooldown_keys.append(
                    f"target_type:{target_type}" if target_type else f"task:{task_id}"
                )
            elif cooldown_scope == "task_type":
                task_type = (
                    metadata.get("task_type")
                    or getattr(task_spec, "task_type", None)
                )
                cooldown_keys.append(
                    f"task_type:{task_type}" if task_type else f"task:{task_id}"
                )
            else:
                cooldown_keys.append(f"task:{task_id}")

        return sorted(set(cooldown_keys)) or ["task:__unknown_task__"]

    def _log_pre_execution_prediction(
        self,
        evaluation: PreExecutionEvaluation,
        env_step: int,
    ) -> None:
        """Record full predicted constraint results for process analysis."""
        if not evaluation.results:
            return
        self._log(
            "pre_execution_prediction",
            env_step=env_step,
            evaluation=evaluation.to_dict(),
            evaluated_tasks=evaluation.metadata.get("tasks_evaluated", []),
            should_repair=evaluation.should_repair,
        )

    def _task_specs_by_id(self) -> Dict[str, Any]:
        get_task_specs = getattr(self.adapter, "get_task_specs", None)
        if get_task_specs is None:
            return {}
        try:
            return {
                str(spec.task_id): spec
                for spec in get_task_specs()
                if getattr(spec, "task_id", None) is not None
            }
        except Exception:
            return {}

    def _normalized_args(self, args: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
        """Return a stable signature for action args that may contain lists."""
        return tuple(
            sorted(
                (
                    str(key),
                    json.dumps(value, sort_keys=True, default=repr),
                )
                for key, value in args.items()
            )
        )

    def _begin_repair(
        self,
        evaluation: PreExecutionEvaluation,
        env_step: int,
        message_broker: Optional[Any] = None,
        repair_dispatcher: Optional[Any] = None,
    ):
        """Send structured repair context to affected agents."""
        affected_agents = sorted(
            set(evaluation.affected_agents)
            | {view.agent_id for view in evaluation.plan_views}
        )
        repair_context = {
            "type": COOP2_REPAIR_CONTENT_TYPE,
            "env_step": env_step,
            "failures": [failure.to_dict() for failure in evaluation.failures],
            "plan_views": [view.to_dict() for view in evaluation.plan_views],
            "message": "Committed plans are predicted to violate cooperative constraints. Replan or coordinate before execution.",
        }
        repair_guidance = self._build_repair_guidance(
            evaluation=evaluation,
            affected_agents=affected_agents,
            env_step=env_step,
        )
        if repair_guidance:
            repair_context["repair_guidance"] = repair_guidance

        self._log(
            "repair_started",
            env_step=env_step,
            affected_agents=affected_agents,
            evaluation=evaluation.to_dict(),
        )

        if repair_dispatcher is not None:
            repair_dispatcher(
                affected_agents=affected_agents,
                repair_context=repair_context,
                env_step=env_step,
            )
        elif message_broker is not None:
            message_broker.send_message(
                sender_id=COOP2_REPAIR_SENDER_ID,
                recipients=affected_agents,
                content=repair_context,
                metadata=coop2_repair_metadata(),
                env_step=env_step,
            )

        self._log(
            "repair_message",
            env_step=env_step,
            affected_agents=affected_agents,
            content=repair_context,
        )

    def _log(
        self,
        event_type: str,
        env_step: int,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **payload: Any,
    ):
        if self.trace_logger is not None:
            self.trace_logger.log(
                event_type,
                env_step=env_step,
                agent_id=agent_id,
                task_id=task_id,
                **payload,
            )

    def _build_repair_guidance(
        self,
        evaluation: PreExecutionEvaluation,
        affected_agents: Sequence[str],
        env_step: int,
    ) -> Dict[str, Any]:
        builder = getattr(self.adapter, "build_repair_guidance", None)
        if callable(builder):
            try:
                guidance = builder(
                    evaluation=evaluation,
                    affected_agents=list(affected_agents),
                    env_step=env_step,
                )
                if guidance:
                    return guidance
            except Exception as exc:
                return {
                    "strategy": "soft_sync",
                    "warning": f"adapter repair guidance unavailable: {exc}",
                    "instructions": self._default_repair_instructions(),
                }

        return {
            "strategy": "soft_sync",
            "instructions": self._default_repair_instructions(),
            "task_hints": self._default_task_hints(evaluation),
        }

    def _default_repair_instructions(self) -> List[str]:
        return [
            "Choose one shared target for each failed cooperative collection.",
            "All required agents should navigate directly to that same target before collecting.",
            "Early agents should wait briefly with noop instead of collecting alone.",
            "If one agent is much farther away, switch to a closer shared target rather than forcing an exact collect step.",
            "The goal is a good-enough synchronized collect window, not an exact primitive-step schedule.",
        ]

    def _default_task_hints(self, evaluation: PreExecutionEvaluation) -> List[Dict[str, Any]]:
        hints_by_task: Dict[str, Dict[str, Any]] = {}
        for failure in evaluation.failures:
            task_id = str(failure.task_id)
            hint = hints_by_task.setdefault(
                task_id,
                {
                    "task_id": task_id,
                    "constraints": [],
                    "agents": set(),
                    "collect_steps": {},
                },
            )
            hint["constraints"].append(failure.constraint_type.value)
            hint["agents"].update(failure.agents)
            collect_steps = failure.metadata.get("collect_steps")
            if isinstance(collect_steps, dict):
                hint["collect_steps"].update(collect_steps)

        task_hints = []
        for hint in hints_by_task.values():
            hint["constraints"] = sorted(set(hint["constraints"]))
            hint["agents"] = sorted(hint["agents"])
            task_hints.append(hint)
        return task_hints
