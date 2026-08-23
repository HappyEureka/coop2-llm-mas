"""Pre-execution COOP2 constraint evaluation.

The evaluator is intentionally environment-neutral. Adapters can provide richer
environment-specific checks while the default path still catches basic temporal
mismatches for PettingZoo-style environments.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .core import AgentPlanView, ConstraintResult, ConstraintType, TaskSpec


@dataclass
class PreExecutionEvaluation:
    """Predicted constraint status for the currently committed plans."""

    env_step: int
    plan_views: List[AgentPlanView]
    results: List[ConstraintResult] = field(default_factory=list)
    failures: List[ConstraintResult] = field(default_factory=list)
    affected_agents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def should_repair(self) -> bool:
        return bool(self.failures)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_step": self.env_step,
            "plan_views": [view.to_dict() for view in self.plan_views],
            "results": [result.to_dict() for result in self.results],
            "failures": [failure.to_dict() for failure in self.failures],
            "affected_agents": self.affected_agents,
            "metadata": self.metadata,
            "should_repair": self.should_repair,
        }


class PreExecutionConstraintEvaluator:
    """Evaluate committed agent plans before primitive environment execution."""

    def __init__(self, adapter: Any):
        self.adapter = adapter

    def evaluate(
        self,
        plan_views: Sequence[AgentPlanView],
        env_step: int = 0,
        infos: Optional[Dict[str, Any]] = None,
    ) -> PreExecutionEvaluation:
        views = list(plan_views)
        task_specs = self._task_specs_by_id()
        plan_level_results = self._adapter_plan_level_pre_execution_results(
            views,
            task_specs,
            env_step,
            infos,
        )
        if plan_level_results is not None:
            results = plan_level_results["results"]
            failures = [result for result in results if not result.satisfied]
            affected_agents = sorted(
                {
                    agent_id
                    for failure in failures
                    for agent_id in failure.agents
                }
            )
            metadata = plan_level_results.get("metadata", {})
            metadata.setdefault(
                "tasks_evaluated",
                sorted({result.task_id for result in results}),
            )
            metadata.setdefault("unresolved_agents", [])
            return PreExecutionEvaluation(
                env_step=env_step,
                plan_views=views,
                results=results,
                failures=failures,
                affected_agents=affected_agents,
                metadata=metadata,
            )

        grouped_views: Dict[str, List[AgentPlanView]] = defaultdict(list)
        unresolved_agents: List[str] = []

        for view in views:
            task_id = view.task_id or self._infer_plan_task_id(view)
            if task_id is None:
                unresolved_agents.append(view.agent_id)
                continue
            view.task_id = str(task_id)
            grouped_views[str(task_id)].append(view)

        results: List[ConstraintResult] = []
        for task_id, task_views in grouped_views.items():
            task_spec = task_specs.get(task_id)
            if task_spec is None:
                task_spec = self._resolve_task_spec(task_id)
            if task_spec is None:
                continue

            adapter_results = self._adapter_pre_execution_results(
                task_spec,
                task_views,
                env_step,
                infos,
            )
            if adapter_results is not None:
                results.extend(adapter_results)
            else:
                results.extend(self._default_results(task_spec, task_views))

        failures = [result for result in results if not result.satisfied]
        affected_agents = sorted(
            {
                agent_id
                for failure in failures
                for agent_id in failure.agents
            }
        )
        return PreExecutionEvaluation(
            env_step=env_step,
            plan_views=views,
            results=results,
            failures=failures,
            affected_agents=affected_agents,
            metadata={
                "tasks_evaluated": sorted(grouped_views.keys()),
                "unresolved_agents": unresolved_agents,
            },
        )

    def _task_specs_by_id(self) -> Dict[str, TaskSpec]:
        if not hasattr(self.adapter, "get_task_specs"):
            return {}
        return {str(spec.task_id): spec for spec in self.adapter.get_task_specs()}

    def _infer_plan_task_id(self, plan_view: AgentPlanView) -> Optional[str]:
        infer = getattr(self.adapter, "infer_plan_task_id", None)
        if infer is not None:
            task_id = infer(plan_view)
            if task_id is not None:
                return str(task_id)

        for action in plan_view.remaining_actions:
            args = self._action_args(action)
            for key in ("task_id", "resource_id", "target_id", "item_id"):
                if args.get(key) is not None:
                    return str(args[key])
        return None

    def _resolve_task_spec(self, task_id: str) -> Optional[TaskSpec]:
        resolve = getattr(self.adapter, "resolve_task_spec", None)
        if resolve is None:
            return None
        return resolve(task_id)

    def _adapter_pre_execution_results(
        self,
        task_spec: TaskSpec,
        task_views: Sequence[AgentPlanView],
        env_step: int,
        infos: Optional[Dict[str, Any]],
    ) -> Optional[List[ConstraintResult]]:
        evaluate = getattr(self.adapter, "get_pre_execution_constraint_results", None)
        if evaluate is None:
            return None
        return list(evaluate(task_spec, task_views, env_step=env_step, infos=infos))

    def _adapter_plan_level_pre_execution_results(
        self,
        plan_views: Sequence[AgentPlanView],
        task_specs: Dict[str, TaskSpec],
        env_step: int,
        infos: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        evaluate = getattr(self.adapter, "get_pre_execution_constraint_results_for_plans", None)
        if evaluate is None:
            return None
        raw = evaluate(plan_views, list(task_specs.values()), env_step=env_step, infos=infos)
        if raw is None:
            return None
        if isinstance(raw, dict):
            return {
                "results": list(raw.get("results", [])),
                "metadata": dict(raw.get("metadata", {})),
            }
        return {"results": list(raw), "metadata": {}}

    def _default_results(
        self,
        task_spec: TaskSpec,
        task_views: Sequence[AgentPlanView],
    ) -> List[ConstraintResult]:
        required = max(int(task_spec.required_agents), 1)
        collecting_agents = [
            view.agent_id
            for view in task_views
            if self._plan_has_action(view, "collect")
        ]
        return [
            ConstraintResult(
                task_id=str(task_spec.task_id),
                constraint_type=ConstraintType.TEMPORAL,
                satisfied=len(collecting_agents) >= required,
                score=len(collecting_agents) / required,
                agents=collecting_agents or [view.agent_id for view in task_views],
                reason=(
                    None
                    if len(collecting_agents) >= required
                    else f"Requires {required} planned collect actions, found {len(collecting_agents)}"
                ),
                metadata={"required_agents": required},
            ),
        ]

    def _plan_has_action(self, plan_view: AgentPlanView, action_type: str) -> bool:
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
