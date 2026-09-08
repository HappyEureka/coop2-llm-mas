"""Regression tests for MA-Crafter COOP2 plan prediction."""

from types import SimpleNamespace
import unittest

from coop2_repair.core import AgentPlanView, ConstraintType, TaskSpec
from coop2_repair.macrafter_prediction import MacrafterPredictionMixin


class _PredictionAdapter(MacrafterPredictionMixin):
    def get_prediction_initial_state(self, agent_id):
        return {
            "position": (0, 0),
            "inventory": {"wood_pickaxe": 1},
            "metadata": {},
        }

    def resolve_prediction_target(self, plan_view, action, task_specs_by_id):
        task_id = str(action["args"]["task_id"])
        spec = task_specs_by_id[task_id]
        return {
            "task_id": task_id,
            "target_id": task_id,
            "target_type": spec.target_type,
            "position": spec.metadata["position"],
            "distance_threshold": spec.metadata["distance_threshold"],
        }

    def estimate_prediction_duration(self, state, action, target):
        return 1

    def get_prediction_temporal_tolerance(self):
        return 2

    def get_prediction_spatial_tolerance(self):
        return 1

    def _get_task_state(self, task_id):
        return SimpleNamespace(
            required_tool="wood_pickaxe",
            required_tool_mode="all",
        )


class DistinctAgentPredictionTest(unittest.TestCase):
    def test_repeated_collects_do_not_count_as_multiple_agents(self):
        adapter = _PredictionAdapter()
        task = TaskSpec(
            task_id="coal-1",
            task_type="collect",
            target_id="coal-1",
            target_type="coal",
            required_agents=2,
            required_capabilities=["wood_pickaxe"],
            metadata={"position": (0, 0), "distance_threshold": 1},
        )
        collect = {
            "action_type": "collect",
            "args": {
                "task_id": "coal-1",
                "resource_type": "coal",
                "steps": 1,
            },
        }
        plan = AgentPlanView(
            agent_id="agent_0",
            plan_id=1,
            task_id="coal-1",
            specification="collect_coal(coal#1)",
            remaining_actions=[collect, collect],
        )

        evaluation = adapter.get_pre_execution_constraint_results_for_plans(
            [plan],
            [task],
        )
        results = {
            result.constraint_type: result
            for result in evaluation["results"]
        }

        for constraint_type in (
            ConstraintType.TEMPORAL,
            ConstraintType.SPATIAL,
            ConstraintType.DEPENDENCY,
        ):
            result = results[constraint_type]
            self.assertFalse(result.satisfied)
            self.assertEqual(result.score, 0.5)
            self.assertEqual(result.agents, ["agent_0"])


if __name__ == "__main__":
    unittest.main()
