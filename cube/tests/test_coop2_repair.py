"""Tests for CUBE COOP2-Repair: plan-effect prediction, repair guidance, the repair channel, and prompt contracts.

Run from the environment directory:

    python -m unittest tests.test_coop2_repair
"""

from types import SimpleNamespace
import unittest

from cognitive.action.action import SymbolicActionExecutor
from cognitive.agent.base_llm_agent import BaseLLMAgent
from cognitive.agent.llm_client import (
    LLMPlanResponse,
    PushAction,
    Task,
    TaskSpecification,
)
from cognitive.agent.prompts import build_observation_prompt, build_system_prompt
from cognitive.plan.coop2_repair_dispatcher import Coop2RepairDispatcher
from coop2_repair.core import AgentPlanView, ConstraintResult, ConstraintType, TaskSpec
from coop2_repair.cube_adapter import CubeCoopAdapter
from coop2_repair.evaluator import PreExecutionEvaluation
from coop2_repair.repair_controller import Coop2RepairController
from env.env import CoopBlockPush


class CubeRepairPredictionTests(unittest.TestCase):
    def setUp(self):
        block = SimpleNamespace(
            id=0,
            weight=2,
            r=5,
            c=5,
            cells=lambda: [(5, 5), (5, 6), (6, 5), (6, 6)],
        )
        base_env = SimpleNamespace(
            _agent_positions={
                "agent_0": (5, 0),
                "agent_1": (6, 1),
            },
            _blocks=[block],
            K=15,
            possible_agents=["agent_0", "agent_1"],
        )
        symbolic_env = SimpleNamespace(
            env=base_env,
            possible_agents=["agent_0", "agent_1"],
            name_map={},
            reverse_name_map={},
        )
        self.adapter = CubeCoopAdapter(
            symbolic_env,
            temporal_tolerance=1,
        )
        self.spec = TaskSpec(
            task_id="block_0_left",
            task_type="push",
            target_id="0",
            target_type="weight_2",
            required_agents=2,
            metadata={
                "block_id": 0,
                "face": "left",
                "weight": 2,
                "face_cells": [(5, 4), (6, 4)],
                "position": (5.5, 4),
            },
        )


    @staticmethod
    def _plan_view(agent_id):
        return AgentPlanView(
            agent_id=agent_id,
            plan_id=1,
            task_id=None,
            specification="deliver block 0",
            remaining_actions=[
                {
                    "action_type": "navigate",
                    "args": {
                        "entity_type": "block",
                        "entity_id": 0,
                        "timeout": 30,
                    },
                },
                {
                    "action_type": "push",
                    "args": {"block_id": 0, "num_steps": 4},
                },
            ],
        )

    def test_navigation_is_predicted_before_spatial_check(self):
        evaluation = self.adapter.get_pre_execution_constraint_results_for_plans(
            [self._plan_view("agent_0"), self._plan_view("agent_1")],
            [self.spec],
            env_step=0,
        )
        by_type = {
            result.constraint_type: result
            for result in evaluation["results"]
        }
        self.assertTrue(by_type[ConstraintType.SPATIAL].satisfied)
        self.assertTrue(by_type[ConstraintType.TEMPORAL].satisfied)
        self.assertEqual(
            by_type[ConstraintType.SPATIAL].agents,
            ["agent_0", "agent_1"],
        )

    def test_missing_partner_is_predicted_as_failure(self):
        evaluation = self.adapter.get_pre_execution_constraint_results_for_plans(
            [self._plan_view("agent_0")],
            [self.spec],
            env_step=0,
        )
        by_type = {
            result.constraint_type: result
            for result in evaluation["results"]
        }
        self.assertFalse(by_type[ConstraintType.SPATIAL].satisfied)
        self.assertFalse(by_type[ConstraintType.TEMPORAL].satisfied)

    def test_infer_task_from_navigate_entity_id(self):
        self.assertEqual(
            self.adapter.infer_plan_task_id(self._plan_view("agent_0")),
            "block_0_left",
        )

    def test_navigation_does_not_route_into_occupied_target_cell(self):
        direction = SymbolicActionExecutor._bfs_navigation_direction(
            agent_pos=(0, 0),
            targets=[(0, 1), (1, 1)],
            blocked_cells={(0, 1)},
            grid_size=3,
        )
        self.assertEqual(direction, "down")

    def test_repair_guidance_assigns_distinct_face_cells(self):
        self.adapter.get_task_specs = lambda: [self.spec]
        evaluation = PreExecutionEvaluation(
            env_step=0,
            plan_views=[
                self._plan_view("agent_0"),
                self._plan_view("agent_1"),
            ],
            failures=[
                ConstraintResult(
                    task_id=self.spec.task_id,
                    constraint_type=ConstraintType.SPATIAL,
                    satisfied=False,
                    score=0.5,
                    agents=["agent_0"],
                    metadata=dict(self.spec.metadata),
                )
            ],
            affected_agents=["agent_0", "agent_1"],
        )

        guidance = self.adapter.build_repair_guidance(
            evaluation=evaluation,
            affected_agents=["agent_0", "agent_1"],
            env_step=0,
        )

        assignments = guidance["recommended_target"]["cell_assignments"]
        self.assertEqual(len({tuple(cell) for cell in assignments.values()}), 2)
        self.assertEqual(
            guidance["recommended_target"]["synchronize_push_at_step"],
            max(
                plan["synchronize_push_at_step"]
                for plan in guidance["recommended_plans"].values()
            ),
        )
        for agent_id, plan in guidance["recommended_plans"].items():
            self.assertEqual(plan["assigned_face_cell"], assignments[agent_id])
            self.assertTrue(
                {
                    action["action_type"]
                    for action in plan["actions"]
                }.issubset({"move", "wait", "push"})
            )
            self.assertEqual(
                sum(
                    action["args"]["num_steps"]
                    for action in plan["actions"]
                    if action["action_type"] == "push"
                ),
                self.adapter._distance_to_goal(0),
            )

    def test_repair_guidance_skips_unreachable_high_priority_target(self):
        unreachable = TaskSpec(
            task_id="block_0_unreachable",
            task_type="push",
            target_id="0",
            target_type="weight_2",
            required_agents=2,
            metadata={
                **self.spec.metadata,
                "face_cells": [(5, 4)],
            },
        )
        reachable = TaskSpec(
            task_id="block_0_reachable",
            task_type="push",
            target_id="0",
            target_type="weight_1",
            required_agents=1,
            metadata={
                **self.spec.metadata,
                "weight": 1,
                "face_cells": [(5, 4)],
            },
        )
        self.adapter.get_task_specs = lambda: [unreachable, reachable]
        evaluation = PreExecutionEvaluation(
            env_step=0,
            plan_views=[self._plan_view("agent_0"), self._plan_view("agent_1")],
            failures=[
                ConstraintResult(
                    task_id=unreachable.task_id,
                    constraint_type=ConstraintType.SPATIAL,
                    satisfied=False,
                    score=0.0,
                    agents=[],
                    metadata=dict(unreachable.metadata),
                )
            ],
            affected_agents=["agent_0", "agent_1"],
        )

        guidance = self.adapter.build_repair_guidance(
            evaluation=evaluation,
            affected_agents=["agent_0", "agent_1"],
            env_step=0,
        )

        self.assertEqual(
            guidance["recommended_target"]["task_id"],
            reachable.task_id,
        )

    def test_adapter_recommendation_is_advisory_prompt_context(self):
        messages = [
            {
                "content": {
                    "type": "coop2_pre_execution_repair",
                    "env_step": 4,
                    "repair_guidance": {
                        "recommended_plans": {
                            "agent_0": {
                                "task": "deliver block 0",
                                "actions": [
                                    {
                                        "action_type": "push",
                                        "args": {
                                            "block_id": 0,
                                            "num_steps": 3,
                                        },
                                    }
                                ],
                            }
                        }
                    },
                }
            }
        ]
        prompt = build_observation_prompt(
            env_step=4,
            agent_id="agent_0",
            agent_names=["agent_0", "agent_1"],
            messages=messages,
        )
        self.assertIn("RECOMMENDED PLAN FOR YOU (advisory)", prompt)
        self.assertIn('"task": "deliver block 0"', prompt)
        self.assertIn('"action_type": "push"', prompt)
        self.assertIn("Generate and commit your own final plan", prompt)
        self.assertIn("not an enforced replacement", prompt)

    def test_llm_plan_is_not_replaced_by_recommendation(self):
        class _LLMClient:
            def generate_plan(self, messages, temperature):
                return (
                    LLMPlanResponse(
                        task=TaskSpecification(
                            task=Task.PUSH_BLOCK,
                            block_id=1,
                        ),
                        actions=[
                            PushAction(
                                block_id=1,
                                num_steps=2,
                            )
                        ],
                        reasoning="Block 1 is feasible from the current observation.",
                    ),
                    {
                        "total_tokens": 1,
                        "prompt_tokens": 1,
                        "completion_tokens": 0,
                    },
                )

        agent = BaseLLMAgent("agent_0", _LLMClient(), verbose=False)
        messages = [
            {
                "content": {
                    "type": "coop2_pre_execution_repair",
                    "repair_guidance": {
                        "recommended_plans": {
                            "agent_0": {
                                "task": "deliver block 0",
                                "actions": [
                                    {
                                        "action_type": "push",
                                        "args": {"block_id": 0, "num_steps": 3},
                                    }
                                ],
                            }
                        }
                    },
                }
            }
        ]
        plan = agent.generate_plan(messages=messages)
        self.assertEqual(plan.specification, "push_block(block#1)")
        self.assertEqual(plan.actions[0].args["block_id"], 1)

    def test_primitive_moves_never_overlap_agents(self):
        env = CoopBlockPush(
            grid_size=7,
            num_agents=2,
            block_specs={1: 1},
            seed=1,
        )
        env.reset(
            seed=1,
            options={
                "fixed_agent_positions": {
                    "agent_0": (0, 0),
                    "agent_1": (0, 1),
                }
            },
        )
        env.step({"agent_0": 4, "agent_1": 0})
        positions = list(env._agent_positions.values())
        self.assertEqual(len(positions), len(set(positions)))


class RepairChannelSelectionTests(unittest.TestCase):
    def test_repair_channel_uses_recommended_cohort(self):
        controller = Coop2RepairController(adapter=None)
        plan_views = [
            AgentPlanView(
                agent_id=f"agent_{index}",
                plan_id=1,
                task_id="block_0_left",
                specification="deliver block 0",
                remaining_actions=[],
            )
            for index in range(6)
        ]
        evaluation = PreExecutionEvaluation(
            env_step=0,
            plan_views=plan_views,
            failures=[
                ConstraintResult(
                    task_id="block_0_left",
                    constraint_type=ConstraintType.TEMPORAL,
                    satisfied=False,
                    score=0.0,
                    agents=[view.agent_id for view in plan_views],
                )
            ],
            affected_agents=[view.agent_id for view in plan_views],
        )
        guidance = {
            "recommended_target": {
                "participants": ["agent_2", "agent_4"],
            },
            "recommended_plans": {
                "agent_2": {"task": "deliver block 0", "actions": []},
                "agent_4": {"task": "deliver block 0", "actions": []},
            },
        }

        selected = controller._repair_channel_agents(
            evaluation=evaluation,
            repair_guidance=guidance,
            available_agents=[view.agent_id for view in plan_views],
        )

        self.assertEqual(selected, ["agent_2", "agent_4"])

    def test_repair_channel_falls_back_to_failed_agents(self):
        controller = Coop2RepairController(adapter=None)
        evaluation = PreExecutionEvaluation(
            env_step=0,
            plan_views=[],
            affected_agents=["agent_1", "agent_3"],
        )

        selected = controller._repair_channel_agents(
            evaluation=evaluation,
            repair_guidance={},
            available_agents=["agent_0", "agent_1", "agent_2", "agent_3"],
        )

        self.assertEqual(selected, ["agent_1", "agent_3"])


class _RepairRoundAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.previous_counts = []

    def describe_repair_intention(self, repair_context, previous_statements):
        self.previous_counts.append(len(previous_statements))
        return f"{self.agent_id} intends to repair"


class _RepairRoundBroker:
    def __init__(self):
        self.calls = []

    def send_message(self, **kwargs):
        self.calls.append(kwargs)


class CubeRepairDispatcherTests(unittest.TestCase):
    def test_ordered_round_is_attached_before_delivery(self):
        agents = {
            agent_id: _RepairRoundAgent(agent_id)
            for agent_id in ("agent_10", "agent_2", "agent_1")
        }
        broker = _RepairRoundBroker()
        dispatcher = Coop2RepairDispatcher(
            agents=agents,
            message_broker_getter=lambda: broker,
        )
        context = {"failures": [{"constraint_type": "spatial"}]}

        dispatcher.dispatch(
            affected_agents=["agent_10", "agent_2", "agent_1"],
            repair_context=context,
            env_step=7,
        )

        channel = context["repair_channel"]
        self.assertEqual(
            channel["protocol"],
            "ordered_one_round_intention_then_global_revision",
        )
        self.assertEqual(
            channel["order"],
            ["agent_1", "agent_2", "agent_10"],
        )
        self.assertEqual(
            [statement["agent_id"] for statement in channel["statements"]],
            channel["order"],
        )
        self.assertEqual(agents["agent_1"].previous_counts, [0])
        self.assertEqual(agents["agent_2"].previous_counts, [1])
        self.assertEqual(agents["agent_10"].previous_counts, [2])
        self.assertEqual(len(broker.calls), 1)
        self.assertIs(broker.calls[0]["content"], context)

    def test_missing_broker_fails_before_repair_round(self):
        agent = _RepairRoundAgent("agent_0")
        dispatcher = Coop2RepairDispatcher(
            agents={"agent_0": agent},
            message_broker_getter=lambda: None,
        )

        with self.assertRaisesRegex(RuntimeError, "message broker"):
            dispatcher.dispatch(
                affected_agents=["agent_0"],
                repair_context={"failures": []},
                env_step=2,
            )

        self.assertEqual(agent.previous_counts, [])

    def test_missing_affected_agent_fails(self):
        dispatcher = Coop2RepairDispatcher(
            agents={},
            message_broker_getter=lambda: _RepairRoundBroker(),
        )

        with self.assertRaisesRegex(RuntimeError, "not configured"):
            dispatcher.dispatch(
                affected_agents=["agent_0"],
                repair_context={"failures": []},
                env_step=2,
            )


class CubePromptContractTests(unittest.TestCase):
    def test_system_prompt_uses_schema_task_names(self):
        prompt = build_system_prompt("agent_0")

        for task_name in ("push_block", "coordinate", "wait"):
            self.assertIn(f"- {task_name}:", prompt)
        for invalid_name in (
            "deliver_block",
            "coordinate_push",
            "approach_block",
            "wait_for_others",
        ):
            self.assertNotIn(invalid_name, prompt)

    def test_plan_response_normalizes_task_name_shorthand(self):
        response = LLMPlanResponse.model_validate(
            {
                "task": "push_block",
                "actions": [
                    {
                        "action_type": "navigate",
                        "entity_type": "block",
                        "entity_id": 4,
                        "timeout": 30,
                    },
                    {
                        "action_type": "push",
                        "block_id": 4,
                        "num_steps": 1,
                    },
                ],
                "reasoning": "Deliver block 4.",
            }
        )

        self.assertEqual(response.task.task, Task.PUSH_BLOCK)
        self.assertEqual(response.task.block_id, 4)


if __name__ == "__main__":
    unittest.main()
