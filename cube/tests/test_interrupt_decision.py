"""Tests for the interrupted stage (I): the LLM decides to resume or replan.

Uses a scripted LLM client, so no API credentials are needed. Run from the
environment directory:

    python -m unittest tests.test_interrupt_decision
"""

import unittest
from typing import List

from cognitive.agent import AgentState, InterruptDecision
from cognitive.agent.cognitive_agent import build_interrupt_prompt
from cognitive.agent.llm_client import (
    LLMInterruptResponse,
    LLMPlanResponse,
    MoveAction,
    Task,
    TaskSpecification,
)
from cognitive.messages import MessageBroker
from cognitive.plan.plan import SymbolicAction, SymbolicPlan
from comm_topology import (
    create_llm_broadcast_chain_topology,
    create_llm_centralized_topology,
    create_llm_individual_topology,
)
from coop2_repair.message_protocol import COOP2_REPAIR_SENDER_ID, coop2_repair_metadata

REVISED_TASK = TaskSpecification(task=Task.PUSH_BLOCK, block_id=99)


def _plan_response(block_id: int, direction: str = "left") -> LLMPlanResponse:
    return LLMPlanResponse(
        task=TaskSpecification(task=Task.PUSH_BLOCK, block_id=block_id),
        actions=[MoveAction(direction=direction, num_steps=1)],
        reasoning="scripted",
    )


class ScriptedLLMClient:
    """Stand-in for LLMClient that returns scripted structured responses."""

    def __init__(self, decision: InterruptDecision, inline_plan: bool = True):
        self.decision = decision
        self.inline_plan = inline_plan
        self.calls: List[str] = []
        self.last_prompt = None
        self.verbose = False
        self.model = "scripted"
        self.backend = "scripted"

    @staticmethod
    def _usage():
        return {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    def generate(self, messages, response_format=None, temperature=0.7):
        self.calls.append(response_format.__name__ if response_format else "text")
        self.last_prompt = messages
        if response_format is None:
            return f"scripted message #{len(self.calls)}", self._usage()
        if response_format is LLMPlanResponse:
            return _plan_response(block_id=len(self.calls)), self._usage()
        if response_format is LLMInterruptResponse:
            new_plan = None
            if self.decision == InterruptDecision.REPLAN and self.inline_plan:
                new_plan = _plan_response(block_id=REVISED_TASK.block_id, direction="right")
            response = LLMInterruptResponse(decision=self.decision, reasoning="scripted", new_plan=new_plan)
            return response, self._usage()
        raise AssertionError(f"unexpected response_format {response_format}")

    def generate_plan(self, messages, temperature=0.7):
        return self.generate(messages, response_format=LLMPlanResponse, temperature=temperature)

    def generate_interrupt_decision(self, messages, temperature=0.7):
        return self.generate(messages, response_format=LLMInterruptResponse, temperature=temperature)


def _committed_plan(agent_id: str) -> SymbolicPlan:
    return SymbolicPlan(
        specification="push_block(block#1)",
        actions=[SymbolicAction("move", {"direction": "left", "num_steps": 2})],
        plan_id=1,
        agent_id=agent_id,
        created_at_step=0,
    )


def _wire(agents) -> MessageBroker:
    """Give every agent a broker and a committed plan, then start execution (state X)."""
    broker = MessageBroker(agents)
    for agent in agents.values():
        agent.message_broker = broker
        agent.observation = {}
        agent.env_step = 5
        agent.plan = _committed_plan(agent.agent_id)
        agent.set_ready()
        agent.start_execution()
        assert agent.state == AgentState.X
    return broker


class CentralizedFollowerInterruptTests(unittest.TestCase):
    def _interrupt_follower(self, decision, inline_plan=True):
        client = ScriptedLLMClient(decision, inline_plan)
        agents = create_llm_centralized_topology(3, client, verbose=False)
        _wire(agents)
        leader, follower = agents["agent_0"], agents["agent_1"]
        # Leader finished its plan and is reasoning again: it broadcasts and waits for acks.
        leader.set_unready(reason="plan_terminated")
        leader.send_message(recipients=leader.send_to, content="new directive", metadata={"type": "leader_broadcast"})
        self.assertEqual(follower.state, AgentState.I)
        return client, leader, follower

    def test_resume_keeps_committed_plan_and_acks_leader(self):
        client, leader, follower = self._interrupt_follower(InterruptDecision.RESUME)
        original = follower.plan
        follower.create_agent_thread()
        self.assertIs(follower.plan, original)
        self.assertEqual(follower.plan_count, 1)
        self.assertEqual(follower.state, AgentState.W)
        acks = [m for m in leader.message_buffer if m["metadata"].get("type") == "follower_response"]
        self.assertEqual([m["sender"] for m in acks], ["agent_1"])
        self.assertIn("LLMInterruptResponse", client.calls)
        self.assertNotIn("LLMPlanResponse", client.calls)

    def test_replan_installs_inline_plan(self):
        client, _leader, follower = self._interrupt_follower(InterruptDecision.REPLAN)
        original = follower.plan
        follower.create_agent_thread()
        self.assertIsNot(follower.plan, original)
        self.assertEqual(follower.plan.specification, str(REVISED_TASK))
        self.assertEqual(follower.plan_count, 2)
        self.assertEqual(follower.state, AgentState.W)
        self.assertNotIn("LLMPlanResponse", client.calls)

    def test_replan_without_inline_plan_uses_follower_planner(self):
        client, _leader, follower = self._interrupt_follower(InterruptDecision.REPLAN, inline_plan=False)
        original = follower.plan
        follower.create_agent_thread()
        self.assertIsNot(follower.plan, original)
        self.assertIn("LLMPlanResponse", client.calls)
        self.assertEqual(follower.plan_count, 2)

    def test_llm_failure_falls_back_to_resume(self):
        client, _leader, follower = self._interrupt_follower(InterruptDecision.REPLAN)
        original = follower.plan

        def boom(messages, temperature=0.7):
            raise RuntimeError("scripted API failure")

        client.generate_interrupt_decision = boom
        follower.create_agent_thread()
        self.assertIs(follower.plan, original)
        self.assertEqual(follower.state, AgentState.W)

    def test_repair_request_always_replans(self):
        client = ScriptedLLMClient(InterruptDecision.RESUME)
        agents = create_llm_centralized_topology(3, client, verbose=False)
        broker = _wire(agents)
        follower = agents["agent_1"]
        original = follower.plan
        broker.send_message(
            sender_id=COOP2_REPAIR_SENDER_ID,
            recipients=["agent_1"],
            content={"type": "coop2_pre_execution_repair", "env_step": 5, "failures": []},
            metadata=coop2_repair_metadata(),
        )
        self.assertEqual(follower.state, AgentState.I)
        follower.create_agent_thread()
        self.assertIsNot(follower.plan, original)
        self.assertNotIn("LLMInterruptResponse", client.calls)
        self.assertIn("LLMPlanResponse", client.calls)


class BroadcastChainInterruptTests(unittest.TestCase):
    def _interrupt_later_speakers(self, decision):
        client = ScriptedLLMClient(decision)
        agents = create_llm_broadcast_chain_topology(3, client, verbose=False)
        _wire(agents)
        first = agents["agent_0"]
        # First speaker finished its plan and broadcasts a new proposal while others execute.
        first.set_unready(reason="plan_terminated")
        first.send_message(
            recipients=first.send_to,
            content="proposal from agent_0",
            metadata={"type": "broadcast_chain", "speaker_order": 0},
        )
        self.assertEqual(agents["agent_1"].state, AgentState.I)
        self.assertEqual(agents["agent_2"].state, AgentState.I)
        return client, agents

    def test_resume_does_not_broadcast(self):
        _client, agents = self._interrupt_later_speakers(InterruptDecision.RESUME)
        second, third = agents["agent_1"], agents["agent_2"]
        original = second.plan
        second.create_agent_thread()
        self.assertIs(second.plan, original)
        self.assertEqual(second.state, AgentState.W)
        self.assertEqual(second.broadcast_history, ["[agent_0]: proposal from agent_0"])
        self.assertEqual([m["sender"] for m in third.message_buffer], ["agent_0"])
        third.create_agent_thread()
        self.assertEqual(third.state, AgentState.W)
        self.assertIs(third.plan, third._committed_plan)

    def test_replan_broadcasts_revised_plan_to_following_speakers(self):
        _client, agents = self._interrupt_later_speakers(InterruptDecision.REPLAN)
        second, third = agents["agent_1"], agents["agent_2"]
        original = second.plan
        second.create_agent_thread()
        self.assertIsNot(second.plan, original)
        self.assertEqual(second.plan_count, 2)
        self.assertEqual(second.state, AgentState.W)
        self.assertEqual([m["sender"] for m in third.message_buffer], ["agent_0", "agent_1"])
        self.assertEqual(third.message_buffer[-1]["metadata"]["type"], "broadcast_chain")


class IndividualAgentTests(unittest.TestCase):
    """Individual agents use the base defaults: plan on reasoning, replan only on COOP2 repair."""

    def test_reasoning_default_plans_with_buffered_messages(self):
        client = ScriptedLLMClient(InterruptDecision.RESUME)
        agents = create_llm_individual_topology(2, client, verbose=False)
        broker = MessageBroker(agents)
        for agent in agents.values():
            agent.message_broker = broker
            agent.observation = {}
            agent.env_step = 5
        first, second = agents["agent_0"], agents["agent_1"]
        self.assertEqual(second.state, AgentState.R)
        first.send_message(recipients=["agent_1"], content="hello from agent_0")
        self.assertEqual(second.state, AgentState.R)  # reasoning agents are not interrupted
        second.create_agent_thread()
        self.assertIsNotNone(second.plan)
        self.assertEqual(second.state, AgentState.W)
        self.assertEqual(client.calls, ["LLMPlanResponse"])
        self.assertIn("From agent_0: hello from agent_0", client.last_prompt[1]["content"])
        self.assertIn("INDEPENDENT AGENT", client.last_prompt[0]["content"])

    def test_repair_request_replans(self):
        client = ScriptedLLMClient(InterruptDecision.RESUME)
        agents = create_llm_individual_topology(2, client, verbose=False)
        broker = _wire(agents)
        agent = agents["agent_1"]
        original = agent.plan
        broker.send_message(
            sender_id=COOP2_REPAIR_SENDER_ID,
            recipients=["agent_1"],
            content={"type": "coop2_pre_execution_repair", "env_step": 5, "failures": []},
            metadata=coop2_repair_metadata(),
        )
        self.assertEqual(agent.state, AgentState.I)
        agent.create_agent_thread()
        self.assertIsNot(agent.plan, original)
        self.assertEqual(agent.state, AgentState.W)
        self.assertNotIn("LLMInterruptResponse", client.calls)


class InterruptPromptTests(unittest.TestCase):
    def test_memory_and_extra_context_are_rendered(self):
        memory = [
            {"type": "message_in", "env_step": 3, "sender": "agent_0", "content": "go left"},
            {"type": "message_out", "env_step": 4, "recipients": ["agent_0"], "content": "ack"},
            {"type": "plan", "env_step": 4, "plan_id": 1, "specification": "push_block(block#1)", "actions": []},
        ]
        prompt = build_interrupt_prompt(
            observation={},
            env_step=5,
            agent_id="agent_1",
            current_plan=_committed_plan("agent_1"),
            received_messages=[{"sender": "agent_0", "content": "new directive"}],
            memory=memory,
            extra_context="## Earlier Proposals\n[agent_0]: proposal",
        )
        user_prompt = prompt[1]["content"]
        self.assertIn("Received from agent_0: go left", user_prompt)
        self.assertIn("Sent to ['agent_0']: ack", user_prompt)
        self.assertIn("Plan #1: push_block(block#1)", user_prompt)
        self.assertIn("## Earlier Proposals", user_prompt)
        self.assertIn("From agent_0: new directive", user_prompt)


if __name__ == "__main__":
    unittest.main()
