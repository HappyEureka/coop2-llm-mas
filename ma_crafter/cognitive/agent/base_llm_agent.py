"""
Base LLM agent for MA-Crafter.

The reusable LLM-backed agent behind every communication topology:
- plan generation with the agent's role, team, and topology context
- the interrupted-stage decision: resume the committed plan or replan
- plain-text message generation for topology protocols and COOP2 repair rounds
- LLM usage accounting

A topology class (see comm_topology/) sets ``role_prompt`` and ``role_name``,
overrides the small ``_plan_*`` hooks for its team and context, and adds its
communication flow on top of ``handle_reasoning`` / ``handle_interrupt``.
"""

import json
import threading
from typing import Any, Dict, List, Optional

from .agent import Agent
from .llm_client import LLMClient, InterruptDecision
from .prompts import build_system_prompt, build_observation_prompt
from .cognitive_agent import (
    build_interrupt_prompt,
    extract_status,
    extract_position,
    extract_facing,
    extract_visible_area,
    parse_plan_response,
    parse_interrupt_response,
)
from ..plan.plan import SymbolicPlan, SymbolicAction


class BaseLLMAgent(Agent):
    """
    LLM-backed planning agent. Communication is added by topology subclasses.

    Defaults: the reasoning stage plans from the current observation, with any
    buffered messages as context, and sends nothing. An interrupt asks the LLM
    whether to resume or replan, except for COOP2 repair requests, which
    always replan.
    """

    _llm_print_lock = threading.Lock()

    # Set by topology classes: role text appended to the system prompt and a
    # short name used in log lines (e.g. "LEADER").
    role_prompt: str = ""
    role_name: str = ""

    def __init__(
        self,
        agent_id: str,
        llm_client: LLMClient,
        max_memory_size: int = 10,
        temperature: float = 0.7,
        verbose: bool = True,
    ):
        """
        Args:
            agent_id: Agent identifier
            llm_client: LLM client for API calls
            max_memory_size: Number of recent events to keep in memory
            temperature: LLM sampling temperature
            verbose: Whether to print LLM inputs and outputs
        """
        super().__init__(agent_id, memory_size=max_memory_size)
        self.llm_client = llm_client
        self.temperature = temperature
        self.verbose = verbose
        self._system_prompt: Optional[str] = None

        # LLM usage accounting
        self.total_tokens_used = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.api_calls = 0
        self.api_latency_seconds = 0.0
        self.max_api_latency_seconds = 0.0
        self.api_retries = 0
        self.api_rate_limit_retries = 0
        self.api_error_retries = 0
        self.llm_errors = 0
        self.guard_filter_events = 0

        # Environment info, refreshed by the planning wrapper every step
        self.coop_config: Optional[str] = None
        self.symbolic_view: Optional[str] = None
        self.target_hints: Optional[str] = None

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------
    def observe(self, observation: Any, env_step: int):
        """Store the latest observation and environment step."""
        self.observation = observation
        self.env_step = env_step

    def set_env_info(
        self,
        coop_config: Optional[str] = None,
        symbolic_view: Optional[str] = None,
        target_hints: Optional[str] = None,
    ):
        """Set environment info used in prompts."""
        self.coop_config = coop_config
        self.symbolic_view = symbolic_view
        self.target_hints = target_hints

    # ------------------------------------------------------------------
    # Topology hooks
    # ------------------------------------------------------------------
    def _get_system_prompt(self) -> str:
        """Environment system prompt plus this agent's role description (cached)."""
        if self._system_prompt is None:
            base = build_system_prompt(self.agent_id, max_actions=6, include_env_description=True)
            self._system_prompt = base + self.role_prompt
        return self._system_prompt

    def _plan_agent_names(self) -> List[str]:
        """Agent ids listed in the plan prompt. Topologies return their team."""
        return [self.agent_id]

    def _plan_context_prefix(self) -> str:
        """Topology context placed before the observation, e.g. the leader's request."""
        return ""

    def _plan_coop_config(self) -> Optional[str]:
        """Cooperative configuration text for the plan prompt; topologies may extend it."""
        return self.coop_config

    # ------------------------------------------------------------------
    # LLM call helpers
    # ------------------------------------------------------------------
    def _should_print_llm_io(self) -> bool:
        """Return True when either agent or client verbose mode wants prompt IO."""
        return bool(self.verbose or getattr(self.llm_client, "verbose", False))

    def _print_llm_messages(self, label: str, messages: List[Dict]):
        """Print the full input messages sent to the model for this agent."""
        with self._llm_print_lock:
            print(f"\n{'=' * 80}")
            print(f"LLM INPUT [{self.agent_id}] - {label}")
            print(f"{'=' * 80}")
            for index, message in enumerate(messages):
                role = str(message.get("role", "unknown")).upper()
                content = str(message.get("content", ""))
                print(f"\n--- MESSAGE {index}: {role} ---")
                print(content)
            print(f"{'=' * 80}\n")

    def _record_llm_usage(self, usage: Dict[str, Any]):
        """Record token/API usage from one LLM call."""
        self.api_calls += 1
        self.total_tokens_used += usage["total_tokens"]
        self.prompt_tokens += usage["prompt_tokens"]
        self.completion_tokens += usage["completion_tokens"]
        latency = float(usage.get("latency_seconds", 0.0) or 0.0)
        self.api_latency_seconds += latency
        self.max_api_latency_seconds = max(self.max_api_latency_seconds, latency)
        self.api_retries += int(usage.get("retry_count", 0) or 0)
        self.api_rate_limit_retries += int(usage.get("rate_limit_retry_count", 0) or 0)
        self.api_error_retries += int(usage.get("api_error_retry_count", 0) or 0)
        self.guard_filter_events += int(usage.get("guard_filter_count", 0) or 0)

    def _record_llm_error(self, error: Exception):
        """Record an LLM-call failure for run-level monitoring."""
        self.llm_errors += 1
        is_guard_filter_error = getattr(self.llm_client, "_is_guard_filter_error", None)
        if callable(is_guard_filter_error) and is_guard_filter_error(error):
            self.guard_filter_events += 1

    def _any_waiting_agent_not_ready(
        self,
        agents_by_id: Dict[str, Any],
        wait_for: Optional[List[str]] = None,
    ) -> bool:
        """Return True if any agent this agent waits for has not committed yet."""
        wait_ids = self.wait_for if wait_for is None else wait_for
        if not wait_ids or not agents_by_id:
            return False
        return any(
            agents_by_id.get(agent_id) and not agents_by_id[agent_id].ready
            for agent_id in wait_ids
        )

    def _build_plan_prompt_messages(
        self,
        system_prompt: str,
        agent_names: List[str],
        messages: Optional[List[Dict]] = None,
        user_prefix: str = "",
        coop_config_override: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build the plan prompt: system prompt, then topology context and observation."""
        coop_config = self.coop_config if coop_config_override is None else coop_config_override
        obs_prompt = build_observation_prompt(
            env_step=self.env_step,
            agent_id=self.agent_id,
            agent_names=agent_names,
            status=extract_status(self.observation),
            position=extract_position(self.observation),
            facing=extract_facing(self.observation),
            visible_area=extract_visible_area(self.observation),
            messages=messages,
            memory=self.memory.get_events(),
            coop_config=coop_config,
            symbolic_view=self.symbolic_view,
            target_hints=self.target_hints,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prefix + obs_prompt},
        ]

    def _generate_plan_from_messages(
        self,
        prompt_messages: List[Dict],
        label: str = "Plan Generation",
        verbose_prefix: str = "Calling LLM for plan...",
    ) -> SymbolicPlan:
        """Call the LLM and parse the plan it commits; fall back to a fixed plan on failure."""
        if self._should_print_llm_io():
            self._print_llm_messages(label, prompt_messages)

        if self.verbose:
            print(f"  [{self.agent_id}] {verbose_prefix}")

        try:
            response, usage = self.llm_client.generate_plan(
                messages=prompt_messages,
                temperature=self.temperature,
            )
            self._record_llm_usage(usage)

            plan = parse_plan_response(
                llm_response=response,
                agent_id=self.agent_id,
                env_step=self.env_step,
                plan_id=self.plan_count + 1,
            )
            if self.verbose:
                print(f"  [{self.agent_id}] Plan: {plan.specification}")
            return plan
        except Exception as e:
            self._record_llm_error(e)
            print(f"  [{self.agent_id}] LLM error: {e}")
            return self._generate_fallback_plan()

    def _generate_text_from_messages(
        self,
        messages: List[Dict],
        fallback: str,
        temperature: Optional[float] = None,
    ) -> str:
        """Call the LLM for a plain text message and record usage consistently."""
        try:
            response, usage = self.llm_client.generate(
                messages=messages,
                response_format=None,
                temperature=self.temperature if temperature is None else temperature,
            )
            self._record_llm_usage(usage)
            return str(response)
        except Exception as e:
            self._record_llm_error(e)
            return fallback

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def generate_plan(self, messages: Optional[List[Dict]] = None) -> SymbolicPlan:
        """
        Generate a plan with the LLM using this agent's role, team, and topology context.

        Args:
            messages: Messages to show in the prompt (e.g. a COOP2 repair request)

        Returns:
            The new plan, also stored in self.plan
        """
        prompt_messages = self._build_plan_prompt_messages(
            system_prompt=self._get_system_prompt(),
            agent_names=self._plan_agent_names(),
            messages=messages,
            user_prefix=self._plan_context_prefix(),
            coop_config_override=self._plan_coop_config(),
        )
        self.plan = self._generate_plan_from_messages(
            prompt_messages=prompt_messages,
            label=f"{self.role_name} Plan Generation".strip(),
            verbose_prefix=(
                f"{self.role_name} calling LLM for plan..." if self.role_name else "Calling LLM for plan..."
            ),
        )
        return self.plan

    def _generate_fallback_plan(self) -> SymbolicPlan:
        """Simple fallback plan when the LLM call fails."""
        return SymbolicPlan(
            specification="Fallback exploration",
            actions=[
                SymbolicAction("move", {"direction": "left", "num_steps": 2}),
                SymbolicAction("collect", {"target": "wood"}),
            ],
            plan_id=self.plan_count + 1,
            agent_id=self.agent_id,
            created_at_step=self.env_step
        )

    # ------------------------------------------------------------------
    # COOP2 repair round
    # ------------------------------------------------------------------
    def describe_repair_intention(
        self,
        repair_context: Dict[str, Any],
        previous_statements: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate one concise statement for the ordered COOP2 repair round."""
        previous_statements = previous_statements or []
        plan_lines = []
        if self.plan is not None:
            plan_lines.append(f"Current plan #{self.plan.plan_id}: {self.plan.specification}")
            for index, action in enumerate(
                self.plan.actions[self.plan.current_action_index:],
                start=self.plan.current_action_index + 1,
            ):
                plan_lines.append(f"  {index}. {action}")
        else:
            plan_lines.append("Current plan: none")

        user_prompt = "\n".join([
            "You are participating in a COOP2 repair channel.",
            "Agents speak once in ascending agent-id order. Later agents can see earlier statements.",
            "State only what you intend to do or change for the predicted failure.",
            "Be concrete and concise; prefer the shared target/timing implied by the repair context.",
            "",
            f"You are: {self.agent_id}",
            f"Environment step: {self.env_step}",
            "",
            "Predicted repair context:",
            json.dumps(repair_context, indent=2, default=str),
            "",
            "Current reachable targets:",
            self.target_hints or "unknown",
            "",
            "Previous repair statements:",
            json.dumps(previous_statements, indent=2, default=str),
            "",
            "Your current plan:",
            "\n".join(plan_lines),
            "",
            "Write your repair intention in 1-3 sentences.",
        ])
        messages = [
            {
                "role": "system",
                "content": self._repair_intention_system_prompt(),
            },
            {"role": "user", "content": user_prompt},
        ]

        try:
            response, usage = self.llm_client.generate(
                messages=messages,
                response_format=None,
                temperature=self.temperature,
            )
            self._record_llm_usage(usage)
            return str(response).strip()
        except Exception as e:
            self._record_llm_error(e)
            if self.verbose:
                print(f"  [{self.agent_id}] LLM repair intention error: {e}")
            raise

    def _repair_intention_system_prompt(self) -> str:
        """Role prompt plus the COOP2 repair-channel response rules."""
        repair_rules = "\n".join([
            "COOP2 repair-channel response rules:",
            "- Keep the role and communication structure from the prompt above.",
            "- State only your repair intention for the current predicted failure.",
            "- Do not assign yourself or others a new permanent role.",
        ])
        return f"{self._get_system_prompt()}\n\n{repair_rules}"

    # ------------------------------------------------------------------
    # Reasoning and interrupted stages
    # ------------------------------------------------------------------
    def _handle_coop2_repair_interrupt(self, messages: Optional[List[Dict]] = None) -> bool:
        """
        Replan when the buffered (or given) messages contain a COOP2 repair request.

        With ``messages=None`` the buffer is only consumed if it holds a repair
        request. Returns True when a repair request was handled.
        """
        if messages is None:
            if not self.has_coop2_repair_request(self.get_messages(clear_buffer=False)):
                return False
            messages = self.get_messages(clear_buffer=True)
        elif not self.has_coop2_repair_request(messages):
            return False

        if self.verbose:
            print(f"  [{self.agent_id}] COOP2 repair request received: replanning")
        self.generate_plan(messages=messages)
        return True

    def decide_interrupt(
        self,
        messages: List[Dict],
        extra_context: Optional[str] = None,
    ) -> InterruptDecision:
        """
        Ask the LLM whether to resume the committed plan or replan.

        This is the decision made in the interrupted stage (I): the agent
        processes the messages that interrupted it and chooses RESUME (keep
        self.plan) or REPLAN (self.plan becomes a new plan). Topology agents
        call this after their own protocol steps, e.g. replying to the leader.

        Args:
            messages: Messages that triggered the interrupt (already taken from the buffer)
            extra_context: Optional extra prompt section (e.g. earlier proposals)

        Returns:
            The decision that was applied. Any LLM failure falls back to RESUME.
        """
        if not messages:
            return InterruptDecision.RESUME

        if self.verbose:
            print(f"\n[{self.agent_id}] Received {len(messages)} message(s):")
            for msg in messages:
                print(f"  From {msg['sender']}: {msg['content']}")
            print(f"  [{self.agent_id}] Asking LLM for interrupt decision...")

        try:
            interrupt_messages = build_interrupt_prompt(
                observation=self.observation,
                env_step=self.env_step,
                agent_id=self.agent_id,
                current_plan=self.plan,
                received_messages=messages,
                system_prompt=self._get_system_prompt(),
                memory=self.memory.get_events(),
                coop_config=self.coop_config,
                symbolic_view=self.symbolic_view,
                target_hints=self.target_hints,
                extra_context=extra_context,
            )

            if self._should_print_llm_io():
                self._print_llm_messages("Interrupt Decision", interrupt_messages)

            response, usage = self.llm_client.generate_interrupt_decision(
                messages=interrupt_messages,
                temperature=self.temperature,
            )
            self._record_llm_usage(usage)

            if self.verbose:
                print(f"    LLM reasoning: {response.reasoning}")
                print(f"    Tokens used: {usage['total_tokens']}")

            decision, new_plan = parse_interrupt_response(
                llm_response=response,
                agent_id=self.agent_id,
                env_step=self.env_step,
                plan_id=self.plan_count + 1,
            )

            if decision == InterruptDecision.RESUME:
                if self.verbose:
                    print(f"  [{self.agent_id}] LLM decided: RESUME current plan")
                return InterruptDecision.RESUME

            if self.verbose:
                print(f"  [{self.agent_id}] LLM decided: REPLAN")
            if new_plan is not None:
                self.plan = new_plan
                if self.verbose:
                    print(f"    New plan: {self.plan.specification}")
                    print(f"    Actions: {[str(a) for a in self.plan.actions]}")
            else:
                # LLM said replan but did not include a plan: generate one
                if self.verbose:
                    print("    Generating new plan...")
                self.generate_plan()
            return InterruptDecision.REPLAN

        except Exception as e:
            self._record_llm_error(e)
            print(f"  [{self.agent_id}] LLM interrupt error: {e}")
            print(f"  [{self.agent_id}] Falling back to resume")
            import traceback
            traceback.print_exc()
            return InterruptDecision.RESUME

    def handle_reasoning(self):
        """
        Reasoning-stage default: plan from the current observation, showing any
        buffered messages as context, and send nothing. Topology classes
        override this to run their communication flow first.
        """
        messages = self.get_messages(clear_buffer=True)
        self.generate_plan(messages=messages or None)

    def handle_interrupt(self):
        """
        Interrupted-stage default: a COOP2 repair request always replans;
        otherwise the LLM decides whether to resume or replan.
        """
        messages = self.get_messages(clear_buffer=True)
        if not messages:
            return
        if self._handle_coop2_repair_interrupt(messages=messages):
            return
        self.decide_interrupt(messages)

    def reset(self):
        """Reset agent state. Usage counters are kept across resets for statistics."""
        super().reset()

    def get_usage_stats(self) -> dict:
        """Get LLM usage statistics."""
        return {
            'api_calls': self.api_calls,
            'total_tokens': self.total_tokens_used,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'api_latency_seconds': self.api_latency_seconds,
            'avg_api_latency_seconds': (
                self.api_latency_seconds / self.api_calls if self.api_calls else 0.0
            ),
            'max_api_latency_seconds': self.max_api_latency_seconds,
            'api_retries': self.api_retries,
            'api_rate_limit_retries': self.api_rate_limit_retries,
            'api_error_retries': self.api_error_retries,
            'llm_errors': self.llm_errors,
            'guard_filter_events': self.guard_filter_events,
        }
