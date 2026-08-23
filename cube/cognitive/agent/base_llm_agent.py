"""
Base LLM Agent implementation.

Provides a reusable Agent that uses LLM for:
- Plan generation based on observations
- Interrupt handling decisions
- Message generation for inter-agent communication
"""

from typing import List, Any, Optional

from .agent import Agent
from .llm_client import LLMClient, InterruptDecision
from .cognitive_agent import (
    build_plan_prompt,
    build_interrupt_prompt,
    parse_plan_response,
    parse_interrupt_response,
)
from ..plan.plan import SymbolicPlan, SymbolicAction


class BaseLLMAgent(Agent):
    """
    Agent that uses real LLM for plan generation and communication.
    
    Uses OpenAI/Azure OpenAI API for:
    - Generating plans based on observations
    - Deciding whether to resume or replan on interrupts
    - Generating communication messages
    """
    
    def __init__(
        self,
        agent_id: str,
        llm_client: LLMClient,
        should_broadcast: bool = True,
        max_memory_size: int = 10,
        temperature: float = 0.7,
        verbose: bool = True,
    ):
        """
        Initialize LLM agent.
        
        Args:
            agent_id: Agent identifier
            llm_client: LLM client for API calls
            should_broadcast: Whether to send messages after plan generation
            max_memory_size: Number of recent events to keep in memory
            temperature: LLM sampling temperature
            verbose: Whether to print LLM inputs and outputs
        """
        super().__init__(agent_id, memory_size=max_memory_size)
        self.llm_client = llm_client
        self.should_broadcast = should_broadcast
        self.temperature = temperature
        self.verbose = verbose
        self.other_agents: List[str] = []  # Set externally
        
        # Token tracking
        self.total_tokens_used = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.api_calls = 0
        
        # Environment info (set externally)
        self.coop_config: Optional[str] = None
        self.symbolic_view: Optional[str] = None
    
    def observe(self, observation: Any, env_step: int):
        """Process observation from environment."""
        self.observation = observation
        self.env_step = env_step
    
    def set_env_info(self, coop_config: Optional[str] = None, symbolic_view: Optional[str] = None):
        """Set environment info for prompts."""
        self.coop_config = coop_config
        self.symbolic_view = symbolic_view
    
    def generate_plan(self) -> SymbolicPlan:
        """
        Generate a plan using LLM based on current observation.
        
        Returns:
            SymbolicPlan: Generated plan
        """
        # Build prompt for LLM
        messages = build_plan_prompt(
            observation=self.observation,
            env_step=self.env_step,
            agent_id=self.agent_id,
            memory=self.memory.get_events(),
            coop_config=self.coop_config,
            symbolic_view=self.symbolic_view,
        )
        
        # Call LLM
        if self.verbose:
            print(f"  [{self.agent_id}] Calling LLM for plan generation...")
            print(f"\n{'='*60}")
            print(f"LLM INPUT [{self.agent_id}] - Plan Generation")
            print(f"{'='*60}")
            for msg in messages:
                print(f"\n--- {msg['role'].upper()} ---")
                print(msg['content'])
            print(f"{'='*60}\n")
        
        try:
            response, usage = self.llm_client.generate_plan(
                messages=messages,
                temperature=self.temperature
            )
            
            # Track usage
            self.api_calls += 1
            self.total_tokens_used += usage['total_tokens']
            self.prompt_tokens += usage['prompt_tokens']
            self.completion_tokens += usage['completion_tokens']
            
            # Parse response to SymbolicPlan
            self.plan = parse_plan_response(
                llm_response=response,
                agent_id=self.agent_id,
                env_step=self.env_step,
                plan_id=self.plan_count + 1
            )
            
            if self.verbose:
                print(f"  [{self.agent_id}] LLM generated plan: {self.plan.specification}")
                print(f"    Actions: {[str(a) for a in self.plan.actions]}")
                print(f"    Tokens used: {usage['total_tokens']}")
            
        except Exception as e:
            print(f"  [{self.agent_id}] LLM error: {e}")
            # Fallback to a simple exploration plan
            self.plan = self._generate_fallback_plan()
        
        return self.plan
    
    def _generate_fallback_plan(self) -> SymbolicPlan:
        """Generate a simple fallback plan when LLM fails (CUBE version)."""
        return SymbolicPlan(
            specification="Fallback: move towards blocks",
            actions=[
                SymbolicAction("move", {"direction": "right", "num_steps": 2}),
                SymbolicAction("move", {"direction": "down", "num_steps": 1}),
            ],
            plan_id=self.plan_count + 1,
            agent_id=self.agent_id,
            created_at_step=self.env_step
        )
    
    def handle_interrupt(self):
        """
        Handle interrupt by asking LLM to decide whether to resume or replan.
        
        Provides LLM with:
        - Current observation
        - Memory of recent events
        - Current plan and its execution status
        - Messages that triggered the interrupt
        """
        messages = self.get_messages(clear_buffer=True)
        
        if not messages:
            return
        
        if self.verbose:
            print(f"\n[{self.agent_id}] Received {len(messages)} message(s):")
            for msg in messages:
                print(f"  From {msg['sender']}: {msg['content']}")
        
        # Always use LLM to decide
        if self.verbose:
            print(f"  [{self.agent_id}] Asking LLM for interrupt decision...")
        
        try:
            interrupt_messages = build_interrupt_prompt(
                observation=self.observation,
                env_step=self.env_step,
                agent_id=self.agent_id,
                current_plan=self.plan,
                received_messages=messages,
                memory=self.memory.get_events(),
                coop_config=self.coop_config,
                symbolic_view=self.symbolic_view,
            )
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"LLM INPUT [{self.agent_id}] - Interrupt Decision")
                print(f"{'='*60}")
                for msg in interrupt_messages:
                    print(f"\n--- {msg['role'].upper()} ---")
                    print(msg['content'])
                print(f"{'='*60}\n")
            
            response, usage = self.llm_client.generate_interrupt_decision(
                messages=interrupt_messages,
                temperature=self.temperature
            )
            
            # Track usage
            self.api_calls += 1
            self.total_tokens_used += usage['total_tokens']
            self.prompt_tokens += usage['prompt_tokens']
            self.completion_tokens += usage['completion_tokens']
            
            if self.verbose:
                print(f"    LLM reasoning: {response.reasoning}")
                print(f"    Tokens used: {usage['total_tokens']}")
            
            decision, new_plan = parse_interrupt_response(
                llm_response=response,
                agent_id=self.agent_id,
                env_step=self.env_step,
                plan_id=self.plan_count + 1
            )
            
            if decision == InterruptDecision.RESUME:
                if self.verbose:
                    print(f"  [{self.agent_id}] LLM decided: RESUME current plan")
            else:
                if self.verbose:
                    print(f"  [{self.agent_id}] LLM decided: REPLAN")
                if new_plan:
                    self.plan = new_plan
                    if self.verbose:
                        print(f"    New plan: {self.plan.specification}")
                        print(f"    Actions: {[str(a) for a in self.plan.actions]}")
                else:
                    # LLM said replan but didn't provide plan, generate one
                    if self.verbose:
                        print(f"    Generating new plan...")
                    self.generate_plan()
                    
        except Exception as e:
            print(f"  [{self.agent_id}] LLM interrupt error: {e}")
            print(f"  [{self.agent_id}] Falling back to resume")
            import traceback
            traceback.print_exc()
    
    def handle_reasoning(self):
        """
        Handle reasoning state - generate plan and optionally broadcast.
        """
        # Generate plan using LLM
        self.generate_plan()
        
        # Optionally broadcast to other agents
        if self.should_broadcast and self.message_broker and self.other_agents:
            try:
                content = f"I'm starting plan: {self.plan.specification}"
                self.send_message(
                    recipients=self.other_agents,
                    content=content,
                    metadata={'plan_id': self.plan.plan_id, 'step': self.env_step}
                )
                if self.verbose:
                    print(f"  [{self.agent_id}] Broadcasted plan to {self.other_agents}")
            except Exception as e:
                print(f"  [{self.agent_id}] Broadcast failed: {e}")
    
    def reset(self):
        """Reset agent state."""
        super().reset()
        # Keep token counts across resets for statistics
    
    def get_usage_stats(self) -> dict:
        """Get LLM usage statistics."""
        return {
            'api_calls': self.api_calls,
            'total_tokens': self.total_tokens_used,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
        }
