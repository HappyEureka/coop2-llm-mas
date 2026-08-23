"""
LLM Client and Response Models for MA-Crafter.

This module provides:
- Pydantic models for structured LLM responses (plans, messages, interrupts)
- LLMClient wrapper for Azure OpenAI and DeepSeek API calls
"""

import os
import time
from pathlib import Path
from typing import Any, Optional, List, Dict, Union, Literal
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI, RateLimitError, APIError
from pydantic import BaseModel, Field


# Load local credentials without overriding variables exported by the caller.
# An environment-specific file takes precedence over the repository-level file.
_ENVIRONMENT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ENVIRONMENT_ROOT / ".env", override=False)
load_dotenv(_ENVIRONMENT_ROOT.parent / ".env", override=False)


# ============================================================================
# Enums for constrained values
# ============================================================================

class Direction(str, Enum):
    """Movement directions."""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class EntityType(str, Enum):
    """Types of entities in CUBE environment."""
    AGENT = "agent"
    BLOCK = "block"
    GOAL = "goal"
    WALL = "wall"


class Task(str, Enum):
    """Available tasks in CUBE block-pushing environment."""
    # Block manipulation tasks
    PUSH_BLOCK = "push_block"  # Push a specific block toward the goal
    COORDINATE = "coordinate"  # Coordinate with other agents on block position
    WAIT = "wait"  # Wait for other agents to get into position
    
    # Navigation
    # NAVIGATE = "navigate"  # Move to a specific position
    # EXPLORE = "explore"  # Explore the environment


class TaskSpecification(BaseModel):
    """Task specification with block ID for CUBE environment."""
    task: Task = Field(description="The task type to accomplish")
    block_id: int = Field(description="ID of the block to manipulate")
    
    def __str__(self) -> str:
        """String representation of the task specification."""
        return f"{self.task.value}(block#{self.block_id})"


class InterruptDecision(str, Enum):
    """Decision options when agent is interrupted."""
    RESUME = "resume"  # Continue with the current plan
    REPLAN = "replan"  # Generate a completely new plan


# ============================================================================
# Pydantic models for structured LLM output - Action Classes
# ============================================================================

class MoveAction(BaseModel):
    """Move in a direction."""
    action_type: Literal["move"] = "move"
    direction: Direction = Field(description="Direction to move")
    num_steps: int = Field(ge=1, le=5, description="Number of steps (1-5)")


class PushAction(BaseModel):
    """Push a block toward the goal zone."""
    action_type: Literal["push"] = "push"
    block_id: int = Field(ge=0, description="ID of the block to push")
    num_steps: int = Field(ge=1, le=10, description="Number of steps to push (1-10)")


class WaitAction(BaseModel):
    """Wait for a number of steps (useful for coordination)."""
    action_type: Literal["wait"] = "wait"
    num_steps: int = Field(ge=1, le=10, description="Number of steps to wait (1-10)")


class NavigateAction(BaseModel):
    """Navigate to a specific entity using pathfinding."""
    action_type: Literal["navigate"] = "navigate"
    entity_type: EntityType = Field(description="Type of entity to navigate to (block, goal, agent)")
    entity_id: int = Field(description="ID of the specific entity instance to navigate to")
    timeout: int = Field(default=30, ge=1, le=100, description="Maximum steps to attempt navigation (default 30)")


class NoopAction(BaseModel):
    """Do nothing this step."""
    action_type: Literal["noop"] = "noop"


# Union type for all CUBE actions
LLMAction = Union[MoveAction, NavigateAction, PushAction, WaitAction, NoopAction]


# ============================================================================
# Response Models
# ============================================================================

class LLMPlanResponse(BaseModel):
    """Structured response from LLM for plan generation."""
    task: TaskSpecification = Field(description="The task specification with type and optional target object")
    actions: List[LLMAction] = Field(description="List of actions to execute")
    reasoning: str = Field(description="Brief explanation of why this plan was chosen")


class LLMMessageResponse(BaseModel):
    """Structured response from LLM for message generation."""
    recipients: List[str] = Field(description="List of agent IDs to send message to")
    content: str = Field(description="Message content to send")
    reasoning: str = Field(description="Brief explanation of why sending this message")


class LLMInterruptResponse(BaseModel):
    """Structured response from LLM for interrupt handling."""
    decision: InterruptDecision = Field(
        description="Whether to resume the current plan or generate a new plan"
    )
    reasoning: str = Field(
        description="Brief explanation of why this decision was made based on the messages received"
    )
    # Optional new plan - only required if decision is REPLAN
    new_plan: Optional[LLMPlanResponse] = Field(
        default=None,
        description="The new plan to execute (required if decision is 'replan')"
    )


# ============================================================================
# LLM Client
# ============================================================================

class LLMClient:
    """
    Wrapper for Azure OpenAI and DeepSeek API calls.
    
    Supports two backends:
    - Azure OpenAI: Azure-hosted OpenAI models with structured output support
    - Foundry: Azure AI Foundry OpenAI-compatible endpoint
    - DeepSeek: DeepSeek models (manual JSON parsing)
    
    Takes messages and a response format, returns structured response.
    """
    
    # Backends that support OpenAI's structured output (beta.chat.completions.parse)
    STRUCTURED_OUTPUT_BACKENDS = {"azure"}
    JSON_MODE_BACKENDS = {"deepseek"}
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5.2-chat",
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        backend: str = "azure",  # "azure", "foundry", or "deepseek"
        base_url: Optional[str] = None,  # Custom base URL for OpenAI-compatible backends
        verbose: bool = False,  # Print API calls and responses
    ):
        """
        Initialize LLM client.
        
        Args:
            api_key: API key (reads from env var if not provided)
            model: Model name to use
            azure_endpoint: Azure OpenAI endpoint URL
            api_version: Azure API version
            backend: Backend type ("azure", "foundry", or "deepseek")
            base_url: Custom base URL for OpenAI-compatible backends
            verbose: If True, print API calls and responses for debugging
        """
        self.model = model
        self.backend = backend.lower()
        self.verbose = verbose
        
        if self.backend == "azure":
            # Azure OpenAI
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            )
        elif self.backend == "deepseek":
            # DeepSeek via OpenAI-compatible API
            self.client = OpenAI(
                base_url=base_url or os.getenv("DEEPSEEK_ENDPOINT"),
                api_key=api_key or os.getenv("DEEPSEEK_API_KEY")
            )
        elif self.backend == "foundry":
            # Azure AI Foundry via OpenAI-compatible API
            self.client = OpenAI(
                base_url=base_url or os.getenv("AZURE_FOUNDRY_ENDPOINT"),
                api_key=api_key or os.getenv("AZURE_FOUNDRY_API_KEY")
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'azure', 'foundry', or 'deepseek'.")
    
    @classmethod
    def from_env(
        cls,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        verbose: bool = False,
    ) -> "LLMClient":
        """
        Create LLM client from environment variables.
        
        Args:
            backend: Override backend type. If None, auto-detect from env vars.
            model: Override model/deployment name. If None, use backend-specific env var.
            verbose: If True, print API calls and responses for debugging.
        
        Environment variables:
            - LLM_BACKEND: "azure", "foundry", or "deepseek"
            - AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_MODEL, AZURE_OPENAI_API_VERSION
            - AZURE_FOUNDRY_ENDPOINT, AZURE_FOUNDRY_API_KEY, AZURE_FOUNDRY_MODEL
            - DEEPSEEK_ENDPOINT, DEEPSEEK_API_KEY, DEEPSEEK_MODEL
        """
        # Determine backend
        if backend is None:
            backend = os.getenv("LLM_BACKEND", "").lower()
        
        # Auto-detect if not specified
        if not backend:
            if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"):
                backend = "azure"
            elif os.getenv("AZURE_FOUNDRY_ENDPOINT") and os.getenv("AZURE_FOUNDRY_API_KEY"):
                backend = "foundry"
            elif os.getenv("DEEPSEEK_ENDPOINT") and os.getenv("DEEPSEEK_API_KEY"):
                backend = "deepseek"
            else:
                raise ValueError(
                    "Missing LLM credentials. Please set one of:\n"
                    "  - Azure: AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY\n"
                    "  - Foundry: AZURE_FOUNDRY_ENDPOINT and AZURE_FOUNDRY_API_KEY\n"
                    "  - DeepSeek: DEEPSEEK_ENDPOINT and DEEPSEEK_API_KEY\n"
                    "Or explicitly set LLM_BACKEND to 'azure', 'foundry', or 'deepseek'"
                )
        
        if backend == "deepseek":
            endpoint = os.getenv("DEEPSEEK_ENDPOINT")
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not endpoint or not api_key:
                raise ValueError(
                    "DeepSeek backend requires DEEPSEEK_ENDPOINT and DEEPSEEK_API_KEY environment variables"
                )
            return cls(
                backend="deepseek",
                base_url=endpoint,
                api_key=api_key,
                model=model or os.getenv("DEEPSEEK_MODEL", "DeepSeek-V3.2"),
                verbose=verbose,
            )
        elif backend == "foundry":
            endpoint = os.getenv("AZURE_FOUNDRY_ENDPOINT")
            api_key = os.getenv("AZURE_FOUNDRY_API_KEY")
            if not endpoint or not api_key:
                raise ValueError(
                    "Foundry backend requires AZURE_FOUNDRY_ENDPOINT and AZURE_FOUNDRY_API_KEY environment variables"
                )
            return cls(
                backend="foundry",
                base_url=endpoint,
                api_key=api_key,
                model=model or os.getenv("AZURE_FOUNDRY_MODEL", "Llama-4-Scout-17B-16E-Instruct"),
                verbose=verbose,
            )
        elif backend == "azure":
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if not endpoint or not api_key:
                raise ValueError(
                    "Azure backend requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables"
                )
            return cls(
                backend="azure",
                azure_endpoint=endpoint,
                api_key=api_key,
                model=model or os.getenv("AZURE_OPENAI_MODEL", "gpt-5.2-chat"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown LLM backend: {backend}. Supported: 'azure', 'foundry', 'deepseek'")
    
    def _parse_json_response(self, content: str, response_format: type):
        """
        Parse JSON response manually for backends without structured output support.
        
        Args:
            content: Raw response content (may contain JSON in markdown code blocks)
            response_format: Pydantic model to parse into
            
        Returns:
            Parsed Pydantic model instance
        """
        import json
        import re
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = content
        
        # Parse JSON and validate with Pydantic
        data = json.loads(json_str)
        return response_format.model_validate(data)
    
    def _get_json_schema_prompt(self, response_format: type) -> str:
        """
        Generate a prompt suffix with JSON schema for models without structured output.
        
        Args:
            response_format: Pydantic model to generate schema for
            
        Returns:
            Prompt string describing the expected JSON format
        """
        schema = response_format.model_json_schema()
        import json
        return f"\n\nYou MUST respond with valid JSON matching this schema:\n```json\n{json.dumps(schema, indent=2)}\n```\nRespond ONLY with the JSON object, no other text."

    @staticmethod
    def _is_guard_filter_error(error: Exception) -> bool:
        """Return True for Azure/OpenAI content-filter guardrail failures."""
        parts = [
            str(error),
            str(getattr(error, "code", "")),
            str(getattr(error, "body", "")),
            str(getattr(error, "message", "")),
        ]
        text = " ".join(parts).lower()
        markers = (
            "content_filter",
            "content filter",
            "responsibleaipolicy",
            "policy violation",
            "filtered due to",
            "safety system",
        )
        return any(marker in text for marker in markers)
    
    def generate(
        self,
        messages: List[Dict],
        response_format: Optional[type] = None,
        temperature: float = 0.7,
        max_retries: int = 6,
        retry_delay: float = 10.0
    ) -> tuple:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            response_format: Pydantic model for structured output (LLMPlanResponse, 
                           LLMMessageResponse, LLMInterruptResponse, or None for free-form)
            temperature: Sampling temperature (ignored for models that don't support it)
            max_retries: Maximum number of retries on rate limit errors
            retry_delay: Seconds to wait before retrying (default: 30)
            
        Returns:
            Tuple of (response, usage_dict)
            - response: Parsed Pydantic model if response_format provided, else string
            - usage_dict: {"prompt_tokens", "completion_tokens", "total_tokens"}
        """
        # Some models (like gpt-5.2-chat) don't support temperature parameter
        # Check if model supports temperature, otherwise omit it
        supports_temperature = "gpt-5" not in self.model.lower()
        
        # Verbose logging: print API call info
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[LLM API CALL] Backend: {self.backend}, Model: {self.model}")
            print(f"  Response format: {response_format.__name__ if response_format else 'None (free-form)'}")
            print(f"  Temperature: {temperature if supports_temperature else 'N/A (model default)'}")
            print(f"  Messages ({len(messages)}):")
            for i, msg in enumerate(messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:200]
                print(f"    [{i}] {role}: {content}{'...' if len(msg.get('content', '')) > 200 else ''}")
            print(f"{'='*60}")
        
        raw_content = None  # For verbose logging
        call_start = time.monotonic()
        retry_count = 0
        rate_limit_retry_count = 0
        api_error_retry_count = 0
        
        # Build optional kwargs (only include temperature if supported)
        temp_kwargs = {"temperature": temperature} if supports_temperature else {}
        
        # Retry loop for rate limit errors
        for attempt in range(max_retries + 1):
            try:
                if response_format is not None and self.backend in self.STRUCTURED_OUTPUT_BACKENDS:
                    # Use OpenAI's structured output (beta.chat.completions.parse)
                    raw_response = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        response_format=response_format,
                        **temp_kwargs
                    )
                    response = raw_response.choices[0].message.parsed
                    raw_content = str(response)
                elif response_format is not None:
                    # For OpenAI-compatible backends without structured output,
                    # add JSON schema to the last user message and parse manually.
                    messages_with_schema = messages.copy()
                    if messages_with_schema and messages_with_schema[-1]["role"] == "user":
                        messages_with_schema[-1] = messages_with_schema[-1].copy()
                        messages_with_schema[-1]["content"] += self._get_json_schema_prompt(response_format)

                    response_kwargs = {}
                    if self.backend in self.JSON_MODE_BACKENDS:
                        response_kwargs["response_format"] = {"type": "json_object"}
                    
                    raw_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages_with_schema,
                        **response_kwargs,
                        **temp_kwargs
                    )
                    raw_content = raw_response.choices[0].message.content
                    response = self._parse_json_response(raw_content, response_format)
                else:
                    # Free-form output
                    raw_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        **temp_kwargs
                    )
                    response = raw_response.choices[0].message.content
                    raw_content = response
                
                # Success - break out of retry loop
                break
                
            except RateLimitError as e:
                if attempt < max_retries:
                    retry_count += 1
                    rate_limit_retry_count += 1
                    print(f"\n[RATE LIMIT] Hit rate limit. Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    print(f"\n[RATE LIMIT] Max retries ({max_retries}) exceeded. Raising error.")
                    raise
            except APIError as e:
                if hasattr(e, "status_code") and e.status_code == 429:
                    if attempt < max_retries:
                        retry_count += 1
                        rate_limit_retry_count += 1
                        api_error_retry_count += 1
                        print(f"\n[RATE LIMIT] Azure 429 error. Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        print(f"\n[RATE LIMIT] Max retries ({max_retries}) exceeded. Raising error.")
                        raise
                else:
                    if self._is_guard_filter_error(e):
                        print(f"\n[GUARD FILTER] LLM request blocked by content filter/policy: {e}")
                    raise
        
        finish_reason = None
        guard_filter_count = 0
        if getattr(raw_response, "choices", None):
            finish_reason = getattr(raw_response.choices[0], "finish_reason", None)
            if str(finish_reason).lower() == "content_filter":
                guard_filter_count = 1
                print("\n[GUARD FILTER] LLM response was filtered by content policy.")

        response_usage = getattr(raw_response, "usage", None)
        prompt_tokens = int(getattr(response_usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(response_usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(response_usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": time.monotonic() - call_start,
            "retry_count": retry_count,
            "rate_limit_retry_count": rate_limit_retry_count,
            "api_error_retry_count": api_error_retry_count,
            "guard_filter_count": guard_filter_count,
            "finish_reason": None if finish_reason is None else str(finish_reason),
        }
        
        # Verbose logging: print response
        if self.verbose:
            print(f"\n[LLM RESPONSE] Backend: {self.backend}")
            print(f"  Usage: {usage}")
            print(f"  Raw content ({len(raw_content) if raw_content else 0} chars):")
            if raw_content:
                # Print first 500 chars
                print(f"    {raw_content[:500]}{'...' if len(raw_content) > 500 else ''}")
            print(f"{'='*60}\n")
        
        return response, usage
    
    def generate_plan(self, messages: List[Dict], temperature: float = 0.7) -> tuple:
        """Convenience method to generate a plan response."""
        return self.generate(messages, response_format=LLMPlanResponse, temperature=temperature)
    
    def generate_message(self, messages: List[Dict], temperature: float = 0.7) -> tuple:
        """Convenience method to generate a message response."""
        return self.generate(messages, response_format=LLMMessageResponse, temperature=temperature)
    
    def generate_interrupt_decision(self, messages: List[Dict], temperature: float = 0.7) -> tuple:
        """Convenience method to generate an interrupt decision response."""
        return self.generate(messages, response_format=LLMInterruptResponse, temperature=temperature)


# ============================================================================
# Utility functions
# ============================================================================

def load_env_file(env_path: str = ".env"):
    """
    Load environment variables from a .env file.
    
    Args:
        env_path: Path to .env file (will also check project root)
    """
    # Try current directory first
    if not os.path.exists(env_path):
        # Try to find .env in project root (relative to this file)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env_path = os.path.join(project_root, ".env")
    
    if not os.path.exists(env_path):
        return
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value


# Auto-load .env file on import
load_env_file()
