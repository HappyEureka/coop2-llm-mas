"""
Cognitive LLM utilities for MA-Crafter.

This module provides utilities for:
- Converting environment observations to LLM prompts
- Parsing LLM responses to SymbolicPlan objects
"""

from typing import Any, Optional, List, Dict

from ..plan.plan import SymbolicPlan, SymbolicAction
from .prompts import (
    build_system_prompt,
    build_observation_prompt,
    format_memory,
)
from .llm_client import (
    LLMPlanResponse,
    LLMInterruptResponse,
    InterruptDecision,
    LLMAction,
    TaskSpecification,
    MoveAction,
    CollectAction,
    PlaceAction,
    CraftAction,
    SleepAction,
    NoopAction,
    NavigateAction,
    ShareAction,
)


# ============================================================================
# Observation to Prompt Conversion
# ============================================================================

def extract_status(observation: Any) -> Optional[Dict]:
    """Extract status dict from observation."""
    if isinstance(observation, dict):
        status = {}
        if 'health' in observation:
            status['health'] = observation['health']
        if 'food' in observation:
            status['food'] = observation['food']
        if 'drink' in observation:
            status['drink'] = observation['drink']
        if 'energy' in observation:
            status['energy'] = observation['energy']
        if 'inventory' in observation:
            status['inventory'] = observation['inventory']
        if 'tools' in observation:
            status['tools'] = observation['tools']
        if 'status' in observation:
            # Nested status dict
            status.update(observation['status'])
        return status if status else None
    return None


def extract_position(observation: Any) -> Any:
    """Extract position from observation."""
    if isinstance(observation, dict):
        return observation.get('position')
    return None


def extract_facing(observation: Any) -> Optional[str]:
    """Extract facing direction from observation."""
    if isinstance(observation, dict):
        return observation.get('facing')
    return None


def extract_visible_area(observation: Any) -> Any:
    """Extract visible area from observation."""
    if isinstance(observation, dict):
        return observation.get('semantic_grid') or observation.get('visible_area')
    return None


def build_plan_prompt(
    observation: Any,
    env_step: int,
    agent_id: str,
    system_prompt: Optional[str] = None,
    max_actions: int = 6,
    all_agents: Optional[Dict[str, Any]] = None,
    agent_names: Optional[List[str]] = None,
    agent_states: Optional[Dict[str, str]] = None,
    messages: Optional[List[Dict]] = None,
    memory: Optional[List[Dict]] = None,
    coop_config: Optional[str] = None,
    symbolic_view: Optional[str] = None,
    target_hints: Optional[str] = None,
) -> List[Dict]:
    """
    Build LLM prompt messages for plan generation from observation.
    
    Args:
        observation: Environment observation dict
        env_step: Current environment step
        agent_id: Agent identifier
        system_prompt: Custom system prompt (uses default if not provided)
        max_actions: Maximum actions per plan
        all_agents: Dict of agent_id -> agent object
        agent_names: List of agent IDs
        agent_states: Dict of agent_id -> state string
        messages: List of received messages
        memory: List of memory events from AgentMemory.get_events()
        coop_config: Formatted cooperative configuration string from env.get_config_observation()
        symbolic_view: Symbolic view text from coop_env info['symbolic_view']
        target_hints: Compact current target list from env info['target_hints']
        
    Returns:
        List of message dicts for LLM API call
    """
    if system_prompt is None:
        system_prompt = build_system_prompt(
            agent_id=agent_id,
            max_actions=max_actions,
            include_env_description=True,
        )
    
    obs_prompt = build_observation_prompt(
        env_step=env_step,
        agent_id=agent_id,
        all_agents=all_agents,
        agent_names=agent_names,
        agent_states=agent_states,
        status=extract_status(observation),
        position=extract_position(observation),
        facing=extract_facing(observation),
        visible_area=extract_visible_area(observation),
        messages=messages,
        memory=memory,
        coop_config=coop_config,
        symbolic_view=symbolic_view,
        target_hints=target_hints,
    )
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": obs_prompt}
    ]


def build_interrupt_prompt(
    observation: Any,
    env_step: int,
    agent_id: str,
    current_plan: Optional[SymbolicPlan],
    received_messages: List[Dict],
    system_prompt: Optional[str] = None,
    max_actions: int = 6,
    memory: Optional[List[Dict]] = None,
    coop_config: Optional[str] = None,
    symbolic_view: Optional[str] = None,
    target_hints: Optional[str] = None,
    extra_context: Optional[str] = None,
) -> List[Dict]:
    """
    Build LLM prompt messages for interrupt handling.
    
    Args:
        observation: Environment observation dict
        env_step: Current environment step
        agent_id: Agent identifier
        current_plan: The current plan being executed
        received_messages: List of messages that triggered the interrupt
        system_prompt: Custom system prompt (uses default if not provided)
        max_actions: Maximum actions per plan
        memory: List of memory events from AgentMemory.get_events()
        coop_config: Formatted cooperative configuration string from env.get_config_observation()
        symbolic_view: Symbolic view text from coop_env info['symbolic_view']
        target_hints: Compact current target list from env info['target_hints']
        extra_context: Optional extra prompt section shown after the memory (e.g. earlier proposals)
        
    Returns:
        List of message dicts for LLM API call
    """
    if system_prompt is None:
        system_prompt = build_system_prompt(
            agent_id=agent_id,
            max_actions=max_actions,
            include_env_description=True,
        )
    
    # Build interrupt-specific user prompt
    lines = [
        f"Step {env_step}: You have been INTERRUPTED by incoming messages.",
        "",
    ]
    
    # Cooperative configuration (important for understanding requirements)
    if coop_config:
        lines.append("## Cooperative Configuration")
        lines.append(coop_config)
        lines.append("")
    
    # Symbolic view (detailed view with entity IDs)
    if symbolic_view:
        lines.append("## Symbolic View")
        lines.append(symbolic_view)
        lines.append("")

    if target_hints:
        lines.append("## Current Reachable Targets")
        lines.append(target_hints)
        lines.append("")
    
    # Current observation
    lines.append("## Current Observation")
    status = extract_status(observation)
    position = extract_position(observation)
    if status:
        lines.append(f"Health: {status.get('health', '?')}, Food: {status.get('food', '?')}, Drink: {status.get('drink', '?')}, Energy: {status.get('energy', '?')}")
        if 'inventory' in status:
            lines.append(f"Inventory: {status['inventory']}")
    if position:
        lines.append(f"Position: {position}")
    lines.append("")
    
    # Memory (recent events)
    memory_text = format_memory(memory, max_events=5) if memory else ""
    if memory_text:
        lines.append("## Recent Memory")
        memory_lines = memory_text.splitlines()
        if memory_lines and memory_lines[0].strip() == "RECENT MEMORY:":
            memory_lines = memory_lines[1:]
        lines.extend(memory_lines)
        lines.append("")
    
    # Extra topology context (e.g. earlier Broadcast Chain proposals)
    if extra_context:
        lines.append(extra_context.rstrip())
        lines.append("")

    # Current plan and status
    lines.append("## Current Plan Status")
    if current_plan:
        lines.append(f"Task: {current_plan.specification}")
        lines.append(f"Plan ID: {current_plan.plan_id}")
        lines.append(f"Status: {current_plan.status.value if current_plan.status else 'in_progress'}")
        
        total_actions = len(current_plan.actions)
        current_idx = current_plan.current_action_index
        completed_actions = current_idx
        remaining_actions = total_actions - current_idx
        
        lines.append(f"Progress: {completed_actions}/{total_actions} actions completed ({remaining_actions} remaining)")
        
        if current_idx < total_actions:
            current_action = current_plan.actions[current_idx]
            lines.append(f"Current action: {current_action}")
        
        if remaining_actions > 0:
            lines.append("Remaining actions:")
            for i in range(current_idx, min(current_idx + 5, total_actions)):  # Show up to 5 remaining
                lines.append(f"  {i+1}. {current_plan.actions[i]}")
            if total_actions - current_idx > 5:
                lines.append(f"  ... and {total_actions - current_idx - 5} more")
    else:
        lines.append("No current plan.")
    lines.append("")
    
    # Messages received
    lines.append("## Messages Received (Reason for Interrupt)")
    for msg in received_messages:
        lines.append(f"From {msg['sender']}: {msg['content']}")
        if msg.get('metadata'):
            lines.append(f"  (metadata: {msg['metadata']})")
    lines.append("")
    
    lines.append("## Your Decision")
    lines.append(
        "Decide whether to resume the current plan or replan. Resume if the "
        "plan is still useful; replan if messages or progress make a different "
        "task/action sequence better. If replanning, provide the replacement plan."
    )
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(lines)}
    ]


# ============================================================================
# LLM Response to Plan Conversion
# ============================================================================

def extract_action_parameters(llm_action: LLMAction) -> Dict[str, Any]:
    """Extract parameters from typed action class to dict for SymbolicAction."""
    if isinstance(llm_action, MoveAction):
        return {"direction": llm_action.direction.value, "num_steps": llm_action.num_steps}
    elif isinstance(llm_action, CollectAction):
        return {"target": llm_action.target.value, "steps": llm_action.steps}
    elif isinstance(llm_action, PlaceAction):
        return {"object_type": llm_action.item.value}
    elif isinstance(llm_action, CraftAction):
        return {"object_type": llm_action.item.value}
    elif isinstance(llm_action, SleepAction):
        return {}
    elif isinstance(llm_action, NoopAction):
        return {}
    elif isinstance(llm_action, NavigateAction):
        return {"object_type": llm_action.object_type, "item_id": llm_action.item_id, "timeout": llm_action.timeout}
    elif isinstance(llm_action, ShareAction):
        return {"recipient_agent_id": llm_action.recipient_agent_id, "resource_type": llm_action.resource_type, "quantity": llm_action.quantity}
    else:
        return {}


def parse_plan_response(
    llm_response: LLMPlanResponse,
    agent_id: str,
    env_step: int,
    plan_id: int = 1,
) -> SymbolicPlan:
    """
    Convert LLM plan response to SymbolicPlan.
    
    Args:
        llm_response: Parsed LLMPlanResponse from LLM
        agent_id: Agent identifier
        env_step: Current environment step
        plan_id: Plan ID number
        
    Returns:
        SymbolicPlan object
    """
    actions = []
    for llm_action in llm_response.actions:
        action_type = llm_action.action_type
        args = extract_action_parameters(llm_action)
        
        action = SymbolicAction(
            action_type=action_type,
            args=args
        )
        actions.append(action)

    _attach_task_target_to_collect_actions(llm_response.task, actions)
    _ensure_task_terminal_action(llm_response.task, actions)
    
    return SymbolicPlan(
        specification=str(llm_response.task),
        actions=actions,
        plan_id=plan_id,
        agent_id=agent_id,
        created_at_step=env_step
    )


def _collect_task_args(task_spec: TaskSpecification) -> Dict[str, Any]:
    task_name = str(task_spec.task.value or "").lower()
    if not task_name.startswith("collect_"):
        return {}
    target = task_name.removeprefix("collect_")
    return {
        "target": target,
        "resource_type": task_spec.object_type,
        "task_id": task_spec.object_id,
        "target_id": task_spec.object_id,
        "item_id": task_spec.object_id,
    }


def _attach_task_target_to_collect_actions(
    task_spec: TaskSpecification,
    actions: List[SymbolicAction],
) -> None:
    """Carry task ids from collect task specifications into collect actions."""
    task_args = _collect_task_args(task_spec)
    if not task_args:
        return
    target = str(task_args["target"]).lower()
    for action in actions:
        if action.action_type != "collect":
            continue
        action_target = str(action.args.get("target") or "").lower()
        if action_target and action_target != target:
            continue
        for key, value in task_args.items():
            action.args.setdefault(key, value)


def _ensure_task_terminal_action(task_spec: TaskSpecification, actions: List[SymbolicAction]) -> None:
    """Ensure task-labeled plans include an action that can achieve the task."""
    task_name = str(task_spec.task.value or "").lower()

    if task_name.startswith("collect_"):
        task_args = _collect_task_args(task_spec)
        target = task_args.get("target", task_name.removeprefix("collect_"))
        has_matching_collect = any(
            action.action_type == "collect"
            and str(action.args.get("target", "")).lower() == target
            for action in actions
        )
        if not has_matching_collect:
            actions.append(SymbolicAction("collect", task_args or {"target": target}))
        return

    if task_name.startswith("make_"):
        item = task_name.removeprefix("make_")
        has_matching_craft = any(
            action.action_type == "craft"
            and str(action.args.get("object_type", "")).lower() == item
            for action in actions
        )
        if not has_matching_craft:
            actions.append(SymbolicAction("craft", {"object_type": item}))
        return

    if task_name.startswith("place_"):
        item = task_name.removeprefix("place_")
        has_matching_place = any(
            action.action_type == "place"
            and str(action.args.get("object_type", "")).lower() == item
            for action in actions
        )
        if not has_matching_place:
            actions.append(SymbolicAction("place", {"object_type": item}))


def parse_interrupt_response(
    llm_response: LLMInterruptResponse,
    agent_id: str,
    env_step: int,
    plan_id: int = 1,
) -> tuple:
    """
    Parse interrupt response and return decision and optional new plan.
    
    Args:
        llm_response: Parsed LLMInterruptResponse from LLM
        agent_id: Agent identifier
        env_step: Current environment step
        plan_id: Plan ID number for new plan
        
    Returns:
        Tuple of (decision: InterruptDecision, new_plan: Optional[SymbolicPlan])
    """
    if llm_response.decision == InterruptDecision.RESUME:
        return (InterruptDecision.RESUME, None)
    
    # Decision is REPLAN
    if llm_response.new_plan is None:
        return (InterruptDecision.REPLAN, None)
    
    new_plan = parse_plan_response(
        llm_response.new_plan,
        agent_id=agent_id,
        env_step=env_step,
        plan_id=plan_id
    )
    return (InterruptDecision.REPLAN, new_plan)
