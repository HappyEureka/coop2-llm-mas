"""
Prompt construction utilities for LLM agents.

This module provides clean, modular functions for building prompts
that describe the environment, agent states, and observations.
"""

from typing import Dict, List, Any, Optional
from enum import Enum


# ============================================================================
# Environment Description
# ============================================================================

ENV_DESCRIPTION = """You are playing CUBE - a cooperative block-pushing game.

GOAL:
Maximize team score by delivering blocks to the goal column (rightmost column).
Delivering a block gives score equal to its weight.

GRID:
- Simple K×K grid (main experiments use 15×15)
- Goal column is the rightmost column (column K-1)
- Agents can move in 4 directions: up, down, left, right
- Blocks are squares with side length = weight

BLOCKS:
- Each block has a weight W, which determines both its size and required push force
- A block of weight W requires W agents pushing the same face in the same step
- Blocks move when sufficient force is applied
- A block is delivered when it reaches the goal column

COOPERATIVE MECHANICS:
- To push a block of weight W, exactly W agents must:
  1. Be adjacent to the same side of the block
  2. All move INTO the block simultaneously (push)
- Agents pushing a block move forward with it
- Coordination is essential for heavier blocks

Use the symbolic view for exact block IDs, positions, required agents, and distance to goal."""


def get_env_description() -> str:
    """Get the environment description."""
    return ENV_DESCRIPTION


# ============================================================================
# Agent State Formatting
# ============================================================================

class AgentStateCode(str, Enum):
    """Single-letter codes for agent states."""
    REASONING = "R"      # Agent is thinking/planning
    INTERRUPTED = "I"    # Agent was interrupted by message
    EXECUTING = "X"      # Agent is executing a plan
    WAITING = "W"        # Agent is waiting (ready for next step)


def get_state_code(state_value: str) -> str:
    """Convert agent state to single-letter code."""
    state_map = {
        "reasoning": "R",
        "interrupted": "I",
        "executing": "X",
        "waiting": "W",
        "idle": "W",
    }
    return state_map.get(state_value.lower(), "?")


def format_agent_states(
    current_agent_id: str,
    all_agents: Dict[str, Any],
) -> str:
    """
    Format information about all agents and their states.
    
    Args:
        current_agent_id: ID of the agent receiving this prompt
        all_agents: Dict mapping agent_id -> agent object (with .state attribute)
    
    Returns:
        Formatted string describing agent states
    """
    lines = []
    lines.append(f"TEAM ({len(all_agents)} agents):")
    lines.append(f"You are: {current_agent_id}")
    
    # Explicitly list collaborators
    collaborators = [a for a in all_agents.keys() if a != current_agent_id]
    if collaborators:
        lines.append(f"Your collaborators: {collaborators}")
    
    lines.append("")
    lines.append("Agent States (R=Reasoning, I=Interrupted, X=Executing, W=Waiting):")
    
    for agent_id, agent in all_agents.items():
        # Get state code
        if hasattr(agent, 'state'):
            state_value = agent.state.value if hasattr(agent.state, 'value') else str(agent.state)
            state_code = get_state_code(state_value)
        else:
            state_code = "?"
        
        # Mark current agent
        marker = " <- YOU" if agent_id == current_agent_id else ""
        lines.append(f"  {agent_id}: [{state_code}]{marker}")
    
    return "\n".join(lines)


def format_agent_states_simple(
    current_agent_id: str,
    agent_names: List[str],
    agent_states: Optional[Dict[str, str]] = None,
) -> str:
    """
    Format agent states from simple lists/dicts.
    
    Args:
        current_agent_id: ID of the agent receiving this prompt
        agent_names: List of all agent IDs
        agent_states: Optional dict mapping agent_id -> state string
    
    Returns:
        Formatted string describing agent states
    """
    lines = []
    lines.append(f"TEAM ({len(agent_names)} agents):")
    lines.append(f"You are: {current_agent_id}")
    
    # Explicitly list collaborators
    collaborators = [a for a in agent_names if a != current_agent_id]
    if collaborators:
        lines.append(f"Your collaborators: {collaborators}")
    
    if agent_states:
        lines.append("")
        lines.append("Agent States (R=Reasoning, I=Interrupted, X=Executing, W=Waiting):")
        for agent_id in agent_names:
            state = agent_states.get(agent_id, "unknown")
            state_code = get_state_code(state)
            marker = " <- YOU" if agent_id == current_agent_id else ""
            lines.append(f"  {agent_id}: [{state_code}]{marker}")
    
    return "\n".join(lines)


# ============================================================================
# Agent Status Formatting (Health, Tools, Resources)
# ============================================================================

def format_agent_status(
    position: Optional[tuple] = None,
    block_info: Optional[List[Dict]] = None,
    goal_column: Optional[int] = None,
    grid_size: Optional[int] = None,
) -> str:
    """
    Format the agent's current status for CUBE.
    
    Args:
        position: Agent position (row, col)
        block_info: List of block dicts with id, weight, position
        goal_column: Goal column index
        grid_size: Grid size K
    
    Returns:
        Formatted string describing agent status
    """
    lines = []
    lines.append("YOUR STATUS:")
    
    if position is not None:
        lines.append(f"  Position: row={position[0]}, col={position[1]}")
    
    if grid_size is not None:
        lines.append(f"  Grid size: {grid_size}×{grid_size}")
    
    if goal_column is not None:
        lines.append(f"  Goal column: {goal_column} (rightmost)")
    
    if block_info:
        lines.append("  Blocks:")
        for block in block_info:
            bid = block.get('id', '?')
            weight = block.get('weight', '?')
            pos = block.get('position', ('?', '?'))
            dist_to_goal = goal_column - pos[1] if goal_column and isinstance(pos[1], int) else '?'
            lines.append(f"    Block {bid}: weight={weight}, pos=({pos[0]},{pos[1]}), distance_to_goal={dist_to_goal}")
    
    return "\n".join(lines)


def format_status_from_dict(status_dict: Dict[str, Any]) -> str:
    """
    Format agent status from a dictionary for CUBE.
    
    Expected keys: position, blocks, goal_column, grid_size
    """
    return format_agent_status(
        position=status_dict.get('position'),
        block_info=status_dict.get('blocks'),
        goal_column=status_dict.get('goal_column'),
        grid_size=status_dict.get('grid_size'),
    )


# ============================================================================
# Observation Formatting
# ============================================================================

def format_visible_area(semantic_grid: Any) -> str:
    """
    Format the visible area from a semantic grid.
    
    Args:
        semantic_grid: Grid representation of what the agent sees
    
    Returns:
        Formatted string describing visible objects
    """
    if semantic_grid is None:
        return "Visible area: unknown"
    
    # If it's a simple string or already formatted
    if isinstance(semantic_grid, str):
        return f"Visible area:\n{semantic_grid}"
    
    # If it's a dict of object counts
    if isinstance(semantic_grid, dict):
        items = [f"{k}: {v}" for k, v in semantic_grid.items() if v > 0]
        if items:
            return f"Nearby objects: {', '.join(items)}"
        return "Nearby objects: none visible"
    
    # Fallback
    return f"Visible area: {semantic_grid}"


def format_position(position: Any, facing: Optional[str] = None) -> str:
    """Format agent position and facing direction."""
    lines = []
    if position is not None:
        lines.append(f"Position: {position}")
    if facing is not None:
        lines.append(f"Facing: {facing}")
    return "\n".join(lines) if lines else ""


def format_memory(memory_events: List[Dict], max_events: int = 5) -> str:
    """
    Format memory events for inclusion in prompts.
    
    Args:
        memory_events: List of event dicts from AgentMemory.get_events()
        max_events: Maximum number of recent events to include
        
    Returns:
        Formatted string describing recent memory
    """
    if not memory_events:
        return ""
    
    # Take most recent events
    recent = memory_events[-max_events:] if len(memory_events) > max_events else memory_events
    
    lines = ["RECENT MEMORY:"]
    for event in recent:
        step = event.get('env_step', '?')
        event_type = event.get('type', '')
        
        if event_type == 'message_in':
            sender = event.get('sender', '?')
            content = event.get('content', '')
            lines.append(f"  [Step {step}] Received from {sender}: {content}")
        elif event_type == 'message_out':
            recipients = event.get('recipients', [])
            content = event.get('content', '')
            lines.append(f"  [Step {step}] Sent to {recipients}: {content}")
        elif event_type == 'plan':
            spec = event.get('specification', '?')
            plan_id = event.get('plan_id', '?')
            actions = event.get('actions', [])
            lines.append(f"  [Step {step}] Plan #{plan_id}: {spec}")
            if actions:
                action_str = ", ".join(str(a) for a in actions[:3])
                if len(actions) > 3:
                    action_str += f"... ({len(actions)} actions)"
                lines.append(f"    Actions: {action_str}")
    
    return "\n".join(lines)


# ============================================================================
# Complete Prompt Building
# ============================================================================

def build_system_prompt(
    agent_id: str,
    max_actions: int = 6,
    include_env_description: bool = True,
) -> str:
    """
    Build the system prompt for an LLM agent in CUBE.
    
    Args:
        agent_id: The agent's identifier
        max_actions: Maximum actions per plan
        include_env_description: Whether to include full env description
    
    Returns:
        Complete system prompt
    """
    parts = []
    
    # Identity
    parts.append(f"You are agent '{agent_id}' in CUBE, a multi-agent cooperative block-pushing game.")
    parts.append("")
    
    # Environment description
    if include_env_description:
        parts.append(ENV_DESCRIPTION)
        parts.append("")
    
    # Available tasks
    parts.append("""AVAILABLE TASKS (choose one as your plan's goal):
- deliver_block: Move a specific block to the goal column
- coordinate_push: Coordinate agents on the same block face
- approach_block: Move to the left push cells of a block
- wait_for_others: Wait briefly for teammates to arrive""")
    parts.append("")
    
    # Available actions
    parts.append("""AVAILABLE ACTIONS:
1. move: direction (up/down/left/right), num_steps (1-10)
   - Moves the agent in the specified direction for num_steps
2. navigate: entity_type='block', entity_id=block_id, timeout (1-100)
   - Move step-by-step toward the target block's left push cells
   - Use this before pushing unless you are already on a listed left_push_cell
3. push: block_id (integer), num_steps (1-10)
   - Push the specified block by moving into it for num_steps
   - Agent must be adjacent to the block
   - Direction is determined automatically based on agent position relative to block
4. wait: num_steps (1-10)
   - Do nothing for num_steps (useful for coordination)""")
    parts.append("")
    
    # Planning Strategy
    parts.append("""PLANNING STRATEGY:
- Choose a feasible block that can increase team score soon.
- Weight-1 blocks can be delivered by one agent; weight-2 and weight-3 blocks require coordination.
- Push toward the goal by standing on the block's left push cells and using push(block_id=...).
- Prefer navigate(entity_type='block', entity_id=...) to reach the left push cells; use manual move only for short obvious moves.
- For heavier blocks, coordinate the same block_id and same left face with enough teammates.
- Use wait only for synchronization; otherwise move or push.""")
    parts.append("")
    
    # Step-by-step planning guide
    parts.append("""ACTION ORDER:
1. Navigate or move to one listed left_push_cell for the target block.
2. If the block needs teammates, wait or coordinate until enough agents are on compatible left_push_cells.
3. Push the target block for several steps.

Do not push a block before reaching its left push cells. A blocked or wrong-block push counts as failure.""")
    parts.append("")
    
    # Instructions
    parts.append("""INSTRUCTIONS:
1. Use exact block IDs from the symbolic view.
2. Make a short plan with 2-6 actions; navigate can cover long movement in one action.
3. Prefer actions that can deliver score under the remaining time.
4. For heavy blocks, only choose them when enough agents can coordinate on the same block.
5. Communicate your intended block_id and role when coordination is needed.""")
    
    return "\n".join(parts)


def build_observation_prompt(
    env_step: int,
    agent_id: str,
    all_agents: Optional[Dict[str, Any]] = None,
    agent_names: Optional[List[str]] = None,
    agent_states: Optional[Dict[str, str]] = None,
    status: Optional[Dict[str, Any]] = None,
    position: Any = None,
    facing: Optional[str] = None,
    visible_area: Any = None,
    messages: Optional[List[Dict]] = None,
    memory: Optional[List[Dict]] = None,
    coop_config: Optional[str] = None,
    symbolic_view: Optional[str] = None,
) -> str:
    """
    Build the observation/user prompt for plan generation.
    
    Args:
        env_step: Current environment step
        agent_id: The agent's identifier
        all_agents: Dict of agent_id -> agent object (alternative to agent_names)
        agent_names: List of agent IDs (alternative to all_agents)
        agent_states: Dict of agent_id -> state string
        status: Agent status dict (health, inventory, etc.)
        position: Agent position
        facing: Agent facing direction
        visible_area: Semantic grid or visible objects
        messages: List of received messages
        memory: List of memory events from AgentMemory.get_events()
        coop_config: Formatted cooperative configuration string from env.get_config_observation()
        symbolic_view: Symbolic view text from coop_env info['symbolic_view']
    
    Returns:
        Complete observation prompt
    """
    parts = []
    parts.append(f"=== STEP {env_step} ===")
    parts.append("")
    
    # Cooperative configuration (important for understanding requirements)
    if coop_config:
        parts.append(coop_config)
        parts.append("")
    
    # Agent states
    if all_agents:
        parts.append(format_agent_states(agent_id, all_agents))
    elif agent_names:
        parts.append(format_agent_states_simple(agent_id, agent_names, agent_states))
    parts.append("")
    
    # Agent status
    if status:
        parts.append(format_status_from_dict(status))
        parts.append("")
    
    # Position
    pos_str = format_position(position, facing)
    if pos_str:
        parts.append(pos_str)
        parts.append("")
    
    # Visible area
    if visible_area is not None:
        parts.append(format_visible_area(visible_area))
        parts.append("")
    
    # Symbolic view (detailed view with entity IDs)
    if symbolic_view:
        parts.append("SYMBOLIC VIEW:")
        parts.append(symbolic_view)
        parts.append("")
    
    # Memory (recent events)
    if memory:
        memory_str = format_memory(memory)
        if memory_str:
            parts.append(memory_str)
            parts.append("")
    
    # Messages from other agents (current step - may overlap with memory)
    if messages:
        parts.append("MESSAGES RECEIVED:")
        for msg in messages:
            sender = msg.get('sender', 'unknown')
            content = msg.get('content', '')
            parts.append(f"  From {sender}: {content}")
        parts.append("")
    
    parts.append("""Generate a short plan to maximize delivery score.

Guidelines:
- Use exact block IDs from the symbolic view.
- Use navigate(entity_type='block', entity_id=...) to reach a listed left_push_cell before push(block_id=...).
- Weight-1 blocks need one pusher; weight-2 needs two; weight-3 needs three.
- Use wait only to synchronize a coordinated push.
- Avoid wait-only plans.""")
    
    return "\n".join(parts)


def build_message_prompt(
    env_step: int,
    agent_id: str,
    other_agents: List[str],
    current_task: Optional[str] = None,
    status: Optional[Dict[str, Any]] = None,
    context: Optional[str] = None,
) -> str:
    """
    Build a prompt for message generation.
    
    Args:
        env_step: Current environment step
        agent_id: The agent's identifier
        other_agents: List of other agent IDs that can receive messages
        current_task: Current task being worked on
        status: Agent status dict
        context: Additional context for message generation
    
    Returns:
        Prompt for message generation
    """
    parts = []
    parts.append(f"=== STEP {env_step} ===")
    parts.append(f"You are: {agent_id}")
    parts.append(f"Other agents you can message: {other_agents}")
    parts.append("")
    
    if current_task:
        parts.append(f"Your current task: {current_task}")
    
    if status:
        parts.append(format_status_from_dict(status))
    
    if context:
        parts.append("")
        parts.append(f"Context: {context}")
    
    parts.append("")
    parts.append("Decide if you should send a message to coordinate with other agents.")
    parts.append("If sending, specify recipients and message content.")
    
    return "\n".join(parts)
