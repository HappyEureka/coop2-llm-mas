"""
Symbolic action interface for CUBE.

This module provides symbolic actions (data structures) and their executor
that translates symbolic actions to primitive actions.

CUBE only has 5 primitive actions: stay(0), up(1), down(2), left(3), right(4)
We support two symbolic actions: move(direction, num_steps) and wait(num_steps)
"""

from typing import List, Optional, Dict, Tuple, Literal, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from ..constants import ACTION_NAME_TO_VALUE
import numpy as np


@dataclass
class SymbolicAction:
    """A single symbolic action within a plan."""
    action_type: str
    args: Dict[str, Any] = field(default_factory=dict)
    
    # Execution tracking
    start_step: Optional[int] = None
    end_step: Optional[int] = None
    status: Optional[str] = None  # "success", "failed", "executing"
    failure_reason: Optional[str] = None
    primitive_action: Optional[str] = None  # Step-level action name (e.g., "move_left", "do")
    primitive_action_history: Dict[int, str] = field(default_factory=dict)  # Per-step primitive actions
    
    def __post_init__(self):
        """Set step-level action name based on action type and args."""
        if self.primitive_action is None:
            self.primitive_action = self._map_to_step_action()
    
    def _map_to_step_action(self) -> str:
        """Map symbolic action to step-level action name.
        For CUBE, we only have move, wait, and push."""
        if self.action_type == 'move':
            direction = self.args.get('direction', 'up')
            return f"move_{direction}"
        elif self.action_type == 'wait':
            return 'noop'
        elif self.action_type == 'push':
            # Direction determined dynamically based on agent/block positions
            return 'push'
        elif self.action_type == 'navigate':
            return 'navigate'
        else:
            return self.action_type
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "action_type": self.action_type,
            "args": self.args,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "duration": self.end_step - self.start_step if self.end_step and self.start_step else None,
            "primitive_action": self.primitive_action,
            "primitive_action_history": self.primitive_action_history
        }


class SymbolicActionStatus(Enum):
    """Status of symbolic action execution."""
    PENDING = "pending"  # Action started but not complete
    SUCCESS = "success"  # Action completed successfully
    FAILED = "failed"    # Action failed
    

class ActionRecord:
    """Record of a symbolic action execution."""
    def __init__(self, action_type: str, args: Dict, start_step: int):
        self.action_type = action_type
        self.args = args
        self.start_step = start_step
        self.end_step: Optional[int] = None
        self.status: SymbolicActionStatus = SymbolicActionStatus.PENDING
        self.failure_reason: Optional[str] = None
        self.primitive_action: Optional[str] = None
        self.primitive_action_history: Dict[int, str] = {}  # Per-step primitive actions
        
    def complete(self, end_step: int, status: SymbolicActionStatus, failure_reason: Optional[str] = None):
        """Mark action as complete."""
        self.end_step = end_step
        self.status = status
        self.failure_reason = failure_reason
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "action_type": self.action_type,
            "args": self.args,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "status": self.status.value,
            "failure_reason": self.failure_reason,
            "duration": self.end_step - self.start_step if self.end_step else None,
            "primitive_action": self.primitive_action,
            "primitive_action_history": self.primitive_action_history
        }


class SymbolicActionExecutor:
    """
    Action executor that translates symbolic actions to primitive actions.
    
    For CUBE, we only support:
    - move(direction, num_steps): Move in a direction for N steps
    - wait(num_steps): Do nothing for N steps
    """
    
    def __init__(self, agent_id: str = None):
        self.agent_id = agent_id
        self.actions = {
            "noop": self.noop,
            "move": self.move,
            "wait": self.wait,
            "push": self.push,
            "navigate": self.navigate,
        }
        
        # State for multi-step actions
        self.current_symbolic_action: Optional[ActionRecord] = None
        self.current_env_step: int = 0
        self.action_history: List[ActionRecord] = []
        
        # Move state
        self.move_steps_remaining: int = 0
        self.move_direction: Optional[str] = None
        
        # Wait state
        self.wait_steps_remaining: int = 0

        # Navigate state
        self.navigate_steps_remaining: int = 0
        self.navigate_entity_type: Optional[str] = None
        self.navigate_entity_id: Optional[int] = None
        
        # Push state
        self.push_steps_remaining: int = 0
        self.push_block_id: Optional[int] = None
        self.push_direction: Optional[str] = None
        self.push_had_success: bool = False
        self.last_push_failure_reason: Optional[str] = None
    
    def start_symbolic_action(self, action_type: str, args: Dict):
        """Start tracking a new symbolic action."""
        # Move completed action to history if any
        if self.current_symbolic_action and self.current_symbolic_action.status != SymbolicActionStatus.PENDING:
            self.action_history.append(self.current_symbolic_action)
            
        # Only start a new action if there's no current pending action
        if self.current_symbolic_action and self.current_symbolic_action.status == SymbolicActionStatus.PENDING:
            # Don't interrupt a pending action - this should not happen in normal flow
            return
        
        # Start new action at current step
        self.current_symbolic_action = ActionRecord(action_type, args, self.current_env_step)
    
    def complete_current_action(self, status: SymbolicActionStatus, failure_reason: Optional[str] = None):
        """Complete the current symbolic action."""
        if self.current_symbolic_action:
            self.current_symbolic_action.complete(self.current_env_step, status, failure_reason)
            self.action_history.append(self.current_symbolic_action)
            self.current_symbolic_action = None
    
    def update_env_step(self, step: int):
        """Update the current environment step counter."""
        self.current_env_step = step
    
    def check_termination_condition(
        self,
        world_state: Optional[Dict] = None,
        action_outcome: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Check if current symbolic action's termination condition is met.
        Called AFTER the environment step executes.
        
        Args:
            world_state: Current world state (not used in CUBE but kept for compatibility)
            action_outcome: Grounded outcome from the environment for this primitive step.
            
        Returns:
            Dict with 'status' ('pending', 'success', 'failed') and optional 'failure_reason'
        """
        if not self.current_symbolic_action or self.current_symbolic_action.status != SymbolicActionStatus.PENDING:
            return {"status": "pending"}
        
        action_type = self.current_symbolic_action.action_type
        
        # Single-step actions (complete after first step)
        if action_type == 'noop':
            self.complete_current_action(SymbolicActionStatus.SUCCESS)
            return {"status": "success"}
        
        # Move action - check if all steps completed
        elif action_type == 'move':
            if self.move_steps_remaining <= 0:
                self.complete_current_action(SymbolicActionStatus.SUCCESS)
                return {"status": "success"}
            return {"status": "pending"}
        
        # Wait action - check if all steps completed
        elif action_type == 'wait':
            if self.wait_steps_remaining <= 0:
                self.complete_current_action(SymbolicActionStatus.SUCCESS)
                return {"status": "success"}
            return {"status": "pending"}

        # Navigate action - succeeds when the agent reaches the target cell.
        elif action_type == 'navigate':
            if self._at_navigation_target(world_state):
                self.complete_current_action(SymbolicActionStatus.SUCCESS)
                return {"status": "success"}
            if self.navigate_steps_remaining <= 0:
                failure_reason = "navigation_target_not_reached"
                self.complete_current_action(SymbolicActionStatus.FAILED, failure_reason)
                return {"status": "failed", "failure_reason": failure_reason}
            return {"status": "pending"}
        
        # Push action - check if all steps completed
        elif action_type == 'push':
            if action_outcome:
                outcome_status = action_outcome.get("status")
                if outcome_status == "success":
                    self.push_had_success = True
                    self.last_push_failure_reason = None
                elif outcome_status in {"blocked", "failed", "mismatch"}:
                    self.last_push_failure_reason = (
                        action_outcome.get("reason")
                        or action_outcome.get("failure_reason")
                        or outcome_status
                    )

            if self.push_steps_remaining <= 0:
                if self.push_had_success:
                    self.complete_current_action(SymbolicActionStatus.SUCCESS)
                    return {"status": "success"}
                failure_reason = self.last_push_failure_reason or "target block did not move"
                self.complete_current_action(SymbolicActionStatus.FAILED, failure_reason)
                return {"status": "failed", "failure_reason": failure_reason}
            return {"status": "pending"}
        
        # Unknown action type - complete immediately
        return {"status": "pending"}
    
    def process_step_completions(self):
        """
        Process action completions after step counter is updated.
        This is now just a wrapper that calls check_termination_condition.
        """
        if self.current_symbolic_action and self.current_symbolic_action.status == SymbolicActionStatus.PENDING:
            result = self.check_termination_condition(None)
            return result
        return {"status": "pending"}
    
    def noop(self) -> str:
        """Do nothing for one step."""
        self.start_symbolic_action("noop", {})
        return "noop"
    
    def wait(self, num_steps: int = 1) -> str:
        """
        Wait (do nothing) for a specified number of steps.
        
        Args:
            num_steps: Number of steps to wait (default: 1)
            
        Returns:
            Primitive action string (noop)
        """
        # Check if we need to start a new wait action
        need_new_action = (
            not self.current_symbolic_action or
            self.current_symbolic_action.action_type != "wait" or
            self.current_symbolic_action.status != SymbolicActionStatus.PENDING or
            self.wait_steps_remaining <= 0
        )
        
        if need_new_action:
            self.start_symbolic_action("wait", {"num_steps": num_steps})
            self.wait_steps_remaining = num_steps
        
        # Execute one step of waiting
        self.wait_steps_remaining -= 1
        
        return "noop"
    
    def move(self, direction: str, num_steps: int = 1) -> str:
        """
        Move in a direction for a specified number of steps.
        
        Args:
            direction: Direction to move (left, right, up, down)
            num_steps: Number of steps to move (default: 1)
            
        Returns:
            Primitive action string
        """
        direction = direction.lower()
        
        # Check if we need to start a new move action
        need_new_action = (
            # No current action
            not self.current_symbolic_action or
            # Current action is not a move
            self.current_symbolic_action.action_type != "move" or
            # Direction has changed
            self.move_direction != direction or
            # Previous move action has completed 
            self.current_symbolic_action.status != SymbolicActionStatus.PENDING or
            # Move parameters have changed
            (self.current_symbolic_action.args and 
             self.current_symbolic_action.args.get("num_steps") != num_steps) or
            # Steps remaining is already 0 or negative (corrupted state)
            self.move_steps_remaining <= 0
        )
        
        if need_new_action:
            # Start completely fresh move action
            self.start_symbolic_action("move", {"direction": direction, "num_steps": num_steps})
            self.move_steps_remaining = num_steps
            self.move_direction = direction
        
        # Execute one step of movement - decrement remaining steps
        self.move_steps_remaining -= 1
        
        # Return primitive action (termination will be checked after step)
        return f"move_{direction}"

    def navigate(
        self,
        entity_type: str,
        entity_id: int,
        timeout: int = 30,
        world_state: Optional[Dict] = None,
    ) -> str:
        """
        Navigate toward an entity.

        For blocks, the target is the nearest left push cell so that a later
        push(block_id=...) moves the block toward the goal.
        """
        entity_type = str(entity_type).lower()
        try:
            entity_id = int(entity_id)
        except (TypeError, ValueError):
            entity_id = -1
        timeout = max(1, int(timeout or 30))

        need_new_action = (
            not self.current_symbolic_action or
            self.current_symbolic_action.action_type != "navigate" or
            self.navigate_entity_type != entity_type or
            self.navigate_entity_id != entity_id or
            self.current_symbolic_action.status != SymbolicActionStatus.PENDING or
            self.navigate_steps_remaining <= 0
        )

        if need_new_action:
            self.start_symbolic_action(
                "navigate",
                {"entity_type": entity_type, "entity_id": entity_id, "timeout": timeout},
            )
            self.navigate_steps_remaining = timeout
            self.navigate_entity_type = entity_type
            self.navigate_entity_id = entity_id

        direction = self._compute_navigation_direction(entity_type, entity_id, world_state)
        self.navigate_steps_remaining -= 1
        if direction == "noop":
            return "noop"
        return f"move_{direction}"
    
    def push(self, block_id: int, num_steps: int = 1, world_state: Optional[Dict] = None) -> str:
        """
        Push a block by moving into it for a specified number of steps.
        
        The agent will move towards the block and continue pushing it.
        Direction is determined by the relative position of agent and block.
        
        Args:
            block_id: ID of the block to push
            num_steps: Number of steps to push (default: 1)
            world_state: Current world state with agent and block positions
            
        Returns:
            Primitive action string (move direction towards block)
        """
        # Check if we need to start a new push action
        need_new_action = (
            not self.current_symbolic_action or
            self.current_symbolic_action.action_type != "push" or
            self.push_block_id != block_id or
            self.current_symbolic_action.status != SymbolicActionStatus.PENDING or
            self.push_steps_remaining <= 0
        )
        
        if need_new_action:
            self.start_symbolic_action("push", {"block_id": block_id, "num_steps": num_steps})
            self.push_steps_remaining = num_steps
            self.push_block_id = block_id
            self.push_direction = None  # Will be computed based on world_state
            self.push_had_success = False
            self.last_push_failure_reason = None
        
        # Determine push direction from world_state
        direction = self._compute_push_direction(block_id, world_state)
        self.push_direction = direction
        
        # Execute one step of pushing
        self.push_steps_remaining -= 1
        
        return f"move_{direction}"

    def _agent_position(self, world_state: Optional[Dict]) -> Optional[Tuple[int, int]]:
        if world_state is None:
            return None
        agent_pos = world_state.get('agent_position')
        if agent_pos is not None:
            return tuple(agent_pos)
        if self.agent_id:
            agents = world_state.get('agents', {})
            agent_info = agents.get(self.agent_id, {})
            pos = agent_info.get('position')
            if pos is not None:
                return tuple(pos)
        return None

    def _navigation_targets(self, entity_type: str, entity_id: int, world_state: Optional[Dict]) -> List[Tuple[int, int]]:
        if world_state is None:
            return []
        entity_type = str(entity_type).lower()
        if entity_type == "block":
            for block in world_state.get('blocks', []):
                if block.get('id') != entity_id:
                    continue
                block_r, block_c = block.get('position')
                block_weight = int(block.get('weight', 1))
                targets = [
                    (block_r + dr, block_c - 1)
                    for dr in range(block_weight)
                    if block_c - 1 >= 0
                ]
                return targets
        if entity_type == "agent":
            for agent_info in world_state.get('agents', {}).values():
                if agent_info.get('env_id') == entity_id or agent_info.get('id') == entity_id:
                    pos = agent_info.get('position')
                    return [tuple(pos)] if pos is not None else []
        if entity_type == "goal":
            agent_pos = self._agent_position(world_state)
            if agent_pos is None:
                return []
            grid_size = world_state.get("grid_size")
            if grid_size is None:
                return []
            return [(agent_pos[0], int(grid_size) - 1)]
        return []

    def _at_navigation_target(self, world_state: Optional[Dict]) -> bool:
        if not self.current_symbolic_action or self.current_symbolic_action.action_type != "navigate":
            return False
        agent_pos = self._agent_position(world_state)
        if agent_pos is None:
            return False
        args = self.current_symbolic_action.args
        targets = self._navigation_targets(
            args.get("entity_type"),
            int(args.get("entity_id", -1)),
            world_state,
        )
        return tuple(agent_pos) in set(targets)

    def _compute_navigation_direction(
        self,
        entity_type: str,
        entity_id: int,
        world_state: Optional[Dict],
    ) -> str:
        agent_pos = self._agent_position(world_state)
        targets = self._navigation_targets(entity_type, entity_id, world_state)
        if agent_pos is None or not targets:
            return "noop"
        if tuple(agent_pos) in set(targets):
            return "noop"

        agent_r, agent_c = agent_pos
        grid_size = world_state.get("grid_size") if world_state else None
        if grid_size is not None:
            next_direction = self._bfs_navigation_direction(
                agent_pos=tuple(agent_pos),
                targets=targets,
                blocked_cells=set(tuple(cell) for cell in world_state.get("blocked_cells", [])),
                grid_size=int(grid_size),
            )
            if next_direction:
                return next_direction

        target_r, target_c = min(
            targets,
            key=lambda pos: abs(pos[0] - agent_r) + abs(pos[1] - agent_c),
        )
        dr = target_r - agent_r
        dc = target_c - agent_c
        if abs(dc) >= abs(dr) and dc != 0:
            return "right" if dc > 0 else "left"
        if dr != 0:
            return "down" if dr > 0 else "up"
        return "noop"

    @staticmethod
    def _bfs_navigation_direction(
        agent_pos: Tuple[int, int],
        targets: List[Tuple[int, int]],
        blocked_cells: set,
        grid_size: int,
    ) -> Optional[str]:
        target_set = set(tuple(target) for target in targets)
        if agent_pos in target_set:
            return "noop"

        blocked = set(blocked_cells)
        blocked.discard(agent_pos)
        blocked -= target_set

        queue = deque([(agent_pos, None)])
        visited = {agent_pos}
        directions = [
            (0, 1, "right"),
            (1, 0, "down"),
            (-1, 0, "up"),
            (0, -1, "left"),
        ]

        while queue:
            (row, col), first_direction = queue.popleft()
            if (row, col) in target_set:
                return first_direction or "noop"
            for dr, dc, direction in directions:
                nr, nc = row + dr, col + dc
                next_pos = (nr, nc)
                if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                    continue
                if next_pos in visited or next_pos in blocked:
                    continue
                visited.add(next_pos)
                queue.append((next_pos, first_direction or direction))
        return None
    
    def _compute_push_direction(self, block_id: int, world_state: Optional[Dict]) -> str:
        """
        Compute the direction to push based on agent and block positions.
        
        The agent pushes by moving INTO the block, so we determine
        which side of the block the agent is on and move towards it.
        
        Default to 'right' (push towards goal) if world_state is unavailable.
        """
        if world_state is None:
            return "right"  # Default: push towards goal column
        
        # Get agent position
        agent_pos = world_state.get('agent_position')
        if agent_pos is None and self.agent_id:
            agents = world_state.get('agents', {})
            agent_info = agents.get(self.agent_id, {})
            agent_pos = agent_info.get('position')
        
        # Get block position
        block_pos = None
        for block in world_state.get('blocks', []):
            if block.get('id') == block_id:
                block_pos = block.get('position')  # (row, col) of top-left
                block_weight = block.get('weight', 1)
                break
        
        if agent_pos is None or block_pos is None:
            return "right"  # Default fallback
        
        agent_r, agent_c = agent_pos
        block_r, block_c = block_pos
        
        # Determine which side of the block the agent is on
        # Agent pushes by moving INTO the block
        # If agent is to the left of block -> push right (move right)
        # If agent is to the right of block -> push left (move left)
        # If agent is above block -> push down (move down)
        # If agent is below block -> push up (move up)
        
        # Calculate center of block
        block_center_r = block_r + (block_weight - 1) / 2
        block_center_c = block_c + (block_weight - 1) / 2
        
        dr = block_center_r - agent_r
        dc = block_center_c - agent_c
        
        # Move in the dominant direction towards block
        if abs(dc) >= abs(dr):
            # Horizontal movement dominates
            return "right" if dc > 0 else "left"
        else:
            # Vertical movement dominates
            return "down" if dr > 0 else "up"
    
    def get_action_value(self, action_name: str) -> int:
        """
        Convert step-level action name to primitive action value for the environment.
        
        Args:
            action_name: Step-level action name (e.g., "move_left", "noop")
            
        Returns:
            Integer primitive action value recognized by CUBE environment
        """
        return ACTION_NAME_TO_VALUE.get(action_name, ACTION_NAME_TO_VALUE["noop"])
    
    def execute(self, action_type: str, world_state: Optional[Dict] = None, 
                current_step: Optional[int] = None, **kwargs) -> str:
        """
        Execute a symbolic action and return the step-level action string.
        
        Args:
            action_type: Type of symbolic action
            world_state: Current world state (needed for navigate)
            current_step: Current environment step (for tracking primitive_action_history)
            **kwargs: Action-specific arguments
            
        Returns:
            Step-level action string (e.g., "move_left", "do")
        """
        action_func = self.actions.get(action_type)
        if action_func is None:
            return "noop"
        
        # Call the action function with provided arguments
        import inspect
        sig = inspect.signature(action_func)
        
        # Filter kwargs to only include parameters that the function accepts
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        
        # Add world_state if function accepts it
        if 'world_state' in sig.parameters and world_state is not None:
            valid_kwargs['world_state'] = world_state
        
        step_action = action_func(**valid_kwargs)
        
        # Store the step-level action in current record
        if self.current_symbolic_action:
            # For multi-step actions (move, wait, push), update primitive_action each step
            if self.current_symbolic_action.action_type in ['move', 'wait', 'push', 'navigate']:
                # Always update for multi-step actions to show current step action
                self.current_symbolic_action.primitive_action = step_action
                # Record per-step history for visualization
                if current_step is not None:
                    self.current_symbolic_action.primitive_action_history[current_step] = step_action
            elif self.current_symbolic_action.primitive_action is None:
                # For single-step actions, set once
                self.current_symbolic_action.primitive_action = step_action
                # Also record in history for consistency
                if current_step is not None:
                    self.current_symbolic_action.primitive_action_history[current_step] = step_action
        
        return step_action
    
    def get_action_records(self) -> List[Dict]:
        """Get all completed action records."""
        records = [record.to_dict() for record in self.action_history]
        if self.current_symbolic_action:
            records.append(self.current_symbolic_action.to_dict())
        return records
    
    def clear_history(self):
        """Clear action history."""
        self.action_history.clear()
