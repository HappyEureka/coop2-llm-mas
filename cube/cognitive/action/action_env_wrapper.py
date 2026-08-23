"""
Simplified symbolic wrapper for CUBE environment.

This wrapper provides a clean interface without validation checks,
focusing on simple symbolic action -> primitive action mapping.
"""

import inspect
from typing import Dict, List, Optional, Any
from .action import SymbolicActionExecutor
from ..constants import ACTION_NAME_TO_VALUE


class SymbolicEnvWrapper:
    """
    Simplified symbolic wrapper for CUBE.
    
    Provides a clean symbolic action interface without validation checks.
    
    Usage:
        env = SymbolicEnvWrapper(CoopBlockPush, agent_names=['agent_0', 'agent_1'])
        obs, info = env.reset()
        
        actions = {
            'agent_0': {'action_type': 'move', 'direction': 'left', 'num_steps': 3},
            'agent_1': {'action_type': 'wait', 'num_steps': 2}
        }
        obs, rewards, terminated, truncated, info = env.step(actions)
    """
    
    def __init__(self, env, agent_names: List[str], **env_kwargs):
        """
        Initialize the simplified symbolic wrapper.
        
        Args:
            env: The CUBE environment class or instance
            agent_names: List of agent names (e.g., ['agent_0', 'agent_1'])
            **env_kwargs: Additional environment arguments (only used if env is a class)
        """
        # Handle both environment class and instance
        if callable(env) and hasattr(env, '__name__'):
            self.env = env(num_agents=len(agent_names), **env_kwargs)
        else:
            self.env = env
            
        self.user_agent_names = [str(a) for a in agent_names]
        self.name_map = {
            user: env_id 
            for user, env_id in zip(self.user_agent_names, self.env.possible_agents)
        }
        self.reverse_name_map = {v: k for k, v in self.name_map.items()}
        
        # Create action executors for each agent
        self.agent_actions = {
            user: SymbolicActionExecutor(agent_id=user) 
            for user in self.user_agent_names
        }
        
        # Track observations and env steps
        self._last_observations = None
        self._env_step_count = 0
    
    @property
    def agents(self) -> List[str]:
        """Return the current active agents using user names."""
        active_env_agents = self.env.agents if hasattr(self.env, 'agents') else self.env.possible_agents
        return [self.reverse_name_map[env_id] for env_id in active_env_agents if env_id in self.reverse_name_map]
    
    @property
    def possible_agents(self) -> List[str]:
        """Return all possible agent names."""
        return self.user_agent_names
    
    @property
    def observation_space(self):
        """Return the observation space."""
        return self.env.observation_space if hasattr(self.env, 'observation_space') else self.env._observation_space
    
    @property
    def action_space(self):
        """Return the action space."""
        return self.env.action_space if hasattr(self.env, 'action_space') else self.env._action_space
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options (e.g., fixed_agent_positions)
            
        Returns:
            tuple: (observations, info) with user agent names
        """
        if seed is not None:
            if options is not None:
                obs, info = self.env.reset(seed=seed, options=options)
            else:
                obs, info = self.env.reset(seed=seed)
        else:
            if options is not None:
                obs, info = self.env.reset(options=options)
            else:
                obs, info = self.env.reset()
        
        # Reset env step counter
        self._env_step_count = 0
        
        # Update step counter for all agents
        for agent_actions in self.agent_actions.values():
            agent_actions.update_env_step(self._env_step_count)
        
        # Map to user names
        self._last_observations = self._map_dict(obs)
        mapped_info = self._map_dict(info)
        
        return self._last_observations, mapped_info
    
    def step(self, symbolic_actions: Dict[str, Dict[str, Any]]):
        """
        Step through the environment with symbolic actions.
        
        Args:
            symbolic_actions: Dict mapping agent names to symbolic actions.
                Format: {'agent_0': {'action_type': 'move', 'direction': 'left'}}
                or simplified: {'agent_0': {'action_type': 'wait'}}
        
        Returns:
            tuple: (observations, rewards, terminated, truncated, info) with user names
        """
        # Build a lightweight symbolic world state from the wrapped env. This is
        # needed for push(block_id), whose primitive direction depends on the
        # agent's position relative to the block.
        world_state = self._build_world_state()
        
        env_action_dict = {}
        symbolic_intents = {}
        
        # Execute symbolic actions for all agents to get primitive actions
        for user_id, symbolic_action in symbolic_actions.items():
            if user_id not in self.name_map:
                continue
                
            env_id = self.name_map[user_id]
            action_handler = self.agent_actions[user_id]
            
            # Get action type and arguments
            action_type = symbolic_action.get("action_type", "noop")
            
            # Handle both nested args format and flat format
            if "args" in symbolic_action:
                args = symbolic_action["args"]
            else:
                # Flat format - all keys except action_type are args
                args = {k: v for k, v in symbolic_action.items() if k != "action_type"}
            
            # Execute the symbolic action to get primitive action
            # Pass current step for tracking primitive_action_history
            # world_state can be None for most actions
            primitive_action = action_handler.execute(
                action_type, 
                world_state=world_state, 
                current_step=self._env_step_count,
                **args
            )
            
            # Convert to action value
            action_value = action_handler.get_action_value(primitive_action)
            env_action_dict[env_id] = action_value
            symbolic_intents[env_id] = self._build_symbolic_intent(
                action_type=action_type,
                args=args,
                primitive_action=primitive_action,
                action_value=action_value,
            )
        
        # Fill in noop for agents that didn't provide actions
        for env_id in self.env.agents:
            if env_id not in env_action_dict:
                env_action_dict[env_id] = ACTION_NAME_TO_VALUE["noop"]
                symbolic_intents[env_id] = {
                    "action_type": "noop",
                    "args": {},
                    "primitive_action": "noop",
                    "action_value": ACTION_NAME_TO_VALUE["noop"],
                }
        
        if hasattr(self.env, "_last_symbolic_action_intents"):
            self.env._last_symbolic_action_intents = symbolic_intents
        
        # Step the environment with primitive actions
        obs, rewards, terminated, truncated, info = self.env.step(env_action_dict)
        grounded_outcomes = self._extract_grounded_action_outcomes(info)
        
        # NOW increment step counter after all primitive actions are complete
        self._env_step_count += 1
        
        # Refresh world state for termination checks after the primitive step.
        world_state = self._build_world_state()
        
        # Update step counter for all agents synchronously
        for user_id in self.agent_actions:
            action_handler = self.agent_actions[user_id]
            action_handler.update_env_step(self._env_step_count)
            # Check termination conditions now that step is complete
            action_handler.check_termination_condition(
                world_state,
                action_outcome=grounded_outcomes.get(user_id),
            )
        
        # Map results back to user agent names
        self._last_observations = self._map_dict(obs)
        mapped_rewards = self._map_dict(rewards)
        mapped_terminated = self._map_dict(terminated)
        mapped_truncated = self._map_dict(truncated)
        mapped_info = self._map_dict(info)
        
        return (
            self._last_observations,
            mapped_rewards,
            mapped_terminated,
            mapped_truncated,
            mapped_info,
        )

    def _build_world_state(self) -> Dict[str, Any]:
        """Build a small state dict for symbolic action executors."""
        agent_positions = getattr(self.env, "_agent_positions", {})
        blocks = getattr(self.env, "_blocks", [])
        blocked_cells = set()
        for block in blocks:
            blocked_cells.update(tuple(cell) for cell in block.cells())
        blocked_cells.update(tuple(pos) for pos in agent_positions.values())
        return {
            "grid_size": getattr(self.env, "K", None),
            "blocked_cells": sorted(blocked_cells),
            "agents": {
                self.reverse_name_map.get(env_id, env_id): {
                    "position": tuple(pos),
                    "env_id": env_id,
                    "id": getattr(self.env, "agent_name_mapping", {}).get(env_id),
                }
                for env_id, pos in agent_positions.items()
            },
            "blocks": [
                {
                    "id": block.id,
                    "position": (block.r, block.c),
                    "weight": block.weight,
                    "cells": block.cells(),
                }
                for block in blocks
            ],
        }

    def _extract_grounded_action_outcomes(self, info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Map CUBE environment push events back to user-agent symbolic outcomes.

        The symbolic executor should only mark push(block_id=X) as successful
        when that target block actually moved or was delivered by the agent.
        """
        base_info = {}
        if isinstance(info, dict):
            for value in info.values():
                if isinstance(value, dict) and (
                    "last_step_events" in value or "last_symbolic_action_intents" in value
                ):
                    base_info = value
                    break

        step_events = (
            base_info.get("last_step_events")
            or getattr(self.env, "_last_step_events", {})
            or {}
        )
        symbolic_intents = (
            base_info.get("last_symbolic_action_intents")
            or getattr(self.env, "_last_symbolic_action_intents", {})
            or {}
        )

        successes = step_events.get("successful_pushes") or []
        failures = step_events.get("failed_pushes") or []
        deliveries = step_events.get("delivered_blocks") or []
        outcomes: Dict[str, Dict[str, Any]] = {}

        for env_id, intent in symbolic_intents.items():
            if not isinstance(intent, dict) or intent.get("action_type") != "push":
                continue
            user_id = self.reverse_name_map.get(env_id, env_id)
            target_block_id = self._safe_int(intent.get("block_id"))
            target_task_id = intent.get("task_id")

            matching_success = self._matching_push_event(successes, env_id, target_block_id)
            matching_delivery = self._matching_push_event(deliveries, env_id, target_block_id)
            matching_failure = self._matching_push_event(failures, env_id, target_block_id)
            any_success = self._matching_push_event(successes, env_id, None)

            if matching_delivery:
                event = matching_delivery
                outcomes[user_id] = {
                    "action_type": "push",
                    "status": "success",
                    "reason": None,
                    "block_id": event.get("block_id"),
                    "task_id": event.get("task_id") or target_task_id,
                    "face": event.get("face") or intent.get("face"),
                    "metadata": dict(event),
                }
            elif matching_success:
                event = matching_success
                outcomes[user_id] = {
                    "action_type": "push",
                    "status": "success",
                    "reason": None,
                    "block_id": event.get("block_id"),
                    "task_id": event.get("task_id") or target_task_id,
                    "face": event.get("face") or intent.get("face"),
                    "metadata": dict(event),
                }
            elif any_success:
                outcomes[user_id] = {
                    "action_type": "push",
                    "status": "mismatch",
                    "reason": (
                        f"pushed block {any_success.get('block_id')} instead of "
                        f"target block {target_block_id}"
                    ),
                    "block_id": target_block_id,
                    "task_id": target_task_id,
                    "face": intent.get("face"),
                    "metadata": dict(any_success),
                }
            elif matching_failure:
                outcomes[user_id] = {
                    "action_type": "push",
                    "status": "blocked",
                    "reason": matching_failure.get("reason") or "push_failed",
                    "block_id": target_block_id,
                    "task_id": target_task_id,
                    "face": intent.get("face"),
                    "metadata": dict(matching_failure),
                }
            else:
                outcomes[user_id] = {
                    "action_type": "push",
                    "status": "blocked",
                    "reason": "no_contact_with_target_block",
                    "block_id": target_block_id,
                    "task_id": target_task_id,
                    "face": intent.get("face"),
                    "metadata": dict(intent),
                }

        return outcomes

    @staticmethod
    def _matching_push_event(events: List[Dict[str, Any]], env_id: str, block_id: Optional[int]) -> Optional[Dict[str, Any]]:
        for event in events:
            if not isinstance(event, dict):
                continue
            if env_id not in (event.get("agents") or []):
                continue
            if block_id is None:
                return event
            try:
                event_block_id = int(event.get("block_id"))
            except (TypeError, ValueError):
                continue
            if event_block_id == block_id:
                return event
        return None

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _build_symbolic_intent(
        self,
        action_type: str,
        args: Dict[str, Any],
        primitive_action: str,
        action_value: int,
    ) -> Dict[str, Any]:
        intent = {
            "action_type": action_type,
            "args": dict(args),
            "primitive_action": primitive_action,
            "action_value": action_value,
        }
        if action_type == "push":
            block_id = args.get("block_id")
            if block_id is not None:
                intent["block_id"] = block_id
            face = args.get("face") or self._face_from_primitive_action(primitive_action)
            if face:
                intent["face"] = face
                intent["task_id"] = f"block_{block_id}_{face}" if block_id is not None else None
        return intent

    @staticmethod
    def _face_from_primitive_action(primitive_action: str) -> Optional[str]:
        return {
            "move_right": "left",
            "move_left": "right",
            "move_down": "up",
            "move_up": "down",
        }.get(primitive_action)
    
    def _map_dict(self, env_dict: Dict) -> Dict:
        """Map environment agent IDs to user agent names."""
        # Handle single-player mode where env returns non-dict
        if not isinstance(env_dict, dict):
            # Single player - wrap in dict with first user name
            if len(self.user_agent_names) == 1:
                return {self.user_agent_names[0]: env_dict}
            else:
                return {}
        return {
            user: env_dict[env_id] 
            for user, env_id in self.name_map.items() 
            if env_id in env_dict
        }
    
    def render(self):
        """Render the environment."""
        if hasattr(self.env, 'render'):
            return self.env.render()
    
    def close(self):
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()
    
    def get_action_records(self, agent_name: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Get action execution records for agents.
        
        Args:
            agent_name: Specific agent name, or None for all agents
            
        Returns:
            Dict mapping agent names to their action records
        """
        if agent_name:
            if agent_name in self.agent_actions:
                return {agent_name: self.agent_actions[agent_name].get_action_records()}
            return {}
        
        return {
            name: handler.get_action_records() 
            for name, handler in self.agent_actions.items()
        }
    
    def clear_action_history(self, agent_name: Optional[str] = None):
        """Clear action history for agents."""
        if agent_name:
            if agent_name in self.agent_actions:
                self.agent_actions[agent_name].clear_history()
        else:
            for handler in self.agent_actions.values():
                handler.clear_history()
    
    def __getattr__(self, name):
        """Forward attribute access to the wrapped environment."""
        return getattr(self.env, name)
