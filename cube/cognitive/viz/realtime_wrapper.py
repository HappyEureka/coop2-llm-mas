"""
Wrapper that integrates real-time visualization with PlanningEnvWrapper.

This wrapper automatically updates the visualization on every step and state change.
"""

from typing import Dict, Optional
import numpy as np
from .realtime_viz import RealtimeAgentVisualizer


class RealtimeVisualizationWrapper:
    """
    Wrapper that integrates real-time visualization with PlanningEnvWrapper.
    
    This wrapper acts as the main environment interface, forwarding all methods
    to the underlying PlanningEnvWrapper while adding visualization updates.
    
    Example:
        plan_env = PlanningEnvWrapper(base_env, agent_names=agent_names, agents=agents)
        env = RealtimeVisualizationWrapper(plan_env, record_video=True)
        
        obs_dict, _ = env.reset()
        
        while not done:
            obs_dict, rewards, terminated, truncated, info = env.step()
        
        env.save_video("output.gif")
        env.close()
    """
    
    def __init__(self, plan_env, record_video: bool = False, show: bool = True):
        """
        Initialize the visualization wrapper.
        
        Args:
            plan_env: PlanningEnvWrapper instance
            record_video: Whether to record frames for video output
            show: Whether to show the visualization window (default: True)
        """
        import time
        self._env = plan_env
        self._show = show
        self._record_video = record_video
        
        # Create visualizer
        agent_ids = list(plan_env.agents.keys())
        self.viz = RealtimeAgentVisualizer(agent_ids, record_video=record_video, show=show)
        
        # Track message indices and timing
        self._last_msg_idx = 0
        self._start_time = time.time()
        
        # Store current observations for update calls
        self._current_obs = {}
    
    # Forward common properties to the underlying env
    @property
    def agents(self):
        return self._env.agents
    
    @property
    def agent_names(self):
        return self._env.agent_names
    
    @property
    def message_broker(self):
        return self._env.message_broker
    
    @property
    def current_step(self):
        return self._env.current_step
    
    @property
    def plan_executors(self):
        return self._env.plan_executors
    
    @property
    def logger(self):
        return self._env.logger
    
    def update(self, obs_dict: Optional[Dict[str, np.ndarray]] = None):
        """
        Update visualization with current observations.
        
        Args:
            obs_dict: Dictionary of agent observations (uses stored obs if None)
        """
        import time
        
        if obs_dict is None:
            obs_dict = self._current_obs
        
        # Get rendered environment image (CUBE uses render() instead of per-agent views)
        # Try to get rendered image from base environment
        base_env = self._env.symbolic_env.env if hasattr(self._env, 'symbolic_env') else None
        rendered_img = None
        if base_env is not None and hasattr(base_env, 'render'):
            rendered_img = base_env.render(return_image=True)
            if rendered_img is not None:
                # Scale up the small CUBE image (8x8) to be visible using numpy repeat
                scale = 10  # 8x8 -> 80x80
                rendered_img = np.repeat(np.repeat(rendered_img, scale, axis=0), scale, axis=1)
        
        # Update views - use rendered env image for all agents (same view in CUBE)
        for agent_id in self.viz.agent_data.keys():
            if rendered_img is not None:
                self.viz.update_view(agent_id, rendered_img)
            elif agent_id in obs_dict:
                self.viz.update_view(agent_id, obs_dict[agent_id])
        
        # Update states, plans, and actions
        for agent_id, agent in self._env.agents.items():
            state = agent.state.value if hasattr(agent.state, 'value') else str(agent.state)
            self.viz.update_state(agent_id, state, agent.ready)
            
            # Update plan number
            self.viz.update_plan(agent_id, agent.plan_count)
            
            # Update current action from plan executor
            if agent_id in self._env.plan_executors:
                executor_status = self._env.plan_executors[agent_id].get_status()
                current_action = executor_status.get('current_action', '-')
                if current_action is None:
                    current_action = '-'
                self.viz.update_action(agent_id, str(current_action))
        
        # Update messages from broker
        if self._env.message_broker is not None:
            msg_log = self._env.message_broker.get_message_log()
            new_messages = msg_log[self._last_msg_idx:]
            self._last_msg_idx = len(msg_log)
            
            for msg in new_messages:
                recipients = msg.get('recipients', [])
                sender = msg.get('sender', 'unknown')
                content = msg.get('content', '')
                msg_time = time.time() - self._start_time
                
                for recipient in recipients:
                    self.viz.add_message(recipient, sender, content, msg_time)
        
        # Update step counter with elapsed time
        elapsed = time.time() - self._start_time
        self.viz.update_step(self._env.current_step, elapsed)
        
        # Refresh display
        self.viz.refresh()
    
    def set_env_state(self, state: str):
        """Set the environment state indicator ('waiting' or 'stepping')."""
        self.viz.update_env_state(state)
    
    def save_video(self, filename: str = "realtime_viz.gif", fps: int = 10):
        """Save recorded frames as video/gif."""
        self.viz.save_video(filename, fps)
    
    def close(self):
        """Close the visualization."""
        self.viz.close()
    
    # Forward environment methods
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment and update visualization."""
        import time
        self._start_time = time.time()
        self._last_msg_idx = 0
        obs_dict, info = self._env.reset(seed=seed, options=options)
        self._current_obs = obs_dict
        self.update(obs_dict)
        return obs_dict, info
    
    def step(self):
        """Step the environment and update visualization."""
        self.set_env_state('stepping')
        obs_dict, rewards, terminated, truncated, info = self._env.step()
        self._current_obs = obs_dict
        self.update(obs_dict)
        return obs_dict, rewards, terminated, truncated, info
    
    def wait_for_state_change(self, timeout: float = 0.1):
        """Wait for agent state change, updating visualization periodically."""
        self.set_env_state('waiting')
        self.update()
        result = self._env.wait_for_state_change(timeout=timeout)
        return result
    
    def get_all_agent_status(self):
        """Get status of all agents."""
        return self._env.get_all_agent_status()
    
    def __getattr__(self, name):
        """Forward any other attribute access to the underlying env."""
        return getattr(self._env, name)
