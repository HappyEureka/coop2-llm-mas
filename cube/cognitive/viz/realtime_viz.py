"""
Real-time visualization for multi-agent system.

Shows each agent's:
- Current view (observation)
- State (reasoning, executing, interrupted, etc.)
- Recent communication messages
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque
import threading
import os


class RealtimeAgentVisualizer:
    """Real-time visualization of agent views, states, and communications."""
    
    # State colors - aligned with cognitive_viz.py
    # R = Reasoning, X = Executing, I = Interrupted, W = Waiting
    STATE_COLORS = {
        'reasoning': '#FF9F40',      # Orange for reasoning (R)
        'waiting': '#FFD93D',        # Yellow for waiting (W)
        'executing': '#6BCB77',      # Green for executing (X)
        'interrupted': '#FF0000',    # Red for interrupted (I)
    }
    
    STATE_LABELS = {
        'reasoning': 'R',
        'waiting': 'W', 
        'executing': 'X',
        'interrupted': 'I',
    }
    
    def __init__(self, agent_ids: List[str], figsize_per_agent=(3.5, 5), record_video: bool = False, show: bool = True):
        """
        Initialize the real-time visualizer.
        
        Args:
            agent_ids: List of agent identifiers
            figsize_per_agent: Figure size per agent (width, height)
            record_video: Whether to record frames for video output
            show: Whether to show the visualization window (default: True)
        """
        self.agent_ids = agent_ids
        self.n_agents = len(agent_ids)
        self._show = show
        
        # Video recording
        self.record_video = record_video
        self.frames = []
        self._frame_counter = 0
        self._frame_skip = 3  # Capture every 3rd refresh call
        
        # Calculate figure size
        n_cols = min(self.n_agents, 4)
        n_rows = (self.n_agents + n_cols - 1) // n_cols
        figsize = (figsize_per_agent[0] * n_cols, figsize_per_agent[1] * n_rows)
        
        # Create figure with GridSpec - each agent gets rows: image, plan/action, state, messages
        from matplotlib.gridspec import GridSpec
        self.fig = plt.figure(figsize=figsize)
        
        # 10 rows per agent: image (5), plan+action (1), state (1), messages (3)
        gs = GridSpec(n_rows * 10, n_cols, figure=self.fig, hspace=0.4, wspace=0.3)
        
        self.fig.suptitle('Multi-Agent Real-Time View | Step 0', fontsize=14, fontweight='bold')
        
        # Store references for each agent
        self.agent_data = {}
        for i, agent_id in enumerate(agent_ids):
            row, col = i // n_cols, i % n_cols
            base_row = row * 10
            
            # Image axes (rows 0-4)
            ax_img = self.fig.add_subplot(gs[base_row:base_row+5, col])
            # Plan/Action axes (row 5)
            ax_plan = self.fig.add_subplot(gs[base_row+5, col])
            # State axes (row 6)
            ax_state = self.fig.add_subplot(gs[base_row+6, col])
            # Message axes (rows 7-9)
            ax_msg = self.fig.add_subplot(gs[base_row+7:base_row+10, col])
            
            self.agent_data[agent_id] = {
                'ax_img': ax_img,
                'ax_plan': ax_plan,
                'ax_state': ax_state,
                'ax_msg': ax_msg,
                'image': None,
                'plan_text': None,
                'action_text': None,
                'state_patch': None,
                'state_text': None,
                'msg_text': None,
                'messages': deque(maxlen=3),
            }
            
            self._setup_agent_subplot(agent_id)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        # Add environment state indicator at top right (beside title)
        self.env_state_text = self.fig.text(
            0.92, 0.96, 'WAITING',
            ha='right', va='center',
            fontsize=11, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFA500', edgecolor='black', linewidth=2)
        )
        
        # Enable interactive mode only if showing
        if self._show:
            plt.ion()
            self.fig.canvas.draw()
            plt.pause(0.01)
        
        self._lock = threading.Lock()
        self.current_step = 0
    
    def _setup_agent_subplot(self, agent_id: str):
        """Set up the subplot for a single agent."""
        data = self.agent_data[agent_id]
        ax_img = data['ax_img']
        ax_plan = data['ax_plan']
        ax_state = data['ax_state']
        ax_msg = data['ax_msg']
        
        # Setup image axes
        ax_img.set_title(agent_id, fontsize=11, fontweight='bold')
        placeholder = np.zeros((84, 84, 3), dtype=np.uint8)
        img = ax_img.imshow(placeholder)
        data['image'] = img
        ax_img.axis('off')
        
        # Setup plan/action axes - box with plan and action side by side
        ax_plan.set_xlim(0, 1)
        ax_plan.set_ylim(0, 1)
        ax_plan.axis('off')
        
        # Add box background
        plan_bg = patches.FancyBboxPatch(
            (0.02, 0.1), 0.96, 0.8,
            boxstyle="round,pad=0.02",
            facecolor='white',
            edgecolor='#999999',
            linewidth=1
        )
        ax_plan.add_patch(plan_bg)
        
        # Plan number on left
        plan_text = ax_plan.text(
            0.25, 0.5, 'Plan: -',
            ha='center', va='center',
            fontsize=9, fontweight='bold',
            color='#333333'
        )
        data['plan_text'] = plan_text
        
        # Separator
        ax_plan.axvline(x=0.5, ymin=0.2, ymax=0.8, color='#cccccc', linewidth=1)
        
        # Action on right
        action_text = ax_plan.text(
            0.75, 0.5, 'Act: -',
            ha='center', va='center',
            fontsize=9,
            color='#333333'
        )
        data['action_text'] = action_text
        
        # Setup state axes - colored box with state text
        ax_state.set_xlim(0, 1)
        ax_state.set_ylim(0, 1)
        ax_state.axis('off')
        
        state_patch = patches.FancyBboxPatch(
            (0.05, 0.1), 0.9, 0.8,
            boxstyle="round,pad=0.02",
            facecolor='#808080',
            edgecolor='black',
            linewidth=2
        )
        ax_state.add_patch(state_patch)
        data['state_patch'] = state_patch
        
        state_text = ax_state.text(
            0.5, 0.5, 'W',
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            color='white'
        )
        data['state_text'] = state_text
        
        # Setup message axes
        ax_msg.set_xlim(0, 1)
        ax_msg.set_ylim(0, 1)
        ax_msg.axis('off')
        
        # Add background for message area
        msg_bg = patches.FancyBboxPatch(
            (0.02, 0.05), 0.96, 0.9,
            boxstyle="round,pad=0.02",
            facecolor='#f0f0f0',
            edgecolor='#cccccc',
            linewidth=1
        )
        ax_msg.add_patch(msg_bg)
        
        # Single text object for all messages - left aligned with wrapping
        msg_text = ax_msg.text(
            0.08, 0.85, 'No messages',
            ha='left', va='top',
            fontsize=7,
            color='#666666',
            style='italic',
            wrap=True
        )
        # Set clip box to keep text inside
        msg_text.set_clip_on(True)
        data['msg_text'] = msg_text
    
    def update_view(self, agent_id: str, observation: np.ndarray):
        """
        Update an agent's view image.
        
        Args:
            agent_id: Agent identifier
            observation: RGB observation array (H, W, 3)
        """
        if agent_id not in self.agent_data:
            return
        
        with self._lock:
            self.agent_data[agent_id]['image'].set_data(observation)
    
    def update_plan(self, agent_id: str, plan_info):
        """
        Update the plan display.
        
        Args:
            agent_id: Agent identifier
            plan_info: Plan number or string to display
        """
        if agent_id not in self.agent_data:
            return
        
        with self._lock:
            self.agent_data[agent_id]['plan_text'].set_text(f'Plan: {plan_info}')
    
    def update_action(self, agent_id: str, action: str):
        """
        Update the current primitive action display.
        
        Args:
            agent_id: Agent identifier
            action: Current primitive action string
        """
        if agent_id not in self.agent_data:
            return
        
        with self._lock:
            # Truncate long action strings
            if len(action) > 20:
                action = action[:17] + "..."
            self.agent_data[agent_id]['action_text'].set_text(f'Act: {action}')
    
    def update_state(self, agent_id: str, state: str, ready: bool = False):
        """
        Update an agent's state indicator.
        
        Args:
            agent_id: Agent identifier
            state: Current state string (reasoning, executing, interrupted, waiting)
            ready: Whether agent is ready (ignored, state is shown directly)
        """
        if agent_id not in self.agent_data:
            return
        
        with self._lock:
            # Get state label (R, X, I, W) and color
            state_lower = state.lower()
            display_state = self.STATE_LABELS.get(state_lower, state_lower[0].upper())
            color = self.STATE_COLORS.get(state_lower, '#808080')
            
            self.agent_data[agent_id]['state_patch'].set_facecolor(color)
            self.agent_data[agent_id]['state_text'].set_text(display_state)
    
    def add_message(self, agent_id: str, sender: str, content: str, timestamp: float = None):
        """
        Add a received message to an agent's display.
        
        Args:
            agent_id: Receiving agent identifier
            sender: Sender agent identifier
            content: Message content
            timestamp: Time when message was sent
        """
        if agent_id not in self.agent_data:
            return
        
        with self._lock:
            # Wrap long content to multiple lines (max ~30 chars per line)
            import textwrap
            max_width = 28
            
            # Format header with timestamp (2 decimal places)
            if timestamp is not None:
                if isinstance(timestamp, float):
                    header = f"[t={timestamp:.2f}] {sender}:"
                else:
                    header = f"[t={timestamp}] {sender}:"
            else:
                header = f"[t={self.current_step}] {sender}:"
            
            # Wrap content
            if len(content) > max_width:
                wrapped = textwrap.fill(content, width=max_width)
                msg = f"{header}\n  {wrapped.replace(chr(10), chr(10) + '  ')}"
            else:
                msg = f"{header} {content}"
            
            self.agent_data[agent_id]['messages'].append(msg)
            
            # Update message text - show all messages (limit to 2 for space)
            messages = list(self.agent_data[agent_id]['messages'])[-2:]
            if messages:
                display_text = "\n".join(reversed(messages))  # Most recent first
                self.agent_data[agent_id]['msg_text'].set_text(display_text)
                self.agent_data[agent_id]['msg_text'].set_color('#333333')
                self.agent_data[agent_id]['msg_text'].set_style('normal')
            else:
                self.agent_data[agent_id]['msg_text'].set_text('No messages')
                self.agent_data[agent_id]['msg_text'].set_color('#666666')
                self.agent_data[agent_id]['msg_text'].set_style('italic')
    
    def clear_messages(self, agent_id: str):
        """Clear displayed messages for an agent."""
        if agent_id not in self.agent_data:
            return
        
        with self._lock:
            self.agent_data[agent_id]['messages'].clear()
            self.agent_data[agent_id]['msg_text'].set_text('No messages')
            self.agent_data[agent_id]['msg_text'].set_color('#666666')
            self.agent_data[agent_id]['msg_text'].set_style('italic')
    
    def update_step(self, step: int, time_elapsed: float = None):
        """Update the current step display."""
        self.current_step = step
        if time_elapsed is not None:
            self.fig.suptitle(f'Multi-Agent Real-Time View | Step {step} | t={time_elapsed:.2f}s', 
                             fontsize=14, fontweight='bold')
        else:
            self.fig.suptitle(f'Multi-Agent Real-Time View | Step {step}', 
                             fontsize=14, fontweight='bold')
    
    def update_env_state(self, state: str):
        """
        Update the environment state indicator.
        
        Args:
            state: 'waiting' or 'stepping'
        """
        with self._lock:
            if state.lower() == 'stepping':
                self.env_state_text.set_text('STEPPING')
                self.env_state_text.get_bbox_patch().set_facecolor('#32CD32')  # Green
            else:
                self.env_state_text.set_text('WAITING')
                self.env_state_text.get_bbox_patch().set_facecolor('#FFA500')  # Orange
    
    def refresh(self):
        """Refresh the display and optionally capture frame for video."""
        with self._lock:
            if self._show:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
                plt.pause(0.001)
            
            # Capture frame for video if recording (only every N frames)
            if self.record_video:
                self._frame_counter += 1
                if self._frame_counter % self._frame_skip == 0:
                    self._capture_frame()
    
    def _capture_frame(self):
        """Capture current figure as a frame for video (fast method)."""
        # Use Agg backend renderer for fast capture
        self.fig.canvas.draw()
        
        # Get the renderer and extract raw data
        renderer = self.fig.canvas.get_renderer()
        w, h = int(renderer.width), int(renderer.height)
        
        # Get buffer as ARGB and convert to RGB
        buf = np.frombuffer(renderer.buffer_rgba(), dtype=np.uint8)
        frame = buf.reshape(h, w, 4)[:, :, :3]  # Drop alpha channel
        
        self.frames.append(frame.copy())
    
    def save_video(self, filename: str = "realtime_viz.gif", fps: int = 10):
        """
        Save recorded frames as a video/gif.
        
        Args:
            filename: Output filename (.gif recommended, .mp4 requires ffmpeg)
            fps: Frames per second
        """
        if not self.frames:
            print("No frames recorded. Set record_video=True in constructor.")
            return
        
        print(f"Saving {len(self.frames)} frames to {filename}...")
        
        # Use PIL to save as GIF (no external dependencies)
        if filename.endswith('.gif'):
            try:
                from PIL import Image
                # Convert frames to PIL images
                images = [Image.fromarray(frame) for frame in self.frames]
                # Save as GIF
                duration = int(1000 / fps)  # milliseconds per frame
                images[0].save(
                    filename,
                    save_all=True,
                    append_images=images[1:],
                    duration=duration,
                    loop=0
                )
                print(f"GIF saved: {filename}")
                return
            except Exception as e:
                print(f"Failed to save GIF: {e}")
        
        # Try imageio for mp4
        try:
            import imageio
            imageio.mimsave(filename, self.frames, fps=fps)
            print(f"Video saved: {filename}")
        except Exception as e:
            print(f"Failed to save video: {e}")
            # Fallback to GIF
            fallback = filename.rsplit('.', 1)[0] + '.gif'
            print(f"Trying to save as GIF instead: {fallback}")
            try:
                from PIL import Image
                images = [Image.fromarray(frame) for frame in self.frames]
                duration = int(1000 / fps)
                images[0].save(fallback, save_all=True, append_images=images[1:], duration=duration, loop=0)
                print(f"GIF saved: {fallback}")
            except Exception as e2:
                print(f"Failed to save GIF: {e2}")
    
    def close(self):
        """Close the visualization."""
        plt.ioff()
        plt.close(self.fig)
