"""
Comprehensive timeline visualization combining agent states, plans, and actions.

This visualization shows:
- Agent state transitions (R/W/X)
- Plan execution
- Symbolic actions
- Primitive actions
All synchronized on a dual-scale timeline (wall clock time + env steps).
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


def visualize_comprehensive_timeline(agents, plan_history, max_step, message_log=None, output_path='comprehensive_timeline.png'):
    """
    Create a comprehensive timeline showing agent states, plans, actions, and messages.
    
    Args:
        agents: Dict of agent_id -> Agent instances with state history
        plan_history: List of plan dictionaries from logger
        max_step: Maximum environment step
        message_log: List of message records from MessageBroker (optional)
        output_path: Path to save the visualization (default: 'comprehensive_timeline.png')
    """
    if message_log is None:
        message_log = []
    n_agents = len(agents)
    
    # Create figure with two sections: simple state view + detailed view
    fig = plt.figure(figsize=(16, 1.5 * n_agents + 2))
    
    # Create gridspec: top row for environment state view, rest for detailed view
    gs = fig.add_gridspec(n_agents + 1, 1, height_ratios=[1.0] + [1.2] * n_agents, hspace=0.2)
    
    # Colors
    state_colors = {
        "reasoning": "#FF9F40",  # Orange for reasoning
        "waiting": "#FFD93D",
        "executing": "#6BCB77",
        "interrupted": "#FF0000"  # Red for interrupted
    }
    
    # Find max time from state history
    max_time = 0
    for agent in agents.values():
        timeline = agent.get_state_timeline()
        if timeline:
            max_time = max(max_time, timeline[-1][0])
    
    # Extend timeline to max_step (terminate step) even if plans aren't finished
    # This ensures we plot until the end of the episode
    for agent in agents.values():
        timeline = agent.get_state_timeline()
        if timeline and len(timeline) > 1:
            last_time, last_step, _ = timeline[-1]
            
            # If last_step < max_step, we need to extrapolate
            if last_step < max_step:
                # Calculate average time per step from the timeline
                first_time, first_step, _ = timeline[0]
                if last_step > first_step:
                    avg_time_per_step = (last_time - first_time) / (last_step - first_step)
                    extrapolated_time = last_time + (max_step - last_step) * avg_time_per_step
                    max_time = max(max_time, extrapolated_time)
    
    # Collect time-step pairs for top axis
    time_step_pairs = []
    for agent in agents.values():
        timeline = agent.get_state_timeline()
        for time_val, step_val, _ in timeline:
            time_step_pairs.append((time_val, step_val))
    time_step_pairs = sorted(set(time_step_pairs), key=lambda x: x[0])
    
    # === SECTION 1: Environment Transition Timeline (top) ===
    ax_simple = fig.add_subplot(gs[0])
    
    # Determine when all agents are executing vs not
    # Collect all state transitions across agents
    all_transitions = []
    for agent_id, agent in agents.items():
        state_history = agent.get_state_timeline()
        for time_val, step_val, state in state_history:
            all_transitions.append((time_val, step_val, agent_id, state))
    all_transitions.sort(key=lambda x: x[0])  # Sort by time
    
    # Build environment state timeline
    env_states_raw = []  # List of (time_start, time_end, step_start, step_end, all_executing)
    
    # Track current state of each agent
    agent_states = {agent_id: None for agent_id in agents.keys()}
    
    for i, (time_val, step_val, agent_id, state) in enumerate(all_transitions):
        # Update the agent's current state
        agent_states[agent_id] = state
        
        # Check if all agents are executing (environment steps forward)
        all_executing = all(s is not None and s.value == "executing" for s in agent_states.values())
        
        # Find the time of next transition
        if i + 1 < len(all_transitions):
            time_next = all_transitions[i + 1][0]
            step_next = all_transitions[i + 1][1]
        else:
            time_next = max_time
            step_next = max_step
        
        # Add environment state period
        if time_next > time_val:
            env_states_raw.append((time_val, time_next, step_val, step_next, all_executing))
    
    # Merge consecutive periods with the same all_executing status
    env_states = []
    if env_states_raw:
        current_start, current_end, current_step_start, current_step_end, current_executing = env_states_raw[0]
        
        for i in range(1, len(env_states_raw)):
            time_start, time_end, step_start, step_end, all_executing = env_states_raw[i]
            
            # If same execution status, merge by extending the current period
            if all_executing == current_executing:
                current_end = time_end
                current_step_end = step_end
            else:
                # Different status, save current period and start new one
                env_states.append((current_start, current_end, current_step_start, current_step_end, current_executing))
                current_start, current_end, current_step_start, current_step_end, current_executing = time_start, time_end, step_start, step_end, all_executing
        
        # Add the last period
        env_states.append((current_start, current_end, current_step_start, current_step_end, current_executing))
    
    # Draw environment state bars and collect boundary labels
    y_pos = 0
    env_boundary_labels = []  # [(time, step), ...] for dashed line labels
    
    for time_start, time_end, step_start, step_end, all_executing in env_states:
        width = time_end - time_start
        if width > 0:
            color = '#90EE90' if all_executing else '#FFB6C6'  # Light green if executing, light red if not
            label_text = 'Env Stepping' if all_executing else 'Waiting'
            
            rect = Rectangle(
                (time_start, y_pos), 
                width, 
                0.8,
                facecolor=color,
                edgecolor='black',
                linewidth=1
            )
            ax_simple.add_patch(rect)
            
            # For env stepping blocks, record boundary labels to align with dashed lines
            if all_executing:
                env_boundary_labels.append((time_start, step_start))
                env_boundary_labels.append((time_end, step_end))
            
            # Show block type label centered (without step info)
            if width > 0.2:
                ax_simple.text(
                    time_start + width/2, 
                    y_pos + 0.4, 
                    label_text,
                    ha='center', 
                    va='center',
                    fontsize=8,
                    fontweight='bold'
                )
    
    # Collect boundary labels for top x-axis (aligned with dashed lines)
    # Remove duplicates and sort by time
    env_boundary_labels = sorted(set(env_boundary_labels), key=lambda x: x[0])
    
    # Ensure max_time is positive
    if max_time <= 0:
        max_time = 1.0
    
    ax_simple.set_xlim(0, max_time * 1.02)
    ax_simple.set_ylim(-0.2, 1.2)
    ax_simple.set_yticks([0.4])
    ax_simple.set_yticklabels(['Environment'])
    ax_simple.set_xlabel('Wall Clock Time (seconds)', fontsize=9)
    ax_simple.set_title('Environment State Timeline (All Agents Executing → Env Steps)', 
                       fontsize=11, fontweight='bold', pad=10)
    ax_simple.grid(True, axis='x', alpha=0.3)
    
    # Top x-axis with env steps - aligned with vertical dashed lines (env boundaries)
    ax_simple_top = ax_simple.twiny()
    ax_simple_top.set_xlim(ax_simple.get_xlim())
    if env_boundary_labels:
        tick_times = [bl[0] for bl in env_boundary_labels]
        tick_labels = [str(bl[1]) for bl in env_boundary_labels]
        
        ax_simple_top.set_xticks(tick_times)
        ax_simple_top.set_xticklabels(tick_labels, fontsize=8)
    ax_simple_top.set_xlabel('Environment Step', fontsize=9)
    
    # === SECTION 2: Detailed Multi-Layer Timeline (bottom) ===
    
    # Store env transition boundaries for vertical lines
    env_transition_times = []
    for time_start, time_end, step_start, step_end, all_executing in env_states:
        if all_executing:  # Only for "Env Stepping" blocks
            env_transition_times.append(time_start)
            env_transition_times.append(time_end)
    
    # Sort agents by ID to ensure consistent ordering (agent_0, agent_1, agent_2, ...)
    sorted_agents = sorted(agents.items(), key=lambda x: x[0])
    
    # Process each agent
    for idx, (agent_id, agent) in enumerate(sorted_agents):
        ax = fig.add_subplot(gs[idx + 1])
        
        # Get agent's plans
        agent_plans = [p for p in plan_history if p['agent_id'] == agent_id]
        
        # --- Layer 1: Plans (bottom layer) ---
        # Build plan segments from state history, using plan_timeline for plan IDs
        state_history = agent.get_state_timeline()
        plan_timeline = getattr(agent, 'plan_timeline', [])  # [(time, step, plan_id), ...]
        # Convert to relative time
        plan_timeline = [(t - agent._start_time, step, pid) for t, step, pid in plan_timeline]
        
        plan_segments = []  # List of (time_start, time_end, plan_id)
        
        # Track W/X segments (when agent has a plan ready)
        in_ready_state = False
        segment_start_time = None
        current_plan_id = None
        plan_timeline_idx = 0  # Track which plan_timeline entry we're on
        
        for i in range(len(state_history)):
            time_val, step_val, state = state_history[i]
            
            # Entering W or X state - agent is ready and in a plan
            if state.value in ["waiting", "executing"] and not in_ready_state:
                segment_start_time = time_val
                # Get plan_id from plan_timeline by index
                # Each W state entry corresponds to a plan_timeline entry in order
                if plan_timeline_idx < len(plan_timeline):
                    _, _, current_plan_id = plan_timeline[plan_timeline_idx]
                    plan_timeline_idx += 1
                else:
                    current_plan_id = None
                in_ready_state = True
                
            # Leaving ready state (W/X) - end current segment
            elif state.value in ["reasoning", "interrupted"] and in_ready_state:
                plan_segments.append((segment_start_time, time_val, current_plan_id))
                in_ready_state = False
                segment_start_time = None
        
        # If still in a ready state at the end
        if in_ready_state and segment_start_time is not None:
            plan_segments.append((segment_start_time, max_time, current_plan_id))
        
        # Draw plan segment boxes
        for time_start, time_end, plan_id in plan_segments:
            width = time_end - time_start
            if width > 0:
                # Find matching plan from history for details
                matching_plan = None
                if plan_id is not None:
                    for p in agent_plans:
                        if p.get('plan_id') == plan_id:
                            matching_plan = p
                            break
                
                plan_box = Rectangle(
                    (time_start, 0.18), 
                    width, 
                    0.25,
                    facecolor='lightblue',
                    edgecolor='darkblue',
                    linewidth=1,
                    alpha=0.6,
                    zorder=1
                )
                ax.add_patch(plan_box)
                
                # Label plan with the actual plan_id from the plan data
                if plan_id is not None:
                    plan_label = f"P{plan_id}"
                    if matching_plan:
                        status = matching_plan.get('status', 'unknown')
                        if status == 'success':
                            plan_label += " [OK]"
                        elif status == 'failed':
                            plan_label += " [FAIL]"
                        elif status == 'interrupted':
                            plan_label += " [INT]"
                    
                    if width > 0.3:
                        ax.text(
                            time_start + width/2, 
                            0.30, 
                            plan_label,
                            ha='center', 
                            va='center',
                            fontsize=7,
                            fontweight='bold',
                            zorder=2
                        )
        
        # --- Layer 1b: Messages (bottom layer) ---
        msg_sent_color = '#9B59B6'  # Purple for sent
        msg_recv_color = '#3498DB'  # Blue for received
        msg_y = 0.08  # Y position for message layer
        msg_width = 0.02  # Thin fixed width for message markers
        msg_height = 0.10  # Height of message markers
        
        # Plot sent messages
        for msg in message_log:
            if msg['sender'] == agent_id:
                # This agent sent this message
                msg_time = msg['timestamp']
                rect = Rectangle(
                    (msg_time - msg_width/2, msg_y - msg_height/2),
                    msg_width,
                    msg_height,
                    facecolor=msg_sent_color,
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.6,
                    zorder=5
                )
                ax.add_patch(rect)
        
        # Plot received messages
        for msg in message_log:
            if agent_id in msg['recipients']:
                # This agent received this message
                msg_time = msg['timestamp']
                rect = Rectangle(
                    (msg_time - msg_width/2, msg_y - msg_height/2),
                    msg_width,
                    msg_height,
                    facecolor=msg_recv_color,
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.6,
                    zorder=5
                )
                ax.add_patch(rect)
        
        # --- Layer 3: Agent States (top layer) ---
        state_history = agent.get_state_timeline()
        for i in range(len(state_history)):
            time_start, step_start, state = state_history[i]
            
            if i + 1 < len(state_history):
                time_end, step_end, _ = state_history[i + 1]
            else:
                time_end = max_time
                step_end = max_step
            
            width = time_end - time_start
            if width > 0:
                rect = Rectangle(
                    (time_start, 0.6), 
                    width, 
                    0.3,
                    facecolor=state_colors[state.value],
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.4,
                    zorder=3
                )
                ax.add_patch(rect)
                
                # Label states
                if width > 0.2:
                    ax.text(
                        time_start + width/2, 
                        0.75, 
                        state.value[0].upper(),
                        ha='center', 
                        va='center',
                        fontsize=8,
                        fontweight='bold',
                        zorder=4
                    )
        
        # Formatting
        ax.set_xlim(0, max_time * 1.02)
        ax.set_ylim(-0.05, 1.0)
        ax.set_ylabel(agent_id, fontweight='bold', fontsize=9)
        ax.set_yticks([0.08, 0.30, 0.75])
        ax.set_yticklabels(['Msg', 'Plan', 'State'], fontsize=7)
        ax.grid(True, axis='x', alpha=0.3, zorder=0)
        
        # Only show x-axis label on bottom plot
        if idx == n_agents - 1:
            ax.set_xlabel('Wall Clock Time (seconds)', fontsize=10)
        else:
            ax.set_xticklabels([])
        
        # Draw vertical dashed lines for env transition boundaries
        for transition_time in env_transition_times:
            ax.axvline(x=transition_time, color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.6, zorder=10)
    
    # Also draw vertical lines on the top environment timeline
    for transition_time in env_transition_times:
        ax_simple.axvline(x=transition_time, color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.6, zorder=10)
    
    # Add overall title
    fig.suptitle('Multi-Agent Timeline (States + Plans + Messages)', 
                 fontsize=12, fontweight='bold', y=0.98)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor=state_colors['reasoning'], label='R (Reasoning)', alpha=0.4),
        mpatches.Patch(facecolor=state_colors['waiting'], label='W (Waiting)', alpha=0.4),
        mpatches.Patch(facecolor=state_colors['executing'], label='X (Executing)', alpha=0.4),
        mpatches.Patch(facecolor='lightblue', edgecolor='darkblue', label='Plan', alpha=0.6),
        mpatches.Patch(facecolor='#9B59B6', edgecolor='black', label='Msg Sent'),
        mpatches.Patch(facecolor='#3498DB', edgecolor='black', label='Msg Received'),
    ]
    
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.96), 
              ncol=3, fontsize=8, framealpha=0.9)
    
    plt.subplots_adjust(top=0.94, bottom=0.05, left=0.08, right=0.85)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nComprehensive timeline saved to {output_path}")
    plt.close()
