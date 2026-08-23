"""
Clean visualization tools for plan execution.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict


# Color mappings for visualization
PLAN_COLORS = {
    'success': 'green',
    'failed': 'red', 
    'executing': 'orange',
    'pending': 'yellow'
}

ACTION_COLORS = {
    'move': 'skyblue',
    'navigate': 'lightseagreen',
    'collect': 'lightcoral', 
    'craft': 'lightgreen',
    'place': 'orange',
    'sleep': 'purple',
    'share': 'gold',
    'default': 'lightblue'
}

PRIMITIVE_COLORS = {
    'noop': '#CED4DA',
    'move_left': '#2E6DB4',
    'move_right': '#C41E3A', 
    'move_up': '#2EAA5E',
    'move_down': '#D68910',
    'do': '#DC143C',
    'navigate': '#20B2AA',  # lightseagreen
    'share': '#FFD700',  # gold
    'make_wood': '#6A4C93',
    'make_stone': '#008C78',
    'make_iron': '#A63A28',
    'place_stone': '#E6B800',
    'place_table': '#5FAD7F',
    'place_furnace': '#C47F3C',
    'sleep': '#6C757D'
}


def get_primitive_action_label(primitive_action):
    """Get short label for primitive action."""
    if primitive_action.startswith('move_'):
        direction = primitive_action.replace('move_', '').upper()
        return direction[0] if direction else 'M'
    elif primitive_action.startswith('make_'):
        return 'Mk'
    elif primitive_action.startswith('place_'):
        return 'Pl'
    elif primitive_action == 'do':
        return 'Do'
    elif primitive_action == 'sleep':
        return 'Sl'
    elif primitive_action == 'noop':
        return 'N'
    else:
        return primitive_action[:2].upper()


def visualize_plan_timeline(plan_history: List[Dict], agent_id: str = None, figsize=(14, 8)):
    """
    Simple plan execution timeline visualization.
    
    Args:
        plan_history: List of plan dictionaries
        agent_id: Optional agent ID to filter by
        figsize: Figure size
    """
    if agent_id:
        plans = [p for p in plan_history if p["agent_id"] == agent_id]
        title = f"Plan Timeline - Agent {agent_id}"
    else:
        plans = plan_history
        title = "Plan Timeline - All Agents"
    
    if not plans:
        print("No plans to visualize")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Timeline of plans
    for i, plan in enumerate(plans):
        start = plan['created_at_step']
        end = plan['end_step'] if plan['end_step'] else plan['created_at_step']
        duration = max(end - start, 1)
        status = plan['status']
        
        # Draw plan bar
        color = PLAN_COLORS.get(status, 'gray')
        ax.barh(i, duration, left=start, height=0.8, 
                color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add plan ID and specification
        spec_short = plan['specification'][:30] + "..." if len(plan['specification']) > 30 else plan['specification']
        ax.text(start + duration/2, i, f"#{plan['plan_id']}: {spec_short}", 
                ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Environment Step', fontsize=12)
    ax.set_ylabel('Plan #', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(plans)))
    ax.set_yticklabels([f"Plan {p['plan_id']}" for p in plans])
    ax.grid(axis='x', alpha=0.3)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='green', alpha=0.7, label='Success'),
        mpatches.Patch(color='red', alpha=0.7, label='Failed'),
        mpatches.Patch(color='orange', alpha=0.7, label='Executing'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig


def visualize_all_agents_progress(plan_history: List[Dict], figsize=None, output_path='agent_timeline.png'):
    """
    Visualize progress of all agents with three-row display per agent:
    1. Plans (high level)
    2. Symbolic actions (medium level) 
    3. Primitive actions (low level)
    """
    # Group by agent
    agents = {}
    for plan in plan_history:
        agent_id = plan['agent_id']
        if agent_id not in agents:
            agents[agent_id] = []
        agents[agent_id].append(plan)
    
    if not agents:
        print("No plans to visualize")
        return
    
    num_agents = len(agents)
    
    # Calculate dynamic figure size if not provided
    if figsize is None:
        figsize = (14, max(10, num_agents * 4))  # Scale height with number of agents
    
    fig, axes = plt.subplots(num_agents * 3, 1, figsize=figsize, sharex=True)
    
    # Handle single agent case
    if num_agents == 1:
        axes = [axes] if not hasattr(axes, '__len__') else axes
    
    # Sort agents by ID to ensure consistent ordering (agent_0, agent_1, agent_2, ...)
    sorted_agents = sorted(agents.items(), key=lambda x: x[0])
    
    for agent_idx, (agent_id, agent_plans) in enumerate(sorted_agents):
        # Three axes for this agent
        plan_ax = axes[agent_idx * 3]
        symbolic_ax = axes[agent_idx * 3 + 1] 
        primitive_ax = axes[agent_idx * 3 + 2]
        
        # Row 1: Plan level
        for plan in agent_plans:
            start = plan.get('start_step') or plan.get('created_at_step') or 0
            end = plan.get('end_step') or (start + 1)
            duration = end - start
            
            # Skip plans with zero duration (interrupted immediately)
            if duration <= 0:
                continue
            
            color = PLAN_COLORS.get(plan['status'], 'gray')
            plan_ax.barh(0, duration, left=start, height=0.6,
                        color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
            
            # Add plan label with specification
            spec = plan.get('specification', '')
            spec_short = spec[:20] + "..." if len(spec) > 20 else spec
            plan_ax.text(start + duration/2, 0, f"P{plan['plan_id']}: {spec_short}", 
                        ha='center', va='center', fontsize=7, fontweight='bold',
                        color='white')
        
        # Row 2: Symbolic actions
        for plan in agent_plans:
            plan_end = plan.get('end_step', plan.get('start_step', plan['created_at_step']))
            
            for action in plan['actions']:
                start = action.get('start_step')
                end = action.get('end_step')
                
                # Skip actions that never started
                if start is None:
                    continue
                
                # For actions that started but didn't complete (interrupted), use plan end
                if end is None:
                    end = plan_end
                
                # Skip actions that started at the exact moment episode ended (no actual execution)
                if start >= end:
                    continue
                    
                duration = end - start
                action_type = action['action_type'].lower()
                color = ACTION_COLORS.get(action_type, ACTION_COLORS['default'])
                
                symbolic_ax.barh(0, duration, left=start, height=0.6,
                               color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
                
                # Add action label
                label = action['action_type'][:3]
                symbolic_ax.text(start + duration/2, 0, label,
                               ha='center', va='center', fontsize=7, fontweight='bold',
                               color='white')
        
        # Row 3: Primitive actions
        step_actions = {}
        for plan in agent_plans:
            plan_end = plan.get('end_step', plan.get('start_step', plan['created_at_step']))
            
            for action in plan['actions']:
                start = action.get('start_step')
                end = action.get('end_step')
                primitive = action.get('primitive_action', 'noop')
                # Get per-step primitive action history if available
                primitive_history = action.get('primitive_action_history', {})
                
                # Skip actions that never started
                if start is None:
                    continue
                
                # For actions that started but didn't complete (interrupted), use plan end
                if end is None:
                    end = plan_end
                
                # Skip actions that started at the exact moment episode ended (no actual execution)
                if start >= end:
                    continue
                
                # Fill steps where action executed (not including completion step)
                # Action executes from start to end-1, then completes at end
                for step in range(start, end):
                    # Use per-step history if available, otherwise fallback to single primitive
                    step_primitive = primitive_history.get(step) or primitive_history.get(str(step)) or primitive
                    step_actions[step] = step_primitive
        
        # Draw primitive action bars
        for step, primitive_action in sorted(step_actions.items()):
            if primitive_action is None:
                primitive_action = 'noop'
            
            color = PRIMITIVE_COLORS.get(primitive_action, 'lightsteelblue')
            primitive_ax.barh(0, 1, left=step, height=0.6,
                           color=color, edgecolor='white', linewidth=0.5)
            
            # Add action label for first occurrence or different action
            if step == 0 or step_actions.get(step - 1) != primitive_action:
                label = get_primitive_action_label(primitive_action)
                primitive_ax.text(step + 0.5, 0, label,
                               ha='center', va='center', fontsize=7, fontweight='bold',
                               color='white', rotation=90)
        
        # Configure axes
        plan_ax.set_ylabel(f"{agent_id}\nPlans", fontsize=10, fontweight='bold')
        symbolic_ax.set_ylabel(f"{agent_id}\nSymbolic\nActions", fontsize=10, fontweight='bold')
        primitive_ax.set_ylabel(f"{agent_id}\nPrimitive\nActions", fontsize=10, fontweight='bold')
        
        for ax in [plan_ax, symbolic_ax, primitive_ax]:
            ax.set_yticks([])
            ax.grid(axis='x', alpha=0.3)
            ax.set_ylim(-0.5, 0.5)
    
    axes[-1].set_xlabel('Environment Step', fontsize=12)
    
    # Add legends
    plan_legend_elements = [
        mpatches.Patch(color='green', alpha=0.7, label='Success'),
        mpatches.Patch(color='red', alpha=0.7, label='Failed'),
        mpatches.Patch(color='skyblue', alpha=0.8, label='Move'),
        mpatches.Patch(color='lightcoral', alpha=0.8, label='Collect'),
        mpatches.Patch(color='lightgreen', alpha=0.8, label='Craft'),
    ]
    axes[0].legend(handles=plan_legend_elements, loc='upper right', fontsize=9)
    
    # Primitive action legend at the bottom
    primitive_legend_elements = [
        mpatches.Patch(facecolor=color, edgecolor='white', label=name.replace('_', ' ').title())
        for name, color in PRIMITIVE_COLORS.items()
    ]
    
    fig.legend(handles=primitive_legend_elements, loc='lower center', ncol=7, 
              bbox_to_anchor=(0.5, -0.02), frameon=True, title='Primitive Actions',
              fontsize=8, title_fontsize=9)
    
    plt.suptitle('All Agents Execution Timeline (Plans | Symbolic Actions | Primitive Actions)', 
                 fontsize=14, fontweight='bold')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Agent timeline saved to {output_path}")
    plt.close()
    return fig


def print_plan_summary(plan_history: List[Dict]):
    """Print a text summary of plan execution."""
    print("\n" + "="*80)
    print("PLAN EXECUTION SUMMARY")
    print("="*80)
    
    # Group by agent
    agents = {}
    for plan in plan_history:
        agent_id = plan['agent_id']
        if agent_id not in agents:
            agents[agent_id] = []
        agents[agent_id].append(plan)
    
    for agent_id, agent_plans in agents.items():
        print(f"\n{'='*40}")
        print(f"Agent: {agent_id}")
        print(f"{'='*40}")
        print(f"Total Plans: {len(agent_plans)}")
        
        successful = sum(1 for p in agent_plans if p['status'] == 'success')
        failed = sum(1 for p in agent_plans if p['status'] == 'failed')
        executing = sum(1 for p in agent_plans if p['status'] == 'executing')
        interrupted = sum(1 for p in agent_plans if p['status'] == 'interrupted')
        
        if agent_plans:
            print(f"Successful: {successful} ({100*successful/len(agent_plans):.1f}%)")
            print(f"Failed: {failed} ({100*failed/len(agent_plans):.1f}%)")
            if interrupted > 0:
                print(f"Interrupted: {interrupted} ({100*interrupted/len(agent_plans):.1f}%)")
            if executing > 0:
                print(f"Executing: {executing} ({100*executing/len(agent_plans):.1f}%)")
        
        print(f"\n{'Plan Details':^40}")
        print("-"*40)
        
        for plan in agent_plans:
            if plan['status'] == 'success':
                status_symbol = "[OK]"
            elif plan['status'] == 'failed':
                status_symbol = "[FAIL]"
            elif plan['status'] == 'interrupted':
                status_symbol = "[INT]"
            elif plan['status'] == 'executing':
                status_symbol = ">"
            else:
                status_symbol = " "
            print(f"\n{status_symbol} Plan #{plan['plan_id']}: {plan['specification']}")
            end_display = plan.get('end_step', 'ongoing') if plan.get('end_step') is not None else 'ongoing'
            print(f"  Steps: {plan['created_at_step']} - {plan.get('start_step', 'N/A')} - {end_display}")
            print(f"  Actions: {len(plan['actions'])}")
            
            for i, action in enumerate(plan['actions']):
                # Build step range display
                start = action.get('start_step')
                end = action.get('end_step')
                if start is not None and end is not None:
                    step_str = f"[{start}-{end}]"
                elif start is not None:
                    step_str = f"[{start}]"
                else:
                    step_str = "[?]"
                
                args_str = f"({action['args']})" if action['args'] else ""
                primitive = action.get('primitive_action')
                primitive_str = f" | {primitive}" if primitive else ""
                print(f"    {i+1}. {step_str} {action['action_type']}{args_str}{primitive_str}")
                
                if action.get('failure_reason'):
                    print(f"       Reason: {action['failure_reason']}")
            
            if plan.get('failure_reason'):
                print(f"  Failure: {plan['failure_reason']}")
    
    print("\n" + "="*80)