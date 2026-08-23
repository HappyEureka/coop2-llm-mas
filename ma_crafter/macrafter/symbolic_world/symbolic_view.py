"""Text representation of symbolic world view for agents."""

import numpy as np
from typing import Tuple

from .world_state import SymbolicWorldState


def _entity_label(world_state: SymbolicWorldState, etype: str, entity_id: int, name: str) -> str:
    if etype == 'agent':
        return f"A{name}#{entity_id}"

    local_id = world_state.get_type_local_id(name, entity_id)
    if local_id is not None:
        return f"{name}#{local_id}"

    return f"{name[:3]}#{entity_id}"


def get_agent_symbolic_view_text(
    world_state: SymbolicWorldState,
    agent_id: int,
    view_size: Tuple[int, int] = (9, 8)
) -> str:
    """
    Get text representation of agent's symbolic view with IDs for all entities.
    
    Args:
        world_state: The symbolic world state
        agent_id: Agent ID (can be int or string, will look up position from agent name)
        view_size: View window size
        
    Returns:
        String representation of the symbolic view
    """
    # Find agent position by looking through all agents to find matching name
    agent_pos = None
    agent_stable_id = None
    agent_name = str(agent_id)
    agent_props = None
    
    for stable_id, props in world_state.agent_properties.items():
        if props['name'] == agent_name:
            agent_pos = props['position']
            agent_stable_id = stable_id
            agent_props = props
            break
    
    if agent_pos is None:
        return f"Agent {agent_id} not found"
    
    symbolic_slice = world_state.get_symbolic_slice(agent_pos, view_size)
    
    output = [f"\n=== Agent {agent_id} Symbolic View at {agent_pos} ==="]
    output.append(f"View size: {view_size}")
    output.append("")
    
    # Transpose symbolic_slice for display: comes as [y, x], transpose to [x, y] for R,C order
    vw, vh = view_size
    
    # Use numpy transpose for object array
    transposed = np.transpose(symbolic_slice)
    
    offset_x, offset_y = vw // 2, vh // 2
    visible_utilities = {}
    
    # Build display rows
    for j in range(1, vh):  # y index (row in display), start from 1
        row = []
        for i in range(vw):  # x index (col in display)
            is_center = (i == offset_x and j == offset_y)
            entities = transposed[i, j]
            
            # Filter to show non-walkable entities (objects/agents) if present, else materials
            if entities:
                non_material = [e for e in entities if e[0] != 'material']
                entities_to_show = non_material if non_material else entities
                
                # Format entity labels
                labels = []
                for etype, entity_id, name in entities_to_show:
                    labels.append(_entity_label(world_state, etype, entity_id, name))
                    local_id = world_state.get_type_local_id(name, entity_id)
                    if local_id is not None:
                        visible_utilities[(name, local_id)] = int(entity_id)
                cell = "[" + ",".join(labels) + "]"
            else:
                cell = "[]"
            
            row.append(f"*{cell}*" if is_center else cell)
        
        output.append(" ".join(row))
    
    output.append("")
    
    # Add agent inventory
    if agent_props:
        inventory_items = [f"{item}:{count}" for item, count in agent_props['inventory'].items() if count > 0]
        if inventory_items:
            output.append(f"Inventory: {', '.join(inventory_items)}")
        else:
            output.append("Inventory: Empty")

    if visible_utilities:
        utility_labels = []
        for (name, local_id), stable_id in sorted(visible_utilities.items()):
            utility_labels.append(
                f"{name} {local_id} (navigate object_type={name}, item_id={local_id}; stable_id={stable_id})"
            )
        output.append(f"Visible utility IDs: {', '.join(utility_labels)}")
    
    output.append("")
    output.append("Legend: Material#ID, Object#ID, Agent#ID, utility#local_id, * = Current position")
    
    return "\n".join(output)
