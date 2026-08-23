"""Text representation of symbolic world view for CUBE agents."""

import numpy as np
from typing import Tuple, Optional

from .world_state import SymbolicWorldState


def get_agent_symbolic_view_text(
    world_state: SymbolicWorldState,
    agent_name: str,
    view_size: Tuple[int, int] = (9, 9)
) -> str:
    """
    Get text representation of agent's symbolic view with IDs for all entities.
    
    Args:
        world_state: The symbolic world state
        agent_name: Agent name (e.g., 'agent_0')
        view_size: View window size (height, width)
        
    Returns:
        String representation of the symbolic view
    """
    # Find agent position
    agent_pos = world_state.get_agent_position(agent_name)
    
    if agent_pos is None:
        return f"Agent {agent_name} not found"
    
    symbolic_slice = world_state.get_symbolic_slice(agent_pos, view_size)
    
    vh, vw = view_size
    offset_r, offset_c = vh // 2, vw // 2
    
    output = [f"\n=== Agent {agent_name} Symbolic View at {agent_pos} ==="]
    output.append(f"View size: {view_size}")
    output.append(f"Legend: [cell_type#id] [block#id] [agent#id]")
    output.append("")
    
    # Build header with column indices
    header = "    "
    for c in range(vw):
        header += f"{c:^12}"
    output.append(header)
    output.append("    " + "-" * (vw * 12))
    
    # Build display rows
    for j in range(vh):
        row = [f"{j:2} |"]
        for i in range(vw):
            is_center = (j == offset_r and i == offset_c)
            entities = symbolic_slice[j, i]
            
            # Filter to show blocks/agents if present, else show cell type
            if entities:
                non_cell = [e for e in entities if e[0] != 'cell']
                
                if non_cell:
                    # Show blocks and agents
                    labels = []
                    for etype, eid, name in non_cell:
                        if etype == 'block':
                            labels.append(f"B{eid}")
                        elif etype == 'agent':
                            labels.append(f"A{eid}")
                    cell = "[" + ",".join(labels) + "]"
                else:
                    # Show cell type
                    cell_info = entities[0]  # Should be ('cell', id, type)
                    cell_type = cell_info[2]
                    cell_id = cell_info[1]
                    if cell_type == "goal":
                        cell = f"[G{cell_id}]"
                    else:
                        cell = f"[.{cell_id}]"
            else:
                cell = "[--]"
            
            if is_center:
                cell = f"*{cell}*"
            
            row.append(f"{cell:^12}")
        
        output.append("".join(row))
    
    output.append("")
    return "\n".join(output)


def get_full_world_view_text(
    world_state: SymbolicWorldState,
    show_cell_ids: bool = False
) -> str:
    """
    Get text representation of the entire world.
    
    Args:
        world_state: The symbolic world state
        show_cell_ids: Whether to show cell IDs (can be verbose)
        
    Returns:
        String representation of the full world
    """
    K = world_state.grid_size
    
    output = [f"\n=== CUBE World State ({K}x{K}) ==="]
    output.append(f"Blocks: {len(world_state.block_positions)}, Agents: {len(world_state.agent_positions)}")
    output.append(f"Goal column: {world_state.goal_column}")
    output.append("")
    
    # Build header
    header = "    "
    for c in range(K):
        header += f"{c:^6}"
    output.append(header)
    output.append("    " + "-" * (K * 6))
    
    # Build rows
    for r in range(K):
        row = [f"{r:2} |"]
        for c in range(K):
            entities = world_state.symbolic_matrix[r, c]
            
            # Priority: agent > block > cell
            non_cell = [e for e in entities if e[0] != 'cell']
            
            if non_cell:
                # Show highest priority entity
                for etype, eid, name in non_cell:
                    if etype == 'agent':
                        cell = f"A{eid}"
                        break
                    elif etype == 'block':
                        cell = f"B{eid}"
                else:
                    cell = "?"
            else:
                # Show cell
                is_goal = (c == world_state.goal_column)
                if show_cell_ids:
                    cell_id = world_state.id_manager._cell_ids.get((r, c), 0)
                    cell = f"G{cell_id}" if is_goal else f".{cell_id}"
                else:
                    cell = "G" if is_goal else "."
            
            row.append(f"{cell:^6}")
        
        output.append("".join(row))
    
    output.append("")
    
    # Summary
    output.append("Blocks:")
    for block_info in world_state.get_all_blocks():
        output.append(f"  Block {block_info['env_id']}: pos={block_info['position']}, weight={block_info['weight']}")
    
    output.append("Agents:")
    for agent_info in world_state.get_all_agents():
        output.append(f"  {agent_info['name']}: pos={agent_info['position']}")
    
    return "\n".join(output)
