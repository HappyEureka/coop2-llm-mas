"""Navigation utilities for symbolic actions."""

from typing import Optional, Tuple, List, Set
import heapq


def find_object_position(
    world_state,
    item_id: int,
    object_type: Optional[str] = None,
) -> Optional[Tuple[int, int]]:
    """
    Find position of specific object in world state.
    
    Args:
        world_state: SymbolicWorldState instance
        item_id: Stable ID of the item, or a type-local utility ID
            when object_type is table/furnace.
        object_type: Optional object/material type to validate. For utility
            materials, this allows readable references like table 1.
        
    Returns:
        (x, y) position or None if not found
    """
    from macrafter.symbolic_world import SymbolicWorldState
    
    if not isinstance(world_state, SymbolicWorldState):
        return None

    try:
        item_id = int(item_id)
    except (TypeError, ValueError):
        return None

    resolver = getattr(world_state, "resolve_entity_position", None)
    if resolver is not None:
        position = resolver(object_type, item_id)
        if position is not None:
            return position
    
    # Check in object positions
    if item_id in world_state.object_positions:
        if object_type is None:
            return world_state.object_positions[item_id]
        props = world_state.object_properties.get(item_id, {})
        if str(props.get("type", "")).lower() == str(object_type).lower().replace(" ", "_"):
            return world_state.object_positions[item_id]
    
    # Check in agent positions
    if object_type is None and item_id in world_state.agent_positions:
        return world_state.agent_positions[item_id]
    
    # Search symbolic matrix for materials using flat iterator
    if world_state.symbolic_matrix is not None:
        shape = world_state.symbolic_matrix.shape
        for idx, entities in enumerate(world_state.symbolic_matrix.flat):
            for entity_type, entity_id, entity_name in entities:
                if entity_id == item_id:
                    if object_type is not None and entity_name != str(object_type).lower().replace(" ", "_"):
                        continue
                    # Convert flat index back to 2D coordinates
                    x = idx // shape[1]
                    y = idx % shape[1]
                    return (x, y)
    
    return None


def get_agent_position(world_state, agent_id: str) -> Optional[Tuple[int, int]]:
    """Return the current grid position for an agent id."""
    from macrafter.symbolic_world import SymbolicWorldState

    if not isinstance(world_state, SymbolicWorldState):
        return None

    numeric_id = agent_id.split('_')[-1] if '_' in str(agent_id) else str(agent_id)
    for props in world_state.agent_properties.values():
        if props.get('name') == numeric_id:
            position = props.get('position')
            if position is not None:
                return tuple(position)
    return None


def _manhattan(pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> int:
    return abs(int(pos_a[0]) - int(pos_b[0])) + abs(int(pos_a[1]) - int(pos_b[1]))


def is_near_target(
    world_state,
    agent_id: str,
    target_pos: Tuple[int, int],
    radius: int = 1,
) -> bool:
    """Return True when the agent is within interaction radius of target_pos."""
    agent_pos = get_agent_position(world_state, agent_id)
    if agent_pos is None:
        return False
    return _manhattan(agent_pos, target_pos) <= max(0, int(radius))


def is_facing_target(world_state, agent_id: str, target_pos: Tuple[int, int]) -> bool:
    """
    Check if agent is adjacent to and facing the target position.
    
    Args:
        world_state: SymbolicWorldState instance
        agent_id: Agent ID string
        target_pos: (x, y) position of target
        
    Returns:
        True if agent is adjacent and facing target, False otherwise
    """
    from macrafter.symbolic_world import SymbolicWorldState
    
    if not isinstance(world_state, SymbolicWorldState):
        return False
    
    # Get agent's position and facing direction from agent properties
    # Extract numeric ID from agent_id (e.g., "agent_0" -> "0")
    numeric_id = agent_id.split('_')[-1] if '_' in str(agent_id) else str(agent_id)
    
    agent_props = None
    for stable_id, props in world_state.agent_properties.items():
        if props['name'] == numeric_id:
            agent_props = props
            break
    
    if not agent_props:
        return False
    
    agent_pos = agent_props['position']
    facing = agent_props['facing']  # (dx, dy) direction tuple
    
    # Calculate the position the agent is facing
    facing_pos = (agent_pos[0] + facing[0], agent_pos[1] + facing[1])
    
    # Check if facing position matches target position
    # OR if agent is at target position (walked into it)
    return facing_pos == target_pos or agent_pos == target_pos


def is_walkable(world_state, pos: Tuple[int, int]) -> bool:
    """
    Check if a position is walkable (no solid objects blocking).
    
    Args:
        world_state: SymbolicWorldState instance
        pos: (x, y) position to check
        
    Returns:
        True if walkable, False otherwise
    """
    from macrafter.symbolic_world import SymbolicWorldState
    
    if not isinstance(world_state, SymbolicWorldState):
        return False
    
    x, y = pos
    # Check bounds
    if not (0 <= x < world_state.world.area[0] and 0 <= y < world_state.world.area[1]):
        return False
    
    # Check if position has blocking entities
    entities = world_state.symbolic_matrix[x, y]
    for entity_type, entity_id, entity_name in entities:
        # Non-walkable materials: stone, water, lava, tree, coal, iron, diamond
        if entity_type == 'material' and entity_name in ['stone', 'water', 'lava', 'tree', 'coal', 'iron', 'diamond']:
            return False
        # Non-walkable objects: tree, stone, table, furnace, plant
        if entity_type == 'object' and entity_name in ['tree', 'stone', 'table', 'furnace', 'plant', 'cow', 'zombie', 'skeleton']:
            return False
        # Agents block movement
        if entity_type == 'agent':
            return False
    
    return True


def astar_pathfind(world_state, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """
    Find path from start to goal using A* algorithm.
    
    Args:
        world_state: SymbolicWorldState instance
        start: (x, y) starting position
        goal: (x, y) goal position
        
    Returns:
        List of (x, y) positions from start to goal, or None if no path exists
    """
    from macrafter.symbolic_world import SymbolicWorldState
    
    if not isinstance(world_state, SymbolicWorldState):
        return None
    
    # Manhattan distance heuristic
    def heuristic(pos: Tuple[int, int]) -> int:
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    # Priority queue: (f_score, counter, current_pos, path)
    counter = 0
    heap = [(heuristic(start), counter, start, [start])]
    visited: Set[Tuple[int, int]] = set()
    
    while heap:
        f_score, _, current, path = heapq.heappop(heap)
        
        if current in visited:
            continue
        
        visited.add(current)
        
        # Check if reached goal
        if current == goal:
            return path
        
        # Try all 4 directions
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # up, down, left, right
            next_pos = (current[0] + dx, current[1] + dy)
            
            if next_pos in visited:
                continue
            
            # Check if walkable (or is the goal itself, which might not be walkable)
            if not is_walkable(world_state, next_pos) and next_pos != goal:
                continue
            
            # Calculate scores
            g_score = len(path)  # Current path length
            h_score = heuristic(next_pos)
            f = g_score + h_score
            
            counter += 1
            heapq.heappush(heap, (f, counter, next_pos, path + [next_pos]))
    
    # No path found
    return None


def shortest_path_to_target_zone(
    world_state,
    start: Tuple[int, int],
    target_pos: Tuple[int, int],
    radius: int = 1,
) -> Optional[List[Tuple[int, int]]]:
    """Find the shortest path to any walkable cell within radius of target_pos."""
    radius = max(0, int(radius))
    if _manhattan(start, target_pos) <= radius:
        return [start]

    candidates: List[Tuple[int, int]] = []
    if radius <= 0:
        candidates.append(target_pos)
    else:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                candidate = (int(target_pos[0]) + dx, int(target_pos[1]) + dy)
                if _manhattan(candidate, target_pos) > radius:
                    continue
                if candidate != start and not is_walkable(world_state, candidate):
                    continue
                candidates.append(candidate)

    best_path: Optional[List[Tuple[int, int]]] = None
    for candidate in candidates:
        path = astar_pathfind(world_state, start, candidate)
        if path is None:
            continue
        if best_path is None or len(path) < len(best_path):
            best_path = path

    return best_path


def get_move_towards_target(
    world_state,
    agent_id: str,
    target_pos: Tuple[int, int],
    interaction_radius: int = 1,
) -> Optional[str]:
    """
    Get the next move action towards target using A* pathfinding.
    
    Strategy: Use A* to find optimal path. Return first step of path.
    If no path exists, return None to signal failure.
    
    Args:
        world_state: SymbolicWorldState instance
        agent_id: Agent ID string
        target_pos: (x, y) position of target object
        interaction_radius: Distance at which navigate can stop.
        
    Returns:
        Primitive action string (move_left, move_right, move_up, move_down) or None if no path
    """
    from macrafter.symbolic_world import SymbolicWorldState
    
    if not isinstance(world_state, SymbolicWorldState):
        return None
    
    agent_pos = get_agent_position(world_state, agent_id)
    if agent_pos is None:
        return None

    path = shortest_path_to_target_zone(
        world_state,
        agent_pos,
        target_pos,
        radius=interaction_radius,
    )
    
    if path is None or len(path) < 2:
        # No path exists or already within interaction radius.
        return None
    
    # Get next step in path
    next_pos = path[1]  # path[0] is current position
    
    # Calculate direction to next position
    dx = next_pos[0] - agent_pos[0]
    dy = next_pos[1] - agent_pos[1]
    
    # Return appropriate move command
    if dx > 0:
        return "move_right"
    elif dx < 0:
        return "move_left"
    elif dy > 0:
        return "move_down"
    elif dy < 0:
        return "move_up"
    
    # Shouldn't reach here, but return None if something is wrong
    return None
