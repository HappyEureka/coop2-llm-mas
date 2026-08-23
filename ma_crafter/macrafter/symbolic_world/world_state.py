"""Symbolic world state with ID tracking."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# Import macrafter components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from macrafter import engine, objects

from .id_manager import IDManager


class SymbolicWorldState:
    """
    Maintains symbolic representation of the world with stable IDs.
    
    Core data structure: symbolic_matrix
    - Matrix where each cell contains list of (entity_type, id, name) tuples
    - entity_type: 'material', 'object', or 'agent'
    - Enables efficient slicing for agent views
    - Stable IDs persist across position changes
    """
    
    def __init__(self, world: engine.World):
        """
        Initialize symbolic world state.
        
        Args:
            world: The macrafter World instance to track
        """
        self.world = world
        self.id_manager = IDManager()
        
        # Core: Symbolic matrix - each cell is list of (type, id, name) tuples
        self.symbolic_matrix: Optional[np.ndarray] = None  # Shape: (area_x, area_y), dtype=object
        
        # Quick lookup dictionaries
        self.object_positions: Dict[int, Tuple[int, int]] = {}  # obj_id -> (x, y)
        self.agent_positions: Dict[int, Tuple[int, int]] = {}  # agent_id -> (x, y)
        self.object_properties: Dict[int, Dict[str, Any]] = {}  # obj_id -> properties
        self.agent_properties: Dict[int, Dict[str, Any]] = {}  # agent_id -> properties

        # Type-local aliases for navigable utilities. These make prompts stable
        # and readable: table 1, table 2, furnace 1, etc. The underlying stable
        # IDs are still retained for backwards compatibility.
        self.aliasable_types = {"table", "furnace"}
        self._type_local_ids: Dict[Tuple[str, int], int] = {}
        self._next_type_local_id: Dict[str, int] = {}
        self.type_local_positions: Dict[str, Dict[int, Tuple[int, int]]] = {}
        self.type_local_stable_ids: Dict[str, Dict[int, int]] = {}
        self.stable_id_types: Dict[int, str] = {}

    def reset_type_local_ids(self):
        """Reset readable per-type aliases at episode boundaries."""
        self._type_local_ids.clear()
        self._next_type_local_id.clear()
        self.type_local_positions.clear()
        self.type_local_stable_ids.clear()
    
    def update(self, agents: List[objects.Player]):
        """
        Update the world state snapshot.
        
        Args:
            agents: List of agent/player objects
        """
        # Initialize symbolic matrix with empty lists
        self.symbolic_matrix = np.empty(self.world.area, dtype=object)
        for x in range(self.world.area[0]):
            for y in range(self.world.area[1]):
                self.symbolic_matrix[x, y] = []
        
        self.type_local_positions.clear()
        self.type_local_stable_ids.clear()
        self.stable_id_types.clear()

        # Step 1: Fill in materials (base layer)
        for x in range(self.world.area[0]):
            for y in range(self.world.area[1]):
                material, obj = self.world[x, y]
                if material:
                    mat_id = self.id_manager.get_material_id((x, y), material)
                    self._register_entity_alias(material, mat_id, (x, y))
                    self.symbolic_matrix[x, y].append(('material', mat_id, material))
        
        # Step 2: Update dynamic objects
        self.object_positions.clear()
        self.object_properties.clear()
        
        for obj in self.world.objects:
            if isinstance(obj, objects.Player):
                continue  # Handle agents separately
            
            obj_type = type(obj).__name__.lower()
            stable_id = self.id_manager.get_object_id(obj, obj_type)
            pos = tuple(obj.pos)
            self._register_entity_alias(obj_type, stable_id, pos)
            
            # Track position
            self.object_positions[stable_id] = pos
            
            # Track properties
            properties = {'type': obj_type, 'position': pos}
            if hasattr(obj, 'health'):
                properties['health'] = obj.health
            if isinstance(obj, objects.Plant):
                properties['grown'] = obj.grown
                properties['ripe'] = obj.ripe
            
            self.object_properties[stable_id] = properties
            
            # Add to symbolic matrix
            x, y = pos
            self.symbolic_matrix[x, y].append(('object', stable_id, obj_type))
        
        # Step 3: Update agents
        self.agent_positions.clear()
        self.agent_properties.clear()
        
        for i, agent in enumerate(agents):
            if agent.removed:
                continue
            
            agent_name = str(i)
            stable_id = self.id_manager.get_agent_id(agent, agent_name)
            pos = tuple(agent.pos)
            self.stable_id_types[int(stable_id)] = f"agent_{agent_name}"
            
            # Track position
            self.agent_positions[stable_id] = pos
            
            # Track properties
            properties = {
                'type': 'player',
                'name': agent_name,
                'position': pos,
                'facing': tuple(agent.facing),
                'health': agent.health,
                'sleeping': agent.sleeping,
                'inventory': agent.inventory.copy(),
                'achievements': agent.achievements.copy(),
            }
            
            self.agent_properties[stable_id] = properties
            
            # Add to symbolic matrix
            x, y = pos
            self.symbolic_matrix[x, y].append(('agent', stable_id, agent_name))

    @staticmethod
    def _normalize_entity_name(name: Any) -> str:
        return str(name).lower().replace(" ", "_")

    def _register_entity_alias(
        self,
        entity_name: Any,
        stable_id: int,
        position: Tuple[int, int],
    ) -> Optional[int]:
        """Register current entity metadata and optional type-local alias."""
        normalized = self._normalize_entity_name(entity_name)
        stable_id = int(stable_id)
        position = (int(position[0]), int(position[1]))
        self.stable_id_types[stable_id] = normalized

        if normalized not in self.aliasable_types:
            return None

        key = (normalized, stable_id)
        if key not in self._type_local_ids:
            next_id = self._next_type_local_id.get(normalized, 1)
            self._type_local_ids[key] = next_id
            self._next_type_local_id[normalized] = next_id + 1

        local_id = self._type_local_ids[key]
        self.type_local_positions.setdefault(normalized, {})[local_id] = position
        self.type_local_stable_ids.setdefault(normalized, {})[local_id] = stable_id
        return local_id

    def get_type_local_id(self, entity_name: Any, stable_id: int) -> Optional[int]:
        """Return the readable local ID for an entity, if it has one."""
        normalized = self._normalize_entity_name(entity_name)
        return self._type_local_ids.get((normalized, int(stable_id)))

    def resolve_entity_position(
        self,
        entity_name: Optional[Any],
        item_id: int,
    ) -> Optional[Tuple[int, int]]:
        """
        Resolve an LLM navigation reference to a world position.

        For aliasable utilities, item_id can be a type-local alias, such as
        table 1. Stable IDs from the original symbolic matrix still work.
        """
        try:
            item_id = int(item_id)
        except (TypeError, ValueError):
            return None

        normalized = (
            self._normalize_entity_name(entity_name)
            if entity_name is not None
            else None
        )

        if normalized in self.aliasable_types:
            position = self.type_local_positions.get(normalized, {}).get(item_id)
            if position is not None:
                return position

        return self.find_stable_id_position(item_id, normalized)

    def find_stable_id_position(
        self,
        stable_id: int,
        entity_name: Optional[str] = None,
    ) -> Optional[Tuple[int, int]]:
        """Find a stable ID, optionally requiring the current entity type."""
        try:
            stable_id = int(stable_id)
        except (TypeError, ValueError):
            return None

        normalized = (
            self._normalize_entity_name(entity_name)
            if entity_name is not None
            else None
        )

        if stable_id in self.object_positions:
            props = self.object_properties.get(stable_id, {})
            obj_type = self._normalize_entity_name(props.get("type", ""))
            if normalized is None or obj_type == normalized:
                return self.object_positions[stable_id]

        if stable_id in self.agent_positions:
            props = self.agent_properties.get(stable_id, {})
            agent_names = {
                "agent",
                "player",
                self._normalize_entity_name(props.get("name", "")),
                f"agent_{self._normalize_entity_name(props.get('name', ''))}",
            }
            if normalized is None or normalized in agent_names:
                return self.agent_positions[stable_id]

        if self.symbolic_matrix is not None:
            shape = self.symbolic_matrix.shape
            for idx, entities in enumerate(self.symbolic_matrix.flat):
                for entity_type, entity_id, entity_current_name in entities:
                    if int(entity_id) != stable_id:
                        continue
                    current_name = self._normalize_entity_name(entity_current_name)
                    if normalized is not None and current_name != normalized:
                        continue
                    x = idx // shape[1]
                    y = idx % shape[1]
                    return (int(x), int(y))

        return None
    
    def get_symbolic_slice(
        self, 
        center: Tuple[int, int], 
        view_size: Tuple[int, int] = (9, 9)
    ) -> np.ndarray:
        """
        Get a sliced view of the symbolic matrix centered on a position.
        
        Args:
            center: (x, y) center position
            view_size: (width, height) of view window
            
        Returns:
            np.ndarray of shape (view_w, view_h) containing lists of (type, id, name) tuples
        """
        vw, vh = view_size
        offset_x, offset_y = vw // 2, vh // 2
        
        # Create output matrix in [y, x] format (row, col) to match crafter convention
        sliced_view = np.empty((vh, vw), dtype=object)
        for j in range(vh):
            for i in range(vw):
                sliced_view[j, i] = []
        
        # Fill in the slice
        for j in range(vh):
            for i in range(vw):
                world_x = center[0] - offset_x + i
                world_y = center[1] - offset_y + j
                
                if 0 <= world_x < self.world.area[0] and 0 <= world_y < self.world.area[1]:
                    sliced_view[j, i] = self.symbolic_matrix[world_x, world_y].copy()
        
        return sliced_view
