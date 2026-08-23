"""ID management for world entities."""

import numpy as np
from typing import Dict, Optional, Tuple, Any


class IDManager:
    """
    Manages unique IDs for world entities (materials, objects, agents).
    
    IDs are short (max 6 digits) and persistent across position changes.
    """
    
    def __init__(self, max_id: int = 999999):
        """
        Initialize ID manager.
        
        Args:
            max_id: Maximum ID value (default 999999 for 6-digit IDs)
        """
        self.max_id = max_id
        
        # Counter for next available ID
        self._next_id = 1
        
        # Mappings: python_id -> assigned_id
        self._object_ids: Dict[int, int] = {}  # Maps id(obj) -> stable_id
        self._agent_ids: Dict[int, int] = {}   # Maps id(agent) -> stable_id
        
        # Reverse mappings: assigned_id -> python_id
        self._id_to_object: Dict[int, int] = {}
        self._id_to_agent: Dict[int, int] = {}
        
        # Material position-based IDs: (x, y) -> id
        self._material_ids: Dict[Tuple[int, int], int] = {}
        
        # Type tracking for better semantics
        self._id_to_type: Dict[int, str] = {}  # Maps stable_id -> type_name
    
    def reset(self):
        """Reset all ID mappings."""
        self._next_id = 1
        self._object_ids.clear()
        self._agent_ids.clear()
        self._id_to_object.clear()
        self._id_to_agent.clear()
        self._material_ids.clear()
        self._id_to_type.clear()
    
    def _get_next_id(self) -> int:
        """Get next available ID."""
        id_val = self._next_id
        self._next_id = (self._next_id % self.max_id) + 1
        return id_val
    
    def get_material_id(self, pos: Tuple[int, int], material_type: str) -> int:
        """
        Get or create ID for a material at a position.
        
        Materials are static, so we use position-based IDs.
        
        Args:
            pos: (x, y) position
            material_type: Type of material (e.g., 'grass', 'stone')
            
        Returns:
            Stable ID for this material tile
        """
        if pos not in self._material_ids:
            id_val = self._get_next_id()
            self._material_ids[pos] = id_val
            self._id_to_type[id_val] = f"material_{material_type}"
        return self._material_ids[pos]
    
    def get_object_id(self, obj: Any, obj_type: str) -> int:
        """
        Get or create ID for a dynamic object.
        
        Objects can move, so we use python id(obj) for tracking.
        
        Args:
            obj: The object instance
            obj_type: Type name (e.g., 'cow', 'zombie', 'plant')
            
        Returns:
            Stable ID for this object
        """
        python_id = id(obj)
        
        if python_id not in self._object_ids:
            id_val = self._get_next_id()
            self._object_ids[python_id] = id_val
            self._id_to_object[id_val] = python_id
            self._id_to_type[id_val] = obj_type
        
        return self._object_ids[python_id]
    
    def get_agent_id(self, agent: Any, agent_name: str) -> int:
        """
        Get or create ID for an agent/player.
        
        Args:
            agent: The agent/player instance
            agent_name: Agent identifier (e.g., '0', '1', 'agent_0')
            
        Returns:
            Stable ID for this agent
        """
        python_id = id(agent)
        
        if python_id not in self._agent_ids:
            id_val = self._get_next_id()
            self._agent_ids[python_id] = id_val
            self._id_to_agent[id_val] = python_id
            self._id_to_type[id_val] = f"agent_{agent_name}"
        
        return self._agent_ids[python_id]
    
    def get_id_info(self, stable_id: int) -> Optional[Dict[str, Any]]:
        """
        Get information about an ID.
        
        Args:
            stable_id: The stable ID to look up
            
        Returns:
            Dict with 'type' and 'category' (material/object/agent), or None if not found
        """
        if stable_id not in self._id_to_type:
            return None
        
        type_str = self._id_to_type[stable_id]
        
        if type_str.startswith("material_"):
            category = "material"
            type_name = type_str[9:]  # Remove "material_" prefix
        elif type_str.startswith("agent_"):
            category = "agent"
            type_name = type_str[6:]  # Remove "agent_" prefix
        else:
            category = "object"
            type_name = type_str
        
        return {
            'id': stable_id,
            'category': category,
            'type': type_name,
        }
    
    def remove_object(self, obj: Any):
        """
        Remove an object from tracking (when it's destroyed).
        
        Args:
            obj: The object instance to remove
        """
        python_id = id(obj)
        if python_id in self._object_ids:
            stable_id = self._object_ids[python_id]
            del self._object_ids[python_id]
            del self._id_to_object[stable_id]
            # Keep type info for historical reference
    
    def remove_agent(self, agent: Any):
        """
        Remove an agent from tracking (when it dies).
        
        Args:
            agent: The agent instance to remove
        """
        python_id = id(agent)
        if python_id in self._agent_ids:
            stable_id = self._agent_ids[python_id]
            del self._agent_ids[python_id]
            del self._id_to_agent[stable_id]
            # Keep type info for historical reference
    
    def get_all_ids(self) -> Dict[str, list]:
        """
        Get all currently tracked IDs.
        
        Returns:
            Dict with 'materials', 'objects', and 'agents' lists
        """
        material_ids = list(self._material_ids.values())
        object_ids = list(self._object_ids.values())
        agent_ids = list(self._agent_ids.values())
        
        return {
            'materials': material_ids,
            'objects': object_ids,
            'agents': agent_ids,
        }
