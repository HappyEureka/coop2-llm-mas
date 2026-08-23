"""ID management for CUBE world entities.

Manages unique IDs for:
- Grid cells (each position gets a unique ID)
- Blocks (persistent across position changes)
- Agents (persistent across position changes)
"""

from typing import Dict, Optional, Tuple, Any


class IDManager:
    """
    Manages unique IDs for world entities (grid cells, blocks, agents).
    
    IDs are short (max 6 digits) and persistent across position changes.
    Each grid cell has a unique ID based on position.
    Blocks and agents have stable IDs that persist when they move.
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
        
        # Grid cell IDs: (row, col) -> id
        self._cell_ids: Dict[Tuple[int, int], int] = {}
        
        # Block IDs: block.id -> stable_id
        self._block_ids: Dict[int, int] = {}
        
        # Agent IDs: agent_name -> stable_id  
        self._agent_ids: Dict[str, int] = {}
        
        # Reverse mappings
        self._id_to_cell: Dict[int, Tuple[int, int]] = {}
        self._id_to_block: Dict[int, int] = {}
        self._id_to_agent: Dict[int, str] = {}
        
        # Type tracking
        self._id_to_type: Dict[int, str] = {}
    
    def reset(self):
        """Reset all ID mappings."""
        self._next_id = 1
        self._cell_ids.clear()
        self._block_ids.clear()
        self._agent_ids.clear()
        self._id_to_cell.clear()
        self._id_to_block.clear()
        self._id_to_agent.clear()
        self._id_to_type.clear()
    
    def _get_next_id(self) -> int:
        """Get next available ID."""
        id_val = self._next_id
        self._next_id = (self._next_id % self.max_id) + 1
        return id_val
    
    def get_cell_id(self, pos: Tuple[int, int]) -> int:
        """
        Get or create ID for a grid cell at a position.
        
        Grid cells are static, so we use position-based IDs.
        
        Args:
            pos: (row, col) position
            
        Returns:
            Stable ID for this grid cell
        """
        if pos not in self._cell_ids:
            id_val = self._get_next_id()
            self._cell_ids[pos] = id_val
            self._id_to_cell[id_val] = pos
            self._id_to_type[id_val] = "cell"
        return self._cell_ids[pos]
    
    def get_block_id(self, block_env_id: int) -> int:
        """
        Get or create stable ID for a block.
        
        Args:
            block_env_id: The block's environment ID (block.id)
            
        Returns:
            Stable ID for this block
        """
        if block_env_id not in self._block_ids:
            id_val = self._get_next_id()
            self._block_ids[block_env_id] = id_val
            self._id_to_block[id_val] = block_env_id
            self._id_to_type[id_val] = "block"
        return self._block_ids[block_env_id]
    
    def get_agent_id(self, agent_name: str) -> int:
        """
        Get or create stable ID for an agent.
        
        Args:
            agent_name: Agent identifier (e.g., 'agent_0', 'agent_1')
            
        Returns:
            Stable ID for this agent
        """
        if agent_name not in self._agent_ids:
            id_val = self._get_next_id()
            self._agent_ids[agent_name] = id_val
            self._id_to_agent[id_val] = agent_name
            self._id_to_type[id_val] = f"agent_{agent_name}"
        return self._agent_ids[agent_name]
    
    def get_id_info(self, stable_id: int) -> Optional[Dict[str, Any]]:
        """
        Get information about an ID.
        
        Args:
            stable_id: The stable ID to look up
            
        Returns:
            Dict with 'type' and 'category' (cell/block/agent), or None if not found
        """
        if stable_id not in self._id_to_type:
            return None
        
        type_str = self._id_to_type[stable_id]
        
        if type_str == "cell":
            return {
                'id': stable_id,
                'category': 'cell',
                'type': 'cell',
                'position': self._id_to_cell.get(stable_id),
            }
        elif type_str == "block":
            return {
                'id': stable_id,
                'category': 'block',
                'type': 'block',
                'env_id': self._id_to_block.get(stable_id),
            }
        elif type_str.startswith("agent_"):
            return {
                'id': stable_id,
                'category': 'agent',
                'type': type_str,
                'name': self._id_to_agent.get(stable_id),
            }
        
        return None
    
    def remove_block(self, block_env_id: int):
        """
        Remove a block from tracking (when it's delivered to goal).
        
        Args:
            block_env_id: The block's environment ID
        """
        if block_env_id in self._block_ids:
            stable_id = self._block_ids[block_env_id]
            del self._block_ids[block_env_id]
            del self._id_to_block[stable_id]
            # Keep type info for historical reference
    
    def remove_agent(self, agent_name: str):
        """
        Remove an agent from tracking.
        
        Args:
            agent_name: The agent name to remove
        """
        if agent_name in self._agent_ids:
            stable_id = self._agent_ids[agent_name]
            del self._agent_ids[agent_name]
            del self._id_to_agent[stable_id]
            # Keep type info for historical reference
    
    def get_all_ids(self) -> Dict[str, list]:
        """
        Get all currently tracked IDs.
        
        Returns:
            Dict with 'cells', 'blocks', and 'agents' lists
        """
        return {
            'cells': list(self._cell_ids.values()),
            'blocks': list(self._block_ids.values()),
            'agents': list(self._agent_ids.values()),
        }
