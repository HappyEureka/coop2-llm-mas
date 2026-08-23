"""Symbolic world state for CUBE environment.

Maintains symbolic representation of the world with stable IDs.
Each grid cell, block, and agent has a unique persistent ID.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from .id_manager import IDManager


class SymbolicWorldState:
    """
    Maintains symbolic representation of the CUBE world with stable IDs.
    
    Core data structure: symbolic_matrix
    - Matrix where each cell contains list of (entity_type, id, name) tuples
    - entity_type: 'cell', 'block', or 'agent'
    - Enables efficient slicing for agent views
    - Stable IDs persist across position changes
    
    Entity types in CUBE:
    - cell: Grid position (floor, goal)
    - block: Pushable block with weight
    - agent: Player agent
    """
    
    def __init__(self, env):
        """
        Initialize symbolic world state.
        
        Args:
            env: The CUBE CoopBlockPush environment instance
        """
        self.env = env
        self.id_manager = IDManager()
        self.grid_size = env.K
        
        # Core: Symbolic matrix - each cell is list of (type, id, name) tuples
        # Shape: (K, K), dtype=object
        self.symbolic_matrix: Optional[np.ndarray] = None
        
        # Quick lookup dictionaries
        self.cell_positions: Dict[int, Tuple[int, int]] = {}  # cell_id -> (row, col)
        self.block_positions: Dict[int, Tuple[int, int]] = {}  # block_id -> (row, col)
        self.agent_positions: Dict[int, Tuple[int, int]] = {}  # agent_id -> (row, col)
        
        # Properties
        self.cell_properties: Dict[int, Dict[str, Any]] = {}  # cell_id -> properties
        self.block_properties: Dict[int, Dict[str, Any]] = {}  # block_id -> properties
        self.agent_properties: Dict[int, Dict[str, Any]] = {}  # agent_id -> properties
        
        # Goal column
        self.goal_column = env.K - 1
    
    def update(self):
        """
        Update the world state snapshot from the environment.
        """
        K = self.grid_size
        
        # Initialize symbolic matrix with empty lists
        self.symbolic_matrix = np.empty((K, K), dtype=object)
        for r in range(K):
            for c in range(K):
                self.symbolic_matrix[r, c] = []
        
        # Step 1: Fill in grid cells (base layer)
        self.cell_positions.clear()
        self.cell_properties.clear()
        
        for r in range(K):
            for c in range(K):
                cell_id = self.id_manager.get_cell_id((r, c))
                is_goal = (c == self.goal_column)
                cell_type = "goal" if is_goal else "floor"
                
                self.cell_positions[cell_id] = (r, c)
                self.cell_properties[cell_id] = {
                    'type': cell_type,
                    'position': (r, c),
                    'is_goal': is_goal,
                }
                
                self.symbolic_matrix[r, c].append(('cell', cell_id, cell_type))
        
        # Step 2: Update blocks
        self.block_positions.clear()
        self.block_properties.clear()
        
        for block in self.env._blocks:
            stable_id = self.id_manager.get_block_id(block.id)
            pos = (block.r, block.c)
            
            self.block_positions[stable_id] = pos
            self.block_properties[stable_id] = {
                'type': 'block',
                'env_id': block.id,
                'position': pos,
                'weight': block.weight,
                'cells': block.cells(),
            }
            
            # Add block to all cells it occupies
            for cell_r, cell_c in block.cells():
                self.symbolic_matrix[cell_r, cell_c].append(
                    ('block', stable_id, f"b{block.id}_w{block.weight}")
                )
        
        # Step 3: Update agents
        self.agent_positions.clear()
        self.agent_properties.clear()
        
        for agent_name, agent_pos in self.env._agent_positions.items():
            stable_id = self.id_manager.get_agent_id(agent_name)
            pos = tuple(agent_pos)
            
            self.agent_positions[stable_id] = pos
            self.agent_properties[stable_id] = {
                'type': 'agent',
                'name': agent_name,
                'env_id': self.env.agent_name_mapping[agent_name],
                'position': pos,
            }
            
            # Add agent to symbolic matrix
            r, c = pos
            self.symbolic_matrix[r, c].append(('agent', stable_id, agent_name))
    
    def get_symbolic_slice(
        self,
        center: Tuple[int, int],
        view_size: Tuple[int, int] = (9, 9)
    ) -> np.ndarray:
        """
        Get a sliced view of the symbolic matrix centered on a position.
        
        Args:
            center: (row, col) center position
            view_size: (height, width) of view window
            
        Returns:
            np.ndarray of shape (vh, vw) containing lists of (type, id, name) tuples
        """
        vh, vw = view_size
        offset_r, offset_c = vh // 2, vw // 2
        
        # Create output matrix
        sliced_view = np.empty((vh, vw), dtype=object)
        for j in range(vh):
            for i in range(vw):
                sliced_view[j, i] = []
        
        # Fill in the slice
        for j in range(vh):
            for i in range(vw):
                world_r = center[0] - offset_r + j
                world_c = center[1] - offset_c + i
                
                if 0 <= world_r < self.grid_size and 0 <= world_c < self.grid_size:
                    sliced_view[j, i] = self.symbolic_matrix[world_r, world_c].copy()
        
        return sliced_view
    
    # Convenience methods
    
    def get_agent_position(self, agent_name: str) -> Optional[Tuple[int, int]]:
        """Get position of an agent by name."""
        stable_id = self.id_manager._agent_ids.get(agent_name)
        if stable_id is not None:
            return self.agent_positions.get(stable_id)
        return None
    
    def get_block_position(self, block_env_id: int) -> Optional[Tuple[int, int]]:
        """Get top-left position of a block by environment ID."""
        stable_id = self.id_manager._block_ids.get(block_env_id)
        if stable_id is not None:
            return self.block_positions.get(stable_id)
        return None
    
    def get_block_by_env_id(self, block_env_id: int) -> Optional[Dict[str, Any]]:
        """Get block properties by environment ID."""
        stable_id = self.id_manager._block_ids.get(block_env_id)
        if stable_id is not None:
            return self.block_properties.get(stable_id)
        return None
    
    def get_all_blocks(self) -> List[Dict[str, Any]]:
        """Get info about all blocks."""
        return list(self.block_properties.values())
    
    def get_all_agents(self) -> List[Dict[str, Any]]:
        """Get info about all agents."""
        return list(self.agent_properties.values())
    
    def is_goal_cell(self, row: int, col: int) -> bool:
        """Check if a cell is in the goal area."""
        return col == self.goal_column
    
    def is_blocked(self, row: int, col: int) -> bool:
        """Check if a cell is blocked (out of bounds or occupied by block)."""
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            return True
        return self.env._occupied[row, col] == 2  # 2 = block
    
    def is_agent_at(self, row: int, col: int) -> bool:
        """Check if an agent is at the given position."""
        return self.env._occupied[row, col] == 1  # 1 = agent
    
    def get_cell_at(self, row: int, col: int) -> Optional[Dict[str, Any]]:
        """Get cell properties at position."""
        cell_id = self.id_manager._cell_ids.get((row, col))
        if cell_id is not None:
            return self.cell_properties.get(cell_id)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "grid_size": self.grid_size,
            "goal_column": self.goal_column,
            "agents": self.get_all_agents(),
            "blocks": self.get_all_blocks(),
        }
    
    def __repr__(self) -> str:
        return f"SymbolicWorldState(grid={self.grid_size}x{self.grid_size}, agents={len(self.agent_positions)}, blocks={len(self.block_positions)})"
