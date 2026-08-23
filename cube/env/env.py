"""
Simplified Cooperative Block Push Environment.

A minimal implementation of the cooperative block-pushing environment.
Agents work together to push blocks to the goal area (rightmost column).
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from .cooperative_tasks import CooperativeTaskTracker


@dataclass
class Block:
    """A block in the environment."""
    id: int
    weight: int  # side length of the square block
    r: int  # top-left row
    c: int  # top-left column

    def cells(self) -> List[Tuple[int, int]]:
        """Get all cells occupied by this block."""
        return [(self.r + dr, self.c + dc) for dr in range(self.weight) for dc in range(self.weight)]

    def border_facing(self, direction: int) -> List[Tuple[int, int]]:
        """Get cells on the face being pushed (incoming side)."""
        if direction == 1:  # up -> bottom row
            return [(self.r + self.weight - 1, self.c + dc) for dc in range(self.weight)]
        if direction == 2:  # down -> top row
            return [(self.r, self.c + dc) for dc in range(self.weight)]
        if direction == 3:  # left -> right column
            return [(self.r + dr, self.c + self.weight - 1) for dr in range(self.weight)]
        if direction == 4:  # right -> left column
            return [(self.r + dr, self.c) for dr in range(self.weight)]
        return []

    def next_pos(self, direction: int) -> Tuple[int, int]:
        """Get next position if moved in direction."""
        if direction == 1:  # up
            return (self.r - 1, self.c)
        if direction == 2:  # down
            return (self.r + 1, self.c)
        if direction == 3:  # left
            return (self.r, self.c - 1)
        if direction == 4:  # right
            return (self.r, self.c + 1)
        return (self.r, self.c)


class CoopBlockPush(ParallelEnv):
    """
    Simplified cooperative block-pushing environment.
    
    Grid: K x K with goal strip on rightmost column (col = K-1)
    Blocks: squares of side length = weight
    Agents: push blocks by applying force. Block of weight W needs W simultaneous force.
    Goal: Push blocks to goal column to remove them.
    
    Observation: (K, K, 5) float32
      - channel 0: agents mask (1.0 at agent cells)
      - channel 1: blocks weight (value = weight)
      - channel 2: goal strip (1.0 in goal column)
      - channel 3: agent id (+1)
      - channel 4: block id (+1)
    
    Action: Discrete(5) - 0=stay, 1=up, 2=down, 3=left, 4=right
    
    Rewards: -step_cost per step, +deliver_reward*weight when block delivered
    """

    metadata = {
        "name": "cooperative_block_push_v0",
        "render_modes": ["rgb_array", "human", None],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        grid_size: int = 15,
        num_agents: int = 3,
        block_specs: Optional[Dict[int, int]] = None,
        max_steps: int = 500,
        render_mode: Optional[str] = None,
        step_cost: float = 0.01,
        deliver_reward: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize environment with simple defaults."""
        self.K = grid_size
        self._init_num_agents = num_agents
        self.block_specs = block_specs or {2: 1, 3: 1}  # Default: 1 block of weight 2, 1 block of weight 3
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.step_cost = step_cost
        self.deliver_reward = deliver_reward
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # PettingZoo API
        self.possible_agents = [f"agent_{i}" for i in range(self._init_num_agents)]
        self.agents = list(self.possible_agents)
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}

        # Spaces
        self.action_spaces = {agent: spaces.Discrete(5) for agent in self.possible_agents}
        max_weight = max(self.block_specs.keys()) if self.block_specs else 10
        obs_space = spaces.Box(
            low=0.0, high=float(max(10, max_weight + 1)),
            shape=(self.K, self.K, 5), dtype=np.float32
        )
        self.observation_spaces = {agent: obs_space for agent in self.possible_agents}

        # State
        self._agent_positions: Dict[str, Tuple[int, int]] = {}
        self._blocks: List[Block] = []
        self._occupied: np.ndarray = None  # 0=empty, 1=agent, 2=block, 3=goal
        self._step_count = 0
        self._team_score = 0.0
        self._score_events: List[Dict] = []
        self._last_actions: Dict[str, int] = {}
        self._last_step_events: Dict[str, List] = {}
        self._last_symbolic_action_intents: Dict[str, Dict] = {}
        
        # Task tracker (initialized after first reset when blocks exist)
        self.task_tracker: Optional[CooperativeTaskTracker] = None

    def _empty_grid(self) -> np.ndarray:
        """Create empty occupancy grid."""
        return np.zeros((self.K, self.K), dtype=np.int8)

    def _rebuild_occupied(self):
        """Rebuild occupancy grid from current state."""
        g = self._empty_grid()
        g[:, self.K - 1] = 3  # goal column
        for b in self._blocks:
            for r, c in b.cells():
                g[r, c] = 2
        for pos in self._agent_positions.values():
            r, c = pos
            g[r, c] = 1
        self._occupied = g

    def _place_entities(self, fixed_agent_positions: Optional[Dict[str, Tuple[int, int]]] = None):
        """
        Place agents and blocks randomly.
        
        Args:
            fixed_agent_positions: Optional dict mapping agent_id -> (row, col) for fixed positions
        """
        self._agent_positions.clear()
        self._blocks.clear()
        occ = np.zeros((self.K, self.K), dtype=bool)

        # Place blocks
        bid = 0
        for weight, count in self.block_specs.items():
            for _ in range(count):
                placed = False
                tries = 0
                while not placed and tries < 1000:
                    tries += 1
                    # Leave margin from walls and goal
                    r = self.rng.randint(1, self.K - weight - 2)
                    c = self.rng.randint(1, self.K - weight - 2)
                    
                    # Check if position is free
                    if all(not occ[r + dr, c + dc] 
                           for dr in range(weight) for dc in range(weight)):
                        # Mark as occupied
                        for dr in range(weight):
                            for dc in range(weight):
                                occ[r + dr, c + dc] = True
                        self._blocks.append(Block(bid, weight, r, c))
                        bid += 1
                        placed = True
                if not placed:
                    raise RuntimeError(f"Could not place block of weight {weight}")

        # Place agents
        if fixed_agent_positions:
            # Use fixed positions if provided
            for agent in self.possible_agents:
                if agent in fixed_agent_positions:
                    r, c = fixed_agent_positions[agent]
                    if 0 <= r < self.K and 0 <= c < self.K and not occ[r, c]:
                        occ[r, c] = True
                        self._agent_positions[agent] = (r, c)
                    else:
                        # Fallback to random if fixed position is invalid
                        rows = list(range(self.K))
                        self.rng.shuffle(rows)
                        r = rows[0]
                        c = 0
                        if not occ[r, c]:
                            occ[r, c] = True
                            self._agent_positions[agent] = (r, c)
                else:
                    # Random placement for agents not in fixed positions
                    rows = list(range(self.K))
                    self.rng.shuffle(rows)
                    r = rows[0]
                    c = 0
                    if not occ[r, c]:
                        occ[r, c] = True
                        self._agent_positions[agent] = (r, c)
        else:
            # Place agents on left wall (col=0) randomly
            rows = list(range(self.K))
            self.rng.shuffle(rows)
            for i, agent in enumerate(self.possible_agents):
                r = rows[i % len(rows)]
                c = 0
                if not occ[r, c]:
                    occ[r, c] = True
                    self._agent_positions[agent] = (r, c)

        self._rebuild_occupied()

    def _entities_info(self) -> dict:
        """Get info about entities."""
        return {
            "grid_size": self.K,
            "goal_column": self.K - 1,
            "team_score": self._team_score,
            "agents": [
                {
                    "name": agent,
                    "id": self.agent_name_mapping[agent],
                    "pos": tuple(self._agent_positions[agent]),
                }
                for agent in self.possible_agents
                if agent in self._agent_positions
            ],
            "blocks": [
                {
                    "id": b.id,
                    "weight": b.weight,
                    "pos": (b.r, b.c),
                }
                for b in self._blocks
            ],
        }

    def get_config_observation(self) -> str:
        """Return a compact, text description of the current CUBE task."""
        block_summary = ", ".join(
            f"{count} block(s) with weight {weight}"
            for weight, count in sorted(self.block_specs.items())
        )
        return (
            "CUBE task configuration:\n"
            f"- Grid: {self.K}x{self.K}; goal column is column {self.K - 1}.\n"
            f"- Blocks: {block_summary}.\n"
            "- Objective: maximize team delivery score before the episode ends.\n"
            "- Delivery score for one block equals its weight.\n"
            "- A weight W block requires W agents pushing the same face in the same step.\n"
            "- To move a block toward the goal, agents should stand on its left face and push right."
        )

    def _block_distance_to_goal(self, block: Block) -> int:
        """Cells the block's right edge must move to reach the goal column."""
        right_edge = block.c + block.weight - 1
        return max(0, (self.K - 1) - right_edge)

    def _blocking_blocks_to_goal(self, block: Block) -> List[int]:
        """Blocks currently occupying rows between this block and the goal."""
        rows = set(range(block.r, block.r + block.weight))
        right_edge_next = block.c + block.weight
        blockers = set()
        for other in self._blocks:
            if other.id == block.id:
                continue
            for row, col in other.cells():
                if row in rows and col >= right_edge_next:
                    blockers.add(other.id)
                    break
        return sorted(blockers)

    def _symbolic_view_for_agent(self, agent: str) -> str:
        """Return a stable symbolic view with IDs for LLM planning."""
        pos = self._agent_positions.get(agent)
        lines = [
            f"You: {agent} at row={pos[0]}, col={pos[1]}" if pos else f"You: {agent}",
            "Agents:",
        ]
        for other in sorted(self._agent_positions, key=lambda a: self.agent_name_mapping.get(a, 999)):
            row, col = self._agent_positions[other]
            marker = " (you)" if other == agent else ""
            lines.append(f"- {other}: row={row}, col={col}{marker}")

        if not self._blocks:
            lines.append("Blocks: none remaining.")
            return "\n".join(lines)

        lines.append("Blocks:")
        for block in sorted(self._blocks, key=lambda b: (self._block_distance_to_goal(b), b.weight, b.id)):
            cells = block.cells()
            face_left = [(block.r + dr, block.c - 1) for dr in range(block.weight)]
            valid_left = [
                (r, c)
                for r, c in face_left
                if 0 <= r < self.K and 0 <= c < self.K
            ]
            distance = self._block_distance_to_goal(block)
            blockers = self._blocking_blocks_to_goal(block)
            lines.append(
                f"- Block {block.id}: weight={block.weight}, required_agents={block.weight}, "
                f"top_left=({block.r},{block.c}), cells={cells}, "
                f"left_push_cells={valid_left}, distance_to_goal={distance}, "
                f"score_if_delivered={block.weight}, "
                f"path_clear_to_goal={not blockers}, blocking_blocks_ahead={blockers}"
            )
        return "\n".join(lines)

    def _info_for_agents(self) -> Dict[str, dict]:
        """Build per-agent info dictionaries without sharing mutable objects."""
        entities = self._entities_info()
        return {
            agent: {
                **entities,
                "symbolic_view": self._symbolic_view_for_agent(agent),
            }
            for agent in self.agents
        }

    def _obs(self) -> Dict[str, np.ndarray]:
        """Generate observations for all agents."""
        obs = np.zeros((self.K, self.K, 5), dtype=np.float32)

        # Goal strip
        obs[:, self.K - 1, 2] = 1.0

        # Blocks
        for b in self._blocks:
            for r, c in b.cells():
                obs[r, c, 1] = float(b.weight)
                obs[r, c, 4] = float(b.id + 1)

        # Agents
        for agent, (r, c) in self._agent_positions.items():
            obs[r, c, 0] = 1.0
            obs[r, c, 3] = float(self.agent_name_mapping[agent] + 1)

        return {agent: obs for agent in self.agents}

    @staticmethod
    def _dir_to_delta(direction: int) -> Tuple[int, int]:
        """Convert direction to (dr, dc)."""
        if direction == 1:  # up
            return (-1, 0)
        if direction == 2:  # down
            return (1, 0)
        if direction == 3:  # left
            return (0, -1)
        if direction == 4:  # right
            return (0, 1)
        return (0, 0)  # stay

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset environment.
        
        Args:
            seed: Random seed for reproducibility
            options: Optional dict with:
                - 'fixed_agent_positions': Dict mapping agent_id -> (row, col) for fixed positions
        """
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self._step_count = 0
        self._team_score = 0.0
        self._score_events = []
        self._last_actions = {}
        self._last_symbolic_action_intents = {}
        self._last_step_events = {
            "moved_blocks": [],
            "successful_pushes": [],
            "delivered_blocks": [],
            "failed_pushes": [],
        }
        
        # Extract fixed agent positions from options
        fixed_agent_positions = None
        if options and 'fixed_agent_positions' in options:
            fixed_agent_positions = options['fixed_agent_positions']
        
        self._place_entities(fixed_agent_positions=fixed_agent_positions)
        
        # Initialize/reset task tracker
        if self.task_tracker is None:
            self.task_tracker = CooperativeTaskTracker(self)
        else:
            self.task_tracker.reset()
        # Initial update at step 0
        self.task_tracker.update(0)
        
        obs = self._obs()
        infos = self._info_for_agents()
        return obs, infos

    def step(self, actions: Dict[str, int]):
        """Execute one step."""
        assert set(actions.keys()) == set(self.agents), "Actions must be provided for all live agents"
        self._step_count += 1
        self._last_actions = {agent: int(action) for agent, action in actions.items()}
        moved_blocks_this_step: set[int] = set()
        successful_pushes_this_step: List[Dict] = []
        delivered_blocks_this_step: List[Dict] = []
        failed_pushes_this_step: List[Dict] = []

        # Validate actions
        for a in self.agents:
            act = int(actions[a])
            if act < 0 or act > 4:
                actions[a] = 0

        pos_to_agent = {pos: agent for agent, pos in self._agent_positions.items()}
        moved_agents: set[str] = set()

        # Process block pushes per direction
        for direction in [1, 2, 3, 4]:  # up, down, left, right
            dr, dc = self._dir_to_delta(direction)
            moved_block_ids: set[int] = set()

            for b in list(self._blocks):
                if b.id in moved_block_ids:
                    continue

                # Get face being pushed
                face = b.border_facing(direction)

                # Get adjacent cells where pushers stand
                adj_cells = []
                for r, c in face:
                    rr, cc = r - dr, c - dc
                    if 0 <= rr < self.K and 0 <= cc < self.K:
                        adj_cells.append((rr, cc))

                # Count force from agents pushing in this direction
                total_force = 0
                pushing_agents = []
                for r, c in adj_cells:
                    if (r, c) in pos_to_agent:
                        agent = pos_to_agent[(r, c)]
                        if actions[agent] == direction:
                            total_force += 1
                            pushing_agents.append(agent)

                # Check if enough force to move block
                if total_force < b.weight:
                    if pushing_agents:
                        failed_pushes_this_step.append({
                            "block_id": b.id,
                            "direction": direction,
                            "agents": list(pushing_agents),
                            "force": total_force,
                            "required": b.weight,
                            "reason": "insufficient_force",
                        })
                    continue

                # Check if destination is free
                nr, nc = b.next_pos(direction)
                dest_cells = [(nr + dr, nc + dc) for dr in range(b.weight) for dc in range(b.weight)]
                current_block_cells = set(b.cells())
                
                can_move = True
                for rr, cc in dest_cells:
                    if not (0 <= rr < self.K and 0 <= cc < self.K):
                        can_move = False
                        break
                    if self._occupied[rr, cc] == 2 and (rr, cc) not in current_block_cells:  # Another block
                        can_move = False
                        break
                    if self._occupied[rr, cc] == 1:  # Agent in front
                        can_move = False
                        break

                if not can_move:
                    failed_pushes_this_step.append({
                        "block_id": b.id,
                        "direction": direction,
                        "agents": list(pushing_agents),
                        "force": total_force,
                        "required": b.weight,
                        "reason": "blocked_destination",
                    })
                    continue

                # Move block
                face_name = {
                    1: "down",
                    2: "up",
                    3: "right",
                    4: "left",
                }.get(direction)
                push_event = {
                    "block_id": b.id,
                    "direction": direction,
                    "face": face_name,
                    "task_id": f"block_{b.id}_{face_name}" if face_name else None,
                    "agents": list(pushing_agents),
                    "force": total_force,
                    "required": b.weight,
                }
                successful_pushes_this_step.append(push_event)
                b.r, b.c = nr, nc
                moved_block_ids.add(b.id)
                moved_blocks_this_step.add(b.id)
                self._rebuild_occupied()
                
                # Move pushing agents forward
                for agent in pushing_agents:
                    r, c = self._agent_positions[agent]
                    nr_a, nc_a = r + dr, c + dc
                    if 0 <= nr_a < self.K and 0 <= nc_a < self.K:
                        if self._occupied[nr_a, nc_a] == 0:  # Free cell
                            self._agent_positions[agent] = (nr_a, nc_a)
                            moved_agents.add(agent)

                # Update occupancy
                self._rebuild_occupied()

                # Check delivery
                if any(c == self.K - 1 for (_, c) in b.cells()):
                    value = self.deliver_reward * b.weight
                    event = {
                        "env_step": self._step_count,
                        "block_id": b.id,
                        "weight": b.weight,
                        "value": value,
                        "task_id": push_event["task_id"],
                        "face": face_name,
                        "agents": list(pushing_agents),
                    }
                    delivered_blocks_this_step.append(event)
                    self._team_score += value
                    self._score_events.append(event)
                    self._blocks.remove(b)
                    self._rebuild_occupied()

        # Move non-pushing agents
        for agent, pos in self._agent_positions.items():
            if agent in moved_agents:
                continue
            
            act = int(actions[agent])
            dr, dc = self._dir_to_delta(act)
            nr, nc = pos[0] + dr, pos[1] + dc
            
            if 0 <= nr < self.K and 0 <= nc < self.K:
                if self._occupied[nr, nc] == 0:  # Free cell
                    self._agent_positions[agent] = (nr, nc)
                elif self._occupied[nr, nc] == 1:  # Another agent - resolve by ID
                    other_agent = pos_to_agent.get((nr, nc))
                    if other_agent and self.agent_name_mapping[agent] < self.agent_name_mapping[other_agent]:
                        # This agent has smaller ID, can move
                        self._agent_positions[agent] = (nr, nc)
                        # Move other agent back if possible
                        or_, oc = self._agent_positions[other_agent]
                        if self._occupied[or_, oc] == 0:
                            self._agent_positions[other_agent] = (or_, oc)

        # Rebuild occupancy
        self._rebuild_occupied()

        # Compute rewards
        rewards = {a: -self.step_cost for a in self.agents}
        
        # Delivery reward (simplified: equal share across live agents).
        if delivered_blocks_this_step:
            bonus = sum(event["value"] for event in delivered_blocks_this_step)
            share = bonus / max(1, len(self.agents))
            for a in rewards:
                rewards[a] += share

        # Terminations
        terminated = {a: False for a in self.agents}
        if len(self._blocks) == 0:
            for a in self.agents:
                terminated[a] = True
        
        truncated = {a: False for a in self.agents}
        if self._step_count >= self.max_steps:
            for a in self.agents:
                truncated[a] = True

        self._last_step_events = {
            "moved_blocks": sorted(moved_blocks_this_step),
            "successful_pushes": successful_pushes_this_step,
            "delivered_blocks": delivered_blocks_this_step,
            "failed_pushes": failed_pushes_this_step,
        }

        # Info
        infos = self._info_for_agents()
        for agent_info in infos.values():
            agent_info.update({
                "last_actions": dict(self._last_actions),
                "last_symbolic_action_intents": dict(self._last_symbolic_action_intents),
                "last_step_events": self._last_step_events,
                "team_score": self._team_score,
            })

        # Update task tracker with actions
        if self.task_tracker is not None:
            self.task_tracker.update(self._step_count, actions)

        # Clear agents if done
        if any(terminated.values()) or any(truncated.values()):
            self.agents = []

        obs = self._obs() if self.agents else {}
        return obs, rewards, terminated, truncated, infos

    def get_team_score_breakdown(self, include_events: bool = False) -> Dict:
        """Return cumulative team score and optional delivery events."""
        data = {
            "score": self._team_score,
            "team_score": self._team_score,
            "blocks_delivered": len(self._score_events),
            "by_weight": {},
        }
        for event in self._score_events:
            key = f"weight_{event['weight']}"
            data["by_weight"][key] = data["by_weight"].get(key, 0.0) + event["value"]
        if include_events:
            data["events"] = list(self._score_events)
        return data
    
    def render(self, return_image=False):
        """Simple render."""
        if self.render_mode == "rgb_array" or return_image:
            img = np.zeros((self.K, self.K, 3), dtype=np.uint8)
            img[:] = 240  # Light gray background
            
            # Goal column
            img[:, self.K - 1, :] = [157, 212, 177]  # Green
            
            # Blocks
            for b in self._blocks:
                for r, c in b.cells():
                    img[r, c, :] = [212, 196, 188]  # Brown
            
            # Agents
            for r, c in self._agent_positions.values():
                img[r, c, :] = [152, 193, 217]  # Blue
            
            return img
        return None

    def close(self):
        """Clean up."""
        pass

    def state(self) -> np.ndarray:
        """Get full state."""
        return self._occupied.copy()
