"""
Cooperative Multi-Agent Crafter Engine

Extends the base macrafter environment to support cooperative collection:
- Multiple agents can collect resources together
- Resources can require multiple agents to collect
- Resources are distributed to all participating agents
- No facing requirement - just need to be within distance
- Integrated task tracking for all 4 dimensions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
import yaml
import os
import time

from macrafter.env import Env
from macrafter import constants, objects
from macrafter.cooperative_tasks import (
    CooperativeTaskTracker,
    ToolRequirement,
    satisfies_tool_requirement,
)
from macrafter.symbolic_world import SymbolicWorldState, get_agent_symbolic_view_text


RESOURCE_SCORE_KEYS = {
    'wood': 'tree',
    'stone': 'stone',
    'coal': 'coal',
    'iron': 'iron',
    'diamond': 'diamond',
}

TOOL_TIERS = {
    'wood_pickaxe': 1,
    'stone_pickaxe': 2,
    'iron_pickaxe': 3,
}

PROMPT_RESOURCE_REQUIREMENTS = {'tree', 'stone', 'coal', 'iron', 'diamond'}

DEFAULT_TEAM_SCORE_CONFIG = {
    'enabled': True,
    'terminate_on_diamond': False,
    'wall_clock_limit_seconds': 120,
    'score_lifetime_collections': True,
    'score_shared_resources_per_agent': True,
    'scale_by_required_agents': True,
    'tool_tier_bonus': 0.5,
    'base_values': {
        'wood': 1.0,
        'stone': 10.0,
        'coal': 25.0,
        'iron': 25.0,
        'diamond': 80.0,
    },
}


class CooperativeEnv(Env):
    """
    Cooperative Multi-Agent Crafter Environment.

    Extends the base Env to support:
    - Cooperative collection requiring multiple agents
    - Resource distribution to all participating agents
    - Distance-based collection (no facing requirement for coop resources)
    """

    def __init__(self, *args, coop_config_path: Optional[str] = None, **kwargs):
        """
        Initialize the cooperative environment.

        Args:
            *args: Arguments passed to base Env
            coop_config_path: "paper" or a path to a custom YAML file
            **kwargs: Keyword arguments passed to base Env
        """
        super().__init__(*args, **kwargs)

        # Resolve config path first
        resolved_config_path = self._resolve_config_path(coop_config_path)

        # Load cooperative configuration
        self.coop_config = self._load_coop_config(resolved_config_path)
        self.coop_enabled = self.coop_config.get('cooperative_collection', {}).get('enabled', True)
        self.distance_threshold = self.coop_config.get('cooperative_collection', {}).get('distance_threshold', 2)
        self.resource_requirements = self.coop_config.get('cooperative_collection', {}).get('resources', {})
        self.team_score_config = self._load_team_score_config()
        self.team_score_enabled = self.team_score_config.get('enabled', True)
        self.terminate_on_diamond = self.team_score_config.get('terminate_on_diamond', False)
        self.score_shared_resources_per_agent = self.team_score_config.get('score_shared_resources_per_agent', True)
        self._team_score_start_time: Optional[float] = None

        # Collection info for this step (for debugging/tracking)
        self._coop_collections: List[Dict] = []
        self._blocked_collections: Dict[str, str] = {}
        self._share_results: List[Dict] = []
        self._place_results: List[Dict] = []
        self._team_score_events: List[Dict] = []

        # Cooperative task tracker - tracks 4 dimensions for each resource (use resolved path)
        self.task_tracker = CooperativeTaskTracker(resolved_config_path)
        self._current_collect_agents: Set[str] = set()  # Track who issues collect each step

        # Symbolic world state tracker - provides symbolic observations
        self.symbolic_world = SymbolicWorldState(self._world)

    def _load_team_score_config(self) -> Dict[str, Any]:
        config = DEFAULT_TEAM_SCORE_CONFIG.copy()
        config['base_values'] = DEFAULT_TEAM_SCORE_CONFIG['base_values'].copy()

        configured = self.coop_config.get('team_score') or {}
        for key, value in configured.items():
            if key == 'base_values':
                config['base_values'].update(value or {})
            else:
                config[key] = value
        return config

    def set_team_score_time_limit(self, seconds: Optional[float]):
        """Set the wall-clock budget used by the team-score task."""
        self.team_score_config['wall_clock_limit_seconds'] = seconds

    def _team_score_elapsed_seconds(self) -> float:
        if self._team_score_start_time is None:
            return 0.0
        return max(0.0, time.monotonic() - self._team_score_start_time)

    def _team_score_remaining_seconds(self) -> Optional[float]:
        limit = self.team_score_config.get('wall_clock_limit_seconds')
        if limit is None or float(limit) <= 0:
            return None
        return max(0.0, float(limit) - self._team_score_elapsed_seconds())

    def is_team_score_time_limit_reached(self) -> bool:
        remaining = self._team_score_remaining_seconds()
        return remaining is not None and remaining <= 0.0

    def _resolve_config_path(self, config_path: Optional[str] = None) -> str:
        """
        Resolve config path to an absolute path.

        Args:
            config_path: "paper" or a path to a custom YAML file

        Returns:
            Resolved absolute path to config file
        """
        # Use the paper task configuration unless a custom file is supplied.
        if config_path in {None, "paper"}:
            return os.path.join(
                os.path.dirname(__file__),
                'paper_config.yaml'
            )
        return config_path

    def get_config_observation(self) -> str:
        """
        Format the cooperative configuration as a readable observation string.

        Returns:
            String describing the current cooperative requirements
        """
        lines = []
        lines.append("COOPERATIVE CONFIGURATION:")
        lines.append(f"  Distance threshold: {self.distance_threshold}")
        lines.append("  Resource requirements:")

        team_size = len(getattr(self, "possible_agents", []) or [])
        impossible_resources = []
        for resource, reqs in self.resource_requirements.items():
            if self.team_score_enabled and resource not in PROMPT_RESOURCE_REQUIREMENTS:
                continue
            agents_needed = reqs.get('required_agents', 1)
            tool_needed = reqs.get('required_tool', None)
            tool_str = tool_needed if tool_needed else "none"
            lines.append(f"    - {resource}: {agents_needed} agent(s), tool: {tool_str}")
            if team_size and int(agents_needed) > team_size:
                impossible_resources.append(resource)

        if impossible_resources:
            lines.append(
                "  Currently impossible with this team size: "
                + ", ".join(sorted(impossible_resources))
            )

        if self.team_score_enabled:
            max_steps = getattr(self, '_length', None)
            current_step = self._step or 0
            remaining_steps = max(0, int(max_steps) - int(current_step)) if max_steps is not None else "unknown"
            wall_clock_limit = self.team_score_config.get('wall_clock_limit_seconds')
            remaining_seconds = self._team_score_remaining_seconds()
            lines.append("")
            lines.append("TEAM SCORE OBJECTIVE:")
            lines.append("  Maximize total team resource score before the step/time budget ends, not diamond-only success.")
            lines.append("  The cooperative configuration above is the single source of distance, participation, and tool requirements.")
            lines.append("  Diamond is valuable but does not end the episode in this scoring task.")
            if self.team_score_config.get('score_lifetime_collections', True):
                lines.append("  Score is based on resources collected over the episode; resources still count after being spent on tools or crafting.")
            if max_steps is not None and wall_clock_limit:
                minutes = float(wall_clock_limit) / 60.0
                lines.append(
                    f"  Deadline: score as much as possible before step {max_steps} "
                    f"or {minutes:g} minutes of wall-clock time, whichever comes first."
                )
                lines.append(
                    f"  Current budget: step {current_step}/{max_steps}, "
                    f"{remaining_steps} step(s) left, about {int(remaining_seconds or 0)} second(s) left."
                )
            scored_items = [
                (item_name, self.get_item_team_score(item_name))
                for item_name in self.team_score_config.get('base_values', {})
            ]
            scored_items = [(item_name, score) for item_name, score in scored_items if score > 0]
            lines.append("  Resource score per gained unit:")
            for item_name, score in scored_items:
                lines.append(f"    - {item_name}: {score:g} point(s)")

        return "\n".join(lines)

    def _load_coop_config(self, config_path: str) -> Dict:
        """
        Load cooperative configuration from yaml file.

        Args:
            config_path: Resolved path to yaml file
        """
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default config - enable cooperative with standard requirements
            return {
                'cooperative_collection': {
                    'enabled': True,
                    'distance_threshold': 2,
                    'resources': {
                        'tree': {'required_agents': 1, 'required_tool': None},
                        'stone': {'required_agents': 1, 'required_tool': 'wood_pickaxe'},
                        'coal': {'required_agents': 1, 'required_tool': 'wood_pickaxe'},
                        'iron': {'required_agents': 1, 'required_tool': 'stone_pickaxe'},
                        'diamond': {'required_agents': 2, 'required_tool': 'iron_pickaxe'},
                    }
                },
                'team_score': DEFAULT_TEAM_SCORE_CONFIG,
            }

    def _tool_tier(self, required_tool: ToolRequirement) -> int:
        tools = required_tool if isinstance(required_tool, (list, tuple)) else [required_tool]
        return max((TOOL_TIERS.get(tool, 0) for tool in tools if tool), default=0)

    def get_item_team_score(self, item_name: str) -> float:
        """Return the configured team score for one gained unit of an item."""
        if not self.team_score_enabled:
            return 0.0

        base_values = self.team_score_config.get('base_values', {})
        base_value = float(base_values.get(item_name, 0.0))
        if base_value <= 0:
            return 0.0

        resource_key = RESOURCE_SCORE_KEYS.get(item_name, item_name)
        required_agents, required_tool, _required_tool_mode = self.get_resource_requirements(resource_key)

        cooperation_multiplier = required_agents if self.team_score_config.get('scale_by_required_agents', True) else 1
        tool_tier_bonus = float(self.team_score_config.get('tool_tier_bonus', 0.5))
        tool_multiplier = 1.0 + self._tool_tier(required_tool) * tool_tier_bonus
        return round(base_value * max(1, cooperation_multiplier) * tool_multiplier, 3)

    @staticmethod
    def _normalize_env_resource_key(name: Any) -> str:
        """Normalize raw environment object keys without score/category aliases."""
        return str(name).lower().replace(" ", "_")

    @staticmethod
    def _format_tool_requirement(required_tool: ToolRequirement, required_tool_mode: str = 'all') -> str:
        if not required_tool:
            return "none"
        if isinstance(required_tool, (list, tuple)):
            tools = [str(tool) for tool in required_tool if tool]
            if not tools:
                return "none"
            joiner = " or " if required_tool_mode == "any" else " + "
            return joiner.join(tools)
        return str(required_tool)

    def _received_items_for_resource(self, resource_type: str) -> Dict[str, int]:
        if resource_type == 'cow':
            return {'food': 6}
        collect_info = constants.collect.get(resource_type) or {}
        return dict(collect_info.get('receive') or {})

    def _resource_target_score(self, resource_type: str) -> float:
        total = 0.0
        for item_name, amount in self._received_items_for_resource(resource_type).items():
            total += self.get_item_team_score(item_name) * int(amount)
        return round(total, 3)

    def _path_distance_to_zone(
        self,
        agent_pos: Tuple[int, int],
        target_pos: Tuple[int, int],
        radius: int,
    ) -> Optional[int]:
        """Return path length to any walkable cell within radius of target_pos."""
        if self._compute_distance(agent_pos, target_pos) <= radius:
            return 0

        try:
            from cognitive.action.navigation import astar_pathfind, is_walkable
        except ImportError:
            return self._compute_distance(agent_pos, target_pos)

        best_distance: Optional[int] = None
        radius = max(0, int(radius))
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) + abs(dy) > radius:
                    continue
                candidate = (int(target_pos[0]) + dx, int(target_pos[1]) + dy)
                if not (0 <= candidate[0] < self._world.area[0] and 0 <= candidate[1] < self._world.area[1]):
                    continue
                if candidate != agent_pos and not is_walkable(self.symbolic_world, candidate):
                    continue
                path = astar_pathfind(self.symbolic_world, agent_pos, candidate)
                if path is None:
                    continue
                distance = max(0, len(path) - 1)
                if best_distance is None or distance < best_distance:
                    best_distance = distance

        return best_distance

    def _agent_index_from_id(self, agent_id: Any) -> Optional[int]:
        try:
            return int(str(agent_id).split('_')[-1])
        except (TypeError, ValueError):
            return None

    def _visible_symbolic_positions(
        self,
        agent_pos: Tuple[int, int],
        view_size: Tuple[int, int] = (9, 8),
    ) -> Set[Tuple[int, int]]:
        """Return world cells displayed in get_agent_symbolic_view_text()."""
        vw, vh = view_size
        offset_x, offset_y = vw // 2, vh // 2
        positions: Set[Tuple[int, int]] = set()
        for j in range(1, vh):
            for i in range(vw):
                x = int(agent_pos[0]) - offset_x + i
                y = int(agent_pos[1]) - offset_y + j
                if 0 <= x < self._world.area[0] and 0 <= y < self._world.area[1]:
                    positions.add((x, y))
        return positions

    def get_reachable_navigation_targets(self, agent_id: Any, limit: int = 8) -> List[Dict[str, Any]]:
        """
        Return current visible, physically reachable scoring-resource targets.

        The list is recomputed from the live symbolic world, so collected or stale
        resource IDs naturally disappear from the prompt. It only considers cells
        currently displayed in the agent's symbolic view, avoiding map leakage.
        """
        limit = max(0, int(limit))
        if limit == 0:
            return []

        symbolic_matrix = getattr(self.symbolic_world, "symbolic_matrix", None)
        agent_idx = self._agent_index_from_id(agent_id)
        if symbolic_matrix is None or agent_idx is None or agent_idx >= len(self._players):
            return []

        player = self._players[agent_idx]
        if player.removed:
            return []

        agent_pos = (int(player.pos[0]), int(player.pos[1]))
        candidates: List[Dict[str, Any]] = []
        for x, y in sorted(self._visible_symbolic_positions(agent_pos)):
            for entity_type, entity_id, entity_name in symbolic_matrix[x, y]:
                resource_type = self._normalize_env_resource_key(entity_name)
                is_survival_target = entity_type == 'object' and resource_type == 'cow'
                if entity_type != 'material' and not is_survival_target:
                    continue
                if entity_type == 'material' and resource_type not in constants.collect:
                    continue

                score_value = self._resource_target_score(resource_type)
                if score_value <= 0 and not is_survival_target:
                    continue

                required_agents, required_tool, required_tool_mode = self.get_resource_requirements(resource_type)
                tool_ready = satisfies_tool_requirement(
                    player.inventory,
                    required_tool,
                    required_tool_mode,
                )
                rough_distance = max(
                    0,
                    self._compute_distance(agent_pos, (int(x), int(y))) - int(self.distance_threshold),
                )
                candidates.append({
                    'object_type': resource_type,
                    'item_id': int(entity_id),
                    'position': [int(x), int(y)],
                    'distance': int(rough_distance),
                    'score': score_value,
                    'required_agents': int(required_agents),
                    'required_tool': required_tool,
                    'required_tool_mode': required_tool_mode,
                    'tool_ready': bool(tool_ready),
                    'source': 'current_symbolic_view',
                    'purpose': 'survival' if is_survival_target else 'score',
                    'received': self._received_items_for_resource(resource_type),
                })

        candidates.sort(key=lambda item: (
            0 if item['tool_ready'] else 1,
            int(item['distance']),
            str(item['object_type']),
            int(item['item_id']),
            -float(item['score']),
        ))

        # Pathfinding against every resource every step is unnecessarily costly.
        # Check the most promising rough candidates and stop when we have enough.
        targets: List[Dict[str, Any]] = []
        candidate_limit = max(limit * 6, 36)
        for candidate in candidates[:candidate_limit]:
            target_pos = tuple(candidate['position'])
            distance = self._path_distance_to_zone(
                agent_pos=agent_pos,
                target_pos=(int(target_pos[0]), int(target_pos[1])),
                radius=self.distance_threshold,
            )
            if distance is None:
                continue
            candidate = candidate.copy()
            candidate['distance'] = int(distance)
            targets.append(candidate)
            if len(targets) >= limit:
                break

        targets.sort(key=lambda item: (
            0 if item['tool_ready'] else 1,
            int(item['distance']),
            str(item['object_type']),
            int(item['item_id']),
            -float(item['score']),
        ))
        return targets[:limit]

    def get_reachable_utility_targets(self, agent_id: Any, limit: int = 3) -> List[Dict[str, Any]]:
        """Return reachable table/furnace local IDs that are useful for crafting."""
        limit = max(0, int(limit))
        if limit == 0:
            return []

        agent_idx = self._agent_index_from_id(agent_id)
        if agent_idx is None or agent_idx >= len(self._players):
            return []
        player = self._players[agent_idx]
        if player.removed:
            return []

        agent_pos = (int(player.pos[0]), int(player.pos[1]))
        visible_positions = self._visible_symbolic_positions(agent_pos)
        utilities: List[Dict[str, Any]] = []
        for object_type in ('table', 'furnace'):
            local_positions = getattr(self.symbolic_world, "type_local_positions", {}).get(object_type, {})
            stable_ids = getattr(self.symbolic_world, "type_local_stable_ids", {}).get(object_type, {})
            for local_id, position in local_positions.items():
                if (int(position[0]), int(position[1])) not in visible_positions:
                    continue
                distance = self._path_distance_to_zone(
                    agent_pos=agent_pos,
                    target_pos=(int(position[0]), int(position[1])),
                    radius=1,
                )
                if distance is None:
                    continue
                utilities.append({
                    'object_type': object_type,
                    'item_id': int(local_id),
                    'stable_id': int(stable_ids.get(local_id, -1)),
                    'position': [int(position[0]), int(position[1])],
                    'distance': int(distance),
                })

        utilities.sort(key=lambda item: (
            int(item['distance']),
            str(item['object_type']),
            int(item['item_id']),
        ))
        return utilities[:limit]

    def format_reachable_targets_for_prompt(
        self,
        reachable_targets: List[Dict[str, Any]],
        reachable_utilities: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        lines = [
            "CURRENT REACHABLE TARGETS:",
            "  These IDs come only from your current SYMBOLIC VIEW. For resource navigation, treat unlisted resource IDs as stale, collected, hidden, or unreachable.",
        ]

        if reachable_targets:
            score_targets = [
                target for target in reachable_targets
                if target.get('purpose', 'score') == 'score'
            ]
            survival_targets = [
                target for target in reachable_targets
                if target.get('purpose') == 'survival'
            ]
            if score_targets:
                lines.append("  Score resources:")
            else:
                lines.append("  Score resources: none reachable now.")
            for target in score_targets:
                tool_text = self._format_tool_requirement(
                    target.get('required_tool'),
                    target.get('required_tool_mode', 'all'),
                )
                readiness = "ready" if target.get('tool_ready') else "needs_tool"
                lines.append(
                    "    - "
                    f"{target['object_type']} item_id={target['item_id']} "
                    f"dist={target['distance']} score={target['score']:g} "
                    f"requires={target['required_agents']} agent(s), tool={tool_text} "
                    f"({readiness}); navigate object_type={target['object_type']} item_id={target['item_id']}"
                )
            if survival_targets:
                lines.append("  Survival targets:")
            else:
                lines.append("  Survival targets: none visible now.")
            for target in survival_targets:
                received = target.get('received') or {}
                restores = ", ".join(f"{item}+{amount}" for item, amount in received.items()) or "food"
                lines.append(
                    "    - "
                    f"{target['object_type']} item_id={target['item_id']} "
                    f"dist={target['distance']} restores={restores} "
                    f"(no score); navigate object_type={target['object_type']} item_id={target['item_id']}"
                )
        else:
            lines.append("  Score resources: none reachable now.")
            lines.append("  Survival targets: none visible now.")

        if reachable_utilities:
            lines.append("  Utilities:")
            for target in reachable_utilities:
                lines.append(
                    "    - "
                    f"{target['object_type']} item_id={target['item_id']} "
                    f"dist={target['distance']}; navigate object_type={target['object_type']} item_id={target['item_id']}"
                )

        return "\n".join(lines)

    def _build_target_context_for_agent(self, agent_id: Any) -> Dict[str, Any]:
        reachable_targets = self.get_reachable_navigation_targets(agent_id)
        reachable_utilities = self.get_reachable_utility_targets(agent_id)
        return {
            'reachable_targets': reachable_targets,
            'reachable_utilities': reachable_utilities,
            'target_hints': self.format_reachable_targets_for_prompt(
                reachable_targets,
                reachable_utilities,
            ),
        }

    def _add_symbolic_info(self, info: Dict[str, Any], agent_id: Any):
        """Attach symbolic world, local view, and target hygiene hints."""
        agent_idx = self._agent_index_from_id(agent_id)
        if agent_idx is None:
            return
        info['symbolic_world_state'] = self.symbolic_world
        info['symbolic_view'] = get_agent_symbolic_view_text(self.symbolic_world, agent_idx)
        info.update(self._build_target_context_for_agent(agent_id))

    def _record_team_score_event(
        self,
        pos: Tuple[int, int],
        resource_type: str,
        participating_agents: List[str],
        received: Dict[str, int],
    ):
        if not self.team_score_enabled:
            return

        participant_multiplier = len(participating_agents) if self.score_shared_resources_per_agent else 1
        item_scores = {}
        scored_amounts = {}
        total_score = 0.0

        for item_name, amount in received.items():
            score_per_unit = self.get_item_team_score(item_name)
            if score_per_unit <= 0:
                continue
            scored_amount = int(amount) * participant_multiplier
            item_scores[item_name] = score_per_unit
            scored_amounts[item_name] = scored_amount
            total_score += score_per_unit * scored_amount

        if not scored_amounts:
            return

        self._team_score_events.append({
            'env_step': self._step,
            'position': [int(pos[0]), int(pos[1])],
            'resource_type': resource_type,
            'participating_agents': list(participating_agents),
            'received': dict(received),
            'scored_amounts': scored_amounts,
            'item_scores': item_scores,
            'score': round(total_score, 3),
        })

    def get_team_score_breakdown(self, include_events: bool = True) -> Dict[str, Any]:
        resource_counts: Dict[str, int] = {}
        resource_scores: Dict[str, float] = {}
        total_score = 0.0

        for event in self._team_score_events:
            total_score += float(event.get('score', 0.0))
            for item_name, amount in event.get('scored_amounts', {}).items():
                resource_counts[item_name] = resource_counts.get(item_name, 0) + int(amount)
            for item_name, score in event.get('item_scores', {}).items():
                resource_scores[item_name] = float(score)

        breakdown = {
            'enabled': self.team_score_enabled,
            'total_score': round(total_score, 3),
            'resource_counts': resource_counts,
            'resource_scores': resource_scores,
            'terminate_on_diamond': self.terminate_on_diamond,
            'score_shared_resources_per_agent': self.score_shared_resources_per_agent,
            'score_lifetime_collections': self.team_score_config.get('score_lifetime_collections', True),
            'max_episode_steps': getattr(self, '_length', None),
            'current_step': self._step,
            'wall_clock_limit_seconds': self.team_score_config.get('wall_clock_limit_seconds'),
            'elapsed_wall_clock_seconds': round(self._team_score_elapsed_seconds(), 3),
            'remaining_wall_clock_seconds': (
                None if self._team_score_remaining_seconds() is None
                else round(self._team_score_remaining_seconds(), 3)
            ),
            'scoring_config': self.team_score_config,
        }
        if include_events:
            breakdown['events'] = self._team_score_events
        return breakdown

    def save_team_score_log(self, output_path: str):
        import json
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.get_team_score_breakdown(include_events=True), f, indent=2)

    def reset(self, seed=None, options=None):
        """Reset the environment, task tracker, and symbolic world."""
        result = super().reset(seed=seed, options=options)

        # Reset task tracker
        self.task_tracker.reset()
        self._current_collect_agents = set()
        self._coop_collections = []
        self._blocked_collections = {}
        self._share_results = []
        self._place_results = []
        self._team_score_events = []
        self._team_score_start_time = time.monotonic()
        if hasattr(self.symbolic_world, "reset_type_local_ids"):
            self.symbolic_world.reset_type_local_ids()

        # Initialize previous inventories for capability tracking (baseline)
        for i, player in enumerate(self._players):
            if not player.removed:
                agent_id = str(i)
                self.task_tracker.previous_inventories[agent_id] = player.inventory.copy()

        # Update symbolic world
        self.symbolic_world.update(self._players)

        # Record step 0 task states
        self._compute_task_states(set())  # No collect actions at reset
        self.task_tracker.finalize_step()

        # Add symbolic information to info for each agent
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            for agent_id in self.possible_agents:
                if agent_id not in info:
                    info[agent_id] = {}
                self._add_symbolic_info(info[agent_id], agent_id)
            return obs, info

        return result

    def get_resource_requirements(self, resource_type: str) -> Tuple[int, ToolRequirement, str]:
        """Get (required_agents, required_tool, required_tool_mode) for a resource type."""
        resource_type = resource_type.lower()
        if resource_type in self.resource_requirements:
            req = self.resource_requirements[resource_type]
            return (
                req.get('required_agents', 1),
                req.get('required_tool'),
                req.get('required_tool_mode', 'all'),
            )
        return (1, None, 'all')  # Default: 1 agent, no tool

    def _compute_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Compute Manhattan distance to match cooperative collection execution."""
        return abs(int(pos1[0]) - int(pos2[0])) + abs(int(pos1[1]) - int(pos2[1]))

    def _compute_task_states(
        self,
        collect_agent_ids: Set[str],
        collect_attempts: Optional[Dict[str, Set[str]]] = None,
    ):
        """
        Compute cooperative task states for all collectable resources.

        This uses the CooperativeTaskTracker to track 4 dimensions:
        - Spatial: agents within distance
        - Temporal: valid collect actions issued
        - Dependency: tool requirements
        - Participation: agents meeting all criteria
        """
        # Import SymbolicWorldState lazily to avoid circular imports
        from macrafter.symbolic_world import SymbolicWorldState

        # Create a temporary world state for the tracker
        world_state = SymbolicWorldState(self._world)
        world_state.update(self._players)

        # Gather agent information
        agent_positions: Dict[str, tuple] = {}
        agent_facings: Dict[str, tuple] = {}
        agent_inventories: Dict[str, Dict[str, int]] = {}

        for i, player in enumerate(self._players):
            if player.removed:
                continue
            agent_id = str(i)
            agent_positions[agent_id] = tuple(player.pos)
            agent_facings[agent_id] = tuple(player.facing)
            agent_inventories[agent_id] = player.inventory.copy()

        # Compute task states
        self.task_tracker.compute_task_states(
            env_step=self._step,
            world_state=world_state,
            agent_positions=agent_positions,
            agent_facings=agent_facings,
            agent_inventories=agent_inventories,
            collect_actions=collect_agent_ids,
            collect_attempts=collect_attempts,
        )

        # Track capability changes (resources/tools gained or lost)
        self.task_tracker.track_capability_changes(agent_inventories)

    def _collect_attempts_by_task(
        self,
        collect_requests: Optional[List[Dict]],
        collect_agent_ids: Set[str],
    ) -> Dict[str, Set[str]]:
        """Map targeted symbolic collect requests to resource/task ids."""
        attempts: Dict[str, Set[str]] = {}
        for request in collect_requests or []:
            if not isinstance(request, dict):
                continue
            raw_agent_id = request.get('agent_id')
            if raw_agent_id is None:
                raw_agent_id = request.get('collector_idx')
            if raw_agent_id is None:
                continue
            agent_id = str(raw_agent_id)
            if agent_id not in collect_agent_ids:
                continue

            target_id = None
            for key in ('task_id', 'resource_id', 'target_id', 'item_id'):
                value = request.get(key)
                if value is not None and str(value) != "":
                    target_id = value
                    break
            if target_id is None:
                continue
            attempts.setdefault(str(target_id), set()).add(agent_id)
        return attempts

    def step(
        self,
        actions: Dict[str, int],
        share_requests: Optional[List[Dict]] = None,
        place_requests: Optional[List[Dict]] = None,
        collect_requests: Optional[List[Dict]] = None,
    ):
        """
        Step the environment with cooperative collection support.

        Overrides base step to:
        1. Execute share requests (inventory transfers)
        2. Identify all collect actions
        3. Group agents by nearby resources
        4. Check cooperative requirements
        5. Execute collection and distribute resources

        Args:
            actions: Dict mapping agent_id to action value
            share_requests: Optional list of share requests, each with:
                - sharer_idx: int - player index of sharer
                - recipient_idx: int - player index of recipient
                - resource_type: str - resource to share
                - quantity: int - amount to share
            place_requests: Optional symbolic place requests. When provided,
                placement can use any valid adjacent tile instead of only the
                player's current facing direction.
            collect_requests: Optional symbolic collect requests carrying the
                intended target/task id for temporal process tracking.
        """
        # Increment step counter at the beginning
        self._step += 1

        self._update_time()

        # Reset collection tracking
        self._coop_collections = []
        self._blocked_collections = {}
        self._share_results = []
        self._place_results = []
        before_inventories = {
            agent_id: self._players[int(agent_id)].inventory.copy()
            for agent_id in self.agents
            if not self._players[int(agent_id)].removed
        }
        before_achievements = {
            agent_id: self._players[int(agent_id)].achievements.copy()
            for agent_id in self.agents
            if not self._players[int(agent_id)].removed
        }

        # Execute share requests FIRST (before any other actions)
        if share_requests:
            self._execute_shares(share_requests)

        if place_requests:
            self._execute_symbolic_places(place_requests)

        # Separate collect actions from others
        collect_agents = []  # List of (agent_id, player_idx) for agents doing collect
        other_actions = {}   # agent_id -> action for non-collect actions
        collect_agent_ids: Set[str] = set()  # For task tracking
        action_names: Dict[str, str] = {}

        for agent_id in self.agents:
            action = actions[agent_id]
            action_name = constants.actions[action] if action < len(constants.actions) else 'noop'
            action_names[agent_id] = action_name

            if action_name == 'do' and self.coop_enabled:
                # This is a collect action - handle cooperatively
                collect_agents.append((agent_id, int(agent_id)))
                collect_agent_ids.add(agent_id)
                # Cooperative collection is handled before Player.update().
                # Clear the previous primitive action so the object update pass
                # does not replay the agent's last craft/place/move command.
                self._players[int(agent_id)].action = 'noop'
            elif (
                place_requests
                and action_name.startswith('place_')
                and any(str(req.get('placer_idx')) == agent_id for req in place_requests)
            ):
                # Symbolic placement has already been applied without requiring
                # the current facing direction. Still log the action as place_*,
                # but execute noop for the primitive player update.
                other_actions[agent_id] = constants.actions.index('noop')
            else:
                other_actions[agent_id] = action

        collect_attempts = self._collect_attempts_by_task(
            collect_requests=collect_requests,
            collect_agent_ids=collect_agent_ids,
        )

        # Process cooperative collections first
        if collect_agents:
            self._process_cooperative_collections(collect_agents)

        # Process other actions normally
        player_rewards = {}
        players_to_remove = []

        for agent_id in self.agents:
            if agent_id in other_actions:
                curr_reward, is_alive = self.step_one_player(other_actions[agent_id], int(agent_id))
            else:
                # Collect agents - reward is based on cooperative collection results
                curr_reward = self._get_coop_reward(agent_id)
                is_alive = not self._players[int(agent_id)].removed

            player_rewards[agent_id] = curr_reward
            if not is_alive:
                players_to_remove.append(agent_id)

        return_terminated = {
            agent_id: (
                self.terminate_on_diamond
                and self._players[int(agent_id)].achievements['collect_diamond'] > 0
            )
            for agent_id in self.agents
        }
        wall_clock_expired = self.is_team_score_time_limit_reached()
        return_truncated = {agent_id: wall_clock_expired for agent_id in self.agents}

        if len(players_to_remove) > 0:
            for agent_id in players_to_remove:
                self.agents.remove(agent_id)
                return_truncated[agent_id] = True

        # Update all objects
        for obj in self._world.objects:
            obj.update()

        # Balance chunks periodically
        if self._step % 10 == 0:
            for chunk, objs in self._world.chunks.items():
                self._balance_chunk(chunk, objs)

        action_outcomes = self._build_action_outcomes(
            action_names=action_names,
            before_inventories=before_inventories,
            before_achievements=before_achievements,
        )

        self.render_all()
        obs = self._obs()

        # Update symbolic world after step
        self.symbolic_world.update(self._players)
        self._annotate_place_results_with_symbolic_ids()

        # Compute task states AFTER all actions (captures post-action state)
        self._compute_task_states(collect_agent_ids, collect_attempts=collect_attempts)

        # Finalize task tracking for this step
        task_summary = self.task_tracker.finalize_step()

        # Get task states for return_info
        task_states = self.task_tracker.get_all_task_states()

        return_obs = {agent_id: obs[agent_id] for agent_id in self.agents + players_to_remove}
        return_reward = {agent_id: player_rewards[agent_id] for agent_id in self.agents + players_to_remove}
        return_info = {agent_id: {
            'coop_collections': self._coop_collections,
            'blocked_collection': self._blocked_collections.get(agent_id),
            'task_states': task_states,
            'task_summary': task_summary,
            'share_results': self._share_results,
            'place_results': self._place_results,
            'action_outcome': action_outcomes.get(agent_id),
            'team_score': self.get_team_score_breakdown(include_events=False),
        } for agent_id in self.agents + players_to_remove}

        for agent_id, agent_info in return_info.items():
            self._add_symbolic_info(agent_info, agent_id)

        if self.n_players == 1:
            agent_id = self.possible_agents[0]
            return return_obs[agent_id], return_reward[agent_id], return_terminated[agent_id], \
                   return_truncated[agent_id], return_info[agent_id]
        return return_obs, return_reward, return_terminated, return_truncated, return_info

    def _build_action_outcomes(
        self,
        action_names: Dict[str, str],
        before_inventories: Dict[str, Dict[str, int]],
        before_achievements: Dict[str, Dict[str, int]],
    ) -> Dict[str, Dict[str, Any]]:
        """Build environment-backed outcomes for the symbolic action logger."""
        outcomes: Dict[str, Dict[str, Any]] = {}

        for agent_id, action_name in action_names.items():
            player = self._players[int(agent_id)]
            before_inv = before_inventories.get(agent_id, {})
            before_ach = before_achievements.get(agent_id, {})
            after_inv = player.inventory.copy()
            after_ach = player.achievements.copy()

            inventory_delta = {
                item: after_inv.get(item, 0) - before_inv.get(item, 0)
                for item in set(before_inv) | set(after_inv)
                if after_inv.get(item, 0) != before_inv.get(item, 0)
            }
            achievement_delta = {
                name: after_ach.get(name, 0) - before_ach.get(name, 0)
                for name in set(before_ach) | set(after_ach)
                if after_ach.get(name, 0) != before_ach.get(name, 0)
            }

            outcome = {
                'agent_id': agent_id,
                'primitive_action': action_name,
                'status': 'success',
                'reason': None,
                'effects': {
                    'inventory_delta': inventory_delta,
                    'achievement_delta': achievement_delta,
                }
            }

            if action_name == 'do' and self.coop_enabled:
                agent_collections = [
                    collection for collection in self._coop_collections
                    if agent_id in collection.get('participating_agents', [])
                ]
                if agent_collections:
                    outcome['effects']['collections'] = agent_collections
                elif agent_id in self._blocked_collections:
                    outcome['status'] = 'failed'
                    outcome['reason'] = self._blocked_collections[agent_id]
                else:
                    outcome['status'] = 'failed'
                    outcome['reason'] = 'No collectable resource in range'

            elif action_name == 'share':
                share_results = [
                    result for result in self._share_results
                    if str(result.get('request', {}).get('sharer_idx')) == agent_id
                ]
                outcome['effects']['share_results'] = share_results
                if not share_results:
                    outcome['status'] = 'failed'
                    outcome['reason'] = 'No share request executed'
                elif not any(result.get('success') for result in share_results):
                    outcome['status'] = 'failed'
                    outcome['reason'] = share_results[0].get('reason', 'Share failed')

            elif action_name.startswith('make_') or action_name.startswith('place_'):
                if achievement_delta.get(action_name, 0) <= 0:
                    outcome['status'] = 'failed'
                    outcome['reason'] = 'Requirements not met or target invalid'
                if action_name.startswith('place_'):
                    place_results = [
                        result for result in self._place_results
                        if str(result.get('request', {}).get('placer_idx')) == agent_id
                    ]
                    outcome['effects']['place_results'] = place_results
                    if place_results:
                        if any(result.get('success') for result in place_results):
                            outcome['status'] = 'success'
                            outcome['reason'] = None
                        else:
                            outcome['status'] = 'failed'
                            outcome['reason'] = place_results[0].get('reason', 'Place failed')

            outcomes[agent_id] = outcome

        return outcomes

    def _execute_symbolic_places(self, place_requests: List[Dict]):
        """
        Execute symbolic placement without depending on facing direction.

        The action remains a generic place(item) action. This helper only
        chooses a valid adjacent target cell for PettingZoo-style symbolic
        control, because LLM agents do not reliably manage low-level facing.
        """
        for request in place_requests:
            try:
                placer_idx = int(request.get('placer_idx'))
            except (TypeError, ValueError):
                placer_idx = None
            object_type = str(request.get('object_type', '')).lower().replace(' ', '_')
            result = {'request': request}

            if placer_idx is None or placer_idx < 0 or placer_idx >= len(self._players):
                result['success'] = False
                result['reason'] = f'Invalid placer index: {placer_idx}'
                self._place_results.append(result)
                continue
            if object_type not in constants.place:
                result['success'] = False
                result['reason'] = f'Unknown place target: {object_type}'
                self._place_results.append(result)
                continue

            player = self._players[int(placer_idx)]
            if player.removed:
                result['success'] = False
                result['reason'] = 'Player is removed'
                self._place_results.append(result)
                continue

            info = constants.place[object_type]
            if any(player.inventory.get(item, 0) < amount for item, amount in info.get('uses', {}).items()):
                result['success'] = False
                result['reason'] = 'Missing place resources'
                self._place_results.append(result)
                continue

            target = self._find_symbolic_place_target(player, info)
            if target is None:
                result['success'] = False
                result['reason'] = 'No valid adjacent place target'
                self._place_results.append(result)
                continue

            for item, amount in info.get('uses', {}).items():
                player.inventory[item] -= amount
            if info['type'] == 'material':
                self._world[target] = object_type
            elif info['type'] == 'object':
                cls = {
                    'plant': objects.Plant,
                }.get(object_type)
                if cls is None:
                    result['success'] = False
                    result['reason'] = f'Unsupported place object: {object_type}'
                    self._place_results.append(result)
                    continue
                self._world.add(cls(self._world, target))

            player.achievements[f'place_{object_type}'] += 1
            result.update({
                'success': True,
                'position': [int(target[0]), int(target[1])],
                'object_type': object_type,
            })
            self._place_results.append(result)

    def _annotate_place_results_with_symbolic_ids(self):
        """Add stable/local navigation references to successful place results."""
        symbolic_matrix = getattr(self.symbolic_world, "symbolic_matrix", None)
        if symbolic_matrix is None:
            return

        for result in self._place_results:
            if not result.get('success'):
                continue
            object_type = str(result.get('object_type', '')).lower().replace(' ', '_')
            position = result.get('position')
            if not object_type or not position or len(position) != 2:
                continue

            x, y = int(position[0]), int(position[1])
            if not (0 <= x < symbolic_matrix.shape[0] and 0 <= y < symbolic_matrix.shape[1]):
                continue

            for _entity_type, stable_id, entity_name in symbolic_matrix[x, y]:
                if str(entity_name).lower().replace(' ', '_') != object_type:
                    continue
                stable_id = int(stable_id)
                result['stable_id'] = stable_id
                local_id = self.symbolic_world.get_type_local_id(entity_name, stable_id)
                if local_id is not None:
                    local_id = int(local_id)
                    result['local_id'] = local_id
                    result['navigation_reference'] = {
                        'object_type': object_type,
                        'item_id': local_id,
                    }
                break

    def _find_symbolic_place_target(self, player, place_info: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """Find a deterministic valid adjacent tile for symbolic placement."""
        directions = [
            tuple(player.facing),
            (0, 1),
            (1, 0),
            (-1, 0),
            (0, -1),
        ]
        seen = set()
        for dx, dy in directions:
            if (dx, dy) in seen:
                continue
            seen.add((dx, dy))
            target = (int(player.pos[0] + dx), int(player.pos[1] + dy))
            material, obj = self._world[target]
            if obj is not None:
                continue
            if material in place_info.get('where', []):
                return target
        return None

    def _execute_shares(self, share_requests: List[Dict]):
        """
        Execute share requests to transfer inventory between players.

        Args:
            share_requests: List of share request dicts with:
                - sharer_idx: int - player index of sharer
                - recipient_idx: int - player index of recipient
                - resource_type: str - resource to share
                - quantity: int - amount to share
        """
        for request in share_requests:
            sharer_idx = request.get('sharer_idx')
            recipient_idx = request.get('recipient_idx')
            resource_type = request.get('resource_type')
            quantity = request.get('quantity', 1)

            result = {'request': request}

            # Cannot share health
            if resource_type == 'health':
                result['success'] = False
                result['reason'] = 'Cannot share health'
                self._share_results.append(result)
                continue

            # Validate player indices
            if sharer_idx is None or sharer_idx < 0 or sharer_idx >= len(self._players):
                result['success'] = False
                result['reason'] = f'Invalid sharer index: {sharer_idx}'
                self._share_results.append(result)
                continue

            if recipient_idx is None or recipient_idx < 0 or recipient_idx >= len(self._players):
                result['success'] = False
                result['reason'] = f'Invalid recipient index: {recipient_idx}'
                self._share_results.append(result)
                continue

            if sharer_idx == recipient_idx:
                result['success'] = False
                result['reason'] = 'Cannot share with yourself'
                self._share_results.append(result)
                continue

            sharer = self._players[sharer_idx]
            recipient = self._players[recipient_idx]

            if sharer.removed or recipient.removed:
                result['success'] = False
                result['reason'] = 'Player is removed'
                self._share_results.append(result)
                continue

            # Check if sharer has the resource
            if resource_type not in sharer.inventory or sharer.inventory[resource_type] <= 0:
                result['success'] = False
                result['reason'] = f"No '{resource_type}' available to share"
                self._share_results.append(result)
                continue

            # Calculate actual amount to share (up to what sharer has)
            available = sharer.inventory[resource_type]
            actual_quantity = min(quantity, available)

            # Execute the transfer
            sharer.inventory[resource_type] -= actual_quantity
            if resource_type in recipient.inventory:
                recipient.inventory[resource_type] += actual_quantity
            else:
                recipient.inventory[resource_type] = actual_quantity

            result['success'] = True
            result['quantity_transferred'] = actual_quantity
            self._share_results.append(result)

    def _process_cooperative_collections(self, collect_agents: List[Tuple[str, int]]):
        """
        Process all collection actions cooperatively.

        Groups agents by nearby resources and checks if cooperative requirements are met.
        If met, executes collection and distributes resources to all participating agents.
        """
        # Find all collectible resources and which agents are near them
        resource_agents: Dict[Tuple[int, int], Dict] = {}  # pos -> {type, agents, ...}

        for agent_id, player_idx in collect_agents:
            player = self._players[player_idx]
            if player.removed:
                continue

            agent_pos = tuple(player.pos)

            # Find resources within distance threshold
            for dx in range(-self.distance_threshold, self.distance_threshold + 1):
                for dy in range(-self.distance_threshold, self.distance_threshold + 1):
                    if abs(dx) + abs(dy) > self.distance_threshold:
                        continue

                    check_pos = (int(agent_pos[0]) + dx, int(agent_pos[1]) + dy)

                    # Check bounds
                    if not (0 <= check_pos[0] < self._world.area[0] and
                            0 <= check_pos[1] < self._world.area[1]):
                        continue

                    material, obj = self._world[check_pos]

                    # Check if this is a collectible resource
                    resource_type = None
                    if material in constants.collect:
                        resource_type = material
                    elif isinstance(obj, objects.Cow):
                        resource_type = 'cow'

                    if resource_type:
                        if check_pos not in resource_agents:
                            resource_agents[check_pos] = {
                                'type': resource_type,
                                'material': material,
                                'object': obj,
                                'agents': [],
                                'agents_with_tools': []
                            }

                        # Add this agent as nearby
                        if agent_id not in resource_agents[check_pos]['agents']:
                            resource_agents[check_pos]['agents'].append(agent_id)

                            # Check if agent has required tool
                            required_agents, required_tool, required_tool_mode = self.get_resource_requirements(resource_type)
                            if satisfies_tool_requirement(player.inventory, required_tool, required_tool_mode):
                                resource_agents[check_pos]['agents_with_tools'].append(agent_id)

        # Now check which resources can be collected
        collected_positions = set()

        for pos, info in resource_agents.items():
            resource_type = info['type']
            required_agents, required_tool, required_tool_mode = self.get_resource_requirements(resource_type)

            # Get agents who meet all requirements (nearby + tool)
            participating_agents = info['agents_with_tools']

            if len(participating_agents) >= required_agents:
                # Collection is possible! Execute it
                self._execute_cooperative_collection(pos, info, participating_agents)
                collected_positions.add(pos)
                for agent_id in info['agents']:
                    if agent_id not in participating_agents:
                        self._blocked_collections[agent_id] = "Missing required tool"
            else:
                # Block collection - record why
                for agent_id in info['agents']:
                    reason = f"Requires {required_agents} agents with tools (have {len(participating_agents)})"
                    self._blocked_collections[agent_id] = reason

    def _execute_cooperative_collection(
        self,
        pos: Tuple[int, int],
        info: Dict,
        participating_agents: List[str]
    ):
        """
        Execute a cooperative collection and distribute resources.

        Args:
            pos: Position of the resource
            info: Resource info dict
            participating_agents: List of agent_ids who get the resource
        """
        resource_type = info['type']
        material = info['material']
        obj = info['object']

        # Determine what resource to give
        collect_info = constants.collect.get(material)

        if collect_info:
            # Material collection (tree, stone, etc.)
            # Check if any agent meets the requirements (already checked in caller)

            # Change the terrain
            self._world[pos] = collect_info['leaves']

            # Give resources to ALL participating agents
            for agent_id in participating_agents:
                player = self._players[int(agent_id)]
                for item_name, amount in collect_info['receive'].items():
                    player.inventory[item_name] = player.inventory.get(item_name, 0) + amount
                    player.achievements[f'collect_{item_name}'] = \
                        player.achievements.get(f'collect_{item_name}', 0) + 1

            # Record the collection
            collection = {
                'position': pos,
                'resource_type': resource_type,
                'participating_agents': participating_agents,
                'received': collect_info['receive']
            }
            self._coop_collections.append(collection)
            self._record_team_score_event(pos, resource_type, participating_agents, collect_info['receive'])

        elif isinstance(obj, objects.Cow):
            # Cow collection - kill the cow and distribute food
            self._world.remove(obj)

            for agent_id in participating_agents:
                player = self._players[int(agent_id)]
                player.inventory['food'] = player.inventory.get('food', 0) + 6
                player.achievements['eat_cow'] = player.achievements.get('eat_cow', 0) + 1

            collection = {
                'position': pos,
                'resource_type': 'cow',
                'participating_agents': participating_agents,
                'received': {'food': 6}
            }
            self._coop_collections.append(collection)
            self._record_team_score_event(pos, 'cow', participating_agents, {'food': 6})

    def _get_coop_reward(self, agent_id: str) -> float:
        """Get reward for an agent based on cooperative collection results."""
        reward = 0.0

        for collection in self._coop_collections:
            if agent_id in collection['participating_agents']:
                # Agent participated in this collection
                for item, amount in collection['received'].items():
                    if self.team_score_enabled:
                        reward += self.get_item_team_score(item) * amount
                    else:
                        reward_map = {
                            'wood': 5, 'stone': 50, 'coal': 50, 'iron': 50, 'diamond': 10,
                            'food': 1, 'drink': 1, 'sapling': 1
                        }
                        reward += reward_map.get(item, 0) * amount

        return reward

    @property
    def coop_info(self) -> Dict:
        """Get cooperative collection info for the current step."""
        return {
            'collections': self._coop_collections,
            'blocked': self._blocked_collections,
            'enabled': self.coop_enabled,
            'distance_threshold': self.distance_threshold
        }
