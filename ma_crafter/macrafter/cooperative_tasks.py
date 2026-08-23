"""
Cooperative task tracking for multi-agent resource collection.

Tracks 4 dimensions for each collectable resource:
- Spatial: Number of agents within distance d
- Temporal: Number of agents attempting the task this step
- Dependency: Number of agents with appropriate tools
- Participation: Required agents denominator for the task
"""

import yaml
import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


ToolRequirement = Optional[Union[str, List[str], Tuple[str, ...]]]


def normalize_tool_requirement(required_tool: ToolRequirement) -> List[str]:
    """Return a stable list representation for a configured tool requirement."""
    if required_tool is None:
        return []
    if isinstance(required_tool, str):
        return [required_tool]
    return [tool for tool in required_tool if tool]


def satisfies_tool_requirement(
    inventory: Dict[str, int],
    required_tool: ToolRequirement,
    mode: str = "all",
) -> bool:
    """
    Check whether an inventory satisfies a configured tool requirement.

    A string means that one tool is required. A list means all listed tools are
    required by default; configs can opt into "any" when the list represents
    alternatives.
    """
    tools = normalize_tool_requirement(required_tool)
    if not tools:
        return True
    if mode == "any":
        return any(inventory.get(tool, 0) > 0 for tool in tools)
    return all(inventory.get(tool, 0) > 0 for tool in tools)


class TaskStatus(Enum):
    """Status of a collection task."""
    PENDING = "pending"  # Resource exists, not being collected
    IN_PROGRESS = "in_progress"  # Some agents nearby/attempting
    READY = "ready"  # All requirements met, can collect
    COMPLETED = "completed"  # Resource collected
    FAILED = "failed"  # Collection attempt failed (requirements not met)


@dataclass
class TaskState:
    """
    State of a collection task for a single resource at a single time step.
    
    Tracks 4 key dimensions:
    - Spatial: agents within distance d
    - Temporal: valid collect actions issued this step  
    - Dependency: tool requirements
    - Participation: agents actively participating
    """
    # Resource identification
    resource_id: int
    resource_type: str
    position: Tuple[int, int]
    env_step: int
    
    # Requirements from config
    required_agents: int = 1
    required_tool: ToolRequirement = None
    required_tool_mode: str = "all"
    distance_threshold: int = 2
    
    # Spatial: agents within distance d
    agents_nearby: List[str] = field(default_factory=list)
    spatial_count: int = 0
    
    # Temporal: collect attempts for this task this step
    collect_actions: List[str] = field(default_factory=list)
    temporal_count: int = 0
    
    # Dependency: agents capable of the task, independent of spatial readiness
    agents_with_tools: List[str] = field(default_factory=list)
    capable_agents_nearby: List[str] = field(default_factory=list)  # agents nearby AND have tool
    dependency_count: int = 0  # count of capable agents
    dependency_met: bool = True  # True if no tool required or count >= required_agents
    
    # Participation: agents nearby (may or may not issue action)
    # Ratio = participation_count / required_agents (can be >1 or <1)
    participating_agents: List[str] = field(default_factory=list)
    participation_count: int = 0
    participation_ratio: float = 0.0  # participation_count / required_agents
    
    # Overall status
    status: TaskStatus = TaskStatus.PENDING
    collection_possible: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "position": self.position,
            "env_step": self.env_step,
            "required_agents": self.required_agents,
            "required_tool": self.required_tool,
            "required_tool_mode": self.required_tool_mode,
            "distance_threshold": self.distance_threshold,
            "spatial": {
                "agents_nearby": self.agents_nearby,
                "count": self.spatial_count,
                "met": self.spatial_count >= self.required_agents
            },
            "temporal": {
                "collect_actions": self.collect_actions,
                "count": self.temporal_count,
                "met": self.temporal_count >= self.required_agents
            },
            "dependency": {
                "agents_with_tools": self.agents_with_tools,
                "capable_agents_nearby": self.capable_agents_nearby,
                "count": self.dependency_count,
                "met": self.dependency_met
            },
            "participation": {
                "participating_agents": self.participating_agents,
                "count": self.participation_count,
                "ratio": self.participation_ratio,
                "met": self.participation_ratio >= 1.0
            },
            "status": self.status.value,
            "collection_possible": self.collection_possible
        }


@dataclass
class CapabilityChange:
    """Record of a capability change for an agent."""
    agent_id: str
    item_name: str
    old_count: int
    new_count: int
    change_type: str  # "gain" or "loss"
    env_step: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "item_name": self.item_name,
            "old_count": self.old_count,
            "new_count": self.new_count,
            "change_type": self.change_type,
            "env_step": self.env_step
        }


@dataclass
class StepMetrics:
    """Aggregated metrics comparing current step to previous step."""
    env_step: int
    
    # Task set changes
    task_set_size: int = 0  # Current number of tasks
    tasks_added: int = 0  # New tasks this step
    tasks_removed: int = 0  # Tasks removed (collected)
    task_set_delta: int = 0  # Net change in task set size
    
    # Current constraint satisfaction counts (tasks with constraint met)
    spatial_satisfied: int = 0  # Tasks where spatial_count >= required_agents
    temporal_satisfied: int = 0  # Tasks where temporal_count >= required_agents
    dependency_satisfied: int = 0  # Tasks where dependency_ratio >= 1.0
    participation_satisfied: int = 0  # Tasks where participation_ratio >= 1.0
    total_satisfied: int = 0  # Tasks where ALL constraints are met
    
    # Average participation ratio across all tasks
    participation_ratio_avg: float = 0.0
    
    # Constraint changes - count of unique tasks with each constraint type change
    spatial_improved: int = 0  # Tasks where spatial_count increased
    spatial_worsened: int = 0  # Tasks where spatial_count decreased
    temporal_improved: int = 0  # Tasks where temporal_count increased
    temporal_worsened: int = 0  # Tasks where temporal_count decreased
    dependency_improved: int = 0  # Tasks where dependency changed False->True
    dependency_worsened: int = 0  # Tasks where dependency changed True->False
    participation_improved: int = 0  # Tasks where participation_ratio increased
    participation_worsened: int = 0  # Tasks where participation_ratio decreased
    
    # Task IDs for each constraint type (for tracking unique tasks across episode)
    spatial_improved_tasks: List[str] = field(default_factory=list)
    spatial_worsened_tasks: List[str] = field(default_factory=list)
    temporal_improved_tasks: List[str] = field(default_factory=list)
    temporal_worsened_tasks: List[str] = field(default_factory=list)
    dependency_improved_tasks: List[str] = field(default_factory=list)
    dependency_worsened_tasks: List[str] = field(default_factory=list)
    participation_improved_tasks: List[str] = field(default_factory=list)
    participation_worsened_tasks: List[str] = field(default_factory=list)
    
    # Aggregate constraint changes
    total_improved: int = 0  # Total tasks with any positive constraint change
    total_worsened: int = 0  # Total tasks with any negative constraint change
    
    # Capability changes (resources/tools, not health/food/drink/energy)
    capability_increased: int = 0  # Number of capability gains
    capability_decreased: int = 0  # Number of capability losses
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_step": self.env_step,
            "task_set": {
                "size": self.task_set_size,
                "added": self.tasks_added,
                "removed": self.tasks_removed,
                "delta": self.task_set_delta
            },
            "constraint_satisfaction": {
                "spatial": self.spatial_satisfied,
                "temporal": self.temporal_satisfied,
                "dependency": self.dependency_satisfied,
                "participation": self.participation_satisfied,
                "participation_ratio_avg": self.participation_ratio_avg,
                "total": self.total_satisfied
            },
            "constraint_changes": {
                "spatial": {"improved": self.spatial_improved, "worsened": self.spatial_worsened,
                           "improved_tasks": self.spatial_improved_tasks, "worsened_tasks": self.spatial_worsened_tasks},
                "temporal": {"improved": self.temporal_improved, "worsened": self.temporal_worsened,
                            "improved_tasks": self.temporal_improved_tasks, "worsened_tasks": self.temporal_worsened_tasks},
                "dependency": {"improved": self.dependency_improved, "worsened": self.dependency_worsened,
                              "improved_tasks": self.dependency_improved_tasks, "worsened_tasks": self.dependency_worsened_tasks},
                "participation": {"improved": self.participation_improved, "worsened": self.participation_worsened,
                                 "improved_tasks": self.participation_improved_tasks, "worsened_tasks": self.participation_worsened_tasks},
                "total_improved": self.total_improved,
                "total_worsened": self.total_worsened
            },
            "capability_changes": {
                "increased": self.capability_increased,
                "decreased": self.capability_decreased
            }
        }


@dataclass 
class StepTaskSummary:
    """Summary of all task states for a single env step."""
    env_step: int
    tasks: Dict[int, TaskState] = field(default_factory=dict)  # resource_id -> TaskState
    successful_collections: List[int] = field(default_factory=list)  # resource_ids collected
    failed_attempts: List[int] = field(default_factory=list)  # resource_ids with failed attempts
    metrics: Optional[StepMetrics] = None  # Aggregated metrics for this step
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_step": self.env_step,
            "tasks": {rid: t.to_dict() for rid, t in self.tasks.items()},
            "successful_collections": self.successful_collections,
            "failed_attempts": self.failed_attempts,
            "metrics": self.metrics.to_dict() if self.metrics else None
        }


class CooperativeTaskTracker:
    """
    Tracks cooperative collection tasks across all resources.
    
    Responsible for:
    1. Loading cooperative config
    2. Computing task states each step
    3. Determining if collection attempts succeed
    4. Maintaining task history
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the tracker.
        
        Args:
            config_path: Resolved cooperative task configuration from `coop_env`
        """
        self.config = self._load_config(config_path)
        self.enabled = self.config.get('cooperative_collection', {}).get('enabled', False)
        self.distance_threshold = self.config.get('cooperative_collection', {}).get('distance_threshold', 2)
        self.resource_requirements = self.config.get('cooperative_collection', {}).get('resources', {})
        self.track_history = self.config.get('task_tracking', {}).get('enabled', True)
        
        # Current step state
        self.current_step: int = 0
        self.current_tasks: Dict[int, TaskState] = {}  # resource_id -> TaskState
        self.previous_tasks: Dict[int, TaskState] = {}  # Previous step tasks for comparison
        
        # Track discovered resources (once seen, always tracked)
        self.discovered_resources: Dict[int, Tuple[Tuple[int, int], str]] = {}  # resource_id -> (pos, type)
        
        # Capability tracking
        self.previous_inventories: Dict[str, Dict[str, int]] = {}  # agent_id -> inventory
        self.capability_history: List[CapabilityChange] = []  # All capability changes
        
        # Ignore these items for capability tracking (survival stats, not capabilities)
        self.ignored_items = {'health', 'food', 'drink', 'energy', 'sapling'}
        
        # History
        self.task_history: List[StepTaskSummary] = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from yaml file.
        
        Args:
            config_path: Resolved path to yaml file (provided by coop_env)
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_resource_requirements(self, resource_type: str) -> Tuple[int, ToolRequirement, str]:
        """
        Get collection requirements for a resource type.
        
        Returns:
            (required_agents, required_tool, required_tool_mode)
        """
        resource_type = resource_type.lower()
        if resource_type in self.resource_requirements:
            req = self.resource_requirements[resource_type]
            return (
                req.get('required_agents', 1),
                req.get('required_tool'),
                req.get('required_tool_mode', 'all'),
            )
        # Default: 1 agent, no tool
        return (1, None, "all")
    
    def compute_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Compute Manhattan distance to match cooperative collection execution."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def compute_task_states(
        self,
        env_step: int,
        world_state: Any,  # SymbolicWorldState
        agent_positions: Dict[str, Tuple[int, int]],  # agent_id -> position
        agent_facings: Dict[str, Tuple[int, int]],  # agent_id -> facing direction
        agent_inventories: Dict[str, Dict[str, int]],  # agent_id -> inventory
        collect_actions: Set[str],  # agent_ids issuing collect this step
        collect_attempts: Optional[Dict[str, Set[str]]] = None,
    ) -> Dict[int, TaskState]:
        """
        Compute task states for all collectable resources.
        
        Args:
            env_step: Current environment step
            world_state: SymbolicWorldState with object positions
            agent_positions: Dict mapping agent_id to (x, y) position
            agent_facings: Dict mapping agent_id to facing direction
            agent_inventories: Dict mapping agent_id to inventory dict
            collect_actions: Set of agent_ids issuing collect action this step
            collect_attempts: Optional mapping from resource/task id to agents
                explicitly attempting that task this step.
            
        Returns:
            Dict mapping resource_id to TaskState
        """
        self.current_step = env_step
        self.current_tasks = {}
        collect_attempts = {
            str(task_id): {str(agent_id) for agent_id in agent_ids}
            for task_id, agent_ids in (collect_attempts or {}).items()
        }
        targeted_collect_agents = {
            agent_id
            for agent_ids in collect_attempts.values()
            for agent_id in agent_ids
        }
        
        if not world_state:
            return self.current_tasks
        
        # Collectable materials (found in terrain, not as objects)
        COLLECTABLE_MATERIALS = {'tree', 'stone', 'coal', 'iron', 'diamond', 'water'}
        
        # First, collect all resources: both objects and collectable materials
        all_resources = []  # List of (resource_id, position, resource_type)
        
        # Get objects from world state (e.g., cows)
        for obj_id, pos in world_state.object_positions.items():
            props = world_state.object_properties.get(obj_id, {})
            obj_type = props.get('type', 'unknown')
            
            # Skip non-collectable types. Cows are handled cooperatively by CooperativeEnv.
            if obj_type in ['player', 'zombie', 'skeleton']:
                continue
            if obj_type not in self.resource_requirements:
                continue
            
            all_resources.append((obj_id, pos, obj_type))
        
        # Get collectable materials from symbolic matrix - track ALL resources
        if world_state.symbolic_matrix is not None:
            # Scan entire world for all collectable resources
            for x in range(world_state.world.area[0]):
                for y in range(world_state.world.area[1]):
                    for entity_type, mat_id, mat_name in world_state.symbolic_matrix[x, y]:
                        if entity_type == 'material' and mat_name in COLLECTABLE_MATERIALS:
                            all_resources.append((mat_id, (x, y), mat_name))
        
        # Now process all resources
        for obj_id, pos, obj_type in all_resources:
            # Create task state for this resource
            required_agents, required_tool, required_tool_mode = self.get_resource_requirements(obj_type)
            
            task = TaskState(
                resource_id=obj_id,
                resource_type=obj_type,
                position=pos,
                env_step=env_step,
                required_agents=required_agents,
                required_tool=required_tool,
                required_tool_mode=required_tool_mode,
                distance_threshold=self.distance_threshold
            )
            
            # Compute spatial: agents within distance
            for agent_id, agent_pos in agent_positions.items():
                distance = self.compute_distance(pos, agent_pos)
                if distance <= self.distance_threshold:
                    task.agents_nearby.append(agent_id)
            task.spatial_count = len(task.agents_nearby)
            
            # Compute dependency: agents capable of this task, independent
            # of whether they are already spatially near the resource.
            if not normalize_tool_requirement(required_tool):
                # No tool required - all live agents are capable.
                task.agents_with_tools = list(agent_positions.keys())
                task.capable_agents_nearby = task.agents_nearby.copy()
                task.dependency_count = len(task.agents_with_tools)
                task.dependency_met = True
            else:
                for agent_id, inventory in agent_inventories.items():
                    if satisfies_tool_requirement(inventory, required_tool, required_tool_mode):
                        task.agents_with_tools.append(agent_id)
                # Keep this diagnostic, but dependency_count itself is not
                # spatially gated.
                task.capable_agents_nearby = list(set(task.agents_nearby) & set(task.agents_with_tools))
                task.dependency_count = len(task.agents_with_tools)
                task.dependency_met = task.dependency_count >= required_agents
            
            # Compute temporal: agents that attempted this task this step.
            # Targeted collect attempts are intent-based and independent of
            # spatial readiness; untargeted primitive collects retain the
            # nearby fallback.
            attempted_agents = set(collect_attempts.get(str(obj_id), set()))
            untargeted_collect_actions = {
                str(agent_id)
                for agent_id in collect_actions
                if str(agent_id) not in targeted_collect_agents
            }
            for agent_id in untargeted_collect_actions:
                if agent_id in task.agents_nearby:
                    attempted_agents.add(agent_id)
            task.collect_actions = sorted(attempted_agents)
            task.temporal_count = len(task.collect_actions)
            
            # Compute participation: agents nearby (spatial constraint)
            # This is the ratio of nearby agents to required agents
            task.participating_agents = task.agents_nearby.copy()
            task.participation_count = len(task.participating_agents)
            task.participation_ratio = task.participation_count / required_agents if required_agents > 0 else 0.0
            
            # Determine overall status
            if task.participation_ratio >= 1.0 and task.temporal_count >= required_agents and task.dependency_met:
                task.collection_possible = True
                task.status = TaskStatus.READY
            elif task.temporal_count > 0:
                # Some agents tried but not enough
                task.status = TaskStatus.IN_PROGRESS
            elif task.spatial_count > 0:
                # Agents nearby but not collecting
                task.status = TaskStatus.IN_PROGRESS
            else:
                task.status = TaskStatus.PENDING
            
            self.current_tasks[obj_id] = task
        
        return self.current_tasks
    
    def track_capability_changes(self, agent_inventories: Dict[str, Dict[str, int]]):
        """
        Track changes in agent capabilities (resources/tools, excluding health/food/drink/energy).
        
        Args:
            agent_inventories: Current inventories for all agents {agent_id -> inventory}
        """
        for agent_id, current_inv in agent_inventories.items():
            # Get previous inventory for this agent
            prev_inv = self.previous_inventories.get(agent_id, {})
            
            # Check all items in current inventory
            all_items = set(current_inv.keys()) | set(prev_inv.keys())
            
            for item_name in all_items:
                # Skip survival stats
                if item_name in self.ignored_items:
                    continue
                
                old_count = prev_inv.get(item_name, 0)
                new_count = current_inv.get(item_name, 0)
                
                # Track changes
                if new_count > old_count:
                    change = CapabilityChange(
                        agent_id=agent_id,
                        item_name=item_name,
                        old_count=old_count,
                        new_count=new_count,
                        change_type="gain",
                        env_step=self.current_step
                    )
                    self.capability_history.append(change)
                elif new_count < old_count:
                    change = CapabilityChange(
                        agent_id=agent_id,
                        item_name=item_name,
                        old_count=old_count,
                        new_count=new_count,
                        change_type="loss",
                        env_step=self.current_step
                    )
                    self.capability_history.append(change)
        
        # Update previous inventories for next step
        self.previous_inventories = {aid: inv.copy() for aid, inv in agent_inventories.items()}
    
    def process_collection_attempt(
        self,
        resource_id: int,
        collecting_agents: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Process a collection attempt for a resource.
        
        Args:
            resource_id: ID of resource being collected
            collecting_agents: List of agent_ids attempting to collect
            
        Returns:
            (success, agents_who_get_resource)
            - success: True if collection succeeded
            - agents_who_get_resource: List of agent_ids who receive the resource
        """
        if resource_id not in self.current_tasks:
            # No task state - allow default behavior (single agent collect)
            return (True, collecting_agents[:1]) if collecting_agents else (False, [])
        
        task = self.current_tasks[resource_id]
        
        if task.collection_possible:
            # All requirements met - all participating agents get the resource
            task.status = TaskStatus.COMPLETED
            return (True, task.participating_agents)
        else:
            # Requirements not met - collection fails
            task.status = TaskStatus.FAILED
            return (False, [])
    
    def _compute_metrics(self) -> StepMetrics:
        """
        Compute aggregated metrics by comparing current and previous tasks.
        
        Returns:
            StepMetrics with aggregated changes
        """
        metrics = StepMetrics(env_step=self.current_step)
        
        # Task set changes
        current_ids = set(self.current_tasks.keys())
        previous_ids = set(self.previous_tasks.keys())
        
        metrics.task_set_size = len(current_ids)
        metrics.tasks_added = len(current_ids - previous_ids)
        metrics.tasks_removed = len(previous_ids - current_ids)
        metrics.task_set_delta = metrics.tasks_added - metrics.tasks_removed
        
        # Current constraint satisfaction counts (how many tasks have each constraint met)
        total_participation_ratio = 0.0
        total_dependency_ratio = 0.0
        for rid, task in self.current_tasks.items():
            required = task.required_agents
            spatial_met = task.spatial_count >= required
            temporal_met = task.temporal_count >= required
            participation_met = task.participation_ratio >= 1.0
            dependency_met = task.dependency_count >= required
            
            # satisfied = count > 0 (any progress)
            if task.spatial_count > 0:
                metrics.spatial_satisfied += 1
            if task.temporal_count > 0:
                metrics.temporal_satisfied += 1
            if task.dependency_count > 0:
                metrics.dependency_satisfied += 1
            if task.participation_ratio > 0:
                metrics.participation_satisfied += 1
            if spatial_met and temporal_met and dependency_met and participation_met:
                metrics.total_satisfied += 1
            
            total_participation_ratio += task.participation_ratio
        
        # Average participation ratio
        if len(self.current_tasks) > 0:
            metrics.participation_ratio_avg = total_participation_ratio / len(self.current_tasks)
        
        # Track which tasks had improvements/worsenings
        tasks_with_improvements = set()
        tasks_with_worsenings = set()
        
        # For step 1 (no previous tasks), use satisfaction counts as "improved" for individual constraints
        # but don't set total_improved (that info is in task_set_delta)
        if not self.previous_tasks:
            metrics.spatial_improved = metrics.spatial_satisfied
            metrics.temporal_improved = metrics.temporal_satisfied
            metrics.dependency_improved = metrics.dependency_satisfied
            metrics.participation_improved = metrics.participation_satisfied
            # total_improved stays 0 - task set change is already shown in task_set_delta
        else:
            # Compare constraints for tasks that exist in both steps
            common_ids = current_ids & previous_ids
            for rid in common_ids:
                prev_task = self.previous_tasks[rid]
                curr_task = self.current_tasks[rid]
                
                # Spatial constraint changes
                if curr_task.spatial_count > prev_task.spatial_count:
                    metrics.spatial_improved += 1
                    metrics.spatial_improved_tasks.append(rid)
                    tasks_with_improvements.add(rid)
                elif curr_task.spatial_count < prev_task.spatial_count:
                    metrics.spatial_worsened += 1
                    metrics.spatial_worsened_tasks.append(rid)
                    tasks_with_worsenings.add(rid)
                
                # Temporal constraint changes
                if curr_task.temporal_count > prev_task.temporal_count:
                    metrics.temporal_improved += 1
                    metrics.temporal_improved_tasks.append(rid)
                    tasks_with_improvements.add(rid)
                elif curr_task.temporal_count < prev_task.temporal_count:
                    metrics.temporal_worsened += 1
                    metrics.temporal_worsened_tasks.append(rid)
                    tasks_with_worsenings.add(rid)
                
                # Dependency constraint changes (using count of capable agents nearby)
                if curr_task.dependency_count > prev_task.dependency_count:
                    metrics.dependency_improved += 1
                    metrics.dependency_improved_tasks.append(rid)
                    tasks_with_improvements.add(rid)
                elif curr_task.dependency_count < prev_task.dependency_count:
                    metrics.dependency_worsened += 1
                    metrics.dependency_worsened_tasks.append(rid)
                    tasks_with_worsenings.add(rid)
                
                # Participation constraint changes (using ratio)
                if curr_task.participation_ratio > prev_task.participation_ratio:
                    metrics.participation_improved += 1
                    metrics.participation_improved_tasks.append(rid)
                    tasks_with_improvements.add(rid)
                elif curr_task.participation_ratio < prev_task.participation_ratio:
                    metrics.participation_worsened += 1
                    metrics.participation_worsened_tasks.append(rid)
                    tasks_with_worsenings.add(rid)
            
            # Aggregate: unique tasks with any improvement/worsening
            metrics.total_improved = len(tasks_with_improvements)
            metrics.total_worsened = len(tasks_with_worsenings)
        
        # Capability changes for this step - total quantity change
        step_capability_changes = [c for c in self.capability_history if c.env_step == self.current_step]
        metrics.capability_increased = sum(c.new_count - c.old_count for c in step_capability_changes if c.change_type == "gain")
        metrics.capability_decreased = sum(c.old_count - c.new_count for c in step_capability_changes if c.change_type == "loss")
        
        return metrics
    
    def finalize_step(self) -> StepTaskSummary:
        """
        Finalize the current step and save to history.
        
        Returns:
            StepTaskSummary for this step with computed metrics
        """
        # Compute metrics by comparing to previous step
        metrics = self._compute_metrics()
        
        summary = StepTaskSummary(
            env_step=self.current_step,
            tasks=self.current_tasks.copy(),
            metrics=metrics
        )
        
        # Categorize tasks
        for rid, task in self.current_tasks.items():
            if task.status == TaskStatus.COMPLETED:
                summary.successful_collections.append(rid)
            elif task.status == TaskStatus.FAILED:
                summary.failed_attempts.append(rid)
        
        if self.track_history:
            self.task_history.append(summary)
        
        # Store current tasks as previous for next step
        self.previous_tasks = self.current_tasks.copy()
        
        return summary
    
    def get_task_state(self, resource_id: int) -> Optional[TaskState]:
        """Get current task state for a resource."""
        return self.current_tasks.get(resource_id)
    
    def get_all_task_states(self) -> Dict[int, TaskState]:
        """Get all current task states."""
        return self.current_tasks
    
    def get_history(self) -> List[StepTaskSummary]:
        """Get full task history."""
        return self.task_history
    
    def reset(self):
        """Reset tracker state."""
        self.current_step = 0
        self.current_tasks = {}
        self.previous_tasks = {}
        self.discovered_resources = {}
        self.previous_inventories = {}
        self.capability_history = []
        self.task_history = []
    
    def print_task_summary(self, verbose: bool = False):
        """Print summary of current task states."""
        print(f"\n=== Task States (Step {self.current_step}) ===")
        
        # Group by status
        by_status = {}
        for task in self.current_tasks.values():
            status = task.status.value
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(task)
        
        for status, tasks in by_status.items():
            print(f"\n[{status.upper()}] ({len(tasks)} resources)")
            if verbose:
                for task in tasks:
                    print(f"  {task.resource_type}#{task.resource_id} @ {task.position}")
                    print(f"    Spatial: {task.spatial_count}/{task.required_agents} agents nearby {task.agents_nearby}")
                    print(f"    Temporal: {task.temporal_count} collect actions {task.collect_actions}")
                    print(f"    Dependency: {task.dependency_met} (tool: {task.required_tool})")
                    print(f"    Participation: {task.participation_count} agents {task.participating_agents}")


def plot_metrics_timeline(metrics_list: List['StepMetrics'], output_path: Optional[str] = None, show: bool = True):
    """
    Plot cooperative task metrics timeline in row-based format.
    
    Args:
        metrics_list: List of StepMetrics from task tracker history
        output_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    if not metrics_list:
        print("No metrics to plot")
        return
    
    # Use all metrics (including step 1 with initial task set)
    metrics_to_plot = metrics_list
    
    if not metrics_to_plot:
        print("Not enough metrics to plot")
        return
    
    # Extract data
    steps = [m.env_step for m in metrics_to_plot]
    task_set_delta = [m.task_set_delta for m in metrics_to_plot]
    total_improved = [m.total_improved for m in metrics_to_plot]
    total_worsened = [m.total_worsened for m in metrics_to_plot]
    capability_increased = [m.capability_increased for m in metrics_to_plot]
    capability_decreased = [m.capability_decreased for m in metrics_to_plot]
    
    # Individual constraint types
    spatial_improved = [m.spatial_improved for m in metrics_to_plot]
    spatial_worsened = [m.spatial_worsened for m in metrics_to_plot]
    temporal_improved = [m.temporal_improved for m in metrics_to_plot]
    temporal_worsened = [m.temporal_worsened for m in metrics_to_plot]
    dependency_improved = [m.dependency_improved for m in metrics_to_plot]
    dependency_worsened = [m.dependency_worsened for m in metrics_to_plot]
    participation_improved = [m.participation_improved for m in metrics_to_plot]
    participation_worsened = [m.participation_worsened for m in metrics_to_plot]
    
    # Satisfaction counts
    total_satisfied = [m.total_satisfied for m in metrics_to_plot]
    spatial_satisfied = [m.spatial_satisfied for m in metrics_to_plot]
    temporal_satisfied = [m.temporal_satisfied for m in metrics_to_plot]
    dependency_satisfied = [m.dependency_satisfied for m in metrics_to_plot]
    participation_satisfied = [m.participation_satisfied for m in metrics_to_plot]
    participation_ratio_avg = [m.participation_ratio_avg for m in metrics_to_plot]
    
    # Create visualization - 8 rows
    fig, axes = plt.subplots(8, 1, figsize=(16, 12), sharex=True)
    
    # Row 1: Env Step
    for step in steps:
        axes[0].barh(0, 1, left=step, height=0.6, color='lightsteelblue', edgecolor='white', linewidth=0.5)
        axes[0].text(step + 0.5, 0, str(step), ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Row 2: Task Set Change (delta)
    for i, step in enumerate(steps):
        delta = task_set_delta[i]
        color = 'green' if delta > 0 else 'red' if delta < 0 else 'gray'
        axes[1].barh(0, 1, left=step, height=0.6, color=color, edgecolor='white', linewidth=0.5, alpha=0.7)
        if delta != 0:
            axes[1].text(step + 0.5, 0, str(delta), ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Helper function for combined improved/worsened rows (2 levels)
    def plot_combined_row(ax, improved_data, worsened_data):
        for i, step in enumerate(steps):
            imp = improved_data[i]
            wor = worsened_data[i]
            # Upper half: improved (green)
            color_imp = 'green' if imp > 0 else 'lightgray'
            alpha_imp = 0.3 + (0.7 * min(imp / 10, 1.0)) if imp > 0 else 0.3
            ax.barh(0.25, 1, left=step, height=0.4, color=color_imp, edgecolor='white', linewidth=0.5, alpha=alpha_imp)
            if imp > 0:
                ax.text(step + 0.5, 0.25, str(imp), ha='center', va='center', fontsize=7, fontweight='bold', color='white')
            # Lower half: worsened (red)
            color_wor = 'red' if wor > 0 else 'lightgray'
            alpha_wor = 0.3 + (0.7 * min(wor / 10, 1.0)) if wor > 0 else 0.3
            ax.barh(-0.25, 1, left=step, height=0.4, color=color_wor, edgecolor='white', linewidth=0.5, alpha=alpha_wor)
            if wor > 0:
                ax.text(step + 0.5, -0.25, str(wor), ha='center', va='center', fontsize=7, fontweight='bold', color='white')
    
    # Helper function for 3-level rows: satisfied (top), improved (middle), worsened (bottom)
    def plot_triple_row(ax, sat_data, improved_data, worsened_data):
        for i, step in enumerate(steps):
            sat = sat_data[i]
            imp = improved_data[i]
            wor = worsened_data[i]
            # Top third: satisfied count (blue)
            color_sat = 'steelblue' if sat > 0 else 'lightgray'
            alpha_sat = 0.3 + (0.7 * min(sat / 100, 1.0)) if sat > 0 else 0.3
            ax.barh(0.33, 1, left=step, height=0.28, color=color_sat, edgecolor='white', linewidth=0.5, alpha=alpha_sat)
            if sat > 0:
                ax.text(step + 0.5, 0.33, str(sat), ha='center', va='center', fontsize=6, fontweight='bold', color='white')
            # Middle third: improved (green)
            color_imp = 'green' if imp > 0 else 'lightgray'
            alpha_imp = 0.3 + (0.7 * min(imp / 10, 1.0)) if imp > 0 else 0.3
            ax.barh(0, 1, left=step, height=0.28, color=color_imp, edgecolor='white', linewidth=0.5, alpha=alpha_imp)
            if imp > 0:
                ax.text(step + 0.5, 0, str(imp), ha='center', va='center', fontsize=6, fontweight='bold', color='white')
            # Bottom third: worsened (red)
            color_wor = 'red' if wor > 0 else 'lightgray'
            alpha_wor = 0.3 + (0.7 * min(wor / 10, 1.0)) if wor > 0 else 0.3
            ax.barh(-0.33, 1, left=step, height=0.28, color=color_wor, edgecolor='white', linewidth=0.5, alpha=alpha_wor)
            if wor > 0:
                ax.text(step + 0.5, -0.33, str(wor), ha='center', va='center', fontsize=6, fontweight='bold', color='white')
    
    # Row 3: Capability Increased/Decreased (combined, 2 levels)
    plot_combined_row(axes[2], capability_increased, capability_decreased)
    
    # Row 4: Tasks (improved/worsened only, 2 levels)
    plot_combined_row(axes[3], total_improved, total_worsened)
    
    # Row 5: Spatial (satisfied/improved/worsened, 3 levels)
    plot_triple_row(axes[4], spatial_satisfied, spatial_improved, spatial_worsened)
    
    # Row 6: Temporal (satisfied/improved/worsened, 3 levels)
    plot_triple_row(axes[5], temporal_satisfied, temporal_improved, temporal_worsened)
    
    # Row 7: Dependency (satisfied/improved/worsened, 3 levels)
    plot_triple_row(axes[6], dependency_satisfied, dependency_improved, dependency_worsened)
    
    # Row 8: Participation (satisfied/improved/worsened, 3 levels)
    plot_triple_row(axes[7], participation_satisfied, participation_improved, participation_worsened)
    
    # Configure axes
    axes[0].set_ylabel('Env Step', fontsize=9, fontweight='bold')
    axes[1].set_ylabel('Task Set\n(+added/-removed)', fontsize=9, fontweight='bold')
    axes[2].set_ylabel('Capability\n(+gained/-lost)', fontsize=9, fontweight='bold')
    axes[3].set_ylabel('# Tasks\n(any +/-)', fontsize=9, fontweight='bold')
    axes[4].set_ylabel('# Tasks\n(spatial sat/+/-)', fontsize=9, fontweight='bold')
    axes[5].set_ylabel('# Tasks\n(temporal sat/+/-)', fontsize=9, fontweight='bold')
    axes[6].set_ylabel('# Tasks\n(depend. sat/+/-)', fontsize=9, fontweight='bold')
    axes[7].set_ylabel('# Tasks\n(particip. sat/+/-)', fontsize=9, fontweight='bold')
    
    for ax in axes:
        ax.set_yticks([])
        ax.grid(axis='x', alpha=0.3)
        ax.set_ylim(-0.5, 0.5)
    
    axes[-1].set_xlabel('Environment Step', fontsize=12)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='steelblue', alpha=0.7, label='Satisfied (top)'),
        mpatches.Patch(color='green', alpha=0.7, label='Improved (middle)'),
        mpatches.Patch(color='red', alpha=0.7, label='Worsened (bottom)'),
        mpatches.Patch(color='lightgray', alpha=0.5, label='Zero'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.suptitle('Cooperative Task Metrics Timeline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    return obj


def save_task_states_log(task_states_history: List[Dict[str, Any]], output_path: str):
    """
    Save task states history to JSON file.
    
    Args:
        task_states_history: List of dicts with 'step' and 'task_states' (Dict[int, TaskState])
        output_path: Path to save JSON file
    """
    task_states_serializable = []
    for snapshot in task_states_history:
        task_states_serializable.append({
            'step': int(snapshot['step']),
            'task_states': {
                str(resource_id): convert_to_serializable(task.to_dict()) 
                for resource_id, task in snapshot['task_states'].items()
            }
        })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(task_states_serializable, f, indent=2)
    print(f"Saved {len(task_states_history)} task state snapshots to {output_path}")


def save_capability_log(capability_history: List['CapabilityChange'], output_path: str):
    """
    Save capability change history to JSON file.
    
    Args:
        capability_history: List of CapabilityChange objects
        output_path: Path to save JSON file
    """
    capability_data = [convert_to_serializable(change.to_dict()) for change in capability_history]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(capability_data, f, indent=2)
    print(f"Saved {len(capability_history)} capability changes to {output_path}")
