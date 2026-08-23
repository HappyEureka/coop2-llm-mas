"""
Cooperative task tracking for CUBE block-pushing environment.

Each FACE of a block is a unique task (4 tasks per block: up, down, left, right).
Tracks:
- Spatial: Number of agents adjacent to the face
- Temporal: Number of agents targeting/pushing the face in the same step
- Dependency: None (always satisfied)
- Participation: agents adjacent to face / block weight (ratio)
"""

from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class TaskStatus(Enum):
    """Status of a face task."""
    PENDING = "pending"      # Face not yet pushed to goal
    IN_PROGRESS = "in_progress"  # Agents are aligned/pushing this face
    COMPLETED = "completed"  # Block reached goal column


@dataclass
class FaceTaskState:
    """State of a single face task (one face of a block)."""
    task_id: str  # Format: "block_{id}_{face}" e.g. "block_0_right"
    block_id: int
    face: str  # "up", "down", "left", "right"
    weight: int  # Block weight (required agents for participation)
    block_position: Tuple[int, int]  # (row, col) of block top-left
    status: TaskStatus = TaskStatus.PENDING
    
    # Spatial: agents adjacent to this face
    adjacent_agents: Set[str] = field(default_factory=set)
    
    # Temporal: agents pushing toward this face
    pushing_agents: Set[str] = field(default_factory=set)
    
    # Participation: ratio of adjacent agents to weight
    participation_ratio: float = 0.0
    participation_met: bool = False  # True if ratio >= 1.0
    
    def __post_init__(self):
        if not self.adjacent_agents:
            self.adjacent_agents = set()
        if not self.pushing_agents:
            self.pushing_agents = set()
    
    def update_spatial(self, agents: Set[str]):
        """Update which agents are adjacent to this face."""
        self.adjacent_agents = agents
        self._update_participation()
    
    def update_temporal(self, agents: Set[str]):
        """Update which agents are pushing toward this face."""
        self.pushing_agents = agents
    
    def _update_participation(self):
        """Recalculate participation metrics."""
        self.participation_ratio = len(self.adjacent_agents) / self.weight if self.weight > 0 else 0
        self.participation_met = self.participation_ratio >= 1.0
    
    def get_spatial_count(self) -> int:
        """Get number of agents adjacent to this face."""
        return len(self.adjacent_agents)
    
    def get_temporal_count(self) -> int:
        """Get number of agents pushing toward this face."""
        return len(self.pushing_agents)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "block_id": self.block_id,
            "face": self.face,
            "weight": self.weight,
            "block_position": self.block_position,
            "status": self.status.value,
            "adjacent_agents": list(self.adjacent_agents),
            "pushing_agents": list(self.pushing_agents),
            "spatial_count": self.get_spatial_count(),
            "temporal_count": self.get_temporal_count(),
            "participation_ratio": self.participation_ratio,
            "participation_met": self.participation_met,
        }


@dataclass
class CooperativeMetrics:
    """Metrics for cooperative task progress."""
    # Spatial metrics (alignment per face)
    total_adjacent: int = 0  # Total agents adjacent to any face
    adjacent_by_task: Dict[str, int] = field(default_factory=dict)  # task_id -> count
    
    # Temporal metrics (pushing per face)
    total_pushing: int = 0  # Total agents currently pushing
    pushing_by_task: Dict[str, int] = field(default_factory=dict)  # task_id -> count
    
    # Participation metrics
    faces_with_sufficient: int = 0  # Faces with participation_met
    avg_participation_ratio: float = 0.0
    
    # Task completion
    blocks_delivered: int = 0
    blocks_remaining: int = 0
    tasks_total: int = 0  # Total face tasks
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "spatial": {
                "total_adjacent": self.total_adjacent,
                "by_task": self.adjacent_by_task,
            },
            "temporal": {
                "total_pushing": self.total_pushing,
                "by_task": self.pushing_by_task,
            },
            "participation": {
                "faces_with_sufficient": self.faces_with_sufficient,
                "avg_ratio": self.avg_participation_ratio,
            },
            "completion": {
                "delivered": self.blocks_delivered,
                "remaining": self.blocks_remaining,
                "tasks_total": self.tasks_total,
            }
        }


@dataclass
class StepMetrics:
    """Aggregated metrics comparing current step to previous step (ma-crafter compatible)."""
    env_step: int
    
    # Task set changes (face tasks)
    task_set_size: int = 0  # Current number of tasks (faces)
    tasks_added: int = 0  # New tasks this step
    tasks_removed: int = 0  # Tasks removed (block delivered)
    task_set_delta: int = 0  # Net change in task set size
    
    # Current constraint satisfaction counts (faces with constraint met)
    spatial_satisfied: int = 0  # Faces with any agent adjacent
    temporal_satisfied: int = 0  # Faces with any agent pushing
    dependency_satisfied: int = 0  # Always = task_set_size (no dependencies)
    participation_satisfied: int = 0  # Faces with participation_ratio >= 1.0
    total_satisfied: int = 0  # Faces where ALL constraints met
    
    # Average participation ratio across all faces
    participation_ratio_avg: float = 0.0
    
    # Constraint changes - count of unique tasks with each constraint type change
    spatial_improved: int = 0  # Faces where adjacent agents increased
    spatial_worsened: int = 0  # Faces where adjacent agents decreased
    temporal_improved: int = 0  # Faces where pushing agents increased
    temporal_worsened: int = 0  # Faces where pushing agents decreased
    dependency_improved: int = 0  # Always 0 (no dependencies)
    dependency_worsened: int = 0  # Always 0 (no dependencies)
    participation_improved: int = 0  # Faces where participation ratio increased
    participation_worsened: int = 0  # Faces where participation ratio decreased
    
    # Task IDs for each constraint type change
    spatial_improved_tasks: List[str] = field(default_factory=list)
    spatial_worsened_tasks: List[str] = field(default_factory=list)
    temporal_improved_tasks: List[str] = field(default_factory=list)
    temporal_worsened_tasks: List[str] = field(default_factory=list)
    dependency_improved_tasks: List[str] = field(default_factory=list)
    dependency_worsened_tasks: List[str] = field(default_factory=list)
    participation_improved_tasks: List[str] = field(default_factory=list)
    participation_worsened_tasks: List[str] = field(default_factory=list)
    
    # Aggregate constraint changes
    total_improved: int = 0  # Total faces with any positive constraint change
    total_worsened: int = 0  # Total faces with any negative constraint change
    
    # Task state changes (block position)
    task_state_changed: int = 0  # Faces whose block moved this step
    task_state_changed_tasks: List[str] = field(default_factory=list)
    task_state_progress: int = 0  # Faces whose block moved towards goal
    task_state_regress: int = 0  # Faces whose block moved away from goal
    
    # Capability changes (always 0 in CUBE - no capability system)
    capability_increased: int = 0
    capability_decreased: int = 0
    
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
            "task_state_changes": {
                "changed": self.task_state_changed,
                "changed_tasks": self.task_state_changed_tasks,
                "progress": self.task_state_progress,
                "regress": self.task_state_regress
            },
            "capability_changes": {
                "increased": self.capability_increased,
                "decreased": self.capability_decreased
            }
        }


@dataclass
class StepSummary:
    """Summary of task states at a single step."""
    env_step: int
    tasks: Dict[str, FaceTaskState]  # task_id -> state
    metrics: Optional[CooperativeMetrics] = None
    step_metrics: Optional[StepMetrics] = None
    delivered_blocks: List[int] = field(default_factory=list)
    successful_collections: List[int] = field(default_factory=list)  # block_ids collected this step
    failed_attempts: List[int] = field(default_factory=list)  # block_ids with failed attempts
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.env_step,
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "delivered_blocks": self.delivered_blocks,
            "successful_collections": self.successful_collections,
            "failed_attempts": self.failed_attempts,
        }


class CooperativeTaskTracker:
    """
    Tracks cooperative task state for CUBE block-pushing.
    
    Each face of a block is a separate task.
    Updates task states based on:
    - Agent positions relative to block faces
    - Agent actions (push toward face)
    - Block positions and deliveries
    """
    
    FACES = ["up", "down", "left", "right"]
    
    def __init__(self, env):
        """
        Initialize tracker with CUBE environment.
        
        Args:
            env: CoopBlockPush environment instance
        """
        self.env = env
        self.history: List[StepSummary] = []
        self.current_tasks: Dict[str, FaceTaskState] = {}  # task_id -> FaceTaskState
        self.delivered_blocks: List[int] = []
        
        # Initialize tasks from blocks
        self._init_tasks()
    
    def _make_task_id(self, block_id: int, face: str) -> str:
        """Create task ID for a block face."""
        return f"block_{block_id}_{face}"
    
    def _init_tasks(self):
        """Initialize face tasks from current blocks in environment."""
        self.current_tasks.clear()
        for block in self.env._blocks:
            for face in self.FACES:
                task_id = self._make_task_id(block.id, face)
                self.current_tasks[task_id] = FaceTaskState(
                    task_id=task_id,
                    block_id=block.id,
                    face=face,
                    weight=block.weight,
                    block_position=(block.r, block.c),
                    status=TaskStatus.PENDING
                )
    
    def reset(self):
        """Reset tracker for new episode."""
        self.history.clear()
        self.delivered_blocks.clear()
        self._init_tasks()
    
    def update(self, env_step: int, actions: Optional[Dict[str, int]] = None):
        """
        Update task states based on current environment state.
        
        Args:
            env_step: Current environment step
            actions: Dict of agent_name -> action_value (optional, for temporal tracking)
        """
        # Update block positions and check for deliveries
        current_block_ids = {block.id for block in self.env._blocks}
        
        # Check for newly delivered blocks - mark their face tasks as completed
        delivered_this_step = []
        for task_id, task in list(self.current_tasks.items()):
            if task.block_id not in current_block_ids:
                task.status = TaskStatus.COMPLETED
                if task.block_id not in self.delivered_blocks:
                    self.delivered_blocks.append(task.block_id)
                    delivered_this_step.append(task.block_id)
        
        # Update remaining face tasks
        for block in self.env._blocks:
            for face in self.FACES:
                task_id = self._make_task_id(block.id, face)
                if task_id in self.current_tasks:
                    task = self.current_tasks[task_id]
                    task.block_position = (block.r, block.c)
                    
                    # Update spatial: agents adjacent to this face
                    adjacent = self._get_agents_adjacent_to_face(block, face)
                    task.update_spatial(adjacent)
                    
                    # Update temporal: agents pushing toward this face
                    if actions:
                        pushing = self._get_agents_pushing_face(block, face, actions)
                        task.update_temporal(pushing)
                    else:
                        task.update_temporal(set())
                    
                    # Update status
                    if task.get_spatial_count() > 0 or task.get_temporal_count() > 0:
                        task.status = TaskStatus.IN_PROGRESS
                    elif task.status != TaskStatus.COMPLETED:
                        task.status = TaskStatus.PENDING
        
        # Compute metrics
        metrics = self._compute_metrics()
        
        # Create tasks copy for history
        tasks_copy = {
            task_id: FaceTaskState(
                task_id=task.task_id,
                block_id=task.block_id,
                face=task.face,
                weight=task.weight,
                block_position=task.block_position,
                status=task.status,
                adjacent_agents=set(task.adjacent_agents),
                pushing_agents=set(task.pushing_agents),
                participation_ratio=task.participation_ratio,
                participation_met=task.participation_met,
            )
            for task_id, task in self.current_tasks.items()
        }
        
        # Compute step metrics (ma-crafter compatible)
        step_metrics = self._compute_step_metrics(env_step, tasks_copy)
        
        summary = StepSummary(
            env_step=env_step,
            tasks=tasks_copy,
            metrics=metrics,
            step_metrics=step_metrics,
            delivered_blocks=list(self.delivered_blocks),
            successful_collections=delivered_this_step,
            failed_attempts=[],  # Detailed failed push attempts live in env info.
        )
        self.history.append(summary)
    
    def _get_face_cells(self, block, face: str) -> Set[Tuple[int, int]]:
        """Get the cells adjacent to a face of a block (outside the block)."""
        r, c = block.r, block.c
        w = block.weight
        
        cells = set()
        if face == "up":
            # Cells above the block
            for dc in range(w):
                cells.add((r - 1, c + dc))
        elif face == "down":
            # Cells below the block
            for dc in range(w):
                cells.add((r + w, c + dc))
        elif face == "left":
            # Cells to the left of the block
            for dr in range(w):
                cells.add((r + dr, c - 1))
        elif face == "right":
            # Cells to the right of the block
            for dr in range(w):
                cells.add((r + dr, c + w))
        
        return cells
    
    def _get_agents_adjacent_to_face(self, block, face: str) -> Set[str]:
        """Get agents adjacent to a specific face of a block."""
        face_cells = self._get_face_cells(block, face)
        adjacent = set()
        
        for agent_name, agent_pos in self.env._agent_positions.items():
            if tuple(agent_pos) in face_cells:
                adjacent.add(agent_name)
        
        return adjacent
    
    def _get_agents_pushing_face(self, block, face: str, actions: Dict[str, int]) -> Set[str]:
        """Get agents that target/push a specific face of a block this step."""
        # Symbolic runs expose target intent through the wrapper. Prefer that so
        # temporal coordination is independent from spatial adjacency.
        symbolic_intents = getattr(self.env, "_last_symbolic_action_intents", {}) or {}
        if symbolic_intents:
            pushing = set()
            for agent_name, intent in symbolic_intents.items():
                if intent.get("action_type") != "push":
                    continue
                try:
                    block_id = int(intent.get("block_id"))
                except (TypeError, ValueError):
                    continue
                if block_id == block.id and intent.get("face") == face:
                    pushing.add(agent_name)
            return pushing

        # Raw primitive runs do not include a symbolic target, so fall back to
        # physical contact plus the primitive action direction.
        # An agent pushes toward a face if:
        # 1. They are adjacent to that face
        # 2. Their action is toward the block (opposite of the face direction)
        
        # Face -> action that pushes into the block from that face
        # Agent on "left" face pushes "right" (action 4) to push block
        # Agent on "right" face pushes "left" (action 3) to push block
        # Agent on "up" face pushes "down" (action 2) to push block
        # Agent on "down" face pushes "up" (action 1) to push block
        push_action_map = {
            "left": 4,   # Push right
            "right": 3,  # Push left
            "up": 2,     # Push down
            "down": 1,   # Push up
        }
        
        required_action = push_action_map.get(face)
        if required_action is None:
            return set()
        
        adjacent = self._get_agents_adjacent_to_face(block, face)
        pushing = set()
        
        for agent_name in adjacent:
            if agent_name in actions and actions[agent_name] == required_action:
                pushing.add(agent_name)
        
        return pushing
    
    def _compute_metrics(self) -> CooperativeMetrics:
        """Compute current cooperative metrics."""
        metrics = CooperativeMetrics()
        
        participation_ratios = []
        
        for task_id, task in self.current_tasks.items():
            if task.status == TaskStatus.COMPLETED:
                continue
            
            # Spatial
            adjacent = task.get_spatial_count()
            metrics.total_adjacent += adjacent
            metrics.adjacent_by_task[task_id] = adjacent
            
            # Temporal
            pushing = task.get_temporal_count()
            metrics.total_pushing += pushing
            metrics.pushing_by_task[task_id] = pushing
            
            # Participation
            participation_ratios.append(task.participation_ratio)
            if task.participation_met:
                metrics.faces_with_sufficient += 1
        
        # Completion
        metrics.blocks_delivered = len(self.delivered_blocks)
        metrics.blocks_remaining = len({
            task.block_id for task in self.current_tasks.values()
            if task.status != TaskStatus.COMPLETED
        })
        metrics.tasks_total = len([t for t in self.current_tasks.values() if t.status != TaskStatus.COMPLETED])
        metrics.avg_participation_ratio = sum(participation_ratios) / len(participation_ratios) if participation_ratios else 0
        
        return metrics
    
    def _compute_step_metrics(self, env_step: int, current_tasks: Dict[str, FaceTaskState]) -> StepMetrics:
        """Compute step metrics comparing to previous step (ma-crafter compatible)."""
        step_metrics = StepMetrics(env_step=env_step)
        
        # Get previous step's tasks for comparison
        prev_tasks = {}
        if len(self.history) > 0:
            prev_tasks = self.history[-1].tasks
        
        # Task set metrics (count active face tasks)
        current_active = {tid for tid, t in current_tasks.items() if t.status != TaskStatus.COMPLETED}
        prev_active = {tid for tid, t in prev_tasks.items() if t.status != TaskStatus.COMPLETED} if prev_tasks else set()
        
        step_metrics.task_set_size = len(current_active)
        step_metrics.tasks_added = len(current_active - prev_active)
        step_metrics.tasks_removed = len(prev_active - current_active)
        step_metrics.task_set_delta = step_metrics.tasks_added - step_metrics.tasks_removed
        
        # Constraint satisfaction and changes
        participation_ratios = []
        improved_tasks = set()
        worsened_tasks = set()
        
        for task_id, task in current_tasks.items():
            if task.status == TaskStatus.COMPLETED:
                continue
            
            participation_ratios.append(task.participation_ratio)
            
            # Satisfaction counts
            if task.get_spatial_count() > 0:
                step_metrics.spatial_satisfied += 1
            if task.get_temporal_count() > 0:
                step_metrics.temporal_satisfied += 1
            step_metrics.dependency_satisfied += 1  # Always satisfied
            if task.participation_met:
                step_metrics.participation_satisfied += 1
            
            # Total satisfied = all constraints met (spatial, temporal, participation)
            # For now: participation_met is the main constraint
            if task.participation_met:
                step_metrics.total_satisfied += 1
            
            # Compare to previous
            if task_id in prev_tasks:
                prev_task = prev_tasks[task_id]
                prev_spatial = prev_task.get_spatial_count()
                curr_spatial = task.get_spatial_count()
                prev_temporal = prev_task.get_temporal_count()
                curr_temporal = task.get_temporal_count()
                prev_participation = prev_task.participation_ratio
                curr_participation = task.participation_ratio
                
                # Task state (block position) changes
                prev_pos = prev_task.block_position
                curr_pos = task.block_position
                if prev_pos != curr_pos:
                    step_metrics.task_state_changed += 1
                    step_metrics.task_state_changed_tasks.append(task_id)
                    # Check if moved towards goal (rightward = progress)
                    if curr_pos[1] > prev_pos[1]:
                        step_metrics.task_state_progress += 1
                    elif curr_pos[1] < prev_pos[1]:
                        step_metrics.task_state_regress += 1
                
                # Spatial changes
                if curr_spatial > prev_spatial:
                    step_metrics.spatial_improved += 1
                    step_metrics.spatial_improved_tasks.append(task_id)
                    improved_tasks.add(task_id)
                elif curr_spatial < prev_spatial:
                    step_metrics.spatial_worsened += 1
                    step_metrics.spatial_worsened_tasks.append(task_id)
                    worsened_tasks.add(task_id)
                
                # Temporal changes
                if curr_temporal > prev_temporal:
                    step_metrics.temporal_improved += 1
                    step_metrics.temporal_improved_tasks.append(task_id)
                    improved_tasks.add(task_id)
                elif curr_temporal < prev_temporal:
                    step_metrics.temporal_worsened += 1
                    step_metrics.temporal_worsened_tasks.append(task_id)
                    worsened_tasks.add(task_id)
                
                # Participation changes
                if curr_participation > prev_participation:
                    step_metrics.participation_improved += 1
                    step_metrics.participation_improved_tasks.append(task_id)
                    improved_tasks.add(task_id)
                elif curr_participation < prev_participation:
                    step_metrics.participation_worsened += 1
                    step_metrics.participation_worsened_tasks.append(task_id)
                    worsened_tasks.add(task_id)
        
        # Aggregate
        step_metrics.total_improved = len(improved_tasks)
        step_metrics.total_worsened = len(worsened_tasks)
        step_metrics.participation_ratio_avg = sum(participation_ratios) / len(participation_ratios) if participation_ratios else 0
        
        return step_metrics
    
    def get_history(self) -> List[StepSummary]:
        """Get full history of task states."""
        return self.history
    
    def get_current_state(self) -> Dict[str, FaceTaskState]:
        """Get current task states."""
        return self.current_tasks
    
    def get_metrics(self) -> CooperativeMetrics:
        """Get current metrics."""
        return self._compute_metrics()
    
    def get_step_metrics_history(self) -> List[StepMetrics]:
        """Get list of StepMetrics from history (for plotting)."""
        return [summary.step_metrics for summary in self.history if summary.step_metrics]


def convert_to_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format."""
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, Enum):
        return obj.value
    elif hasattr(obj, 'to_dict'):
        return convert_to_serializable(obj.to_dict())
    else:
        return obj


def save_task_states_log(history: List[StepSummary], output_path: str):
    """Save task states history to JSON file."""
    data = [convert_to_serializable(summary.to_dict()) for summary in history]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} task state snapshots to {output_path}")


def plot_metrics_timeline(metrics_list: List[StepMetrics], output_path: Optional[str] = None, show: bool = True):
    """
    Plot cooperative task metrics timeline in row-based format (ma-crafter compatible).
    
    Args:
        metrics_list: List of StepMetrics from task tracker history
        output_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    if not metrics_list:
        print("No metrics to plot")
        return
    
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
    axes[3].set_ylabel('# Faces\n(any +/-)', fontsize=9, fontweight='bold')
    axes[4].set_ylabel('# Faces\n(spatial sat/+/-)', fontsize=9, fontweight='bold')
    axes[5].set_ylabel('# Faces\n(temporal sat/+/-)', fontsize=9, fontweight='bold')
    axes[6].set_ylabel('# Faces\n(depend. sat/+/-)', fontsize=9, fontweight='bold')
    axes[7].set_ylabel('# Faces\n(particip. sat/+/-)', fontsize=9, fontweight='bold')
    
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
    
    plt.suptitle('Cooperative Task Metrics Timeline (CUBE - Face Tasks)', fontsize=14, fontweight='bold')
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
