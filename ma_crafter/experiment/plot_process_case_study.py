"""
Plot one COOP2 episode from the compact per-step process log.

The plot intentionally depends on ``coop2_process_log.json``. Older
reconstruction paths from plan/task/trace logs are removed so the figure uses
the same process formulation that the framework records during execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cognitive.resource_utils import (
    RESOURCE_TARGET_PATTERN,
    normalize_process_resource,
    process_resource_from_text,
)
from cognitive.coop2_attempt_events import extract_attempt_constraint_events
from coop2_repair import COOP2_CONSTRAINT_TYPES

# Display order is top-to-bottom. This makes the y-axis read bottom-to-top as
# lower-score resources, higher-score resources, then tools.
RESOURCE_ORDER = ["tools", "diamond", "iron", "coal", "stone", "wood"]
CONSTRAINTS = [constraint.value for constraint in COOP2_CONSTRAINT_TYPES]
RESOURCE_COLORS = {
    "wood": "#8D6E63",
    "stone": "#8E99A4",
    "coal": "#3C4043",
    "iron": "#C96934",
    "diamond": "#00ACC1",
    "tools": "#7E57C2",
}
CONSTRAINT_COLORS = {
    "spatial": "#4C78A8",
    "temporal": "#F58518",
    "dependency": "#E45756",
}
CONSTRAINT_LINESTYLES = {
    "spatial": "-",
    "temporal": "-",
    "dependency": "--",
}
REPAIR_COLOR = "#FFB000"
REPAIR_EDGE_COLOR = "#3A2A00"
REPAIR_LABEL_FACE = "#FFF4BF"
CONSTRAINT_ABBREVIATIONS = {
    "spatial": "spat",
    "temporal": "temp",
    "dependency": "dep",
}
FALLBACK_RESOURCE_REQUIRED_AGENTS = {
    "wood": 1,
    "tools": 1,
    "stone": 1,
    "coal": 2,
    "iron": 3,
    "diamond": 3,
}
FALLBACK_RESOURCE_REQUIRED_TOOL = {
    "wood": None,
    "tools": None,
    "stone": "wood_pickaxe",
    "coal": "wood_pickaxe",
    "iron": "stone_pickaxe",
    "diamond": "iron_pickaxe",
}
TOOL_TIERS = {
    "wood_pickaxe": 1,
    "stone_pickaxe": 2,
    "iron_pickaxe": 3,
}


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_resource(value: Any) -> Optional[str]:
    resource = normalize_process_resource(value)
    return resource if resource in RESOURCE_ORDER else None


def _safe_step(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def process_log_max_step(process_log: List[Dict[str, Any]]) -> int:
    steps = [_safe_step(entry.get("env_step")) for entry in process_log if isinstance(entry, dict)]
    valid = [step for step in steps if step is not None]
    return max(valid) if valid else 1


def extract_score_events_from_process_log(process_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Read score events directly from the compact per-step process trace."""
    raw_events: List[Dict[str, Any]] = []
    for entry in process_log:
        for event in entry.get("score_events") or []:
            if isinstance(event, dict):
                raw_events.append(event)

    events: List[Dict[str, Any]] = []
    cumulative = 0.0
    for event in sorted(raw_events, key=lambda item: item.get("env_step", 0)):
        step = _safe_step(event.get("env_step"))
        if step is None:
            continue
        score = _safe_float(event.get("score"))
        cumulative += score
        scored_amounts = event.get("scored_amounts") or {}
        resources = [name for name, amount in scored_amounts.items() if amount]
        resource = _normalize_resource(resources[0]) if resources else _normalize_resource(event.get("resource_type"))
        if resource not in RESOURCE_ORDER:
            continue
        events.append(
            {
                "step": step,
                "resource": resource,
                "score": score,
                "cumulative_score": cumulative,
                "participants": ",".join(str(agent) for agent in event.get("participating_agents") or []),
            }
        )
    return events


def build_process_log_series(
    process_log: List[Dict[str, Any]],
    max_step: int,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, str]], Dict[Tuple[int, str, str], float]]:
    """Build plot-ready focus and attempt-time constraint series."""
    focus_records: List[Dict[str, Any]] = []
    focus_by_step: Dict[int, Dict[str, str]] = defaultdict(dict)
    constraint_values: Dict[Tuple[int, str, str], List[float]] = defaultdict(list)

    for entry in process_log:
        step = _safe_step(entry.get("env_step"))
        if step is None or step < 0 or step > max_step:
            continue

        for agent_view in entry.get("agents") or []:
            if not isinstance(agent_view, dict):
                continue
            resource = _normalize_resource(agent_view.get("resource_focus"))
            if resource not in RESOURCE_ORDER:
                continue
            agent_id = str(agent_view.get("agent_id") or "unknown")
            focus_by_step[step][agent_id] = resource
            action = agent_view.get("action") or {}
            focus_records.append(
                {
                    "step": step,
                    "agent_id": agent_id,
                    "plan_id": agent_view.get("plan_id"),
                    "specification": agent_view.get("specification") or "",
                    "resource": resource,
                    "action_type": action.get("action_type") or "",
                    "action_status": ((agent_view.get("action_outcome") or {}).get("status") or ""),
                    "plan_status": agent_view.get("plan_status_after_step") or agent_view.get("plan_status") or "",
                }
            )

        for event in extract_attempt_constraint_events([entry]):
            resource = _normalize_resource(event.get("resource_type"))
            if resource not in RESOURCE_ORDER:
                continue
            for constraint, score in (event.get("constraint_scores") or {}).items():
                constraint = str(constraint)
                if constraint not in CONSTRAINTS:
                    continue
                constraint_values[(step, resource, constraint)].append(min(max(_safe_float(score), 0.0), 1.0))

    constraint_readiness = {
        key: sum(values) / len(values)
        for key, values in constraint_values.items()
        if values
    }
    return focus_records, focus_by_step, constraint_readiness


def build_score_aim_series(
    process_log: List[Dict[str, Any]],
    max_step: int,
) -> Dict[int, float]:
    """Return average configured score value over unique active tasks per step."""
    aims: Dict[int, float] = {}
    for entry in process_log:
        step = _safe_step(entry.get("env_step"))
        if step is None or step < 0 or step > max_step:
            continue
        task_values = _task_score_values_from_agent_specs(entry)
        if task_values:
            aims[step] = sum(task_values.values()) / len(task_values)
    return aims


def build_requirement_series(
    process_log: List[Dict[str, Any]],
    max_step: int,
) -> Dict[int, float]:
    """Return average required agents over active tasks per step.

    Kept as a data helper for tests and backward compatibility; the current
    figure style no longer plots this series directly.
    """
    requirements: Dict[int, float] = {}
    for entry in process_log:
        step = _safe_step(entry.get("env_step"))
        if step is None or step < 0 or step > max_step:
            continue
        values = [
            _safe_float(task.get("required_agents"), default=1.0)
            for task in entry.get("active_tasks") or []
            if isinstance(task, dict)
        ]
        if values:
            requirements[step] = sum(values) / len(values)
    return requirements


def _task_score_values_from_agent_specs(entry: Dict[str, Any]) -> Dict[str, float]:
    scoring_config = _score_config_from_entry(entry)
    known_values = _task_score_values_from_active_tasks(entry, scoring_config)
    known_resource_values = _resource_score_values_from_active_tasks(entry, scoring_config)
    task_values: Dict[str, float] = {}
    active_tasks = _active_spec_tasks(entry)

    for agent_view in entry.get("agents") or []:
        if not isinstance(agent_view, dict):
            continue
        specification = agent_view.get("specification")
        resource = _resource_from_specification(specification) or _normalize_resource(agent_view.get("resource_focus"))
        if resource not in RESOURCE_ORDER:
            continue

        target_ids = _target_ids_from_agent_spec(agent_view)
        if target_ids:
            for target_id in target_ids:
                target_id = str(target_id)
                if target_id in active_tasks:
                    task_values[target_id] = known_values.get(
                        target_id,
                        known_resource_values.get(resource, _fallback_resource_score(resource, scoring_config)),
                    )
            continue

        fallback_key = (
            f"{agent_view.get('agent_id')}:{agent_view.get('plan_id')}:"
            f"{resource}:{specification or ''}"
        )
        task_values[fallback_key] = known_resource_values.get(
            resource,
            _fallback_resource_score(resource, scoring_config),
        )
    return task_values


def _score_config_from_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    score = entry.get("score") or {}
    config = score.get("scoring_config") or {}
    return config if isinstance(config, dict) else {}


def _task_score_values_from_active_tasks(
    entry: Dict[str, Any],
    scoring_config: Dict[str, Any],
) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for task in entry.get("active_tasks") or []:
        task_id = str(task.get("task_id") or "")
        resource = _normalize_resource(task.get("resource_type"))
        if not task_id or resource not in RESOURCE_ORDER:
            continue
        values[task_id] = _resource_score_value(
            resource,
            _safe_float(task.get("required_agents"), default=1.0),
            task.get("required_tool"),
            scoring_config,
        )
    return values


def _resource_score_values_from_active_tasks(
    entry: Dict[str, Any],
    scoring_config: Dict[str, Any],
) -> Dict[str, float]:
    values: Dict[str, List[float]] = defaultdict(list)
    for task in entry.get("active_tasks") or []:
        resource = _normalize_resource(task.get("resource_type"))
        if resource not in RESOURCE_ORDER:
            continue
        values[resource].append(
            _resource_score_value(
                resource,
                _safe_float(task.get("required_agents"), default=1.0),
                task.get("required_tool"),
                scoring_config,
            )
        )
    return {
        resource: sum(resource_values) / len(resource_values)
        for resource, resource_values in values.items()
        if resource_values
    }


def _fallback_resource_score(resource: str, scoring_config: Dict[str, Any]) -> float:
    return _resource_score_value(
        resource,
        FALLBACK_RESOURCE_REQUIRED_AGENTS.get(resource, 1),
        FALLBACK_RESOURCE_REQUIRED_TOOL.get(resource),
        scoring_config,
    )


def _resource_score_value(
    resource: str,
    required_agents: float,
    required_tool: Any,
    scoring_config: Dict[str, Any],
) -> float:
    if resource == "tools":
        return 0.0
    base_values = scoring_config.get("base_values") or {}
    base_value = _safe_float(base_values.get(resource), default=0.0)
    if base_value <= 0:
        return 0.0
    cooperation_multiplier = required_agents if scoring_config.get("scale_by_required_agents", True) else 1.0
    tool_tier_bonus = _safe_float(scoring_config.get("tool_tier_bonus"), default=0.5)
    tool_multiplier = 1.0 + _tool_tier(required_tool) * tool_tier_bonus
    return base_value * max(1.0, cooperation_multiplier) * tool_multiplier


def _tool_tier(required_tool: Any) -> int:
    if required_tool is None:
        return 0
    if isinstance(required_tool, str):
        return TOOL_TIERS.get(required_tool, 0)
    if isinstance(required_tool, (list, tuple, set)):
        return max((_tool_tier(tool) for tool in required_tool), default=0)
    return 0


def _active_spec_tasks(entry: Dict[str, Any]) -> Dict[str, str]:
    """Return unique active tasks from plan specifications mapped to resource."""
    active_tasks: Dict[str, str] = {}
    for agent_view in entry.get("agents") or []:
        if not isinstance(agent_view, dict):
            continue
        specification = agent_view.get("specification")
        resource = _resource_from_specification(specification) or _normalize_resource(agent_view.get("resource_focus"))
        if resource not in RESOURCE_ORDER:
            continue
        target_ids = _target_ids_from_agent_spec(agent_view)
        if target_ids:
            for target_id in target_ids:
                active_tasks[str(target_id)] = resource
            continue
        fallback_key = (
            f"{agent_view.get('agent_id')}:{agent_view.get('plan_id')}:"
            f"{resource}:{specification or ''}"
        )
        active_tasks[fallback_key] = resource
    return active_tasks


def _target_ids_from_specification(specification: Any, resource: Optional[str]) -> List[str]:
    matches = []
    for target_type, target_id in RESOURCE_TARGET_PATTERN.findall(str(specification or "")):
        target_resource = _normalize_resource(target_type)
        if resource is None or target_resource == resource:
            matches.append(str(target_id))
    return matches[:1]


def _target_ids_from_agent_spec(agent_view: Dict[str, Any]) -> List[str]:
    resource = _resource_from_specification(agent_view.get("specification")) or _normalize_resource(agent_view.get("resource_focus"))
    target_ids = _target_ids_from_specification(agent_view.get("specification"), resource)
    if target_ids:
        return target_ids
    active_target_ids = agent_view.get("active_target_ids") or []
    if len(active_target_ids) == 1:
        return [str(active_target_ids[0])]
    return []


def _resource_from_specification(specification: Any) -> Optional[str]:
    text = str(specification or "")
    for target_type, _target_id in RESOURCE_TARGET_PATTERN.findall(text):
        resource = _normalize_resource(target_type)
        if resource in RESOURCE_ORDER:
            return resource
    resource = process_resource_from_text(text)
    return resource if resource in RESOURCE_ORDER else None


def aggregate_constraint_readiness(
    constraint_readiness: Dict[Tuple[int, str, str], float],
) -> Dict[Tuple[int, str], float]:
    grouped: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    for (step, _resource, constraint), value in constraint_readiness.items():
        grouped[(step, constraint)].append(value)
    return {key: sum(values) / len(values) for key, values in grouped.items() if values}


def build_series_rows(
    focus_by_step: Dict[int, Dict[str, str]],
    constraint_readiness: Dict[Tuple[int, str, str], float],
    score_aim_by_step: Dict[int, float],
    score_events: List[Dict[str, Any]],
    max_step: int,
) -> List[Dict[str, Any]]:
    score_by_step_resource: Dict[Tuple[int, str], float] = defaultdict(float)
    for event in score_events:
        score_by_step_resource[(event["step"], event["resource"])] += event["score"]

    rows = []
    for step in range(max_step + 1):
        counts = defaultdict(int)
        for resource in focus_by_step.get(step, {}).values():
            counts[resource] += 1
        for resource in RESOURCE_ORDER:
            rows.append(
                {
                    "step": step,
                    "resource": resource,
                    "agents_working": counts.get(resource, 0),
                    "spatial_readiness": constraint_readiness.get((step, resource, "spatial"), ""),
                    "temporal_readiness": constraint_readiness.get((step, resource, "temporal"), ""),
                    "dependency_readiness": constraint_readiness.get((step, resource, "dependency"), ""),
                    "avg_score_aim_per_task": score_aim_by_step.get(step, ""),
                    "score_event": score_by_step_resource.get((step, resource), 0.0),
                }
            )
    return rows


def build_cumulative_score_series(
    score_events: List[Dict[str, Any]],
    max_step: int,
) -> List[float]:
    """Return cumulative score after each environment step."""
    score_delta_by_step: Dict[int, float] = defaultdict(float)
    for event in score_events:
        step = _safe_step(event.get("step"))
        if step is None or step < 0 or step > max_step:
            continue
        score_delta_by_step[step] += _safe_float(event.get("score"))

    cumulative = 0.0
    values: List[float] = []
    for step in range(max_step + 1):
        cumulative += score_delta_by_step.get(step, 0.0)
        values.append(cumulative)
    return values


def save_series_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "resource",
                "agents_working",
                "spatial_readiness",
                "temporal_readiness",
                "dependency_readiness",
                "avg_score_aim_per_task",
                "score_event",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_agent_focus_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "agent_id",
                "plan_id",
                "resource",
                "action_type",
                "action_status",
                "plan_status",
                "specification",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def extract_repair_markers(coop2_trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return compact repair annotations from the COOP2 trace."""
    failures_by_step: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for event in coop2_trace:
        if not isinstance(event, dict) or event.get("event_type") != "constraint_predicted_failure":
            continue
        step = _safe_step(event.get("env_step"))
        constraint = (event.get("payload") or {}).get("constraint") or {}
        if step is None or not isinstance(constraint, dict):
            continue
        failures_by_step[step].append(constraint)

    markers: List[Dict[str, Any]] = []
    seen_steps = set()
    for event in coop2_trace:
        if not isinstance(event, dict) or event.get("event_type") != "repair_started":
            continue
        step = _safe_step(event.get("env_step"))
        if step is None or step in seen_steps:
            continue
        seen_steps.add(step)
        markers.append(
            {
                "step": step,
                "failures": failures_by_step.get(step, []),
            }
        )
    return markers


def format_repair_marker_label(index: int, marker: Dict[str, Any]) -> str:
    return f"R{index}"


def draw_repair_markers(
    ax: Any,
    repair_markers: List[Dict[str, Any]],
    *,
    annotate: bool = False,
) -> None:
    for index, marker in enumerate(repair_markers, start=1):
        step = marker["step"]
        ax.axvspan(
            step - 0.32,
            step + 0.32,
            facecolor=REPAIR_COLOR,
            edgecolor=REPAIR_EDGE_COLOR,
            linewidth=0.8,
            hatch="///",
            alpha=0.28,
            zorder=1,
        )
        ax.axvline(step, color=REPAIR_EDGE_COLOR, linestyle="-", linewidth=2.1, alpha=0.95, zorder=7)
        ax.plot(
            [step],
            [1.015],
            marker="*",
            markersize=11,
            color=REPAIR_COLOR,
            markeredgecolor=REPAIR_EDGE_COLOR,
            markeredgewidth=0.9,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            zorder=9,
        )
        if annotate:
            ax.annotate(
                format_repair_marker_label(index, marker),
                xy=(step, 1.035),
                xycoords=ax.get_xaxis_transform(),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color=REPAIR_EDGE_COLOR,
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": REPAIR_LABEL_FACE,
                    "edgecolor": REPAIR_EDGE_COLOR,
                    "linewidth": 0.7,
                    "alpha": 0.96,
                },
                clip_on=False,
                zorder=10,
            )


def plot_case_study(
    run_dir: Path,
    output_path: Path,
    csv_path: Optional[Path] = None,
    agent_focus_csv_path: Optional[Path] = None,
    title: Optional[str] = None,
    show_repair: bool = True,
    show_title: bool = True,
    show_summary: bool = True,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    import numpy as np

    process_log_path = run_dir / "coop2_process_log.json"
    process_log = _read_json(process_log_path, [])
    if not isinstance(process_log, list) or not process_log:
        raise FileNotFoundError(
            f"{process_log_path} is required. Re-run the experiment with the current logger."
        )

    coop2_trace = _read_json(run_dir / "coop2_trace.json", []) if show_repair else []
    team_score = _read_json(run_dir / "team_score.json", {})
    max_step = process_log_max_step(process_log)

    focus_records, focus_by_step, constraint_readiness = build_process_log_series(process_log, max_step)
    score_events = extract_score_events_from_process_log(process_log)
    score_aim_by_step = build_score_aim_series(process_log, max_step)
    rows = build_series_rows(
        focus_by_step,
        constraint_readiness,
        score_aim_by_step,
        score_events,
        max_step,
    )
    if csv_path is not None:
        save_series_csv(rows, csv_path)
    if agent_focus_csv_path is not None:
        save_agent_focus_csv(focus_records, agent_focus_csv_path)

    resources = RESOURCE_ORDER
    resource_index = {resource: idx for idx, resource in enumerate(resources)}
    constraint_labels = [CONSTRAINT_ABBREVIATIONS.get(constraint, constraint) for constraint in CONSTRAINTS]
    resource_offset = len(constraint_labels)
    y_rows = constraint_labels + list(resources)
    constraint_index = {constraint: idx for idx, constraint in enumerate(CONSTRAINTS)}

    focus_matrix = np.full((len(y_rows), max_step + 1), np.nan, dtype=float)
    resource_focus = np.zeros((len(resources), max_step + 1), dtype=float)
    for step, agent_map in focus_by_step.items():
        for resource in agent_map.values():
            if resource in resource_index:
                resource_focus[resource_index[resource], step] += 1
    focus_matrix[resource_offset : resource_offset + len(resources), :] = resource_focus

    steps = np.arange(max_step + 1)
    cumulative_scores = build_cumulative_score_series(score_events, max_step)
    termination_step = _safe_step(team_score.get("current_step"))
    if termination_step is None:
        termination_step = max_step
    termination_step = min(max(termination_step, 0), max_step)

    if title is None:
        title = "COOP2 Repair Trace" if show_repair else "COOP2 Process Case Study"

    fig = plt.figure(figsize=(13.6, 5.0))
    if show_title and title:
        fig.text(
            0.095,
            0.94,
            title,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
        )
    ax_bottom = 0.16 if not show_summary else 0.22
    ax_height = 0.70 if not show_title else 0.59
    ax = fig.add_axes([0.095, ax_bottom, 0.69, ax_height])
    focus_cmap = mcolors.LinearSegmentedColormap.from_list(
        "focus",
        ["#FFFFFF", "#D7E8FF", "#2F80ED"],
    )
    focus_cmap.set_bad("#FFFFFF")
    focus_im = ax.imshow(
        np.ma.masked_invalid(focus_matrix),
        aspect="auto",
        interpolation="nearest",
        cmap=focus_cmap,
        vmin=0,
        vmax=max(1, float(np.nanmax(resource_focus))),
        extent=[0, max_step, len(y_rows) - 0.5, -0.5],
    )
    ax.set_ylabel("constraint / resource", fontsize=13)
    ax.set_xlabel("environment step", fontsize=13)
    ax.set_yticks(range(len(y_rows)))
    ax.set_yticklabels(y_rows, fontsize=12)
    for row_idx, tick_label in enumerate(ax.get_yticklabels()):
        if row_idx < resource_offset:
            tick_label.set_color("#0B6E69")
            tick_label.set_fontweight("bold")
    ax.set_xlim(0, max_step)
    ax.set_ylim(len(y_rows) - 0.5, -0.5)
    ax.grid(axis="x", alpha=0.14)
    ax.axhline(resource_offset - 0.5, color="#111111", linewidth=1.6, alpha=0.85, zorder=7)

    satisfaction_cmap = plt.get_cmap("RdYlGn").copy()
    satisfaction_cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    satisfaction_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    attempt_points = []
    attempt_events = extract_attempt_constraint_events(process_log)
    for event_idx, event in enumerate(attempt_events):
        if str(event.get("action_type") or "").lower() != "collect":
            continue
        if not event.get("attempting_agents"):
            continue
        step = _safe_step(event.get("env_step"))
        if step is None or step > termination_step:
            continue
        scores = event.get("constraint_scores") or {}
        offset = ((event_idx % 5) - 2) * 0.08
        for constraint in CONSTRAINTS:
            if constraint not in scores:
                continue
            attempt_points.append(
                (
                    step + offset,
                    constraint_index[constraint],
                    min(max(_safe_float(scores[constraint]), 0.0), 1.0),
                )
            )
    if attempt_points:
        xs, ys, values = zip(*attempt_points)
        ax.scatter(
            xs,
            ys,
            c=values,
            cmap=satisfaction_cmap,
            norm=satisfaction_norm,
            s=44,
            marker="o",
            edgecolor="black",
            linewidth=0.55,
            zorder=8,
        )

    score_ax = ax.twinx()
    score_max = max(1.0, max(cumulative_scores, default=0.0))
    resource_fraction = len(resources) / max(len(y_rows), 1)
    (score_line,) = score_ax.step(
        steps,
        cumulative_scores,
        where="post",
        color="#111111",
        linewidth=2.8,
        label="score",
        zorder=9,
    )
    score_ax.set_ylabel("cumulative score", fontsize=13, labelpad=10)
    score_ax.set_ylim(0, score_max / max(resource_fraction, 1e-6))
    score_ax.set_yticks(np.linspace(0, score_max, 4))
    score_ax.patch.set_alpha(0.0)
    score_ax.grid(False)

    if termination_step < max_step:
        ax.axvspan(termination_step, max_step, color="white", alpha=1.0, zorder=10)
    ax.axvline(termination_step, color="#D32F2F", linestyle="--", linewidth=2.4, alpha=0.9, zorder=11)

    repair_markers = extract_repair_markers(coop2_trace) if show_repair else []
    repair_markers = [marker for marker in repair_markers if marker["step"] < termination_step]
    repair_steps = [marker["step"] for marker in repair_markers]
    if show_repair:
        draw_repair_markers(ax, repair_markers, annotate=True)

    termination_handle = mlines.Line2D(
        [],
        [],
        color="#D32F2F",
        linestyle="--",
        linewidth=2.4,
        label="end",
    )
    handles = [score_line, termination_handle]
    if show_repair and repair_markers:
        repair_handle = mlines.Line2D(
            [],
            [],
            color=REPAIR_EDGE_COLOR,
            marker="*",
            linestyle="-",
            linewidth=2.1,
            markerfacecolor=REPAIR_COLOR,
            markeredgecolor=REPAIR_EDGE_COLOR,
            markersize=8,
            label="COOP2 repair",
        )
        handles.append(repair_handle)
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower left",
        bbox_to_anchor=(0.865, 0.18),
        ncol=1,
        fontsize=11,
        framealpha=0.95,
        borderaxespad=0.0,
        handlelength=1.7,
    )

    cax_focus = fig.add_axes([0.865, 0.58, 0.018, 0.22])
    focus_cbar = fig.colorbar(focus_im, cax=cax_focus)
    focus_cbar.set_label("# agents", fontsize=11)
    focus_cbar.ax.tick_params(labelsize=10)
    sat_cax = fig.add_axes([0.865, 0.34, 0.018, 0.16])
    sat_sm = plt.cm.ScalarMappable(norm=satisfaction_norm, cmap=satisfaction_cmap)
    sat_sm.set_array([])
    sat_cbar = fig.colorbar(sat_sm, cax=sat_cax)
    sat_cbar.set_label("constraint\nsatisfaction", fontsize=11)
    sat_cbar.ax.tick_params(labelsize=10)
    ax.tick_params(axis="both", labelsize=12, width=1.3, length=5.5)
    score_ax.tick_params(axis="y", labelsize=12, width=1.2, length=4.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)

    score = team_score.get("total_score")
    if score is None and process_log:
        score = (process_log[-1].get("score") or {}).get("total_score")
    counts = team_score.get("resource_counts") or (process_log[-1].get("score") or {}).get("resource_counts", {})
    repair_text = f"; repair steps={repair_steps or 'none'}" if show_repair else ""
    subtitle = f"score={_safe_float(score):g}; resources={counts}{repair_text}"
    if show_summary:
        fig.text(0.44, 0.045, subtitle, ha="center", fontsize=11, color="#444444")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, facecolor="white")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot a COOP2 process case study from one run directory.")
    parser.add_argument("run_dir", type=Path, help="Experiment run directory containing coop2_process_log.json.")
    parser.add_argument("--output", type=Path, default=None, help="PNG output path.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path for the plotted time series.")
    parser.add_argument("--agent-focus-csv", type=Path, default=None, help="Optional CSV with per-step per-agent normalized plan focus.")
    parser.add_argument("--title", default=None, help="Optional figure title.")
    parser.add_argument("--no-title", action="store_true", help="Do not draw the figure title.")
    parser.add_argument("--hide-summary", action="store_true", help="Hide the bottom score/resource summary.")
    parser.add_argument(
        "--hide-repair",
        action="store_true",
        help="Hide repair-start markers from coop2_trace.json.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output = args.output or run_dir / "coop2_process_case_study.png"
    csv_output = args.csv or output.with_suffix(".csv")
    agent_focus_csv = args.agent_focus_csv or output.with_name(f"{output.stem}_agent_focus.csv")

    path = plot_case_study(
        run_dir,
        output.resolve(),
        csv_output.resolve(),
        agent_focus_csv.resolve(),
        args.title,
        not args.hide_repair,
        not args.no_title,
        not args.hide_summary,
    )
    print(f"Process case-study plot: {path}")
    print(f"Process case-study series: {csv_output.resolve()}")
    print(f"Agent focus series: {agent_focus_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
