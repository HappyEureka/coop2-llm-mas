"""Shared CUBE COOP2 process-table and process-plot helpers."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from cognitive.coop2_attempt_events import (
    ATTEMPT_CONSTRAINTS,
    extract_attempt_constraint_events,
    normalize_cube_target,
    target_label,
)


LLAMA_SCOUT_MODEL = "Llama-4-Scout-17B-16E-Instruct"
DEFAULT_MODELS = ["gpt-5.4-mini", LLAMA_SCOUT_MODEL, "gpt-5.4"]
MODEL_LABELS = {
    LLAMA_SCOUT_MODEL: "Llama-Scout",
}
TOPOLOGY_ORDER = ["individual", "centralized", "broadcast_chain"]
TOPOLOGY_LABELS = {
    "individual": "Individual",
    "centralized": "Centralized",
    "broadcast_chain": "Broadcast Chain",
}
CONSTRAINT_LABELS = {
    "spatial": "spat.",
    "temporal": "temp.",
    "dependency": "dep.",
}
TARGET_ORDER = ["weight_3", "weight_2", "weight_1"]
RESULT_RE = re.compile(
    r"^(?P<topology>individual|centralized|broadcast_chain)_agents(?P<agents>\d+)_"
    r"repair_(?P<repair>on|off)_seed(?P<seed>\d+)_"
)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def process_log_max_step(process_log: Iterable[Dict[str, Any]]) -> int:
    steps = [
        safe_int(entry.get("env_step"), -1)
        for entry in process_log or []
        if isinstance(entry, dict)
    ]
    valid = [step for step in steps if step >= 0]
    return max(valid) if valid else 0


def score_from_team_score(team_score: Dict[str, Any]) -> float:
    return safe_float(
        team_score.get("total_score", team_score.get("score", team_score.get("team_score", 0.0)))
    )


def run_model_topology(run_dir: Path) -> Optional[Tuple[str, str]]:
    usage = read_json(run_dir / "llm_usage.json", {})
    model = usage.get("model")
    if not model:
        return None
    topology = usage.get("topology") or run_dir.name.split("_agents", 1)[0].split("_", 1)[0]
    return str(model), str(topology)


def run_metadata(run_dir: Path) -> Optional[Dict[str, Any]]:
    usage = read_json(run_dir / "llm_usage.json", {})
    if not isinstance(usage, dict) or not usage.get("model"):
        return None
    match = RESULT_RE.match(run_dir.name)
    topology = str(usage.get("topology") or (match.group("topology") if match else run_dir.name.split("_", 1)[0]))
    agents = safe_int(
        usage.get("n_agents")
        or usage.get("agents")
        or (len(usage.get("per_agent") or {}) if isinstance(usage.get("per_agent"), dict) else 0)
        or (match.group("agents") if match else 0)
    )
    return {
        "topology": topology,
        "agents": agents,
        "model": str(usage.get("model")),
        "repair": match.group("repair") if match else str(usage.get("repair", "off")),
        "seed": safe_int(match.group("seed") if match else usage.get("seed", 0)),
    }


def find_runs(
    results_dir: Path,
    *,
    model: str,
    topology: str,
    agents: int,
    seed: Optional[int],
    repair: str,
) -> List[Path]:
    runs = []
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        metadata = run_metadata(run_dir)
        if metadata is None:
            continue
        if metadata["model"] != model or metadata["topology"] != topology:
            continue
        if metadata["agents"] != agents or metadata["repair"] != repair:
            continue
        if seed is not None and metadata["seed"] != seed:
            continue
        if not (run_dir / "coop2_process_log.json").exists():
            continue
        runs.append(run_dir)
    return sorted(runs, key=lambda path: path.stat().st_mtime)


def extract_score_events(process_log: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    raw_events: List[Dict[str, Any]] = []
    for entry in process_log or []:
        if not isinstance(entry, dict):
            continue
        for event in entry.get("score_events") or []:
            if isinstance(event, dict):
                raw_events.append(event)

    events = []
    cumulative = 0.0
    for event in sorted(raw_events, key=lambda item: item.get("env_step", 0)):
        step = safe_int(event.get("env_step"), -1)
        if step < 0:
            continue
        value = safe_float(event.get("value", event.get("score", 0.0)))
        if value <= 0:
            continue
        target = normalize_cube_target(
            event.get("resource_type") or event.get("target_type"),
            weight=event.get("weight"),
        )
        cumulative += value
        events.append(
            {
                "step": step,
                "target": target,
                "score": value,
                "cumulative_score": cumulative,
                "block_id": event.get("block_id"),
            }
        )
    return events


def build_cumulative_score_series(score_events: List[Dict[str, Any]], max_step: int) -> List[float]:
    score_delta_by_step: Dict[int, float] = defaultdict(float)
    for event in score_events:
        step = safe_int(event.get("step"), -1)
        if 0 <= step <= max_step:
            score_delta_by_step[step] += safe_float(event.get("score"))

    cumulative = 0.0
    values: List[float] = []
    for step in range(max_step + 1):
        cumulative += score_delta_by_step.get(step, 0.0)
        values.append(cumulative)
    return values


def build_focus_by_step(process_log: Iterable[Dict[str, Any]], max_step: int) -> Dict[int, Dict[str, str]]:
    focus_by_step: Dict[int, Dict[str, str]] = defaultdict(dict)
    for entry in process_log or []:
        if not isinstance(entry, dict):
            continue
        step = safe_int(entry.get("env_step"), -1)
        if step < 0 or step > max_step:
            continue
        active_tasks = {
            str(task.get("task_id")): task
            for task in entry.get("active_tasks") or []
            if isinstance(task, dict) and task.get("task_id") is not None
        }
        block_to_target = {
            str(task.get("block_id")): normalize_cube_target(
                task.get("resource_type") or task.get("target_type"),
                weight=task.get("required_agents"),
            )
            for task in active_tasks.values()
            if task.get("block_id") is not None
        }
        for agent_view in entry.get("agents") or []:
            if not isinstance(agent_view, dict):
                continue
            target = _target_from_agent_view(agent_view, active_tasks, block_to_target)
            if target:
                focus_by_step[step][str(agent_view.get("agent_id") or "unknown")] = target
    return focus_by_step


def build_target_rows(process_log: Iterable[Dict[str, Any]], score_events: List[Dict[str, Any]]) -> List[str]:
    seen = set(TARGET_ORDER)
    for entry in process_log or []:
        for task in (entry.get("active_tasks") or []) if isinstance(entry, dict) else []:
            if not isinstance(task, dict):
                continue
            target = normalize_cube_target(
                task.get("resource_type") or task.get("target_type"),
                weight=task.get("required_agents"),
            )
            if target:
                seen.add(target)
    for event in score_events:
        if event.get("target"):
            seen.add(event["target"])
    weights = sorted(
        [target for target in seen if str(target).startswith("weight_")],
        key=lambda item: safe_int(str(item).split("_")[-1]),
        reverse=True,
    )
    rest = sorted(target for target in seen if target not in weights and target != "block")
    if "block" in seen:
        rest.append("block")
    return weights + rest


def load_trace(run_dir: Path) -> Dict[str, Any]:
    process_log = read_json(run_dir / "coop2_process_log.json", [])
    if not isinstance(process_log, list) or not process_log:
        raise FileNotFoundError(f"{run_dir / 'coop2_process_log.json'} is required.")
    team_score = read_json(run_dir / "team_score.json", {})
    max_step = process_log_max_step(process_log)
    score_events = extract_score_events(process_log)
    termination_step = safe_int(team_score.get("current_step"), max_step)
    return {
        "run_dir": run_dir,
        "process_log": process_log,
        "team_score": team_score,
        "max_step": max(max_step, termination_step),
        "termination_step": termination_step,
        "focus_by_step": build_focus_by_step(process_log, max_step),
        "attempt_events": extract_attempt_constraint_events(process_log),
        "score_events": score_events,
    }


def model_label(model: str) -> str:
    return MODEL_LABELS.get(str(model), str(model))


def topology_label(topology: str) -> str:
    return TOPOLOGY_LABELS.get(str(topology), str(topology).title())


def target_tick_label(target: str) -> str:
    return target_label(target)


def _target_from_agent_view(
    agent_view: Dict[str, Any],
    active_tasks: Dict[str, Dict[str, Any]],
    block_to_target: Dict[str, str],
) -> Optional[str]:
    task_id = agent_view.get("task_id")
    if task_id is not None and str(task_id) in active_tasks:
        task = active_tasks[str(task_id)]
        return normalize_cube_target(
            task.get("resource_type") or task.get("target_type"),
            weight=task.get("required_agents"),
        )

    focus = normalize_cube_target(agent_view.get("resource_focus"))
    if focus and focus != "block":
        return focus

    for source in (agent_view.get("action") or {}, agent_view.get("symbolic_action") or {}):
        if not isinstance(source, dict):
            continue
        args = source.get("args") if isinstance(source.get("args"), dict) else {}
        block_id = source.get("block_id", args.get("block_id", args.get("target_id")))
        if block_id is not None and str(block_id) in block_to_target:
            return block_to_target[str(block_id)]

    return focus
