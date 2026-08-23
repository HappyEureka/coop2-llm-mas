"""CUBE COOP2 attempt-event extraction for tables and process plots."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


ATTEMPT_CONSTRAINTS = ("spatial", "temporal", "dependency")
WEIGHT_PATTERN = re.compile(r"weight[_\s-]*(\d+)|w\s*=?\s*(\d+)", re.IGNORECASE)


def normalize_cube_target(value: Any, *, weight: Any = None) -> Optional[str]:
    """Normalize CUBE task labels to ``weight_N``."""
    parsed_weight = _safe_int(weight)
    if parsed_weight is not None and parsed_weight > 0:
        return f"weight_{parsed_weight}"

    text = str(value or "").strip().lower()
    if not text:
        return None
    match = WEIGHT_PATTERN.search(text)
    if match:
        parsed = next((item for item in match.groups() if item), None)
        if parsed:
            return f"weight_{int(parsed)}"
    if text in {"block", "push", "push_block"}:
        return "block"
    return text.replace(" ", "_")


def target_label(target: str) -> str:
    match = WEIGHT_PATTERN.search(str(target or ""))
    if match:
        parsed = next((item for item in match.groups() if item), None)
        if parsed:
            return f"w={int(parsed)}"
    return str(target or "block")


def extract_attempt_constraint_events(process_log: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return constraint scores for concrete CUBE push attempts."""
    events: List[Dict[str, Any]] = []
    for entry in process_log or []:
        if not isinstance(entry, dict):
            continue
        stored = entry.get("constraint_attempts")
        if isinstance(stored, list) and stored:
            events.extend(_events_from_stored_attempts(entry, stored))
        else:
            events.extend(_events_from_action_outcomes(entry))
    return events


def aggregate_attempt_violation_rates(events: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate attempt-event scores into violation-rate columns."""
    event_list = list(events or [])
    totals = {constraint: 0 for constraint in ATTEMPT_CONSTRAINTS}
    deficits = {constraint: 0.0 for constraint in ATTEMPT_CONSTRAINTS}

    for event in event_list:
        scores = event.get("constraint_scores") or {}
        for constraint in ATTEMPT_CONSTRAINTS:
            if constraint not in scores:
                continue
            totals[constraint] += 1
            deficits[constraint] += max(0.0, 1.0 - min(_safe_float(scores[constraint]), 1.0))

    rates: Dict[str, float] = {"attempt_constraint_events": float(len(event_list))}
    for constraint in ATTEMPT_CONSTRAINTS:
        total = totals[constraint]
        violation_rate = deficits[constraint] / total if total else 0.0
        rates[f"{constraint}_violation_rate"] = violation_rate
        rates[f"{constraint}_score"] = 1.0 - violation_rate if total else 0.0
        rates[f"{constraint}_checks"] = total
    return rates


def _events_from_stored_attempts(entry: Dict[str, Any], attempts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events = []
    step = _safe_int(entry.get("env_step"))
    if step is None:
        return events

    outcomes_by_task = _push_outcomes_by_task(entry)
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        task_id = str(attempt.get("task_id") or "")
        if not task_id:
            continue
        scores = _score_map(attempt.get("constraint_scores") or attempt.get("scores"))
        if not scores:
            continue
        required_agents = max(_safe_int(attempt.get("required_agents"), 1) or 1, 1)
        target = normalize_cube_target(
            attempt.get("resource_type") or attempt.get("target_type"),
            weight=attempt.get("weight") or required_agents,
        )
        agents = sorted(str(agent) for agent in attempt.get("agents") or [])
        task_outcomes = outcomes_by_task.get(task_id, [])
        successful = any(outcome.get("status") == "success" for outcome in task_outcomes)
        if not agents:
            agents = sorted(
                str(outcome.get("agent_id"))
                for outcome in task_outcomes
                if outcome.get("agent_id") is not None
            )
        event = {
            "env_step": step,
            "action_type": "push",
            "task_id": task_id,
            "attempt_label": f"push({target or 'block'}#{task_id})",
            "resource_type": target,
            "target_resource_type": target,
            "target_type": target,
            "required_agents": required_agents,
            "block_id": attempt.get("block_id"),
            "face": attempt.get("face"),
            "attempting_agents": agents,
            "successful_agents": sorted(
                str(outcome.get("agent_id"))
                for outcome in task_outcomes
                if outcome.get("status") == "success" and outcome.get("agent_id") is not None
            ),
            "success": successful,
            "constraint_scores": scores,
            "constraint_violations": {
                name: max(0.0, 1.0 - min(value, 1.0))
                for name, value in scores.items()
            },
        }
        events.append(event)
    return events


def _events_from_action_outcomes(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    step = _safe_int(entry.get("env_step"))
    if step is None:
        return []

    constraints = _constraints_by_task(entry)
    active_tasks = _active_tasks_by_id(entry)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for outcome in (entry.get("action_outcomes") or {}).values():
        if not isinstance(outcome, dict) or str(outcome.get("action_type")) != "push":
            continue
        task_id = outcome.get("task_id")
        if task_id is None:
            continue
        grouped[str(task_id)].append(outcome)

    events = []
    for task_id, outcomes in sorted(grouped.items()):
        task = active_tasks.get(task_id, {})
        metadata = _merged_metadata(outcomes)
        required_agents = max(
            _safe_int(
                task.get("required_agents")
                or metadata.get("required_agents")
                or metadata.get("weight"),
                1,
            )
            or 1,
            1,
        )
        scores = {
            constraint: _constraint_score(constraints.get(task_id, {}).get(constraint), required_agents)
            for constraint in ATTEMPT_CONSTRAINTS
            if constraints.get(task_id, {}).get(constraint) is not None
        }
        scores.setdefault("dependency", 1.0)
        if "temporal" not in scores:
            scores["temporal"] = len(outcomes) / required_agents
        if "spatial" not in scores:
            scores["spatial"] = 1.0 if any(outcome.get("status") == "success" for outcome in outcomes) else 0.0
        scores = _score_map(scores)
        target = normalize_cube_target(
            task.get("resource_type") or metadata.get("resource_type") or metadata.get("target_type"),
            weight=required_agents,
        )
        events.append(
            {
                "env_step": step,
                "action_type": "push",
                "task_id": task_id,
                "attempt_label": f"push({target or 'block'}#{task_id})",
                "resource_type": target,
                "target_resource_type": target,
                "target_type": target,
                "required_agents": required_agents,
                "block_id": task.get("block_id") or metadata.get("block_id"),
                "face": task.get("face") or metadata.get("face"),
                "attempting_agents": sorted(str(outcome.get("agent_id")) for outcome in outcomes),
                "successful_agents": sorted(
                    str(outcome.get("agent_id"))
                    for outcome in outcomes
                    if outcome.get("status") == "success"
                ),
                "success": any(outcome.get("status") == "success" for outcome in outcomes),
                "constraint_scores": scores,
                "constraint_violations": {
                    name: max(0.0, 1.0 - min(value, 1.0))
                    for name, value in scores.items()
                },
            }
        )
    return events


def _push_outcomes_by_task(entry: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for outcome in (entry.get("action_outcomes") or {}).values():
        if not isinstance(outcome, dict) or str(outcome.get("action_type")) != "push":
            continue
        task_id = outcome.get("task_id")
        if task_id is not None:
            grouped[str(task_id)].append(outcome)
    return grouped


def _constraints_by_task(entry: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for result in entry.get("observed_constraints") or []:
        if not isinstance(result, dict):
            continue
        task_id = result.get("task_id")
        constraint = result.get("constraint_type")
        if task_id is None or constraint not in ATTEMPT_CONSTRAINTS:
            continue
        grouped[str(task_id)][str(constraint)] = result
    return grouped


def _active_tasks_by_id(entry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    tasks: Dict[str, Dict[str, Any]] = {}
    for task in entry.get("active_tasks") or []:
        if not isinstance(task, dict) or task.get("task_id") is None:
            continue
        task_copy = dict(task)
        task_copy["resource_type"] = normalize_cube_target(
            task.get("resource_type") or task.get("target_type"),
            weight=task.get("required_agents"),
        )
        tasks[str(task["task_id"])] = task_copy
    return tasks


def _merged_metadata(outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for outcome in outcomes:
        metadata = outcome.get("metadata")
        if isinstance(metadata, dict):
            merged.update(metadata)
    return merged


def _score_map(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    scores = {}
    for constraint in ATTEMPT_CONSTRAINTS:
        if constraint in value:
            scores[constraint] = max(0.0, _safe_float(value[constraint]))
    return scores


def _constraint_score(result: Optional[Dict[str, Any]], required_agents: int) -> float:
    if not isinstance(result, dict):
        return 0.0
    if result.get("score") is not None:
        return _safe_float(result.get("score"))
    agents = result.get("agents")
    if isinstance(agents, list):
        return len(agents) / max(required_agents, 1)
    return 1.0 if result.get("satisfied") else 0.0


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
