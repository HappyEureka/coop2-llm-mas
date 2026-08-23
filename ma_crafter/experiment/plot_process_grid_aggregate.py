"""Plot aggregated COOP2 process traces across repeated runs."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np

from cognitive.coop2_attempt_events import extract_attempt_constraint_events
from experiment.plot_process_case_study import (
    CONSTRAINTS,
    RESOURCE_ORDER,
    _read_json,
    _safe_float,
    _safe_step,
    build_cumulative_score_series,
    build_process_log_series,
    extract_score_events_from_process_log,
    process_log_max_step,
)
from experiment.plot_process_grid import (
    AXIS_LABEL_FONTSIZE,
    CELL_TITLE_FONTSIZE,
    COLORBAR_LABEL_FONTSIZE,
    COLORBAR_TICK_FONTSIZE,
    COLUMN_FONTSIZE,
    CONSTRAINT_LABEL_COLOR,
    CONSTRAINT_ROW_LABELS,
    DEFAULT_MODELS,
    LEGEND_FONTSIZE,
    MODEL_LABELS,
    ROW_FONTSIZE,
    TICK_FONTSIZE,
    _focus_matrix,
    _model_label,
    _run_key,
    _topology_label,
)


def _find_runs(
    results_dir: Path,
    model: str,
    topology: str,
    agents: int,
    seed: Optional[int],
    repair: str,
) -> List[Path]:
    seed_part = "*" if seed is None else str(seed)
    pattern = f"{topology}_agents{agents}_repair_{repair}_seed{seed_part}_*"
    runs = []
    for run_dir in results_dir.glob(pattern):
        if _run_key(run_dir) == (model, topology):
            runs.append(run_dir)
    return sorted(runs, key=lambda path: path.stat().st_mtime)


def _load_trace(run_dir: Path) -> Dict[str, Any]:
    process_log = _read_json(run_dir / "coop2_process_log.json", [])
    if not isinstance(process_log, list) or not process_log:
        raise FileNotFoundError(f"{run_dir / 'coop2_process_log.json'} is required.")
    max_step = process_log_max_step(process_log)
    _focus_records, focus_by_step, _constraint_readiness = build_process_log_series(
        process_log,
        max_step,
    )
    team_score = _read_json(run_dir / "team_score.json", {})
    termination_step = _safe_step(team_score.get("current_step")) or max_step
    return {
        "run_dir": run_dir,
        "team_score": team_score,
        "max_step": max(max_step, termination_step),
        "termination_step": termination_step,
        "focus_by_step": focus_by_step,
        "attempt_events": extract_attempt_constraint_events(process_log),
        "score_events": extract_score_events_from_process_log(process_log),
    }


def _mean_ci(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0, ddof=1)
    counts = np.sum(~np.isnan(values), axis=0)
    ci = np.divide(
        1.96 * std,
        np.sqrt(np.maximum(counts, 1)),
        out=np.zeros_like(mean, dtype=float),
        where=counts > 1,
    )
    return mean, ci


def _aggregate_focus(
    traces: List[Dict[str, Any]],
    resource_index: Dict[str, int],
    resources: List[str],
    global_max_step: int,
) -> np.ndarray:
    matrices = []
    for trace in traces:
        trace_max = int(trace["max_step"])
        matrix = np.full((len(resources), global_max_step + 1), np.nan, dtype=float)
        local = _focus_matrix(trace["focus_by_step"], resource_index, trace_max)
        copy_until = min(trace_max, global_max_step)
        matrix[:, : copy_until + 1] = local[:, : copy_until + 1]
        matrices.append(matrix)
    if not matrices:
        return np.full((len(resources), global_max_step + 1), np.nan, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.stack(matrices, axis=0), axis=0)


def _aggregate_scores(
    traces: List[Dict[str, Any]],
    global_max_step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    series = []
    for trace in traces:
        values = build_cumulative_score_series(trace["score_events"], global_max_step)
        series.append(np.array(values, dtype=float))
    if not series:
        empty = np.full(global_max_step + 1, np.nan, dtype=float)
        return empty, empty
    return _mean_ci(np.stack(series, axis=0))


def _aggregate_attempt_points(
    traces: List[Dict[str, Any]],
) -> Dict[Tuple[int, str], Dict[str, float]]:
    values: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    for trace in traces:
        for event in trace.get("attempt_events") or []:
            if str(event.get("action_type") or "").lower() != "collect":
                continue
            if not event.get("attempting_agents"):
                continue
            step = _safe_step(event.get("env_step"))
            if step is None:
                continue
            scores = event.get("constraint_scores") or {}
            for constraint in CONSTRAINTS:
                if constraint not in scores:
                    continue
                values[(step, constraint)].append(min(max(_safe_float(scores[constraint]), 0.0), 1.0))
    return {
        key: {
            "mean": float(sum(items) / len(items)),
            "count": float(len(items)),
        }
        for key, items in values.items()
        if items
    }


def plot_aggregate_grid(
    results_dir: Path,
    output_path: Path,
    models: List[str],
    topologies: List[str],
    agents: int,
    seed: Optional[int],
    repair: str,
) -> Path:
    traces_by_cell: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    missing: List[Tuple[str, str]] = []
    for topology in topologies:
        for model in models:
            runs = _find_runs(results_dir, model, topology, agents, seed, repair)
            if not runs:
                missing.append((model, topology))
                continue
            traces_by_cell[(model, topology)] = [_load_trace(run) for run in runs]

    if not traces_by_cell:
        raise FileNotFoundError("No matching traces found.")

    cell_mean_ends = [
        float(np.mean([int(trace["termination_step"]) for trace in traces]))
        for traces in traces_by_cell.values()
        if traces
    ]
    global_max_step = max(1, int(np.ceil(max(cell_mean_ends))))
    row_max_scores: Dict[str, float] = {}
    for topology in topologies:
        row_scores = [
            max(build_cumulative_score_series(trace["score_events"], global_max_step))
            for model in models
            for trace in traces_by_cell.get((model, topology), [])
        ]
        row_max_scores[topology] = max(1.0, max(row_scores, default=0.0))
    resources = list(RESOURCE_ORDER)
    constraint_rows = [CONSTRAINT_ROW_LABELS.get(constraint, constraint) for constraint in CONSTRAINTS]
    resource_offset = len(constraint_rows)
    y_rows = constraint_rows + resources
    focus_resource_index = {resource: idx for idx, resource in enumerate(resources)}
    constraint_row_index = {constraint: idx for idx, constraint in enumerate(CONSTRAINTS)}

    nrows = len(topologies)
    ncols = len(models)
    fig_width = 9.6 if ncols == 1 else 7.2 * ncols + 2.2
    fig_height = 3.45 * nrows + 1.1
    fig = plt.figure(figsize=(fig_width, fig_height))
    left = 0.17 if ncols == 1 else 0.105
    right = 0.82 if ncols == 1 else 0.872
    top = 0.88
    bottom = 0.08
    outer = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        hspace=0.14,
        wspace=0.045,
    )

    for col_idx, model in enumerate(models):
        x = left + (col_idx + 0.5) * (right - left) / max(ncols, 1)
        fig.text(x, 0.965, _model_label(model), ha="center", va="center", fontsize=COLUMN_FONTSIZE, fontweight="bold")
    if ncols > 1:
        for topo_idx, topology in enumerate(topologies):
            y = top - (topo_idx + 0.5) * (top - bottom) / max(nrows, 1)
            fig.text(0.045, y, _topology_label(topology), ha="center", va="center", rotation=90, fontsize=ROW_FONTSIZE, fontweight="bold")

    focus_cmap = mcolors.LinearSegmentedColormap.from_list("focus", ["#FFFFFF", "#D7E8FF", "#2F80ED"])
    focus_cmap.set_bad("#FFFFFF")
    satisfaction_cmap = plt.get_cmap("RdYlGn").copy()
    satisfaction_cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    satisfaction_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    score_handle = None
    ci_handle = None
    termination_handle = mlines.Line2D([], [], color="#D32F2F", linestyle="--", linewidth=2.0, label="mean end")
    last_im = None

    for topo_idx, topology in enumerate(topologies):
        for col_idx, model in enumerate(models):
            ax = fig.add_subplot(outer[topo_idx, col_idx])
            traces = traces_by_cell.get((model, topology), [])
            if not traces:
                ax.axis("off")
                ax.set_title("missing")
                continue

            final_scores = [_safe_float(trace["team_score"].get("total_score")) for trace in traces]
            final_steps = [int(trace["termination_step"]) for trace in traces]
            mean_end = float(np.mean(final_steps))
            step_values = np.arange(global_max_step + 1)
            after_cutoff = step_values > mean_end

            focus = _aggregate_focus(traces, focus_resource_index, resources, global_max_step)
            focus[:, after_cutoff] = np.nan
            panel_matrix = np.full((len(y_rows), global_max_step + 1), np.nan, dtype=float)
            panel_matrix[resource_offset : resource_offset + len(resources), :] = focus
            last_im = ax.imshow(
                np.ma.masked_invalid(panel_matrix),
                aspect="auto",
                interpolation="nearest",
                cmap=focus_cmap,
                vmin=0,
                vmax=agents,
                extent=[0, global_max_step, len(y_rows) - 0.5, -0.5],
            )

            score_mean, score_ci = _aggregate_scores(traces, global_max_step)
            score_mean[after_cutoff] = np.nan
            score_ci[after_cutoff] = np.nan
            ax.set_title(
                f"{_topology_label(topology)}: n={len(traces)}, score={np.mean(final_scores):.1f}, end={mean_end:.1f}",
                fontsize=CELL_TITLE_FONTSIZE,
                fontweight="bold",
                loc="left",
                pad=2,
            )

            ax.set_yticks(range(len(y_rows)))
            if col_idx == 0:
                ax.set_ylabel("resource / constraint", fontsize=AXIS_LABEL_FONTSIZE)
                ax.set_yticklabels(y_rows)
                for row_idx, tick_label in enumerate(ax.get_yticklabels()):
                    if row_idx < resource_offset:
                        tick_label.set_color(CONSTRAINT_LABEL_COLOR)
                        tick_label.set_fontweight("bold")
            else:
                ax.set_yticklabels([])

            ax.set_xlim(0, global_max_step)
            ax.set_ylim(len(y_rows) - 0.5, -0.5)
            ax.grid(axis="x", alpha=0.14)

            attempt_points = _aggregate_attempt_points(traces)
            constraint_matrix = np.full((len(y_rows), global_max_step + 1), np.nan, dtype=float)
            if attempt_points:
                for (step, constraint), point in sorted(attempt_points.items()):
                    if step > mean_end:
                        continue
                    constraint_matrix[constraint_row_index[constraint], step] = point["mean"]
                ax.imshow(
                    np.ma.masked_invalid(constraint_matrix),
                    aspect="auto",
                    interpolation="nearest",
                    cmap=satisfaction_cmap,
                    norm=satisfaction_norm,
                    extent=[0, global_max_step, len(y_rows) - 0.5, -0.5],
                    alpha=0.52,
                    zorder=3,
                )

            score_ax = ax.twinx()
            x_values = np.arange(global_max_step + 1)
            resource_fraction = len(resources) / max(len(y_rows), 1)
            score_axis_max = row_max_scores.get(topology, 1.0)
            score_ax.set_ylim(0, score_axis_max / max(resource_fraction, 1e-6))
            score_ax.set_yticks(np.linspace(0, score_axis_max, 4))
            ci = score_ax.fill_between(
                x_values,
                np.maximum(score_mean - score_ci, 0),
                score_mean + score_ci,
                step="post",
                color="#111111",
                alpha=0.22,
                linewidth=0,
                label="95% CI",
                zorder=6,
            )
            (score_line,) = score_ax.step(
                x_values,
                score_mean,
                where="post",
                color="#111111",
                linewidth=2.8,
                label="mean score",
                zorder=7,
            )
            score_handle = score_handle or score_line
            ci_handle = ci_handle or ci
            if col_idx == len(models) - 1 and ncols > 1:
                score_ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, width=1.2, length=4.5)
            else:
                score_ax.set_yticklabels([])
                score_ax.tick_params(axis="y", length=0)
            score_ax.patch.set_alpha(0.0)
            score_ax.grid(False)
            score_ax.spines["right"].set_linewidth(1.25)

            if mean_end < global_max_step:
                ax.axvspan(mean_end, global_max_step, color="white", alpha=1.0, zorder=8)
            ax.hlines(
                resource_offset - 0.5,
                xmin=0,
                xmax=min(mean_end, global_max_step),
                color="#111111",
                linewidth=2.0,
                alpha=0.86,
                zorder=9,
            )
            ax.axvline(mean_end, color="#D32F2F", linestyle="--", linewidth=2.8, alpha=0.88, zorder=9)

            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=1.55, length=6.2)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
            for spine in ax.spines.values():
                spine.set_linewidth(1.45)
            if topo_idx == len(topologies) - 1:
                ax.set_xlabel("environment step", fontsize=AXIS_LABEL_FONTSIZE)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)

    legend_items = [termination_handle]
    if score_handle is not None:
        legend_items.append(score_handle)
    if ci_handle is not None:
        legend_items.append(ci_handle)
    for handle in legend_items:
        if hasattr(handle, "set_label"):
            label = handle.get_label()
            if label == "mean end":
                handle.set_label("end")
            elif label == "min end":
                handle.set_label("end")
            elif label == "mean score":
                handle.set_label("score")
    fig.legend(
        legend_items,
        [handle.get_label() for handle in legend_items],
        loc="lower left",
        bbox_to_anchor=(right + 0.02, bottom),
        ncol=1,
        fontsize=LEGEND_FONTSIZE,
        framealpha=0.95,
        borderaxespad=0.0,
        handlelength=1.7,
    )

    if last_im is not None:
        cbar_x = right + 0.04
        cax = fig.add_axes([cbar_x, 0.55, 0.018, 0.26])
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label("mean # agents", fontsize=COLORBAR_LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
        sat_cax = fig.add_axes([cbar_x, 0.29, 0.018, 0.18])
        sat_sm = plt.cm.ScalarMappable(norm=satisfaction_norm, cmap=satisfaction_cmap)
        sat_sm.set_array([])
        sat_cbar = fig.colorbar(sat_sm, cax=sat_cax)
        sat_cbar.set_label("mean constraint\nsatisfaction", fontsize=COLORBAR_LABEL_FONTSIZE)
        sat_cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    if missing:
        missing_text = ", ".join(f"{MODEL_LABELS.get(model, model)}/{_topology_label(topology)}" for model, topology in missing)
        fig.text(0.5, 0.025, f"Missing traces: {missing_text}", ha="center", fontsize=TICK_FONTSIZE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot aggregated COOP2 process traces.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--topologies", nargs="+", default=["individual", "centralized", "broadcast_chain"])
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--repair", choices=["off", "on"], default="off")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    seed = None if args.all_seeds else args.seed
    output = args.output or results_dir / "figures" / f"coop2_process_aggregate_repair_{args.repair}.png"
    path = plot_aggregate_grid(
        results_dir=results_dir,
        output_path=output.resolve(),
        models=args.models,
        topologies=args.topologies,
        agents=args.agents,
        seed=seed,
        repair=args.repair,
    )
    print(f"Aggregate COOP2 process grid: {path}")
    pdf_path = path.with_suffix(".pdf")
    if pdf_path.exists():
        print(f"Aggregate COOP2 process grid PDF: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
