"""
Plot a compact COOP2 process grid across models and topologies.

Each cell is one aligned panel:
- resource focus heatmap with cumulative score line
- observed constraint satisfaction dots in constraint rows
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment.plot_process_case_study import (  # noqa: E402
    CONSTRAINTS,
    CONSTRAINT_COLORS,
    RESOURCE_ORDER,
    _read_json,
    _safe_float,
    _safe_step,
    aggregate_constraint_readiness,
    build_cumulative_score_series,
    build_process_log_series,
    extract_score_events_from_process_log,
    process_log_max_step,
)
from cognitive.coop2_attempt_events import extract_attempt_constraint_events  # noqa: E402

LLAMA_SCOUT_MODEL = "Llama-4-Scout-17B-16E-Instruct"
DEFAULT_MODELS = ["gpt-5.4-mini", LLAMA_SCOUT_MODEL, "gpt-5.4"]
MODEL_LABELS = {
    LLAMA_SCOUT_MODEL: "Llama-Scout",
}
TOPOLOGY_LABELS = {
    "individual": "Individual",
    "centralized": "Centralized",
    "broadcast_chain": "Broadcast Chain",
}
FIGSIZE = (18, 11.4)
GRID_LEFT = 0.105
GRID_RIGHT = 0.872
GRID_TOP = 0.885
GRID_BOTTOM = 0.06
COLUMN_FONTSIZE = 22
ROW_FONTSIZE = 21
CELL_TITLE_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 15
COLORBAR_LABEL_FONTSIZE = 15
COLORBAR_TICK_FONTSIZE = 13
CONSTRAINT_LABEL_COLOR = "#0B6E69"
CONSTRAINT_ROW_LABELS = {
    "spatial": "spat.",
    "temporal": "temp.",
    "dependency": "dep.",
}


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(str(model), str(model))


def _topology_label(topology: str) -> str:
    return TOPOLOGY_LABELS.get(str(topology), str(topology).title())


def _run_key(run_dir: Path) -> Optional[Tuple[str, str]]:
    usage_path = run_dir / "llm_usage.json"
    if not usage_path.exists():
        return None
    usage = _read_json(usage_path, {})
    model = usage.get("model")
    if not model:
        return None
    topology = run_dir.name.split("_agents", 1)[0]
    return str(model), topology


def _find_run(
    results_dir: Path,
    model: str,
    topology: str,
    agents: int,
    seed: int,
    repair: str,
) -> Optional[Path]:
    pattern = f"{topology}_agents{agents}_repair_{repair}_seed{seed}_*"
    candidates: List[Path] = []
    for run_dir in results_dir.glob(pattern):
        key = _run_key(run_dir)
        if key == (model, topology):
            candidates.append(run_dir)
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def _load_trace(run_dir: Path) -> Dict[str, Any]:
    process_log = _read_json(run_dir / "coop2_process_log.json", [])
    if not isinstance(process_log, list) or not process_log:
        raise FileNotFoundError(f"{run_dir / 'coop2_process_log.json'} is required.")
    max_step = process_log_max_step(process_log)
    _focus_records, focus_by_step, constraint_readiness = build_process_log_series(
        process_log,
        max_step,
    )
    return {
        "run_dir": run_dir,
        "team_score": _read_json(run_dir / "team_score.json", {}),
        "max_step": max_step,
        "focus_by_step": focus_by_step,
        "attempt_events": extract_attempt_constraint_events(process_log),
        "observed_by_constraint": aggregate_constraint_readiness(constraint_readiness),
        "score_events": extract_score_events_from_process_log(process_log),
    }


def _focus_matrix(
    focus_by_step: Dict[int, Dict[str, str]],
    resource_index: Dict[str, int],
    max_step: int,
):
    import numpy as np

    matrix = np.zeros((len(RESOURCE_ORDER), max_step + 1), dtype=float)
    for step, agent_map in focus_by_step.items():
        if step < 0 or step > max_step:
            continue
        for resource in agent_map.values():
            if resource in resource_index:
                matrix[resource_index[resource], step] += 1
    return matrix


def plot_grid(
    results_dir: Path,
    output_path: Path,
    models: List[str],
    topologies: List[str],
    agents: int,
    seed: int,
    repair: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.lines as mlines
    import matplotlib.ticker as mticker
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.size": TICK_FONTSIZE,
            "axes.titlesize": CELL_TITLE_FONTSIZE,
            "axes.labelsize": AXIS_LABEL_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
        }
    )

    traces: Dict[Tuple[str, str], Dict[str, Any]] = {}
    missing: List[Tuple[str, str]] = []
    for topology in topologies:
        for model in models:
            run_dir = _find_run(results_dir, model, topology, agents, seed, repair)
            if run_dir is None:
                missing.append((model, topology))
                continue
            traces[(model, topology)] = _load_trace(run_dir)

    if not traces:
        raise FileNotFoundError("No matching traces found.")

    global_max_step = max(trace["max_step"] for trace in traces.values())
    global_max_score = max(
        max(build_cumulative_score_series(trace["score_events"], global_max_step))
        for trace in traces.values()
    )
    resources = list(RESOURCE_ORDER)
    constraint_rows = [CONSTRAINT_ROW_LABELS.get(constraint, constraint) for constraint in CONSTRAINTS]
    resource_offset = len(constraint_rows)
    y_rows = constraint_rows + resources
    focus_resource_index = {resource: idx for idx, resource in enumerate(resources)}
    constraint_row_index = {
        constraint: idx
        for idx, constraint in enumerate(CONSTRAINTS)
    }

    fig = plt.figure(figsize=FIGSIZE)
    outer = fig.add_gridspec(
        nrows=len(topologies),
        ncols=len(models),
        left=GRID_LEFT,
        right=GRID_RIGHT,
        top=GRID_TOP,
        bottom=GRID_BOTTOM,
        hspace=0.14,
        wspace=0.035,
    )
    for col_idx, model in enumerate(models):
        x = GRID_LEFT + (col_idx + 0.5) * (GRID_RIGHT - GRID_LEFT) / max(len(models), 1)
        fig.text(
            x,
            0.947,
            _model_label(model),
            ha="center",
            va="center",
            fontsize=COLUMN_FONTSIZE,
            fontweight="bold",
        )
    for topo_idx, topology in enumerate(topologies):
        y = GRID_TOP - (topo_idx + 0.5) * (GRID_TOP - GRID_BOTTOM) / max(len(topologies), 1)
        fig.text(
            0.028,
            y,
            _topology_label(topology),
            ha="center",
            va="center",
            rotation=90,
            fontsize=ROW_FONTSIZE,
            fontweight="bold",
        )
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "focus",
        ["#FFFFFF", "#D7E8FF", "#2F80ED"],
    )
    cmap.set_bad("#FFFFFF")
    satisfaction_cmap = plt.get_cmap("RdYlGn").copy()
    satisfaction_cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    satisfaction_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    last_im = None
    score_handle = None
    termination_handle = mlines.Line2D(
        [],
        [],
        color="#D32F2F",
        linestyle="--",
        linewidth=2.0,
        label="end",
    )
    for topo_idx, topology in enumerate(topologies):
        for col_idx, model in enumerate(models):
            ax = fig.add_subplot(outer[topo_idx, col_idx])
            trace = traces.get((model, topology))
            if trace is None:
                ax.axis("off")
                ax.set_title(f"{model} | {topology}: missing trace")
                continue

            focus_matrix = np.full((len(y_rows), global_max_step + 1), np.nan, dtype=float)
            focus_matrix[resource_offset : resource_offset + len(resources), :] = _focus_matrix(
                trace["focus_by_step"],
                focus_resource_index,
                global_max_step,
            )
            last_im = ax.imshow(
                np.ma.masked_invalid(focus_matrix),
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=0,
                vmax=agents,
                extent=[0, global_max_step, len(y_rows) - 0.5, -0.5],
            )
            team_score = trace["team_score"]
            score = team_score.get("total_score")
            termination_step = _safe_step(team_score.get("current_step")) or trace["max_step"]
            ax.set_title(
                f"score={_safe_float(score):g}, step={termination_step}",
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
            ax.axhline(resource_offset - 0.5, color="#333333", linewidth=1.1, alpha=0.72)

            score_ax = ax.twinx()
            cumulative_scores = build_cumulative_score_series(trace["score_events"], global_max_step)
            cumulative_scores = [
                value if step <= termination_step else np.nan
                for step, value in enumerate(cumulative_scores)
            ]
            (score_line,) = score_ax.step(
                range(global_max_step + 1),
                cumulative_scores,
                where="post",
                color="#111111",
                linewidth=2.8,
                label="score",
                zorder=7,
            )
            score_handle = score_handle or score_line
            score_axis_max = max(1.0, global_max_score)
            # Keep the score curve inside the resource-focus band. Constraint
            # rows sit above it so wood can remain the literal bottom row.
            resource_fraction = len(resources) / max(len(y_rows), 1)
            score_ax.set_ylim(0, score_axis_max / max(resource_fraction, 1e-6))
            score_ax.set_yticks(np.linspace(0, score_axis_max, 4))
            score_ax.patch.set_alpha(0.0)
            score_ax.grid(False)
            if col_idx == len(models) - 1:
                score_ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, width=1.2, length=4.5)
            else:
                score_ax.set_yticklabels([])
                score_ax.tick_params(axis="y", length=0)
            score_ax.spines["right"].set_linewidth(1.25)

            attempt_points = []
            for event_idx, event in enumerate(trace.get("attempt_events") or []):
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
                            constraint_row_index[constraint],
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
                    s=33,
                    marker="o",
                    edgecolor="black",
                    linewidth=0.45,
                    zorder=6,
                )

            if termination_step < global_max_step:
                ax.axvspan(
                    termination_step,
                    global_max_step,
                    color="white",
                    alpha=1.0,
                    zorder=8,
                )
            ax.axvline(
                termination_step,
                color="#D32F2F",
                linestyle="--",
                linewidth=2.8,
                alpha=0.88,
                zorder=9,
            )
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=1.55, length=6.2)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
            for spine in ax.spines.values():
                spine.set_linewidth(1.45)
            if topo_idx == len(topologies) - 1:
                ax.set_xlabel("environment step", fontsize=AXIS_LABEL_FONTSIZE)
            else:
                ax.set_xlabel("")
                plt.setp(ax.get_xticklabels(), visible=False)

            legend_items = [termination_handle]
    if score_handle is not None:
        legend_items.insert(2, score_handle)
    fig.legend(
        legend_items,
        [handle.get_label() for handle in legend_items],
        loc="lower left",
        bbox_to_anchor=(0.905, 0.055),
        ncol=1,
        fontsize=LEGEND_FONTSIZE,
        framealpha=0.95,
        borderaxespad=0.0,
        handlelength=1.5,
        labelspacing=0.55,
        borderpad=0.4,
    )

    if last_im is not None:
        cax = fig.add_axes([0.925, 0.51, 0.014, 0.30])
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label("# agents", fontsize=COLORBAR_LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
        sat_cax = fig.add_axes([0.925, 0.25, 0.014, 0.18])
        sat_sm = plt.cm.ScalarMappable(norm=satisfaction_norm, cmap=satisfaction_cmap)
        sat_sm.set_array([])
        sat_cbar = fig.colorbar(sat_sm, cax=sat_cax)
        sat_cbar.set_label("constraint\nsatisfaction", fontsize=COLORBAR_LABEL_FONTSIZE)
        sat_cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    if missing:
        missing_text = ", ".join(f"{model}/{_topology_label(topology)}" for model, topology in missing)
        fig.text(0.5, 0.025, f"Missing traces: {missing_text}", ha="center", fontsize=TICK_FONTSIZE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot a COOP2 process grid.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--topologies", nargs="+", default=["individual", "centralized", "broadcast_chain"])
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repair", choices=["off", "on"], default="off")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output = args.output or results_dir / "figures" / f"coop2_process_grid_repair_{args.repair}_seed_{args.seed}.png"
    path = plot_grid(
        results_dir=results_dir,
        output_path=output.resolve(),
        models=args.models,
        topologies=args.topologies,
        agents=args.agents,
        seed=args.seed,
        repair=args.repair,
    )
    print(f"COOP2 process grid: {path}")
    pdf_path = path.with_suffix(".pdf")
    if pdf_path.exists():
        print(f"COOP2 process grid PDF: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
