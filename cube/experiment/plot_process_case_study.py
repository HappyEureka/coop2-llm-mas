"""Plot one CUBE COOP2 process trace with the same compact style as the grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cognitive.coop2_attempt_events import ATTEMPT_CONSTRAINTS
from experiment.process_utils import (
    CONSTRAINT_LABELS,
    build_cumulative_score_series,
    build_target_rows,
    load_trace,
    safe_float,
    safe_int,
    score_from_team_score,
    target_tick_label,
)
from experiment.plot_process_grid import (
    AXIS_LABEL_FONTSIZE,
    CELL_TITLE_FONTSIZE,
    COLORBAR_LABEL_FONTSIZE,
    COLORBAR_TICK_FONTSIZE,
    CONSTRAINT_LABEL_COLOR,
    LEGEND_FONTSIZE,
    TICK_FONTSIZE,
    _focus_matrix,
)


def plot_case_study(run_dir: Path, output_path: Path, *, show_title: bool = False) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    import numpy as np

    trace = load_trace(run_dir)
    max_step = trace["max_step"]
    targets = build_target_rows(trace["process_log"], trace["score_events"])
    constraint_rows = [CONSTRAINT_LABELS.get(constraint, constraint) for constraint in ATTEMPT_CONSTRAINTS]
    target_offset = len(constraint_rows)
    y_rows = constraint_rows + [target_tick_label(target) for target in targets]
    target_index = {target: idx for idx, target in enumerate(targets)}
    constraint_index = {constraint: idx for idx, constraint in enumerate(ATTEMPT_CONSTRAINTS)}

    focus_matrix = np.full((len(y_rows), max_step + 1), np.nan, dtype=float)
    focus_matrix[target_offset : target_offset + len(targets), :] = _focus_matrix(
        trace["focus_by_step"],
        target_index,
        max_step,
    )

    fig = plt.figure(figsize=(13.6, 4.9))
    ax = fig.add_axes([0.095, 0.18, 0.69, 0.66])
    if show_title:
        fig.text(0.095, 0.94, "CUBE COOP2 Process Trace", ha="left", va="top", fontsize=14, fontweight="bold")

    focus_cmap = mcolors.LinearSegmentedColormap.from_list("focus", ["#FFFFFF", "#D7E8FF", "#2F80ED"])
    focus_cmap.set_bad("#FFFFFF")
    focus_im = ax.imshow(
        np.ma.masked_invalid(focus_matrix),
        aspect="auto",
        interpolation="nearest",
        cmap=focus_cmap,
        vmin=0,
        vmax=max(1.0, float(np.nanmax(focus_matrix[target_offset:, :])) if targets else 1.0),
        extent=[0, max_step, len(y_rows) - 0.5, -0.5],
    )

    ax.set_ylabel("constraint / block", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlabel("environment step", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_yticks(range(len(y_rows)))
    ax.set_yticklabels(y_rows, fontsize=TICK_FONTSIZE)
    for row_idx, tick_label in enumerate(ax.get_yticklabels()):
        if row_idx < target_offset:
            tick_label.set_color(CONSTRAINT_LABEL_COLOR)
            tick_label.set_fontweight("bold")
    ax.set_xlim(0, max_step)
    ax.set_ylim(len(y_rows) - 0.5, -0.5)
    ax.grid(axis="x", alpha=0.14)
    ax.axhline(target_offset - 0.5, color="#111111", linewidth=1.6, alpha=0.85, zorder=7)

    satisfaction_cmap = plt.get_cmap("RdYlGn").copy()
    satisfaction_cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    satisfaction_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    attempt_points = []
    termination_step = min(max(safe_int(trace["termination_step"], max_step), 0), max_step)
    for event_idx, event in enumerate(trace.get("attempt_events") or []):
        if str(event.get("action_type") or "").lower() != "push":
            continue
        if not event.get("attempting_agents"):
            continue
        step = safe_int(event.get("env_step"), -1)
        if step < 0 or step > termination_step:
            continue
        scores = event.get("constraint_scores") or {}
        offset = ((event_idx % 5) - 2) * 0.08
        for constraint in ATTEMPT_CONSTRAINTS:
            if constraint in scores:
                attempt_points.append(
                    (
                        step + offset,
                        constraint_index[constraint],
                        min(max(safe_float(scores[constraint]), 0.0), 1.0),
                    )
                )
    if attempt_points:
        xs, ys, values = zip(*attempt_points)
        ax.scatter(xs, ys, c=values, cmap=satisfaction_cmap, norm=satisfaction_norm, s=44, marker="o", edgecolor="black", linewidth=0.55, zorder=8)

    score_ax = ax.twinx()
    cumulative_scores = build_cumulative_score_series(trace["score_events"], max_step)
    score_max = max(1.0, max(cumulative_scores, default=0.0))
    target_fraction = len(targets) / max(len(y_rows), 1)
    (score_line,) = score_ax.step(range(max_step + 1), cumulative_scores, where="post", color="#111111", linewidth=2.8, label="score", zorder=9)
    score_ax.set_ylabel("cumulative score", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    score_ax.set_ylim(0, score_max / max(target_fraction, 1e-6))
    score_ax.set_yticks(np.linspace(0, score_max, 4))
    score_ax.patch.set_alpha(0.0)
    score_ax.grid(False)

    if termination_step < max_step:
        ax.axvspan(termination_step, max_step, color="white", alpha=1.0, zorder=10)
    ax.axvline(termination_step, color="#D32F2F", linestyle="--", linewidth=2.4, alpha=0.9, zorder=11)

    end_handle = mlines.Line2D([], [], color="#D32F2F", linestyle="--", linewidth=2.4, label="end")
    fig.legend([score_line, end_handle], ["score", "end"], loc="lower left", bbox_to_anchor=(0.865, 0.18), ncol=1, fontsize=LEGEND_FONTSIZE, framealpha=0.95, borderaxespad=0.0, handlelength=1.7)

    cax_focus = fig.add_axes([0.865, 0.58, 0.018, 0.22])
    focus_cbar = fig.colorbar(focus_im, cax=cax_focus)
    focus_cbar.set_label("# agents", fontsize=COLORBAR_LABEL_FONTSIZE)
    focus_cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
    sat_cax = fig.add_axes([0.865, 0.34, 0.018, 0.16])
    sat_sm = plt.cm.ScalarMappable(norm=satisfaction_norm, cmap=satisfaction_cmap)
    sat_sm.set_array([])
    sat_cbar = fig.colorbar(sat_sm, cax=sat_cax)
    sat_cbar.set_label("constraint\nsatisfaction", fontsize=COLORBAR_LABEL_FONTSIZE)
    sat_cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    ax.set_title(f"score={score_from_team_score(trace['team_score']):g}, step={termination_step}", fontsize=CELL_TITLE_FONTSIZE, fontweight="bold", loc="left", pad=2)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, width=1.3, length=5.5)
    score_ax.tick_params(axis="y", labelsize=TICK_FONTSIZE, width=1.2, length=4.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, facecolor="white")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot one CUBE COOP2 process trace.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--title", action="store_true")
    args = parser.parse_args()
    output = args.output or args.run_dir / "cube_coop2_process_case_study.png"
    path = plot_case_study(args.run_dir.resolve(), output.resolve(), show_title=args.title)
    print(f"CUBE process case-study plot: {path}")
    pdf_path = path.with_suffix(".pdf")
    if pdf_path.exists():
        print(f"CUBE process case-study PDF: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
