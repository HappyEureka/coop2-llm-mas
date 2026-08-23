"""Plot score versus decision and communication cost from the COOP2 table."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List


MODEL_LABELS = {
    "gpt-5.4-mini": "GPT-5.4-mini",
    "gpt-5.4": "GPT-5.4",
    "Llama-4-Scout-17B-16E-Instruct": "Llama-Scout",
}

MODEL_COLORS = {
    "gpt-5.4-mini": "#3B82F6",
    "gpt-5.4": "#EF7D22",
    "Llama-4-Scout-17B-16E-Instruct": "#2CA25F",
}

TOPOLOGY_LABELS = {
    "individual": "Individual",
    "centralized": "Centralized",
    "broadcast_chain": "Broadcast Chain",
}

TOPOLOGY_MARKERS = {
    "individual": "o",
    "centralized": "s",
    "broadcast_chain": "^",
}

AGENT_MARKER_SIZES = {
    3: 78,
    6: 142,
}


def _float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _plot_panel(
    ax,
    rows: Iterable[Dict[str, str]],
    *,
    x_key: str,
    x_ci_key: str | None,
    x_label: str,
    y_key: str = "score",
    y_ci_key: str = "score_ci95",
    show_ylabel: bool = False,
):
    rows = list(rows)
    for row in rows:
        model = row["model"]
        topology = row["topology"]
        x = _float(row, x_key)
        xerr = _float(row, x_ci_key) if x_ci_key else 0.0
        y = _float(row, y_key)
        yerr = _float(row, y_ci_key)

        ax.errorbar(
            x,
            y,
            xerr=xerr if xerr > 0 else None,
            yerr=yerr if yerr > 0 else None,
            fmt="none",
            ecolor="#B6BEC9",
            elinewidth=0.95,
            capsize=1.8,
            alpha=0.48,
            zorder=1,
        )
        agents = int(float(row["agents"]))
        ax.scatter(
            [x],
            [y],
            s=AGENT_MARKER_SIZES.get(agents, 100),
            marker=TOPOLOGY_MARKERS.get(topology, "o"),
            color=MODEL_COLORS.get(model, "#666666"),
            edgecolor="#1F2937",
            linewidth=1.05,
            alpha=0.94,
            zorder=3,
        )

    ax.set_xlabel(x_label)
    if show_ylabel:
        ax.set_ylabel("Team score")
    ax.grid(True, alpha=0.18, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=0.9, length=3.4)


def plot_tradeoff(input_csv: Path, output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    rows = _load_rows(input_csv)
    for row in rows:
        messages = _float(row, "messages")
        interruptions = _float(row, "interruptions")
        msg_ci = _float(row, "messages_ci95")
        intr_ci = _float(row, "interruptions_ci95")
        row["communication_events"] = str(messages + interruptions)
        row["communication_events_ci95"] = str(math.sqrt(msg_ci * msg_ci + intr_ci * intr_ci))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True, constrained_layout=False)
    column_specs = [
        ("total_decision_time", "total_decision_time_ci95", "Decision time (s)"),
        ("communication_events", "communication_events_ci95", "Comm. events"),
    ]
    x_limits = {}
    for x_key, x_ci_key, _label in column_specs:
        x_limits[x_key] = max(_float(row, x_key) + _float(row, x_ci_key) for row in rows) * 1.06

    max_score = max(_float(row, "score") + _float(row, "score_ci95") for row in rows)
    for col_idx, (x_key, x_ci_key, label) in enumerate(column_specs):
        ax = axes[col_idx]
        _plot_panel(
            ax,
            rows,
            x_key=x_key,
            x_ci_key=x_ci_key,
            x_label=label,
            show_ylabel=col_idx == 0,
        )
        ax.set_ylim(bottom=0, top=max(max_score * 1.08, 1.0))
        ax.set_xlim(left=-0.02 * x_limits[x_key], right=x_limits[x_key])
        ax.set_title("Decision cost" if col_idx == 0 else "Communication cost", fontweight="bold", pad=5)
        ax.text(
            0.02,
            0.94,
            chr(ord("A") + col_idx),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            color="#111827",
        )

    model_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor="#111111",
            markersize=7.2,
            label=MODEL_LABELS.get(model, model),
        )
        for model, color in MODEL_COLORS.items()
        if any(row["model"] == model for row in rows)
    ]
    topology_handles = [
        mlines.Line2D(
            [],
            [],
            marker=marker,
            linestyle="None",
            markerfacecolor="#E5E7EB",
            markeredgecolor="#111111",
            markersize=7.2,
            label=TOPOLOGY_LABELS.get(topology, topology.title()),
        )
        for topology, marker in TOPOLOGY_MARKERS.items()
        if any(row["topology"] == topology for row in rows)
    ]
    agent_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markerfacecolor="#9CA3AF",
            markeredgecolor="#111111",
            markersize=math.sqrt(size),
            label=f"{agents} agents",
        )
        for agents, size in AGENT_MARKER_SIZES.items()
        if any(int(float(row["agents"])) == agents for row in rows)
    ]
    fig.subplots_adjust(left=0.10, right=0.99, top=0.86, bottom=0.30, wspace=0.16)
    fig.legend(
        handles=model_handles + topology_handles + agent_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.14),
        ncol=4,
        frameon=False,
        columnspacing=1.15,
        handletextpad=0.45,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    png_path = output.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot COOP2 performance-cost tradeoff.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiment/results/tables/coop2_table_repair_off_seed_42.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiment/results/figures/performance_cost_tradeoff.pdf"),
    )
    args = parser.parse_args()
    path = plot_tradeoff(args.input.resolve(), args.output.resolve())
    print(f"Saved {path}")
    print(f"Saved {path.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
