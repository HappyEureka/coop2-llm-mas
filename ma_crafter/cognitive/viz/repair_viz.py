"""Compatibility wrappers for COOP2 repair visualizations.

Repair runs now use the shared compact process figure. The previous bespoke
repair timeline/report figure was removed so there is only one visual grammar
for single-run and grid process traces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def visualize_repair_interventions(
    run_dir: str | Path,
    output_path: str | Path | None = None,
) -> Optional[Path]:
    """Save the current COOP2 repair process figure for one run."""
    from experiment.plot_coop2_process_case_study import plot_case_study

    run_path = Path(run_dir)
    if not (run_path / "coop2_process_log.json").exists():
        return None
    output = Path(output_path) if output_path is not None else run_path / "coop2_repair_process_case_study.png"
    return plot_case_study(
        run_dir=run_path,
        output_path=output,
        csv_path=output.with_suffix(".csv"),
        agent_focus_csv_path=output.with_name(f"{output.stem}_agent_focus.csv"),
        title=None,
        show_repair=True,
    )


def build_repair_intervention_report(
    run_dir: str | Path,
    html_output_path: str | Path | None = None,
    timeline_output_path: str | Path | None = None,
) -> Optional[Path]:
    """Backward-compatible entry point returning the new repair figure path.

    ``html_output_path`` is accepted for the old API but intentionally ignored.
    ``timeline_output_path`` can still be used to choose the output image path.
    """
    del html_output_path
    return visualize_repair_interventions(run_dir, timeline_output_path)
