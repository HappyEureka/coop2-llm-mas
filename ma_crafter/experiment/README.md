# MA-Crafter experiment scripts

Run these commands from `ma_crafter/`. See the repository-level README for installation, credentials, terminology, and adaptation guidance.

## Run traces

```bash
python experiment/run_grid.py \
  --model your-deployment-name \
  --topologies individual centralized broadcast_chain \
  --agent-counts 3,6 \
  --seeds 42 \
  --repair off \
  --dry-run
```

Remove `--dry-run` to execute. Set `--repair both` for matched repair-off and repair-on conditions. Outputs go under `experiment/results/` unless `--output-root` is set.

## Run one condition

```bash
python experiment/run_condition.py --topology centralized --agents 3 --seed 42 --model your-deployment-name
```

Add `--coop2-repair` for the repair condition, `--record-video` to save an episode GIF, and `--verbose` or `--llm-verbose` to print agent and LLM output. The grid script above calls this runner once per condition.

## Build tables

```bash
python experiment/build_results_table.py \
  --results-dir path/to/results \
  --models model-a model-b model-c \
  --topologies individual centralized broadcast_chain \
  --agents 3 6 \
  --repair off \
  --hide-runs
```

## Generate figures

```bash
python experiment/plot_process_grid_aggregate.py \
  --results-dir path/to/results \
  --models model-a model-b model-c \
  --topologies individual centralized broadcast_chain \
  --agents 3 \
  --repair off
```

Use `plot_process_case_study.py` for one trace and `plot_performance_cost_tradeoff.py` for the score-cost figure.
