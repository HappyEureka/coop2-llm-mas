"""Run a CUBE COOP2 experiment grid with weighted parallelism."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
CONDITION_SCRIPT = EXPERIMENT_DIR / "run_condition.py"
TOPOLOGIES = ("individual", "centralized", "broadcast_chain")


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def build_command(args, topology: str, agent_count: int, repair_enabled: bool, seed: int) -> list[str]:
    command = [
        sys.executable,
        str(CONDITION_SCRIPT),
        "--topology",
        topology,
        "--agents",
        str(agent_count),
        "--steps",
        str(args.steps),
        "--time-limit-seconds",
        str(args.time_limit_seconds),
        "--grid-size",
        str(args.grid_size),
        "--seed",
        str(seed),
        "--output-root",
        str(args.output_root),
    ]
    if args.model:
        command.extend(["--model", args.model])
    if args.backend:
        command.extend(["--backend", args.backend])
    if args.quiet:
        command.append("--quiet")
    else:
        command.append("--verbose")
    if args.llm_quiet:
        command.append("--llm-quiet")
    else:
        command.append("--llm-verbose")
    if args.show:
        command.append("--show")
    if args.record_video:
        command.append("--record-video")
    if repair_enabled:
        command.append("--coop2-repair")
    return command


def save_manifest(manifest_path: Path, records: list[dict]) -> None:
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)


def run_sequential(args, runs: list[dict], manifest_path: Path) -> int:
    records = []
    for run in runs:
        print(
            f"[{run['index']}/{len(runs)}] {run['topology']}, agents={run['agents']}, "
            f"repair={'on' if run['repair'] else 'off'}, seed={run['seed']}"
        )
        if args.dry_run:
            records.append({**run, "status": "dry_run"})
            continue
        started = time.monotonic()
        result = subprocess.run(run["command"], cwd=ROOT)
        duration = time.monotonic() - started
        record = {
            **run,
            "status": "completed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "duration_seconds": round(duration, 3),
        }
        records.append(record)
        save_manifest(manifest_path, records)
        if result.returncode != 0 and not args.continue_on_error:
            return result.returncode
    save_manifest(manifest_path, records)
    return 0


def run_parallel(args, runs: list[dict], manifest_dir: Path, manifest_path: Path) -> int:
    pending = list(runs)
    active = []
    records = []
    safe_model = str(args.model or "env_model").replace("/", "_").replace(":", "_")
    runner_logs_dir = manifest_dir / f"runner_logs_{safe_model}"
    runner_logs_dir.mkdir(parents=True, exist_ok=True)
    stop_launching = False

    def active_slots() -> int:
        return sum(item["agents"] for item in active)

    def can_launch(run: dict) -> bool:
        if stop_launching:
            return False
        if args.max_parallel_runs and len(active) >= args.max_parallel_runs:
            return False
        return active_slots() + run["agents"] <= args.max_concurrent_agents

    while pending or active:
        launched = False
        while pending and can_launch(pending[0]):
            run = pending.pop(0)
            print(
                f"[{run['index']}/{len(runs)}] launch {run['topology']}, agents={run['agents']}, "
                f"repair={'on' if run['repair'] else 'off'}, seed={run['seed']} "
                f"(slots {active_slots() + run['agents']}/{args.max_concurrent_agents})"
            )
            log_path = runner_logs_dir / (
                f"{run['index']:03d}_{run['topology']}_agents{run['agents']}_"
                f"repair{'on' if run['repair'] else 'off'}_seed{run['seed']}.log"
            )
            log_file = log_path.open("w", encoding="utf-8")
            started = time.monotonic()
            process = subprocess.Popen(
                run["command"],
                cwd=ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
            active.append({**run, "process": process, "started": started, "runner_log": str(log_path), "log_file": log_file})
            launched = True

        finished = []
        for item in active:
            returncode = item["process"].poll()
            if returncode is None:
                continue
            item["log_file"].close()
            duration = time.monotonic() - item["started"]
            record = {key: value for key, value in item.items() if key not in {"process", "started", "log_file"}}
            record.update({
                "status": "completed" if returncode == 0 else "failed",
                "returncode": returncode,
                "duration_seconds": round(duration, 3),
            })
            records.append(record)
            finished.append(item)
            print(
                f"[{item['index']}/{len(runs)}] done {item['topology']}, "
                f"agents={item['agents']}, repair={'on' if item['repair'] else 'off'} -> {record['status']}"
            )
            if returncode != 0 and not args.continue_on_error:
                stop_launching = True

        for item in finished:
            active.remove(item)

        if finished:
            save_manifest(
                manifest_path,
                records + [
                    {key: value for key, value in item.items() if key not in {"process", "started", "log_file"}}
                    for item in active
                ],
            )

        if stop_launching and active:
            for item in active:
                item["process"].terminate()
            for item in active:
                try:
                    item["process"].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    item["process"].kill()
                    item["process"].wait()
                item["log_file"].close()
            return 1
        if stop_launching:
            return 1

        if not launched and not finished:
            if pending and not active and pending[0]["agents"] > args.max_concurrent_agents:
                print(
                    f"Run {pending[0]['index']} needs {pending[0]['agents']} slots, "
                    f"but max-concurrent-agents is {args.max_concurrent_agents}."
                )
                return 2
            time.sleep(args.poll_seconds)

    save_manifest(manifest_path, records)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a CUBE COOP2 experiment grid.")
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=list(TOPOLOGIES),
        choices=TOPOLOGIES,
        help="Communication structures evaluated in the paper.",
    )
    parser.add_argument("--agent-counts", type=parse_int_list, default=parse_int_list("3"))
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("42"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--time-limit-seconds", type=float, default=120)
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--repair", choices=["both", "off", "on"], default="off")
    parser.add_argument("--quiet", action="store_true", default=True)
    parser.add_argument("--verbose", dest="quiet", action="store_false")
    parser.add_argument("--llm-quiet", action="store_true", default=True)
    parser.add_argument("--llm-verbose", dest="llm_quiet", action="store_false")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--max-concurrent-agents", type=int, default=1)
    parser.add_argument("--max-parallel-runs", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()

    repair_modes = [False, True]
    if args.repair == "off":
        repair_modes = [False]
    elif args.repair == "on":
        repair_modes = [True]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_model = str(args.model or "env_model").replace("/", "_").replace(":", "_")
    if args.output_root is None:
        manifest_dir = EXPERIMENT_DIR / "results" / f"cube_coop2_traces_{safe_model}_{timestamp}"
        args.output_root = manifest_dir
    else:
        manifest_dir = args.output_root.resolve()
        args.output_root = manifest_dir
    manifest_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    index = 1
    for seed in args.seeds:
        for topology in args.topologies:
            for agent_count in args.agent_counts:
                for repair_enabled in repair_modes:
                    runs.append(
                        {
                            "index": index,
                            "topology": topology,
                            "agents": agent_count,
                            "repair": repair_enabled,
                            "seed": seed,
                            "command": build_command(args, topology, agent_count, repair_enabled, seed),
                        }
                    )
                    index += 1

    print(f"Planned runs: {len(runs)}")
    manifest_path = manifest_dir / f"manifest_{safe_model}.json"
    print(f"Manifest: {manifest_path}")
    print(f"Agent concurrency budget: {args.max_concurrent_agents}")

    if args.dry_run or args.max_concurrent_agents <= 1:
        return run_sequential(args, runs, manifest_path)
    return run_parallel(args, runs, manifest_dir, manifest_path)


if __name__ == "__main__":
    raise SystemExit(main())
