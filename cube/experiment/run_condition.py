"""Run one CUBE COOP2 LLM condition with standardized logs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cognitive import (
    LLMClient,
    PlanningEnvWrapper,
    RealtimeVisualizationWrapper,
    compute_all_metrics,
    save_metrics,
    save_metrics_csv,
)
from comm_topology import (
    LLMFollowerAgent,
    LLMLeaderAgent,
    create_llm_centralized_topology,
    create_llm_broadcast_chain_topology,
    create_llm_individual_topology,
)
from env.env import CoopBlockPush
from experiment.config import paper_block_specs


TOPOLOGY_BUILDERS = {
    "individual": create_llm_individual_topology,
    "centralized": create_llm_centralized_topology,
    "broadcast_chain": create_llm_broadcast_chain_topology,
}


def _make_agents(topology: str, n_agents: int, llm_client: LLMClient, verbose: bool):
    return TOPOLOGY_BUILDERS[topology](
        n_agents=n_agents,
        llm_client=llm_client,
        temperature=0.7,
        verbose=verbose,
    )


def _role_for_agent(topology: str, agent_id: str, agent) -> str:
    if topology == "individual":
        return "individual"
    if topology == "centralized":
        return "leader" if isinstance(agent, LLMLeaderAgent) else "follower"
    if topology == "broadcast_chain":
        return f"speaker-{getattr(agent, 'speaker_order', agent_id)}"
    return topology


def _agent_usage(agent) -> Dict[str, int]:
    return {
        "plan_count": int(getattr(agent, "plan_count", 0)),
        "api_calls": int(getattr(agent, "api_calls", 0)),
        "total_tokens": int(getattr(agent, "total_tokens_used", 0)),
        "prompt_tokens": int(getattr(agent, "prompt_tokens", 0)),
        "completion_tokens": int(getattr(agent, "completion_tokens", 0)),
    }


def _save_llm_usage(output_dir: Path, topology: str, agents: Dict[str, object], llm_client: LLMClient, args, timed_out: bool) -> None:
    per_agent = {}
    totals = {
        "total_api_calls": 0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    for agent_id, agent in agents.items():
        usage = _agent_usage(agent)
        usage["role"] = _role_for_agent(topology, agent_id, agent)
        per_agent[agent_id] = usage
        totals["total_api_calls"] += usage["api_calls"]
        totals["total_tokens"] += usage["total_tokens"]
        totals["prompt_tokens"] += usage["prompt_tokens"]
        totals["completion_tokens"] += usage["completion_tokens"]

    data = {
        "model": llm_client.model,
        "backend": llm_client.backend,
        "topology": topology,
        "n_agents": args.agents,
        "max_steps": args.steps,
        "time_limit_seconds": args.time_limit_seconds,
        "timed_out": timed_out,
        "seed": args.seed,
        "environment_config": "paper",
        "repair": "on" if args.coop2_repair else "off",
        "record_video": args.record_video,
        **totals,
        "per_agent": per_agent,
    }
    with (output_dir / "llm_usage.json").open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def run_condition(args, llm_client: Optional[LLMClient] = None) -> Path:
    """
    Run one condition and return its results directory.

    ``llm_client`` can be injected (tests use a scripted client); by default it
    is built from the .env credentials via ``LLMClient.from_env``.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    repair_label = "repair_on" if args.coop2_repair else "repair_off"
    output_root = args.output_root or Path(__file__).resolve().parent / "results"
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / f"{args.topology}_agents{args.agents}_{repair_label}_seed{args.seed}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"CUBE COOP2 RUN: {args.topology}, agents={args.agents}, repair={repair_label}, seed={args.seed}")
    print(f"Results: {output_dir}")
    print("=" * 80)

    if llm_client is None:
        llm_client = LLMClient.from_env(backend=args.backend, model=args.model, verbose=not args.llm_quiet)
    print(f"LLM: backend={llm_client.backend}, model={llm_client.model}")

    block_specs = paper_block_specs()
    agent_names = [f"agent_{idx}" for idx in range(args.agents)]
    base_env = CoopBlockPush(
        num_agents=args.agents,
        grid_size=args.grid_size,
        block_specs=block_specs,
        max_steps=args.steps,
        seed=args.seed,
    )
    agents = _make_agents(args.topology, args.agents, llm_client, verbose=not args.quiet)
    plan_env = PlanningEnvWrapper(
        base_env,
        agent_names=agent_names,
        agents=agents,
        coop2_precheck_enabled=args.coop2_repair,
    )
    if args.record_video or args.show:
        env = RealtimeVisualizationWrapper(plan_env, record_video=args.record_video, show=args.show)
    else:
        env = plan_env

    obs_dict, _info = env.reset(seed=args.seed)
    done = False
    timed_out = False
    deadline = None
    if args.time_limit_seconds and args.time_limit_seconds > 0:
        deadline = time.monotonic() + float(args.time_limit_seconds)
    running_threads: Dict[str, threading.Thread] = {}

    try:
        while env.current_step < args.steps and not done:
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                break

            while not all(env.agents[aid].ready for aid in agent_names):
                if deadline is not None and time.monotonic() >= deadline:
                    timed_out = True
                    break
                for aid in agent_names:
                    agent = env.agents[aid]
                    if aid not in running_threads or not running_threads[aid].is_alive():
                        running_threads[aid] = threading.Thread(target=agent.create_agent_thread)
                        running_threads[aid].start()
                env.wait_for_state_change(timeout=0.05)

            if timed_out:
                break

            for thread in running_threads.values():
                thread.join()
            running_threads.clear()

            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                break

            obs_dict, _rewards, terminated, truncated, _info = env.step()
            done = any(terminated.values()) or any(truncated.values())

            if not args.quiet and env.current_step % 10 == 0:
                print(f"step {env.current_step}/{args.steps}")

    except KeyboardInterrupt:
        print("Interrupted by user")

    print(f"Episode ended at step {env.current_step}" + (" by time limit" if timed_out else ""))
    env.terminate_unfinished_plans()
    env.save_logs(str(output_dir))
    _save_llm_usage(output_dir, args.topology, agents, llm_client, args, timed_out)

    metrics = compute_all_metrics(str(output_dir))
    save_metrics(metrics, str(output_dir / "coop2_metrics.json"))
    save_metrics_csv(metrics, str(output_dir / "coop2_metrics.csv"))

    if args.record_video:
        env.save_video(str(output_dir / "episode.gif"), fps=12)
    if hasattr(env, "close"):
        env.close()

    return output_dir


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface; also used by the grid script's manifest check and by tests."""
    parser = argparse.ArgumentParser(description="Run one CUBE COOP2 condition.")
    parser.add_argument("--topology", choices=sorted(TOPOLOGY_BUILDERS), required=True)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--time-limit-seconds", type=float, default=120)
    parser.add_argument("--grid-size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--coop2-repair", action="store_true")
    parser.add_argument("--quiet", action="store_true", default=True)
    parser.add_argument("--verbose", dest="quiet", action="store_false")
    parser.add_argument("--llm-quiet", action="store_true", default=True)
    parser.add_argument("--llm-verbose", dest="llm_quiet", action="store_false")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = run_condition(args)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
