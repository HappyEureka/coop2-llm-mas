"""Run one MA-Crafter COOP2 LLM condition with standardized logs.

One script serves every communication structure. The topology only changes
which agent factory is used and how roles are labeled in llm_usage.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cognitive import (
    LLMClient,
    PlanningEnvWrapper,
    build_repair_intervention_report,
    compute_all_metrics,
    print_metrics_summary,
    print_plan_summary,
    save_metrics,
    save_metrics_csv,
    visualize_all_agents_progress,
    visualize_comprehensive_timeline,
)
from cognitive.viz import RealtimeVisualizationWrapper
from comm_topology import (
    LLMLeaderAgent,
    create_llm_broadcast_chain_topology,
    create_llm_centralized_topology,
    create_llm_individual_topology,
)
from macrafter.coop_env import CooperativeEnv
from macrafter.cooperative_tasks import plot_metrics_timeline

try:
    from llm_usage import print_llm_usage_summary
except ImportError:
    from experiment.llm_usage import print_llm_usage_summary


TOPOLOGY_BUILDERS = {
    "individual": create_llm_individual_topology,
    "centralized": create_llm_centralized_topology,
    "broadcast_chain": create_llm_broadcast_chain_topology,
}


def _make_agents(args, llm_client: LLMClient):
    builder = TOPOLOGY_BUILDERS[args.topology]
    kwargs = dict(n_agents=args.agents, llm_client=llm_client, temperature=0.7, verbose=not args.quiet)
    if args.topology == "individual" and args.goal:
        kwargs["goal_instruction"] = args.goal
    return builder(**kwargs)


def _role_for_agent(topology: str, agent) -> str:
    if topology == "individual":
        return "individual"
    if topology == "centralized":
        return "leader" if isinstance(agent, LLMLeaderAgent) else "follower"
    if topology == "broadcast_chain":
        return f"speaker-{agent.speaker_order}"
    return topology


def _extra_for_agent(topology: str, agent) -> Dict[str, Any]:
    if topology == "broadcast_chain":
        return {"speaker_order": agent.speaker_order}
    return {}


def _describe_topology(topology: str, agent_names) -> str:
    if topology == "individual":
        return "all agents operate independently (no communication)"
    if topology == "centralized":
        return f"leader {agent_names[0]}, followers {agent_names[1:]}"
    return f"speaking order {' -> '.join(agent_names)}, each agent broadcasts to all following agents"


def run_condition(args, llm_client: Optional[LLMClient] = None) -> Path:
    """
    Run one condition and return its results directory.

    ``llm_client`` can be injected (tests use a scripted client); by default it
    is built from the .env credentials via ``LLMClient.from_env``.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    repair_label = "repair_on" if args.coop2_repair else "repair_off"
    output_root = Path(args.output_root) if args.output_root else Path(__file__).resolve().parent / "results"
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / f"{args.topology}_agents{args.agents}_{repair_label}_seed{args.seed}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"MA-CRAFTER COOP2 RUN: {args.topology}, agents={args.agents}, repair={repair_label}, seed={args.seed}")
    print(f"Results: {output_dir}")
    print("=" * 80)

    if llm_client is None:
        llm_client = LLMClient.from_env(backend=args.backend, model=args.model, verbose=not args.llm_quiet)
    print(f"LLM: backend={llm_client.backend}, model={llm_client.model}")

    agent_names = [f"agent_{idx}" for idx in range(args.agents)]
    base_env = CooperativeEnv(
        area=(64, 64),
        view=(9, 9),
        size=(84, 84),
        reward=True,
        length=args.steps,
        n_players=args.agents,
        seed=args.seed,
        coop_config_path="paper",
    )
    if hasattr(base_env, "set_team_score_time_limit"):
        base_env.set_team_score_time_limit(args.time_limit_seconds)

    print(f"Topology: {_describe_topology(args.topology, agent_names)}")
    if args.coop2_repair:
        print("COOP2 repair: enabled")
    if args.time_limit_seconds:
        print(f"Team-score deadline: {args.steps} env steps or {args.time_limit_seconds} seconds, whichever comes first")
    if args.goal:
        print(f"Goal: {args.goal}")

    agents = _make_agents(args, llm_client)
    plan_env = PlanningEnvWrapper(
        base_env,
        agent_names=agent_names,
        agents=agents,
        log_file=str(output_dir / "plan_logs.json"),
        coop2_precheck_enabled=args.coop2_repair,
    )
    env = RealtimeVisualizationWrapper(plan_env, record_video=args.record_video, show=args.show)

    env.reset()
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

            # Reasoning, interrupt handling, and waiting happen in agent threads.
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

            _obs_dict, _rewards, terminated, truncated, _info = env.step()
            done = any(terminated.values()) or any(truncated.values())

            if not args.quiet and env.current_step % 10 == 0:
                print(f"\n--- Step {env.current_step}/{args.steps} ---")
                for agent_id, agent in agents.items():
                    role = _role_for_agent(args.topology, agent).upper()
                    print(f"  {agent_id} ({role}): {agent.api_calls} API calls, {agent.total_tokens_used} tokens")

    except KeyboardInterrupt:
        print("Interrupted by user")

    for agent in agents.values():
        agent.shutdown()
    for thread in running_threads.values():
        thread.join()
    running_threads.clear()

    last_step = env.current_step
    print(f"Episode ended at step {last_step}" + (" by time limit" if timed_out else ""))
    env.terminate_unfinished_plans()
    env.save_logs(str(output_dir))

    if args.coop2_repair:
        report_path = build_repair_intervention_report(str(output_dir))
        if report_path:
            print(f"COOP2 repair process figure: {report_path}")

    stats = env.get_plan_statistics()
    print(f"\nPlan Statistics: total={stats['total_plans']}", end="")
    if stats["total_plans"] > 0:
        print(
            f", successful={stats['successful']}, failed={stats['failed']}, "
            f"success_rate={stats['success_rate'] * 100:.1f}%, average_duration={stats['average_duration']:.1f} steps"
        )
    else:
        print()

    usage_summary = print_llm_usage_summary(
        agents,
        role_getter=lambda _agent_id, agent: _role_for_agent(args.topology, agent),
        extra_getter=lambda _agent_id, agent: _extra_for_agent(args.topology, agent),
    )
    print_plan_summary(env.logger.plan_history)

    visualize_all_agents_progress(env.logger.plan_history, output_path=str(output_dir / "agent_timeline.png"))
    visualize_comprehensive_timeline(
        env.agents,
        env.logger.plan_history,
        last_step,
        env.message_broker.get_message_log(),
        output_path=str(output_dir / "comprehensive_timeline.png"),
    )
    task_history = plan_env.symbolic_env.env.task_tracker.get_history()
    metrics_history = [step_summary.metrics for step_summary in task_history if step_summary.metrics]
    if metrics_history:
        plot_metrics_timeline(metrics_history, output_path=str(output_dir / "metrics_timeline.png"), show=False)

    if args.record_video:
        env.save_video(str(output_dir / "episode.gif"), fps=12)
    env.close()

    llm_stats = {
        "model": llm_client.model,
        "topology": args.topology,
        "max_steps": args.steps,
        "time_limit_seconds": args.time_limit_seconds,
        "timed_out": timed_out,
        "seed": args.seed,
        "record_video": args.record_video,
        **usage_summary,
    }
    with (output_dir / "llm_usage.json").open("w", encoding="utf-8") as handle:
        json.dump(llm_stats, handle, indent=2)

    # COOP2 metrics read llm_usage.json, so they are computed last.
    metrics = compute_all_metrics(str(output_dir))
    print_metrics_summary(metrics)
    save_metrics(metrics, str(output_dir / "coop2_metrics.json"))
    save_metrics_csv(metrics, str(output_dir / "coop2_metrics.csv"))

    print(f"Results saved to: {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface; also used by the grid script's manifest check and by tests."""
    parser = argparse.ArgumentParser(description="Run one MA-Crafter COOP2 condition.")
    parser.add_argument("--topology", choices=sorted(TOPOLOGY_BUILDERS), required=True)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--time-limit-seconds", type=float, default=120, help="Wall-clock budget; 0 disables it")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=None, help="LLM model/deployment override")
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--goal", type=str, default="", help="Global goal instruction for individual agents")
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
