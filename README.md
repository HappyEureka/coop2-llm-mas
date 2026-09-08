# COOP²: Cooperation in LLM Multi-Agent Systems

Code repository for [COOP²: Defining, Observing, and Repairing Cooperation in LLM Multi-Agent Systems](https://arxiv.org/abs/2603.00349).

COOP² connects the cognitive activity of LLM agents, including planning, communication, interruption, and replanning, to grounded environment actions and cooperative task progress. This repository contains the two environment instantiations used in the paper, shared cooperation records and metrics, and COOP²-Repair.

Project page: <https://happyeureka.github.io/coop2/>

## What is included

| Component | Purpose |
| --- | --- |
| `cube/` | CUBE cooperative block pushing environment, agents, COOP² adapter, repair logic, and experiments |
| `ma_crafter/` | MA-Crafter environment, agents, COOP² adapter, repair logic, and experiments |
| `*/cognitive/` | The cognitive side of COOP²: agents, symbolic plans, communication, grounding, process records, and metrics |
| `*/comm_topology/` | Individual, centralized, and Broadcast Chain communication structures |
| `*/coop2_repair/` | COOP²-Repair evaluation, messaging, and environment-specific plan-effect adapters |
| `*/experiment/` | Paper experiment runners and scripts for tables and figures |

Generated traces, figures, tables, videos, and API credentials are excluded from version control. New runs write to a timestamped results directory unless `--output-root` is supplied.

## Setup

Python 3.10 is recommended. From the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configure an LLM backend

Copy the credential template and edit the copy:

```bash
cp .env.example .env
```

The credential template covers Azure OpenAI and Azure AI Foundry. Set `LLM_BACKEND` and the corresponding variables in `.env`. Shell environment variables take precedence over file values. You may also place a separate `.env` inside `cube/` or `ma_crafter/`; the environment-specific file takes precedence over the repository-level file.

Never commit `.env`. It is covered by `.gitignore`.

## Run experiments

Run commands from the environment directory. Start with `--dry-run`, which creates a manifest without making API calls.

### CUBE

```bash
cd cube

python experiment/run_grid.py \
  --topologies individual centralized broadcast_chain \
  --agent-counts 3 \
  --seeds 42 \
  --repair off \
  --model your-deployment-name \
  --dry-run
```

Remove `--dry-run` to execute. To compare the repair case study against its matched baseline, use `--repair both`.

### MA-Crafter

```bash
cd ma_crafter

python experiment/run_grid.py \
  --topologies individual centralized broadcast_chain \
  --agent-counts 3 \
  --seeds 42 \
  --repair off \
  --model your-deployment-name \
  --dry-run
```

Remove `--dry-run` to execute. Use `--repair both` for matched repair-off and repair-on conditions.

`--max-concurrent-agents` is a weighted concurrency budget. A six-agent run consumes six slots. The default value of `1` runs conditions sequentially.

## Communication structure names

The paper's **Broadcast Chain** uses a fixed speaker order. Each agent receives earlier messages and broadcasts its contribution to every later agent.

The Broadcast Chain implementation is named `llm_broadcast_chain.py` throughout the repository.

| Paper name | CLI and result name |
| --- | --- |
| Individual | `individual` |
| Centralized | `centralized` |
| Broadcast Chain | `broadcast_chain` |

## Interrupt handling

A message that arrives while an agent is waiting (`W`) or executing (`X`) moves it to the interrupted stage (`I`). The agent then re-enters the cognitive layer before the next primitive step:

| Role | On interrupt |
| --- | --- |
| Individual | Resumes. Individual agents exchange no messages, so only a COOP²-Repair request can interrupt them, and that triggers replanning. |
| Centralized leader | Re-runs its planning round: request, wait for follower responses, plan. Nothing interrupts a leader except a COOP²-Repair request. |
| Centralized follower | Replies to the leader, then asks the LLM whether to resume its committed plan or replan. |
| Broadcast Chain speaker | Asks the LLM whether to resume its committed plan or replan. Only a revised plan is broadcast to the later speakers. |

The resume-or-replan decision is `BaseLLMAgent.decide_interrupt` in `*/cognitive/agent/base_llm_agent.py`. It uses the structured `LLMInterruptResponse` schema (Listing 1 in the paper). COOP²-Repair requests always trigger replanning.

Both environments ship unit tests for this stage that use a scripted LLM client, so they need no API credentials:

```bash
cd cube && python -m unittest tests.test_interrupt_decision
cd ma_crafter && python -m unittest tests.test_interrupt_decision
```

## Outputs and analysis

Each run records files such as:

- `plan_logs.json`: symbolic plans and transitions
- `llm_usage.json`: per-agent and aggregate API usage
- `coop2_metrics.json` and `coop2_metrics.csv`: outcome and process metrics
- `episode.gif`: optional recording when video is enabled
- `manifest_*.json`: grid commands, condition metadata, status, and duration

Environment-specific table and figure commands are documented in:

- `cube/experiment/README.md`
- `ma_crafter/experiment/README.md`

## Add a new environment

A COOP² environment combines cognitive dynamics with grounded environment and cooperative task dynamics. CUBE and MA-Crafter use the same organization:

```text
COOP²
├── cognitive dynamics
│   ├── agents, messages, and symbolic plans
│   └── PlanningEnvWrapper
│       └── SymbolicEnvWrapper
├── grounded environment and cooperative task dynamics
│   ├── base parallel environment
│   └── task and constraint definitions
└── aligned cognitive-primitive process record
```

The links below use CUBE as the compact reference implementation; MA-Crafter mirrors the same structure.

| Directory | Responsibility |
| --- | --- |
| [`env`](cube/env/) | Implements primitive environment state, observations, actions, transitions, and task state. A new environment starts here or supplies an existing parallel environment. |
| [`cognitive/action`](cube/cognitive/action/) | Defines symbolic actions and their controllers. `SymbolicEnvWrapper` translates each agent's current symbolic action into a primitive environment action, advances the base environment, and returns grounded outcomes. |
| [`cognitive/agent`](cube/cognitive/agent/) | Implements the agent state machine, memory, prompts, LLM calls, plan generation, interruption handling, and replanning. Environment-specific observations and tool descriptions enter agent prompts here. |
| [`cognitive/plan`](cube/cognitive/plan/) | Defines `SymbolicPlan`, plan executors, plan logging, and `PlanningEnvWrapper`. The wrapper owns the agents and message broker, executes their plans through `SymbolicEnvWrapper`, and aligns cognitive events with primitive steps. |
| [`comm_topology`](cube/comm_topology/) | Defines who communicates with whom and in what order. The included factories construct Individual, Centralized, and Broadcast Chain agents. A topology class sets `role_prompt`, overrides the `_plan_*` hooks of `BaseLLMAgent` for its team and context, and adds its communication flow in `handle_reasoning` and `handle_interrupt`. |
| [`cognitive/viz`](cube/cognitive/viz/) | Optionally wraps `PlanningEnvWrapper` to display agent states, plans, actions, and messages or record an episode. It is not required to run an environment. |

`SymbolicEnvWrapper` is the **action-grounding wrapper**: it converts high-level operations such as `navigate`, `collect`, or `push` into the primitive action accepted by the environment at each step. `PlanningEnvWrapper` is the **cognitive-to-primitive orchestration wrapper**: it manages plans and agents above that action layer. Together, `cognitive/` and the grounded environment and task implementation form COOP².

To add an environment:

1. Provide a parallel environment with `reset()` and `step(actions)`.
2. Define its symbolic action vocabulary and implement the corresponding primitive controllers in `cognitive/action/`.
3. Add environment observations, tool descriptions, and plan parsing needed by the agents in `cognitive/agent/`.
4. Implement a COOP² adapter that returns `TaskSpec` objects from `get_task_specs()`. Use `required_agents` for the participation threshold, `required_capabilities` for capability or prerequisite needs, and `ConstraintResult` for the applicable spatial, temporal, and dependency checks.
5. Implement `get_action_outcomes(infos)` and `get_constraint_results(infos)` so grounded execution and task progress enter the process record, then connect the adapter through `PlanningEnvWrapper(..., coop2_adapter=adapter)`.

### Example: Simple Spread

Consider adapting COOP² to [Simple Spread](https://github.com/openai/multiagent-particle-envs), where *N* agents must cover *N* landmarks.

1. **Ground symbolic tools.** Define operations such as `move_to_landmark(landmark_id)` and `hold(num_steps)`, then translate them into primitive environment actions.
2. **Define the cooperative task.** Represent `cover_all_landmarks` as one task with participation requirement *N*, assigning each agent a distinct landmark.
3. **Instantiate the requirements.** The spatial requirement holds when each agent is within ε of its assigned landmark. The temporal requirement holds when all agents occupy their landmarks at the same environment step.
4. **Record cooperation.** Supply these definitions through the environment adapter so plans, grounded actions, task progress, and constraint satisfaction are recorded on the aligned cognitive-primitive timeline.

## COOP²-Repair

COOP²-Repair is an optional use of the formulation. After agents commit plans, it predicts whether those plans will satisfy the intended task requirements and can open a targeted communication channel before primitive execution.

```text
COOP²-Repair
├── plan-effect estimator
├── pre-execution constraint evaluator
├── repair controller
└── targeted repair messages
```

The shared flow is implemented by `coop2_repair/repair_controller.py`, `coop2_repair/evaluator.py`, and `coop2_repair/message_protocol.py`. The environment adapters provide the plan-effect estimates and repair guidance that depend on the domain.

The paper demonstrates this repair flow as a case study of what the COOP² formulation makes possible. Applying it to another domain requires three additional pieces beyond the base environment integration:

1. Extract the state needed to predict the effects of committed plans.
2. Implement `get_pre_execution_constraint_results_for_plans(...)` to estimate which cooperative requirements those plans will satisfy.
3. When needed, implement `build_repair_guidance(...)` to explain the predicted failure to the relevant agents.

For the Simple Spread example above, the estimator would predict destinations and arrival times from the current states and planned movements. The shared evaluator, controller, and message protocol would then handle triggering and targeted communication.

## Citation

```bibtex
@misc{yang2026coop2definingobservingrepairing,
  title={COOP$^2$: Defining, Observing, and Repairing Cooperation in LLM Multi-Agent Systems},
  author={Hanqing Yang and Narjes Nourzad and Shiyu Chen and Marie Siew and Jingdi Chen and Carlee Joe-Wong},
  year={2026},
  eprint={2603.00349},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2603.00349},
}
```

## License

Released under the [MIT License](LICENSE).
