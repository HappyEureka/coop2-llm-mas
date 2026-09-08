"""
Planning environment wrapper for CUBE.

This wrapper sits on top of SymbolicEnvWrapper and manages plan execution:
- Steps through agent plans one primitive action at a time
- Tracks plan status (in progress, success, failure) after each step
- Triggers plan regeneration when plans terminate
"""

import time
from typing import Dict, List, Optional, Any, Callable, Union
from coop2_repair import (
    Coop2RepairController,
    Coop2TraceLogger,
    CubeCoopAdapter,
    PettingZooParallelAdapter,
)
from ..action.action_env_wrapper import SymbolicEnvWrapper
from .plan import SymbolicPlan, SymbolicPlanExecutor, SymbolicPlanLogger
from .coop2_process_logger import Coop2ProcessLogger
from .coop2_repair_dispatcher import Coop2RepairDispatcher
from ..agent import Agent


class PlanningEnvWrapper:
    """
    Wrapper that manages plan-based agent execution.
    
    Handles:
    - Executing one primitive action per agent per step (from their current plans)
    - Tracking plan status and detecting termination
    - Triggering plan generation when needed
    - Synchronous stepping of all agents
    
    Usage:
        env = PlanningEnvWrapper(base_env, agent_names=['alice', 'bob'])
        env.set_plan_generators({'alice': alice_planner, 'bob': bob_planner})
        
        obs, info = env.reset()
        
        while not done:
            # Wrapper handles plan execution and regeneration internally
            obs, rewards, terminated, truncated, info = env.step()
    """
    
    def __init__(
        self,
        base_env,
        agent_names: List[str],
        agents: Optional[Dict[str, Agent]] = None,
        logger: Optional[SymbolicPlanLogger] = None,
        coop2_adapter: Optional[Any] = None,
        coop2_precheck_enabled: bool = False,
        **env_kwargs
    ):
        """
        Initialize the planning wrapper.
        
        Args:
            base_env: The base CUBE environment (CoopBlockPush) class or instance
            agent_names: List of agent names
            agents: Optional dict mapping agent_id to Agent instances
            logger: Optional shared logger for plan tracking
            **env_kwargs: Additional environment arguments
        """
        # Create the symbolic wrapper
        self.symbolic_env = SymbolicEnvWrapper(base_env, agent_names=agent_names, **env_kwargs)
        self.agent_names = agent_names
        
        # Store agents dict (initially empty)
        self.agents: Dict[str, Optional[Agent]] = {aid: None for aid in agent_names}
        
        # Create shared logger
        self.logger = logger or SymbolicPlanLogger()
        self.coop2_trace = Coop2TraceLogger()
        self.coop2_adapter = coop2_adapter or self._default_coop2_adapter()
        self.coop2_repair_controller = Coop2RepairController(
            adapter=self.coop2_adapter,
            trace_logger=self.coop2_trace,
            enabled=coop2_precheck_enabled,
        )
        
        # Create plan executors for each agent (with None agent references initially)
        self.plan_executors: Dict[str, SymbolicPlanExecutor] = {}
        for agent_id in agent_names:
            self.plan_executors[agent_id] = SymbolicPlanExecutor(
                agent_id=agent_id,
                agent=None,  # Will be set when agents are provided
                plan_generator=None,  # Set later via set_plan_generators or use agents
                logger=self.logger
            )
        
        # Track previous action results for each agent
        self._prev_action_results: Dict[str, Optional[Dict]] = {aid: None for aid in agent_names}
        
        # Track current observations for plan generation
        self._current_obs: Dict[str, Any] = {}
        self._current_step: int = 0
        self._current_info: Dict[str, Any] = {}
        self.coop2_process_logger = Coop2ProcessLogger(
            symbolic_env=self.symbolic_env,
            agent_names=self.agent_names,
            agents=self.agents,
            coop2_adapter=self.coop2_adapter,
        )
        self.coop2_repair_dispatcher = Coop2RepairDispatcher(
            agents=self.agents,
            message_broker_getter=lambda: self._message_broker,
        )
        
        # Message broker for coordinated message handling (created when agents are set)
        self._message_broker = None
        
        # Event notification for state changes (event-driven interrupt handling)
        import threading
        self._state_change_event = threading.Condition()
        
        # Set agents if provided (this creates the message broker and assigns it to agents)
        if agents:
            self.set_agents(agents)

    def _default_coop2_adapter(self):
        """Use the CUBE adapter when the wrapped env exposes face-task tracking."""
        base_env = getattr(self.symbolic_env, "env", self.symbolic_env)
        if hasattr(base_env, "task_tracker"):
            return CubeCoopAdapter(self.symbolic_env)
        return PettingZooParallelAdapter(self.symbolic_env)

    def set_coop2_precheck_enabled(self, enabled: bool):
        """Enable or disable optional COOP2 pre-execution repair checks."""
        self.coop2_repair_controller.set_enabled(enabled)

    def enable_coop2_precheck(self):
        self.set_coop2_precheck_enabled(True)

    def disable_coop2_precheck(self):
        self.set_coop2_precheck_enabled(False)

    @property
    def coop2_precheck_enabled(self) -> bool:
        return self.coop2_repair_controller.enabled

    @coop2_precheck_enabled.setter
    def coop2_precheck_enabled(self, enabled: bool):
        self.coop2_repair_controller.set_enabled(enabled)

    @property
    def coop2_process_log(self) -> List[Dict[str, Any]]:
        return self.coop2_process_logger.records
    
    def set_plan_generators(self, generators: Dict[str, Callable]):
        """
        Set plan generator functions for agents.
        
        Args:
            generators: Dict mapping agent_id to generator function
                Generator signature: (agent_id, observation, env_step) -> SymbolicPlan
        """
        for agent_id, generator in generators.items():
            if agent_id in self.plan_executors:
                self.plan_executors[agent_id].plan_generator = generator
    
    def set_plan_generator(self, agent_id: str, generator: Callable):
        """Set plan generator for a single agent."""
        if agent_id in self.plan_executors:
            self.plan_executors[agent_id].plan_generator = generator
    
    def set_agents(self, agents: Dict[str, Agent]):
        """
        Set Agent instances for autonomous plan generation.
        Automatically creates a MessageBroker and assigns it to all agents.
        
        Args:
            agents: Dict mapping agent_id to Agent instance
        """
        for agent_id, agent in agents.items():
            if agent_id in self.agents:
                self.agents[agent_id] = agent
                # Update executor's agent reference
                if agent_id in self.plan_executors:
                    self.plan_executors[agent_id].agent = agent
        
        # Create message broker and assign to all agents
        from ..messages import MessageBroker
        self._message_broker = MessageBroker(agents)
        self._message_broker.wrapper = self
        for agent in agents.values():
            agent.message_broker = self._message_broker
    
    def set_agent(self, agent_id: str, agent: Agent):
        """Set a single Agent instance."""
        if agent_id in self.agents:
            self.agents[agent_id] = agent
            # Update executor's agent reference
            if agent_id in self.plan_executors:
                self.plan_executors[agent_id].agent = agent
    
    def set_message_broker(self, message_broker):
        """
        Set the message broker for coordinated message handling.
        
        Args:
            message_broker: MessageBroker instance
        """
        self._message_broker = message_broker
        self._message_broker.wrapper = self
        # Also assign to all agents
        for agent in self.agents.values():
            if agent is not None:
                agent.message_broker = message_broker
    
    @property
    def message_broker(self):
        """Get the message broker."""
        return self._message_broker
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment. Plans need to be set manually after reset."""
        obs_dict, info = self.symbolic_env.reset(seed=seed, options=options)
        
        self._current_obs = obs_dict
        self._current_info = info  # Store info for accessing symbolic_view
        self._current_step = 0
        self._prev_action_results = {aid: None for aid in self.agent_names}
        self.coop2_repair_controller.reset()
        self.coop2_process_logger.reset(info)
        
        # Get coop config from base env if available
        coop_config = None
        base_env = self.symbolic_env.env
        if hasattr(base_env, 'get_config_observation'):
            coop_config = base_env.get_config_observation()
        
        # Reset agents if they exist
        for agent_id, agent in self.agents.items():
            if agent is not None:
                agent.reset()
                # Set env info if agent supports it
                if hasattr(agent, 'set_env_info'):
                    symbolic_view = info.get(agent_id, {}).get('symbolic_view')
                    agent.set_env_info(coop_config=coop_config, symbolic_view=symbolic_view)
                # Give initial observation to agents
                agent.observe(obs_dict[agent_id], self._current_step)
        
        # Reset plan executors to need new plans
        for agent_id in self.agent_names:
            self.plan_executors[agent_id].needs_new_plan = True
        
        return obs_dict, info

    def _idle_step_return(self, extra_info: Optional[Dict[str, Any]] = None):
        """Return current observations without advancing the underlying env."""
        info = {}
        for agent_id in self.agent_names:
            agent_info = self._current_info.get(agent_id, {}) if isinstance(self._current_info, dict) else {}
            info[agent_id] = dict(agent_info) if isinstance(agent_info, dict) else {}
            if extra_info:
                info[agent_id].update(extra_info)
        return (
            self._current_obs,
            {agent_id: 0.0 for agent_id in self.agent_names},
            {agent_id: False for agent_id in self.agent_names},
            {agent_id: False for agent_id in self.agent_names},
            info,
        )
    
    def wait_for_state_change(self, timeout: float = 0.1) -> bool:
        """
        Wait for any agent state change (event-driven).
        
        Args:
            timeout: Max time to wait in seconds
        
        Returns:
            True if notified, False if timeout
        """
        with self._state_change_event:
            return self._state_change_event.wait(timeout)
    
    def notify_state_change(self):
        """Notify that an agent state has changed."""
        with self._state_change_event:
            self._state_change_event.notify_all()
    
    def step(self):
        """
        Execute one environment step for all agents.
        
        Reads plans from agent.plan directly.
        
        Steps through each agent's plan one primitive action at a time.
        After execution, checks symbolic action status and updates plan status.
        
        Returns:
            tuple: (observations, rewards, terminated, truncated, info)
        """
        # Check if all agents are ready (in W state) and transition to X before execution
        all_ready = all(
            self.agents[aid] is not None and self.agents[aid].ready 
            for aid in self.agent_names
        )
        managed_agents = any(agent is not None for agent in self.agents.values())
        if managed_agents and not all_ready:
            waiting_for = [
                aid
                for aid in self.agent_names
                if self.agents[aid] is None or not self.agents[aid].ready
            ]
            return self._idle_step_return({"waiting_for_agents": waiting_for})

        if all_ready:
            evaluation = self.coop2_repair_controller.before_execution(
                agents=self.agents,
                env_step=self._current_step,
                current_info=getattr(self, "_current_info", None),
                repair_dispatcher=self.coop2_repair_dispatcher.dispatch,
            )
            if evaluation is not None and evaluation.should_repair:
                return self._idle_step_return(
                    {"coop2_pre_execution": evaluation.to_dict()}
                )

            # Use same timestamp for all agents transitioning together
            transition_time = time.time()
            for agent_id in self.agent_names:
                agent = self.agents[agent_id]
                if agent is not None and agent.state.value == "waiting":
                    agent.start_execution(timestamp=transition_time, env_step=self._current_step)
        
        # Get next action from each agent's current plan
        actions = {}
        for agent_id in self.agent_names:
            agent = self.agents[agent_id]
            plan = agent.plan if agent is not None else None
            old_plan = self.logger.current_plans.get(agent_id)
            if (
                plan is not None
                and plan.status.value == "pending"
                and old_plan is not None
                and old_plan is not plan
            ):
                self.symbolic_env.agent_actions[agent_id].cancel_current_action(
                    "plan_replaced"
                )
            action = self.plan_executors[agent_id].step(
                observation=self._current_obs[agent_id],
                env_step=self._current_step,
                action_results=self._prev_action_results[agent_id]
            )
            actions[agent_id] = action
        process_agent_views = self.coop2_process_logger.build_agent_views(actions)
        
        # Execute all actions in the environment (one step)
        obs_dict, rewards, terminated, truncated, info = self.symbolic_env.step(actions)
        
        # Update current state
        self._current_obs = obs_dict
        self._current_step += 1
        
        # NOW check symbolic action status and update plan status
        for agent_id in self.agent_names:
            # Get the agent FIRST before accessing its plan
            agent = self.agents[agent_id]
            action_handler = self.symbolic_env.agent_actions[agent_id]
            action_records = action_handler.get_action_records()
            
            # Get the latest symbolic action status
            if action_records:
                latest_record = action_records[-1]
                action_status = {
                    "status": latest_record["status"],
                    "failure_reason": latest_record.get("failure_reason")
                }
                # Update the plan's SymbolicAction with the current primitive_action for visualization
                if agent is not None and agent.plan:
                    current_action = agent.plan.get_current_action()
                    if current_action and latest_record.get("primitive_action"):
                        current_action.primitive_action = latest_record["primitive_action"]
                    # Also sync primitive_action_history for multi-step action visualization
                    if current_action and latest_record.get("primitive_action_history"):
                        current_action.primitive_action_history.update(latest_record["primitive_action_history"])
            else:
                action_status = None
            
            # Store for next iteration
            self._prev_action_results[agent_id] = action_status
            
            # Check plan status and handle completion/failure
            if agent is not None and agent.plan and action_status:
                plan = agent.plan
                executor = self.plan_executors[agent_id]  # Reference for setting needs_new_plan
                current_action = plan.get_current_action()
                
                # If symbolic action succeeded, advance and check completion
                if action_status["status"] == "success" and current_action is not None and current_action.start_step is not None:
                    # Log action completion
                    self.logger.log_action_completed(plan, current_action, self._current_step, True, None)
                    # Advance to next action
                    plan.advance_action()
                    
                    # Check if plan is complete (was last action)
                    if plan.is_complete():
                        if plan.status.value == "executing":
                            plan.complete_success(self._current_step)
                            self.logger.log_plan_completed(plan, self._current_step, True)
                            executor.needs_new_plan = True
                            # Transition agent to R (reasoning) state when plan completes
                            if agent_id in self.agents and self.agents[agent_id] is not None:
                                self.agents[agent_id].set_unready(
                                    reason='plan_terminated',
                                    env_step=self._current_step
                                )
                    else:
                        # There's a next action - log it as starting for next iteration
                        next_action = plan.get_current_action()
                        if next_action is not None:
                            self.logger.log_action_started(plan, next_action, self._current_step)
                
                # If symbolic action failed, plan fails
                elif action_status["status"] == "failed":
                    if plan.status.value == "executing" and current_action is not None:
                        failure_reason = action_status.get("failure_reason", "Action failed")
                        self.logger.log_action_completed(plan, current_action, self._current_step, False, failure_reason)
                        plan.complete_failed(self._current_step, failure_reason)
                        self.logger.log_plan_completed(plan, self._current_step, False, failure_reason)
                        executor.needs_new_plan = True
                        # Transition agent to R (reasoning) state when plan fails
                        if agent_id in self.agents and self.agents[agent_id] is not None:
                            self.agents[agent_id].set_unready(
                                reason='plan_terminated',
                                env_step=self._current_step
                            )
        
        # Sync agent observations and env info
        # Get coop config from base env if available
        coop_config = None
        base_env = self.symbolic_env.env
        if hasattr(base_env, 'get_config_observation'):
            coop_config = base_env.get_config_observation()
        
        for agent_id, agent in self.agents.items():
            if agent is not None and agent_id in obs_dict:
                # Set env info if agent supports it
                if hasattr(agent, 'set_env_info'):
                    symbolic_view = info.get(agent_id, {}).get('symbolic_view')
                    agent.set_env_info(coop_config=coop_config, symbolic_view=symbolic_view)
                agent.observe(obs_dict[agent_id], self._current_step)
        
        self._current_info = info  # Store info for later access
        self.coop2_process_logger.record_step(
            env_step=self._current_step,
            info=info,
            agent_views=process_agent_views,
        )
        
        return obs_dict, rewards, terminated, truncated, info
    
    def get_agent_status(self, agent_id: str) -> Dict:
        """Get current status of an agent's plan execution."""
        if agent_id in self.plan_executors:
            return self.plan_executors[agent_id].get_status()
        return {"status": "unknown"}
    
    def get_all_agent_status(self) -> Dict[str, Dict]:
        """Get status for all agents."""
        return {
            agent_id: self.plan_executors[agent_id].get_status()
            for agent_id in self.agent_names
        }
    
    def get_agents_needing_plans(self) -> List[str]:
        """Get list of agents that need new plans."""
        return [
            agent_id for agent_id in self.agent_names
            if self.plan_executors[agent_id].needs_new_plan
        ]
    
    def agents_need_plans(self) -> Dict[str, bool]:
        """Get dict mapping agent_id to whether they need a new plan."""
        return {
            agent_id: self.plan_executors[agent_id].needs_new_plan
            for agent_id in self.agent_names
        }
    
    def set_agent_plan(self, agent_id: str, plan: SymbolicPlan):
        """
        Manually set a plan for an agent (external plan generation).
        
        Args:
            agent_id: Agent to set plan for
            plan: Plan to execute
        """
        if agent_id in self.plan_executors:
            # Get plan_id from agent's plan_count if agent exists
            plan_id = None
            if agent_id in self.agents and self.agents[agent_id] is not None:
                plan_id = self.agents[agent_id].plan_count
            self.plan_executors[agent_id].set_plan(plan, self._current_step, plan_id=plan_id)
    
    def terminate_unfinished_plans(self):
        """Terminate all unfinished plans (call at episode end)."""
        self.logger.terminate_unfinished_plans(self._current_step)
    
    def save_plan_logs(self, filename: str):
        """Save plan execution logs."""
        self.logger.save_logs(filename)
    
    def save_agent_log(self, output_path: str):
        """Save agent state timelines to JSON."""
        import json
        
        agent_states = {}
        for agent_id, agent in self.agents.items():
            if agent is not None:
                # Convert AgentState enum to string value
                timeline = agent.get_state_timeline()
                agent_states[agent_id] = [
                    (time, step, state.value) for time, step, state in timeline
                ]
        
        with open(output_path, 'w') as f:
            json.dump(agent_states, f, indent=2)
        print(f"Saved agent state timelines to {output_path}")
    
    def save_message_log(self, output_path: str):
        """Save inter-agent messages to JSON."""
        import json
        
        if self._message_broker is not None:
            message_log = self._message_broker.get_message_log()
            with open(output_path, 'w') as f:
                json.dump(message_log, f, indent=2)
            print(f"Saved {len(message_log)} messages to {output_path}")
    
    def save_task_log(self, output_path: str):
        """Save task states history to JSON (only changed tasks to reduce file size)."""
        import json
        from env.cooperative_tasks import convert_to_serializable
        
        # Access through wrapper chain: symbolic_env -> env (CooperativeEnv)
        base_env = self.symbolic_env.env
        if hasattr(base_env, 'task_tracker'):
            task_history = base_env.task_tracker.get_history()
            if task_history:
                # Save metrics + only changed tasks at each step
                lightweight_data = []
                previous_tasks = {}
                
                for summary in task_history:
                    step_data = {
                        'step': summary.env_step,
                        'metrics': convert_to_serializable(summary.metrics.to_dict()) if summary.metrics else None,
                        'changed_tasks': {},
                        'collected': summary.successful_collections,
                        'failed_attempts': summary.failed_attempts
                    }
                    
                    # Find tasks that changed (new, modified, or about to be collected)
                    current_task_ids = set(summary.tasks.keys())
                    previous_task_ids = set(previous_tasks.keys())
                    
                    # New tasks
                    new_tasks = current_task_ids - previous_task_ids
                    # Removed tasks (collected)
                    removed_tasks = previous_task_ids - current_task_ids
                    # Common tasks - check if they changed
                    common_tasks = current_task_ids & previous_task_ids
                    
                    changed_task_ids = set(new_tasks) | set(removed_tasks)
                    
                    # Check common tasks for changes
                    for task_id in common_tasks:
                        curr = summary.tasks[task_id]
                        prev = previous_tasks[task_id]
                        
                        # Check if any dimension changed (FaceTaskState uses methods for counts)
                        if (curr.get_spatial_count() != prev.get_spatial_count() or
                            curr.get_temporal_count() != prev.get_temporal_count() or
                            curr.participation_met != prev.participation_met or
                            curr.participation_ratio != prev.participation_ratio or
                            curr.status != prev.status):
                            changed_task_ids.add(task_id)
                    
                    # Save only changed tasks
                    for task_id in changed_task_ids:
                        if task_id in summary.tasks:
                            step_data['changed_tasks'][str(task_id)] = convert_to_serializable(
                                summary.tasks[task_id].to_dict()
                            )
                    
                    lightweight_data.append(step_data)
                    previous_tasks = summary.tasks.copy()
                
                import os
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(lightweight_data, f, indent=2)
                print(f"Saved {len(lightweight_data)} task state snapshots (changed tasks only) to {output_path}")
    
    def save_capability_log(self, output_path: str):
        """Save capability change history to JSON (if CooperativeEnv is used)."""
        # CUBE doesn't have capability changes (all agents have same capabilities)
        # This method is a no-op for CUBE environments
        base_env = self.symbolic_env.env
        if hasattr(base_env, 'task_tracker') and hasattr(base_env.task_tracker, 'capability_history'):
            capability_history = base_env.task_tracker.capability_history
            if capability_history:
                save_capability_log(capability_history, output_path)

    def save_coop2_process_log(self, output_path: str):
        """Save compact COOP2 step-aligned process records."""
        self.coop2_process_logger.save(output_path)

    def save_team_score_log(self, output_path: str):
        """Save cumulative team score if the env provides it."""
        import json
        import os

        base_env = self.symbolic_env.env
        if not hasattr(base_env, "get_team_score_breakdown"):
            return
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(base_env.get_team_score_breakdown(include_events=True), f, indent=2)
        print(f"Saved team score summary to {output_path}")
    
    def save_logs(self, output_dir: str):
        """
        Save all experiment logs and data.
        
        Saves:
        - plan_logs.json: Plan execution history
        - agent_states.json: Agent state timelines (R/W/X/I transitions)
        - message_log.json: Inter-agent messages
        - task_states.json: Task states history (if CooperativeEnv is used)
        - capability_changes.json: Agent capability changes (if CooperativeEnv is used)
        
        Args:
            output_dir: Directory to save all logs
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plan logs
        plan_logs_path = os.path.join(output_dir, "plan_logs.json")
        self.save_plan_logs(plan_logs_path)
        
        # Save agent state histories
        agent_states_path = os.path.join(output_dir, "agent_states.json")
        self.save_agent_log(agent_states_path)
        
        # Save message log
        message_log_path = os.path.join(output_dir, "message_log.json")
        self.save_message_log(message_log_path)
        
        # Save task states
        task_states_path = os.path.join(output_dir, "task_states.json")
        self.save_task_log(task_states_path)
        
        # Save capability changes
        capability_path = os.path.join(output_dir, "capability_changes.json")
        self.save_capability_log(capability_path)

        # Save COOP2 process and event traces
        self.save_coop2_process_log(os.path.join(output_dir, "coop2_process_log.json"))
        self.coop2_trace.save(os.path.join(output_dir, "coop2_trace.json"))
        self.save_team_score_log(os.path.join(output_dir, "team_score.json"))
    
    def get_plan_statistics(self) -> Dict:
        """Get statistics about plan execution."""
        return self.logger.get_statistics()
    
    @property
    def current_step(self) -> int:
        """Get current environment step."""
        return self._current_step
    
    # Forward attribute access to symbolic env
    def __getattr__(self, name):
        """Forward attribute access to the wrapped symbolic environment."""
        return getattr(self.symbolic_env, name)
