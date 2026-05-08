import re
import time
import traceback

import numpy as np

from ablations.reptraj.context_builder import (
    build_critic_llm_reflection_context,
    format_trajectory_summary,
)
from ablations.reptraj.llm_brain_reflective import ReflectiveLLMBrain
from agent.llm_num_optim_q_table_semantics import LLMNumOptimQTableSemanticsAgent
from agent.policy.replay_buffer import ReplayBuffer


class ReflectiveQTableSemanticsAgent(LLMNumOptimQTableSemanticsAgent):
    def __init__(
        self,
        *args,
        jinja2_env,
        critic_llm_template_name,
        critic_llm_env_desc_file=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.jinja2_env = jinja2_env
        self.critic_llm_template_name = critic_llm_template_name
        self.critic_llm_env_desc_file = critic_llm_env_desc_file or self.env_desc_file
        self.max_iterations = None
        # Replace the brain with the reflective variant
        self.llm_brain = ReflectiveLLMBrain(
            self.llm_brain.llm_si_template,
            self.llm_brain.llm_output_conversion_template,
            self.llm_brain.llm_model_name,
        )
        # Temporary trajectory buffer for the initial proposal rollout
        self._proposal_traj_buffer = ReplayBuffer(
            1, self.traj_buffer.max_traj_length
        )

    def train_policy(self, world, logdir):
        def parse_parameters(input_text):
            print("response:", input_text)
            pattern = re.compile(r"params\[(\d+)\]\s*[:=]\s*([+-]?\d+(?:\.\d+)?)")
            matches = pattern.findall(input_text)
            indexed_results = {}
            for idx_str, value_str in matches:
                idx = int(idx_str)
                if 0 <= idx < self.rank:
                    indexed_results[idx] = float(value_str)
            if len(indexed_results) != self.rank:
                raise AssertionError(
                    f"Expected {self.rank} params, parsed {len(indexed_results)}."
                )
            return np.array(
                [indexed_results[i] for i in range(self.rank)]
            ).reshape((self.rank,))

        def str_nd_examples(replay_buffer, n):
            all_parameters = []
            for weights, reward in replay_buffer.buffer:
                parameters = weights
                all_parameters.append((parameters.reshape(-1), reward))
            text = ""
            for idx, (parameters, reward) in enumerate(all_parameters):
                l = ""
                for i in range(n):
                    l += f"params[{i}]: {parameters[i]:.5g}; "
                l += f"f(params): {reward:.2f}\n"
                text += l
            return text

        # Phase 1: Search-LLM proposes parameters (standard ProPS, no env desc)
        print("RepTraj: Phase 1 - Search-LLM proposing params...")
        try:
            initial_params, search_llm_reasoning, search_llm_api_time = (
                self.llm_brain.search_llm_propose_initial(
                    str_nd_examples(self.replay_buffer, self.rank),
                    parse_parameters,
                    self.training_episodes,
                    self.rank,
                    self.optimum,
                    actions=self.actions,
                )
            )
            self.api_call_time += search_llm_api_time
        except Exception as e:
            print("Exception in Phase 1 (Search-LLM proposal)")
            print(traceback.format_exc())
            raise e

        # Phase 1 evaluation: rollout ALL K proposals, record every trajectory
        print("RepTraj: evaluating proposal (all rollouts)...")
        self.q_table.update_policy(initial_params)
        initial_results = []
        all_traj_buffers = []

        logging_filename = f"{logdir}/initial_proposal_rollout.txt"
        with open(logging_filename, "w") as logging_file:
            for idx in range(self.num_evaluation_episodes):
                buf = ReplayBuffer(1, self.traj_buffer.max_traj_length)
                buf.start_new_trajectory()
                result = self._rollout_episode_to_buffer(world, logging_file, buf)
                initial_results.append(result)
                all_traj_buffers.append(buf)

        initial_reward = float(np.mean(initial_results))

        # Select the trajectory whose individual reward is closest to the mean
        diffs = np.abs(np.array(initial_results) - initial_reward)
        repr_idx = int(np.argmin(diffs))
        self._proposal_traj_buffer = all_traj_buffers[repr_idx]
        repr_traj_reward = initial_results[repr_idx]
        print(
            f"Initial proposal reward (mean): {initial_reward:.4f} | "
            f"representative rollout: idx={repr_idx}, r={repr_traj_reward:.4f}"
        )

        # Phase 2: Critic-LLM reflects on outcome and proposes revision
        print("RepTraj: Phase 2 - Critic-LLM reflecting...")
        trajectory_summary = format_trajectory_summary(self._proposal_traj_buffer)

        critic_llm_context = build_critic_llm_reflection_context(
            initial_params=initial_params,
            initial_reward=initial_reward,
            trajectory_summary=trajectory_summary,
            stats_summary=None,
            replay_buffer=self.replay_buffer,
            step_number=self.training_episodes,
            env_desc_file=self.critic_llm_env_desc_file,
            max_iterations=self.max_iterations,
            rank=self.rank,
            optimum=self.optimum,
            actions=self.actions,
            traj_buffer=self._critic_llm_traj_buffer(),
            **( {"traj_history_last_n": self.traj_history_last_n} if getattr(self, "traj_history_last_n", None) is not None else {}),
        )

        try:
            revised_params, critic_llm_reasoning, critic_llm_api_time = (
                self.llm_brain.critic_llm_reflect_and_revise(
                    parse_parameters=parse_parameters,
                    jinja2_env=self.jinja2_env,
                    critic_llm_template_name=self.critic_llm_template_name,
                    critic_llm_context=critic_llm_context,
                )
            )
            self.api_call_time += critic_llm_api_time
        except Exception as e:
            print("Exception in Phase 2 (Critic-LLM reflection)")
            print(traceback.format_exc())
            raise e

        # Phase 2 evaluation: rollout ALL K revised proposals, record every trajectory
        print("RepTraj: evaluating revised proposal (all rollouts)...")
        self.q_table.update_policy(revised_params)
        revised_results = []
        all_revised_traj_buffers = []

        logging_filename = f"{logdir}/revised_proposal_rollout.txt"
        with open(logging_filename, "w") as logging_file:
            for idx in range(self.num_evaluation_episodes):
                buf = ReplayBuffer(1, self.traj_buffer.max_traj_length)
                buf.start_new_trajectory()
                result = self._rollout_episode_to_buffer(world, logging_file, buf)
                revised_results.append(result)
                all_revised_traj_buffers.append(buf)

        revised_reward = float(np.mean(revised_results))

        # Select representative trajectory for revised proposal too
        revised_diffs = np.abs(np.array(revised_results) - revised_reward)
        revised_repr_idx = int(np.argmin(revised_diffs))
        self._revised_traj_buffer = all_revised_traj_buffers[revised_repr_idx]
        print(f"Revised proposal reward (mean): {revised_reward:.4f}")

        # Keep best
        if revised_reward >= initial_reward:
            winner = "revision_won"
            best_params = revised_params
            best_reward = revised_reward
            winning_traj_buffer = self._revised_traj_buffer
        else:
            winner = "original_won"
            best_params = initial_params
            best_reward = initial_reward
            winning_traj_buffer = self._proposal_traj_buffer

        print(f"Winner: {winner} (reward={best_reward:.4f})")

        # Record winning trajectory to self.traj_buffer (used in optimization history)
        self.traj_buffer.start_new_trajectory()
        for state, action, reward in winning_traj_buffer.buffer[0].buffer:
            self.traj_buffer.add_step(state, action, reward)

        # Store winner in replay buffer
        self.replay_buffer.add(best_params, best_reward)

        # Log parameters and reasoning
        with open(f"{logdir}/parameters.txt", "w") as f:
            f.write(str(self.q_table.mapping))
        reasoning = (
            f"Winner: {winner}\n"
            f"Initial reward: {initial_reward:.4f}\n"
            f"Revised reward: {revised_reward:.4f}\n\n"
            f"--- Search-LLM (Proposal) ---\n{search_llm_reasoning}\n\n"
            f"--- Critic-LLM (Reflection) ---\n{critic_llm_reasoning}"
        )
        with open(f"{logdir}/parameters_reasoning.txt", "w") as f:
            f.write(reasoning)

        # Log reflection outcome
        with open(f"{logdir}/reflection_outcome.txt", "w") as f:
            f.write(
                f"initial_reward={initial_reward:.4f}\n"
                f"revised_reward={revised_reward:.4f}\n"
                f"winner={winner}\n"
                f"repr_traj_rollout_idx={repr_idx}\n"
                f"repr_traj_individual_reward={repr_traj_reward:.4f}\n"
            )

        self.training_episodes += 1
        return (
            time.process_time() - self.start_time,
            self.api_call_time,
            self.total_episodes,
            self.total_steps,
            best_reward,
        )

    def _critic_llm_traj_buffer(self):
        return self.traj_buffer

    def _rollout_episode_to_buffer(self, world, logging_file, traj_buffer):
        """Rollout episode recording trajectory to a specific buffer."""
        state = world.reset()
        logging_file.write(f"state | action | reward\n")
        done = False
        while not done:
            action = self.q_table.get_action(state)
            action = self.format_env_action(action)
            next_state, reward, done = world.step(action)
            logging_file.write(f"{state} | {action} | {reward}\n")
            traj_buffer.add_step(state, action, reward)
            state = next_state
            self.total_steps += 1
        logging_file.write(f"Total reward: {world.get_accu_reward()}\n")
        self.total_episodes += 1
        return world.get_accu_reward()
