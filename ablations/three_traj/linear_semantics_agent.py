import re
import time
import traceback

import numpy as np

from ablations.three_traj.context_builder import (
    build_search_llm_history_text,
    build_critic_llm_reflection_context,
    build_three_trajectory_summary,
)
from ablations.three_traj.llm_brain_reflective import (
    ReflectiveLLMBrain,
)
from agent.llm_num_optim_linear_policy_semantics import LLMNumOptimSemanticAgent
from agent.policy.replay_buffer import ReplayBuffer


class ReflectiveThreeTrajectoriesLinearSemanticsAgent(LLMNumOptimSemanticAgent):
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
        self.llm_brain = ReflectiveLLMBrain(
            self.llm_brain.llm_si_template,
            self.llm_brain.llm_output_conversion_template,
            self.llm_brain.llm_model_name,
        )
        self._proposal_traj_buffer = ReplayBuffer(
            1, self.traj_buffer.max_traj_length
        )

    def train_policy(self, world, logdir):
        def parse_parameters(input_text):
            print("response:", input_text[:200])
            pattern = re.compile(
                r"params\[(\d+)\]:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
            )
            matches = pattern.findall(input_text)
            indexed_results = {}
            for idx_str, value_str in matches:
                idx = int(idx_str)
                if 0 <= idx < self.rank and idx not in indexed_results:
                    indexed_results[idx] = float(value_str)
            if len(indexed_results) != self.rank:
                raise AssertionError(
                    f"Expected {self.rank} params, got {len(indexed_results)}."
                )
            return np.array(
                [indexed_results[i] for i in range(self.rank)]
            ).reshape(-1)

        print("Reflective three trajectories: Phase 1 - Search-LLM proposing params...")
        try:
            initial_params, search_llm_reasoning, search_llm_api_time = (
                self.llm_brain.search_llm_propose_initial(
                    build_search_llm_history_text(
                        self.replay_buffer,
                        warmup_replay_count=getattr(self, "warmup_replay_count", 0),
                    ),
                    parse_parameters,
                    self.training_episodes,
                    self.rank,
                    self.optimum,
                    search_step_size=self.search_step_size,
                )
            )
            self.api_call_time += search_llm_api_time
        except Exception as e:
            print("Exception in Phase 1 (Search-LLM proposal)")
            print(traceback.format_exc())
            raise e

        print("Reflective three trajectories: evaluating proposal...")
        self.policy.update_policy(initial_params)
        initial_results = []
        all_traj_buffers = []

        with open(f"{logdir}/initial_proposal_rollout.txt", "w") as logging_file:
            for _ in range(self.num_evaluation_episodes):
                buf = ReplayBuffer(1, self.traj_buffer.max_traj_length)
                buf.start_new_trajectory()
                result = self._rollout_episode_to_buffer(world, logging_file, buf)
                initial_results.append(result)
                all_traj_buffers.append(buf)

        initial_reward = float(np.mean(initial_results))
        diffs = np.abs(np.array(initial_results) - initial_reward)
        repr_idx = int(np.argmin(diffs))
        self._proposal_traj_buffer = all_traj_buffers[repr_idx]
        repr_traj_reward = initial_results[repr_idx]

        trajectory_summary = build_three_trajectory_summary(
            initial_results, all_traj_buffers
        )

        print("Reflective three trajectories: Phase 2 - Critic-LLM reflecting...")
        critic_llm_context = build_critic_llm_reflection_context(
            initial_params=initial_params,
            initial_reward=initial_reward,
            trajectory_summary=trajectory_summary,
            replay_buffer=self.replay_buffer,
            step_number=self.training_episodes,
            env_desc_file=self.critic_llm_env_desc_file,
            max_iterations=self.max_iterations,
            rank=self.rank,
            optimum=self.optimum,
            search_step_size=self.search_step_size,
            warmup_replay_count=getattr(self, "warmup_replay_count", 0),
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

        print("Reflective three trajectories: evaluating revised proposal...")
        self.policy.update_policy(revised_params)
        revised_results = []
        all_revised_traj_buffers = []

        with open(f"{logdir}/revised_proposal_rollout.txt", "w") as logging_file:
            for _ in range(self.num_evaluation_episodes):
                buf = ReplayBuffer(1, self.traj_buffer.max_traj_length)
                buf.start_new_trajectory()
                result = self._rollout_episode_to_buffer(world, logging_file, buf)
                revised_results.append(result)
                all_revised_traj_buffers.append(buf)

        revised_reward = float(np.mean(revised_results))
        revised_diffs = np.abs(np.array(revised_results) - revised_reward)
        revised_repr_idx = int(np.argmin(revised_diffs))
        self._revised_traj_buffer = all_revised_traj_buffers[revised_repr_idx]

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

        self.traj_buffer.start_new_trajectory()
        for state, action, reward in winning_traj_buffer.buffer[0].buffer:
            self.traj_buffer.add_step(state, action, reward)

        self.replay_buffer.add(best_params, best_reward)

        with open(f"{logdir}/parameters.txt", "w") as f:
            f.write(str(self.policy))
        with open(f"{logdir}/parameters_reasoning.txt", "w") as f:
            f.write(
                f"Winner: {winner}\n"
                f"Initial reward: {initial_reward:.4f}\n"
                f"Revised reward: {revised_reward:.4f}\n\n"
                f"--- Search-LLM (Proposal) ---\n{search_llm_reasoning}\n\n"
                f"--- Critic-LLM (Reflection) ---\n{critic_llm_reasoning}"
            )
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

    def _rollout_episode_to_buffer(self, world, logging_file, traj_buffer):
        state = world.reset()
        state = np.expand_dims(state, axis=0)
        logging_file.write(
            f"{', '.join([str(x) for x in self.policy.get_parameters().reshape(-1)])}\n"
        )
        logging_file.write("parameter ends\n\nstate | action | reward\n")
        done = False
        while not done:
            action = self.policy.get_action(state.T)
            action = np.reshape(action, (1, self.dim_action))
            if world.discretize:
                action = np.argmax(action)
                action = np.array([action])
            next_state, reward, done = world.step(action)
            logging_file.write(f"{state.T[0]} | {action[0]} | {reward}\n")
            traj_buffer.add_step(state.squeeze(0), action, reward)
            state = np.expand_dims(next_state, axis=0)
            self.total_steps += 1
        logging_file.write(f"Total reward: {world.get_accu_reward()}\n")
        self.total_episodes += 1
        return world.get_accu_reward()
