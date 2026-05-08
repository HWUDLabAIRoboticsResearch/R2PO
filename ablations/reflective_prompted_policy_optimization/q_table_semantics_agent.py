import re
import time
import traceback

import numpy as np

from ablations.reflective_prompted_policy_optimization.context_builder import (
    build_search_llm_history_text,
    build_critic_llm_reflection_context,
    build_median_trajectory_stats_summary,
    select_median_rollout_index,
)
from ablations.reflective_prompted_policy_optimization.llm_brain_reflective import (
    ReflectiveLLMBrain,
)
from agent.llm_num_optim_q_table_semantics import LLMNumOptimQTableSemanticsAgent
from agent.policy.replay_buffer import ReplayBuffer


class ReflectivePromptedPolicyOptimizationQTableSemanticsAgent(
    LLMNumOptimQTableSemanticsAgent
):
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

        print("R2PO: Phase 1 - Search-LLM proposing params...")
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
                    actions=self.actions,
                    max_iterations=self.max_iterations,
                )
            )
            self.api_call_time += search_llm_api_time
        except Exception as e:
            print("Exception in Phase 1 (Search-LLM proposal)")
            print(traceback.format_exc())
            raise e

        print("R2PO: evaluating proposal...")
        self.q_table.update_policy(initial_params)
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
        median_idx = select_median_rollout_index(initial_results)
        median_traj_reward = initial_results[median_idx]

        trajectory_summary = build_median_trajectory_stats_summary(
            initial_results, all_traj_buffers, optimum=self.optimum
        )

        print("R2PO: Phase 2 - Critic-LLM reflecting...")
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
            actions=self.actions,
            warmup_replay_count=getattr(self, "warmup_replay_count", 0),
            conservative_threshold=getattr(self, "critic_llm_conservative_threshold", None),
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

        print("R2PO: evaluating revised proposal...")
        self.q_table.update_policy(revised_params)
        revised_results = []

        with open(f"{logdir}/revised_proposal_rollout.txt", "w") as logging_file:
            for _ in range(self.num_evaluation_episodes):
                buf = ReplayBuffer(1, self.traj_buffer.max_traj_length)
                buf.start_new_trajectory()
                result = self._rollout_episode_to_buffer(world, logging_file, buf)
                revised_results.append(result)

        revised_reward = float(np.mean(revised_results))
        revised_median_idx = select_median_rollout_index(revised_results)

        if revised_reward >= initial_reward:
            winner = "revision_won"
            best_params = revised_params
            best_reward = revised_reward
        else:
            winner = "original_won"
            best_params = initial_params
            best_reward = initial_reward

        self.q_table.update_policy(best_params)
        self.replay_buffer.add(best_params, best_reward)

        with open(f"{logdir}/parameters.txt", "w") as f:
            f.write(str(self.q_table.mapping))
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
                f"median_traj_rollout_idx={median_idx}\n"
                f"median_traj_individual_reward={median_traj_reward:.4f}\n"
                f"revised_median_traj_rollout_idx={revised_median_idx}\n"
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
        logging_file.write("state | action | reward\n")
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
