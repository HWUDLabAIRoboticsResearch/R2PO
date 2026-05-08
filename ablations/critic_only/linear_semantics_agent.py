import re
import time
import traceback

import numpy as np

from ablations.critic_only.context_builder import (
    build_critic_only_context,
)
from ablations.critic_only.llm_brain_reflective import (
    ReflectiveLLMBrain,
)
from agent.llm_num_optim_linear_policy_semantics import LLMNumOptimSemanticAgent
from agent.policy.replay_buffer import ReplayBuffer


class ReflectiveCriticOnlyLinearSemanticsAgent(LLMNumOptimSemanticAgent):
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
            return np.array([indexed_results[i] for i in range(self.rank)]).reshape(-1)

        critic_llm_context = build_critic_only_context(
            replay_buffer=self.replay_buffer,
            step_number=self.training_episodes,
            env_desc_file=self.critic_llm_env_desc_file,
            max_iterations=self.max_iterations,
            rank=self.rank,
            optimum=self.optimum,
            search_step_size=self.search_step_size,
            traj_buffer=self.traj_buffer,
            **(
                {"traj_history_last_n": self.traj_history_last_n}
                if getattr(self, "traj_history_last_n", None) is not None
                else {}
            ),
        )

        try:
            params, reasoning, api_time = self.llm_brain.llm_single_pass_reflective(
                parse_parameters=parse_parameters,
                jinja2_env=self.jinja2_env,
                critic_llm_template_name=self.critic_llm_template_name,
                critic_llm_context=critic_llm_context,
            )
            self.api_call_time += api_time
        except Exception as e:
            print("Exception in single-pass CriticOnly proposal")
            print(traceback.format_exc())
            raise e

        self.policy.update_policy(params)

        results = []
        all_traj_buffers = []
        with open(f"{logdir}/training_rollout.txt", "w") as logging_file:
            for _ in range(self.num_evaluation_episodes):
                buf = ReplayBuffer(1, self.traj_buffer.max_traj_length)
                buf.start_new_trajectory()
                result = self._rollout_episode_to_buffer(world, logging_file, buf)
                results.append(result)
                all_traj_buffers.append(buf)

        reward = float(np.mean(results))
        repr_idx = int(np.argmin(np.abs(np.array(results) - reward)))
        winning_traj_buffer = all_traj_buffers[repr_idx]

        self.traj_buffer.start_new_trajectory()
        for state, action, step_reward in winning_traj_buffer.buffer[0].buffer:
            self.traj_buffer.add_step(state, action, step_reward)

        self.replay_buffer.add(params, reward)

        with open(f"{logdir}/parameters.txt", "w") as f:
            f.write(str(self.policy))
        with open(f"{logdir}/parameters_reasoning.txt", "w") as f:
            f.write(reasoning)
        with open(f"{logdir}/critic_only_outcome.txt", "w") as f:
            f.write(
                f"reward={reward:.4f}\n"
                f"repr_traj_rollout_idx={repr_idx}\n"
            )

        self.training_episodes += 1
        return (
            time.process_time() - self.start_time,
            self.api_call_time,
            self.total_episodes,
            self.total_steps,
            reward,
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
