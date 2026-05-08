import re
import time
import traceback

import numpy as np

from ablations.actor_second_pass.llm_brain_two_pass import (
    TwoPassLLMBrain,
)
from agent.llm_num_optim_linear_policy_semantics import LLMNumOptimSemanticAgent
from agent.policy.replay_buffer import ReplayBuffer


class ActorSecondPassLinearSemanticsAgent(LLMNumOptimSemanticAgent):
    """Budget-matched control: evaluate A, then run a generic second optimizer step."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_iterations = None
        self.llm_brain = TwoPassLLMBrain(
            self.llm_brain.llm_si_template,
            self.llm_brain.llm_output_conversion_template,
            self.llm_brain.llm_model_name,
        )
        self._proposal_a_traj_buffer = ReplayBuffer(
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
            return np.array([indexed_results[i] for i in range(self.rank)]).reshape(-1)

        def history_string(extra_rows=None):
            rows = []
            for parameters, reward in self.replay_buffer.buffer:
                params = parameters.reshape(-1)
                row = "".join(
                    f"params[{i}]: {params[i]:.5g}; " for i in range(self.rank)
                )
                row += f"f(params): {reward:.2f}\n"
                rows.append(row)
            if extra_rows:
                rows.extend(extra_rows)
            return "".join(rows)

        def history_row(parameters, reward):
            params = parameters.reshape(-1)
            row = "".join(f"params[{i}]: {params[i]:.5g}; " for i in range(self.rank))
            row += f"f(params): {reward:.2f}\n"
            return row

        print("Actor second pass: proposal A...")
        try:
            proposal_a, reasoning_a, api_time_a = (
                self.llm_brain.llm_propose_standard(
                    history_string(),
                    parse_parameters,
                    self.training_episodes,
                    self.rank,
                    self.optimum,
                    search_step_size=self.search_step_size,
                )
            )
            self.api_call_time += api_time_a
        except Exception as e:
            print("Exception in proposal A")
            print(traceback.format_exc())
            raise e

        print("Actor second pass: evaluating proposal A...")
        self.policy.update_policy(proposal_a)
        proposal_a_results = []
        proposal_a_trajs = []

        with open(f"{logdir}/proposal_a_rollout.txt", "w") as logging_file:
            for _ in range(self.num_evaluation_episodes):
                buf = ReplayBuffer(1, self.traj_buffer.max_traj_length)
                buf.start_new_trajectory()
                result = self._rollout_episode_to_buffer(world, logging_file, buf)
                proposal_a_results.append(result)
                proposal_a_trajs.append(buf)

        proposal_a_reward = float(np.mean(proposal_a_results))
        proposal_a_repr_idx = int(
            np.argmin(np.abs(np.array(proposal_a_results) - proposal_a_reward))
        )
        self._proposal_a_traj_buffer = proposal_a_trajs[proposal_a_repr_idx]

        print("Actor second pass: proposal B using updated history...")
        try:
            proposal_b, reasoning_b, api_time_b = (
                self.llm_brain.llm_propose_standard(
                    history_string([history_row(proposal_a, proposal_a_reward)]),
                    parse_parameters,
                    self.training_episodes,
                    self.rank,
                    self.optimum,
                    search_step_size=self.search_step_size,
                )
            )
            self.api_call_time += api_time_b
        except Exception as e:
            print("Exception in proposal B")
            print(traceback.format_exc())
            raise e

        print("Actor second pass: evaluating proposal B...")
        self.policy.update_policy(proposal_b)
        proposal_b_results = []
        proposal_b_trajs = []

        with open(f"{logdir}/proposal_b_rollout.txt", "w") as logging_file:
            for _ in range(self.num_evaluation_episodes):
                buf = ReplayBuffer(1, self.traj_buffer.max_traj_length)
                buf.start_new_trajectory()
                result = self._rollout_episode_to_buffer(world, logging_file, buf)
                proposal_b_results.append(result)
                proposal_b_trajs.append(buf)

        proposal_b_reward = float(np.mean(proposal_b_results))
        proposal_b_repr_idx = int(
            np.argmin(np.abs(np.array(proposal_b_results) - proposal_b_reward))
        )
        proposal_b_traj_buffer = proposal_b_trajs[proposal_b_repr_idx]

        if proposal_b_reward >= proposal_a_reward:
            winner = "proposal_b"
            best_params = proposal_b
            best_reward = proposal_b_reward
            winning_traj_buffer = proposal_b_traj_buffer
        else:
            winner = "proposal_a"
            best_params = proposal_a
            best_reward = proposal_a_reward
            winning_traj_buffer = self._proposal_a_traj_buffer

        self.policy.update_policy(best_params)
        self.traj_buffer.start_new_trajectory()
        for state, action, reward in winning_traj_buffer.buffer[0].buffer:
            self.traj_buffer.add_step(state, action, reward)

        self.replay_buffer.add(best_params, best_reward)

        with open(f"{logdir}/parameters.txt", "w") as f:
            f.write(str(self.policy))
        with open(f"{logdir}/parameters_reasoning.txt", "w") as f:
            f.write(
                f"Winner: {winner}\n"
                f"Proposal A reward: {proposal_a_reward:.4f}\n"
                f"Proposal B reward: {proposal_b_reward:.4f}\n\n"
                f"--- Proposal A ---\n{reasoning_a}\n\n"
                f"--- Proposal B ---\n{reasoning_b}"
            )
        with open(f"{logdir}/selection_outcome.txt", "w") as f:
            f.write(
                f"proposal_a_reward={proposal_a_reward:.4f}\n"
                f"proposal_b_reward={proposal_b_reward:.4f}\n"
                f"winner={winner}\n"
                f"proposal_a_repr_idx={proposal_a_repr_idx}\n"
                f"proposal_b_repr_idx={proposal_b_repr_idx}\n"
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
        logging_file.write("parameter ends\n\n")
        logging_file.write("state | action | reward\n")
        done = False
        while not done:
            action = self.policy.get_action(state.T)
            action = np.reshape(action, (1, self.dim_action))
            if world.discretize:
                action = np.argmax(action)
                action = np.array([action])
            next_state, reward, done = world.step(action)
            logging_file.write(f"{state.T[0]} | {action[0]} | {reward}\n")
            traj_buffer.add_step(state, action, reward)
            state = next_state
            self.total_steps += 1
        logging_file.write(f"Total reward: {world.get_accu_reward()}\n")
        self.total_episodes += 1
        return world.get_accu_reward()
