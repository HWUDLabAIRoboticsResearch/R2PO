from agent.policy.linear_policy_no_bias import LinearPolicy as LinearPolicyNoBias
from agent.policy.linear_policy import LinearPolicy
from agent.policy.projected_mlp_policy import ProjectedMLPPolicy
from agent.policy.replay_buffer import EpisodeRewardBufferNoBias
from agent.policy.llm_brain_linear_policy import LLMBrain
from world.base_world import BaseWorld
import numpy as np
import re
import time


class LLMNumOptimAgent:
    def __init__(
        self,
        logdir,
        dim_action,
        dim_state,
        max_traj_count,
        max_traj_length,
        llm_si_template,
        llm_output_conversion_template,
        llm_model_name,
        num_evaluation_episodes,
        bias,
        optimum,
        search_step_size,
        policy_variant=None,
        hidden_dim=None,
        latent_dim_layer1=None,
        latent_dim_layer2=None,
        projection_seed=None,
        activation="tanh",
        param_min=-6.0,
        param_max=6.0,
        latent_init_std=2.0,
        decoded_weight_scale=1.0,
    ):
        self.start_time = time.process_time()
        self.api_call_time = 0
        self.total_steps = 0
        self.total_episodes = 0
        self.dim_action = dim_action
        self.dim_state = dim_state
        self.obs_dim = dim_state
        self.bias = bias
        self.optimum = optimum
        self.search_step_size = search_step_size
        self.hidden_dim = hidden_dim
        self.latent_dim_layer1 = latent_dim_layer1
        self.latent_dim_layer2 = latent_dim_layer2
        self.projection_seed = projection_seed
        self.activation = activation
        self.param_min = float(param_min)
        self.param_max = float(param_max)
        self.latent_init_std = float(latent_init_std)
        self.decoded_weight_scale = float(decoded_weight_scale)
        self.max_iterations = None

        if policy_variant is None:
            if hidden_dim is not None and latent_dim_layer1 is not None and latent_dim_layer2 is not None:
                policy_variant = "projected_mlp_qr"
            else:
                policy_variant = "linear"
        self.policy_variant = str(policy_variant)

        if self.policy_variant == "projected_mlp_qr":
            if hidden_dim is None or latent_dim_layer1 is None or latent_dim_layer2 is None:
                raise ValueError(
                    "Projected MLP policy requires hidden_dim, latent_dim_layer1, and latent_dim_layer2."
                )
            if projection_seed is None:
                raise ValueError("Projected MLP policy requires projection_seed.")
            self.rank = int(latent_dim_layer1) + int(latent_dim_layer2)
            self.policy = ProjectedMLPPolicy(
                dim_states=dim_state,
                dim_actions=dim_action,
                hidden_dim=hidden_dim,
                latent_dim_layer1=latent_dim_layer1,
                latent_dim_layer2=latent_dim_layer2,
                projection_seed=projection_seed,
                activation=activation,
                bias=bias,
                latent_param_min=param_min,
                latent_param_max=param_max,
                latent_init_std=latent_init_std,
                decoded_weight_scale=decoded_weight_scale,
            )
            self.parameter_layout_text = self._build_parameter_layout_text()
        else:
            if not self.bias:
                param_count = dim_action * dim_state
            else:
                param_count = dim_action * dim_state + dim_action
            self.rank = param_count
            if not self.bias:
                self.policy = LinearPolicyNoBias(
                    dim_actions=dim_action, dim_states=dim_state
                )
            else:
                self.policy = LinearPolicy(dim_actions=dim_action, dim_states=dim_state)
            self.parameter_layout_text = ""
        self.replay_buffer = EpisodeRewardBufferNoBias(max_size=max_traj_count)
        self.llm_brain = LLMBrain(
            llm_si_template, llm_output_conversion_template, llm_model_name
        )
        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0

        if self.bias and self.policy_variant == "linear":
            self.dim_state += 1

    def _build_parameter_layout_text(self):
        if self.policy_variant != "projected_mlp_qr":
            return ""
        layer1_end = self.latent_dim_layer1 - 1
        layer2_start = self.latent_dim_layer1
        layer2_end = self.rank - 1
        metadata = self.policy.get_projection_metadata()
        return "\n".join(
            [
                f"- params[0] through params[{layer1_end}] are the latent code z1 for layer 1.",
                f"  z1 is mapped through a fixed orthonormal matrix Q1 into the full first-layer parameter vector with {metadata['layer1_param_count']} values.",
                f"- params[{layer2_start}] through params[{layer2_end}] are the latent code z2 for layer 2.",
                f"  z2 is mapped through a fixed orthonormal matrix Q2 into the full second-layer parameter vector with {metadata['layer2_param_count']} values.",
                f"- The reconstructed network is {self.obs_dim} -> {self.hidden_dim} -> {self.dim_action} with {self.activation} activation in the hidden layer.",
            ]
        )

    def _build_template_context(self):
        if self.policy_variant != "projected_mlp_qr":
            return None
        metadata = self.policy.get_projection_metadata()
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.dim_action,
            "hidden_dim": self.hidden_dim,
            "latent_dim_layer1": self.latent_dim_layer1,
            "latent_dim_layer2": self.latent_dim_layer2,
            "layer1_param_count": metadata["layer1_param_count"],
            "layer2_param_count": metadata["layer2_param_count"],
            "projection_seed": self.projection_seed,
            "activation": self.activation,
            "param_min": self.param_min,
            "param_max": self.param_max,
            "parameter_layout_text": self.parameter_layout_text,
        }

    def rollout_episode(self, world: BaseWorld, logging_file, record=True):
        state = world.reset()
        state = np.expand_dims(state, axis=0)
        logging_file.write(
            f"{', '.join([str(x) for x in self.policy.get_parameters().reshape(-1)])}\n"
        )
        logging_file.write(f"parameter ends\n\n")
        logging_file.write(f"state | action | reward\n")
        done = False
        step_idx = 0
        while not done:
            action = self.policy.get_action(state.T)
            action = np.reshape(action, (1, self.dim_action))
            if world.discretize:
                action = np.argmax(action)
                action = np.array([action])
                env_action = action
            else:
                env_action = np.clip(
                    action,
                    world.env.action_space.low,
                    world.env.action_space.high,
                )
            next_state, reward, done = world.step(env_action)
            logging_file.write(f"{state.T[0]} | {env_action[0]} | {reward}\n")
            state = next_state
            step_idx += 1
            self.total_steps += 1
        logging_file.write(f"Total reward: {world.get_accu_reward()}\n")
        self.total_episodes += 1
        if record:
            self.replay_buffer.add(
                self.policy.get_parameters(), world.get_accu_reward()
            )
        return world.get_accu_reward()

    def random_warmup(self, world: BaseWorld, logdir, num_episodes):
        for episode in range(num_episodes):
            self.policy.initialize_policy()
            # Run the episode and collect the trajectory
            print(f"Rolling out warmup episode {episode}...")
            logging_filename = f"{logdir}/warmup_rollout_{episode}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logging_file)
            print(f"Result: {result}")

    def train_policy(self, world: BaseWorld, logdir):

        def parse_parameters(input_text):
            s = input_text.split("\n")[0]
            print("response:", s)
            pattern = re.compile(r"params\[(\d+)\]:\s*([+-]?\d+(?:\.\d+)?)")
            matches = pattern.findall(s)

            results = []
            for match in matches:
                results.append(float(match[1]))
            if not results:
                array_match = re.search(r"params\s*=\s*\[([^\]]+)\]", s)
                if array_match:
                    values = [
                        token.strip()
                        for token in array_match.group(1).split(",")
                        if token.strip()
                    ]
                    results = [float(token) for token in values]
            print(results)
            assert len(results) == self.rank
            return np.array(results).reshape(-1)

        def str_nd_examples(replay_buffer: EpisodeRewardBufferNoBias, n):

            all_parameters = []
            for weights, reward in replay_buffer.buffer:
                parameters = weights
                all_parameters.append((parameters.reshape(-1), reward))

            text = ""
            for parameters, reward in all_parameters:
                l = ""
                for i in range(n):
                    l += f"params[{i}]: {parameters[i]:.5g}; "
                fxy = reward
                l += f"f(params): {fxy:.2f}\n"
                text += l
            return text

        print("Updating the policy...")
        new_parameter_list, reasoning, api_time = self.llm_brain.llm_update_parameters_num_optim(
            str_nd_examples(self.replay_buffer, self.rank),
            parse_parameters,
            self.training_episodes,
            self.rank,
            self.optimum,
            self.search_step_size,
            max_iterations=self.max_iterations,
            param_min=self.param_min,
            param_max=self.param_max,
            template_context=self._build_template_context(),
        )
        self.api_call_time += api_time

        print(self.policy.get_parameters().shape)
        print(new_parameter_list.shape)
        self.policy.update_policy(new_parameter_list)
        print(self.policy.get_parameters().shape)
        logging_q_filename = f"{logdir}/parameters.txt"
        logging_q_file = open(logging_q_filename, "w")
        logging_q_file.write(str(self.policy))
        logging_q_file.close()
        q_reasoning_filename = f"{logdir}/parameters_reasoning.txt"
        q_reasoning_file = open(q_reasoning_filename, "w")
        q_reasoning_file.write(reasoning)
        q_reasoning_file.close()
        print("Policy updated!")

        # Run the episode and collect the trajectory
        print(f"Rolling out episode {self.training_episodes}...")
        logging_filename = f"{logdir}/training_rollout.txt"
        logging_file = open(logging_filename, "w")
        results = []
        for idx in range(self.num_evaluation_episodes):
            if idx == 0:
                result = self.rollout_episode(world, logging_file, record=False)
            else:
                result = self.rollout_episode(world, logging_file, record=False)
            results.append(result)
        print(f"Results: {results}")
        result = np.mean(results)
        self.replay_buffer.add(new_parameter_list, result)

        self.training_episodes += 1

        _cpu_time = time.process_time() - self.start_time
        _api_time = self.api_call_time
        _total_episodes = self.total_episodes
        _total_steps = self.total_steps
        _total_reward = result
        return _cpu_time, _api_time, _total_episodes, _total_steps, _total_reward
    

    def evaluate_policy(self, world: BaseWorld, logdir):
        results = []
        for idx in range(self.num_evaluation_episodes):
            logging_filename = f"{logdir}/evaluation_rollout_{idx}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logging_file, record=False)
            results.append(result)
        return results
