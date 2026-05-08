import numpy as np

from agent.policy.base_policy import Policy


class RandomFeaturePolicy(Policy):
    """Frozen tanh random-feature encoder with a trainable linear readout."""

    def __init__(
        self,
        dim_states,
        dim_actions,
        hidden_dim,
        projection_seed,
        bias=True,
        param_min=-2.0,
        param_max=2.0,
        readout_init_std=0.75,
        obs_scale=3.0,
        obs_clip=5.0,
        output_gain=0.5,
    ):
        super().__init__(dim_states, dim_actions)
        self.dim_states = dim_states
        self.dim_actions = dim_actions
        self.hidden_dim = hidden_dim
        self.projection_seed = projection_seed
        self.use_bias = bias
        self.param_min = float(param_min)
        self.param_max = float(param_max)
        self.readout_init_std = float(readout_init_std)
        self.obs_scale = float(obs_scale)
        self.obs_clip = float(obs_clip)
        self.output_gain = float(output_gain)

        rng = np.random.RandomState(self.projection_seed)
        self.random_weight = rng.randn(self.hidden_dim, self.dim_states) / np.sqrt(
            self.dim_states
        )
        self.random_bias = rng.uniform(-0.1, 0.1, size=self.hidden_dim)

        self.readout = np.zeros((self.dim_actions, self.hidden_dim), dtype=float)
        self.bias = np.zeros(self.dim_actions, dtype=float)

    def initialize_policy(self):
        self.readout = np.round(
            np.clip(
                np.random.normal(
                    0.0,
                    self.readout_init_std,
                    size=(self.dim_actions, self.hidden_dim),
                ),
                self.param_min,
                self.param_max,
            ),
            1,
        )
        if self.use_bias:
            self.bias = np.round(
                np.clip(
                    np.random.normal(0.0, self.readout_init_std, size=self.dim_actions),
                    self.param_min,
                    self.param_max,
                ),
                1,
            )
        else:
            self.bias = np.zeros(self.dim_actions, dtype=float)

    def _features(self, state):
        state = np.asarray(state, dtype=float).reshape(-1)
        if state.shape[0] != self.dim_states:
            raise ValueError(
                f"Expected state with {self.dim_states} values, got {state.shape[0]}"
            )
        normalized_state = np.clip(state / self.obs_scale, -self.obs_clip, self.obs_clip)
        return np.tanh(self.random_weight @ normalized_state + self.random_bias)

    def get_action(self, state):
        hidden = self._features(state)
        action = self.output_gain * (self.readout @ hidden)
        if self.use_bias:
            action = action + self.output_gain * self.bias
        return action.reshape(1, self.dim_actions)

    def update_policy(self, readout_and_bias_list):
        if readout_and_bias_list is None:
            return

        flat = np.asarray(readout_and_bias_list, dtype=float).reshape(-1)
        expected = self.dim_actions * self.hidden_dim + (
            self.dim_actions if self.use_bias else 0
        )
        if flat.size != expected:
            raise ValueError(f"Expected {expected} parameters, got {flat.size}")

        readout_count = self.dim_actions * self.hidden_dim
        flat = np.clip(flat, self.param_min, self.param_max)
        self.readout = flat[:readout_count].reshape(self.dim_actions, self.hidden_dim)
        if self.use_bias:
            self.bias = flat[readout_count:]
        else:
            self.bias = np.zeros(self.dim_actions, dtype=float)

    def get_parameters(self):
        if self.use_bias:
            return np.concatenate([self.readout.reshape(-1), self.bias.reshape(-1)])
        return self.readout.reshape(-1)

    def __str__(self):
        lines = [
            "Frozen random-feature policy",
            f"projection_seed={self.projection_seed}",
            f"hidden_dim={self.hidden_dim}",
            "activation=tanh",
            f"param_range=[{self.param_min}, {self.param_max}]",
            f"readout_init_std={self.readout_init_std}",
            f"obs_scale={self.obs_scale}",
            f"output_gain={self.output_gain}",
            "Readout:",
        ]
        for row in self.readout:
            lines.append(", ".join(str(v) for v in row))
        if self.use_bias:
            lines.append("Bias:")
            lines.append(", ".join(str(v) for v in self.bias))
        return "\n".join(lines)
