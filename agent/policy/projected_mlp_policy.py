import numpy as np

from agent.policy.base_policy import Policy


class ProjectedMLPPolicy(Policy):
    """Two-layer MLP whose full weights are reconstructed from low-dim latent codes via fixed QR projections."""

    def __init__(
        self,
        dim_states,
        dim_actions,
        hidden_dim,
        latent_dim_layer1,
        latent_dim_layer2,
        projection_seed,
        activation="tanh",
        bias=True,
        latent_param_min=-6.0,
        latent_param_max=6.0,
        latent_init_std=2.0,
        decoded_weight_scale=1.0,
    ):
        super().__init__(dim_states, dim_actions)
        self.dim_states = int(dim_states)
        self.dim_actions = int(dim_actions)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim_layer1 = int(latent_dim_layer1)
        self.latent_dim_layer2 = int(latent_dim_layer2)
        self.projection_seed = int(projection_seed)
        self.activation = str(activation)
        self.use_bias = bool(bias)
        self.latent_param_min = float(latent_param_min)
        self.latent_param_max = float(latent_param_max)
        self.latent_init_std = float(latent_init_std)
        self.decoded_weight_scale = float(decoded_weight_scale)

        if self.activation != "tanh":
            raise ValueError(
                f"Only tanh activation is currently supported, got {self.activation}"
            )

        self.layer1_param_count = self.dim_states * self.hidden_dim + (
            self.hidden_dim if self.use_bias else 0
        )
        self.layer2_param_count = self.hidden_dim * self.dim_actions + (
            self.dim_actions if self.use_bias else 0
        )

        rng = np.random.RandomState(self.projection_seed)
        self.layer1_projection = self._build_projection(
            rng, self.layer1_param_count, self.latent_dim_layer1
        )
        self.layer2_projection = self._build_projection(
            rng, self.layer2_param_count, self.latent_dim_layer2
        )

        self.latent = np.zeros(self.rank, dtype=float)
        self.weight1 = np.zeros((self.dim_states, self.hidden_dim), dtype=float)
        self.bias1 = np.zeros(self.hidden_dim, dtype=float)
        self.weight2 = np.zeros((self.hidden_dim, self.dim_actions), dtype=float)
        self.bias2 = np.zeros(self.dim_actions, dtype=float)

    @property
    def rank(self):
        return self.latent_dim_layer1 + self.latent_dim_layer2

    def _build_projection(self, rng, full_dim, latent_dim):
        gaussian = rng.randn(full_dim, latent_dim)
        q, _ = np.linalg.qr(gaussian)
        return q[:, :latent_dim]

    def initialize_policy(self):
        latent = np.random.normal(0.0, self.latent_init_std, size=self.rank)
        latent = np.clip(latent, self.latent_param_min, self.latent_param_max)
        self.update_policy(np.round(latent, 1))

    def _decode_layer(self, latent, projection, out_dim, in_dim):
        flat = self.decoded_weight_scale * (projection @ latent.reshape(-1))
        if self.use_bias:
            weight_count = in_dim * out_dim
            weight = flat[:weight_count].reshape(in_dim, out_dim)
            bias = flat[weight_count:]
        else:
            weight = flat.reshape(in_dim, out_dim)
            bias = np.zeros(out_dim, dtype=float)
        return weight, bias

    def _activate(self, x):
        return np.tanh(x)

    def get_action(self, state):
        state = np.asarray(state, dtype=float).reshape(-1)
        if state.shape[0] != self.dim_states:
            raise ValueError(
                f"Expected state with {self.dim_states} values, got {state.shape[0]}"
            )
        hidden = self._activate(state @ self.weight1 + self.bias1)
        action = hidden @ self.weight2 + self.bias2
        return action.reshape(1, self.dim_actions)

    def update_policy(self, latent_list):
        if latent_list is None:
            return

        latent = np.asarray(latent_list, dtype=float).reshape(-1)
        if latent.size != self.rank:
            raise ValueError(f"Expected {self.rank} latent params, got {latent.size}")

        latent = np.clip(latent, self.latent_param_min, self.latent_param_max)
        self.latent = latent

        z1 = latent[: self.latent_dim_layer1]
        z2 = latent[self.latent_dim_layer1 :]
        self.weight1, self.bias1 = self._decode_layer(
            z1,
            self.layer1_projection,
            self.hidden_dim,
            self.dim_states,
        )
        self.weight2, self.bias2 = self._decode_layer(
            z2,
            self.layer2_projection,
            self.dim_actions,
            self.hidden_dim,
        )

    def get_parameters(self):
        return self.latent.reshape(-1)

    def get_projection_metadata(self):
        return {
            "projection_seed": self.projection_seed,
            "hidden_dim": self.hidden_dim,
            "latent_dim_layer1": self.latent_dim_layer1,
            "latent_dim_layer2": self.latent_dim_layer2,
            "layer1_param_count": self.layer1_param_count,
            "layer2_param_count": self.layer2_param_count,
            "activation": self.activation,
            "latent_param_min": self.latent_param_min,
            "latent_param_max": self.latent_param_max,
            "latent_init_std": self.latent_init_std,
            "decoded_weight_scale": self.decoded_weight_scale,
        }

    def __str__(self):
        lines = [
            "Projected MLP policy",
            f"projection_seed={self.projection_seed}",
            f"hidden_dim={self.hidden_dim}",
            f"latent_dim_layer1={self.latent_dim_layer1}",
            f"latent_dim_layer2={self.latent_dim_layer2}",
            f"activation={self.activation}",
            f"latent_param_range=[{self.latent_param_min}, {self.latent_param_max}]",
            f"latent_init_std={self.latent_init_std}",
            f"decoded_weight_scale={self.decoded_weight_scale}",
            "Latent:",
            ", ".join(str(v) for v in self.latent.tolist()),
            "Layer1 weight:",
        ]
        for row in self.weight1:
            lines.append(", ".join(str(v) for v in row))
        if self.use_bias:
            lines.append("Layer1 bias:")
            lines.append(", ".join(str(v) for v in self.bias1.tolist()))
        lines.append("Layer2 weight:")
        for row in self.weight2:
            lines.append(", ".join(str(v) for v in row))
        if self.use_bias:
            lines.append("Layer2 bias:")
            lines.append(", ".join(str(v) for v in self.bias2.tolist()))
        return "\n".join(lines)
