import gymnasium as gym
import numpy as np
from gymnasium import spaces

import gym_maze  # noqa: F401
from envs import nim, pong  # noqa: F401


class DiscreteToOneHotObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Discrete)
        self.n = int(env.observation_space.n)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n,),
            dtype=np.float32,
        )

    def observation(self, observation):
        obs = np.zeros(self.n, dtype=np.float32)
        obs[int(observation)] = 1.0
        return obs


class MazeCoordinateToIndexObservation(gym.ObservationWrapper):
    def __init__(self, env, width=3):
        super().__init__(env)
        self.width = int(width)
        self.observation_space = spaces.Discrete(self.width * self.width)

    def observation(self, observation):
        x = int(observation[0])
        y = int(observation[1])
        return y * self.width + x


class NimActionShiftWrapper(gym.ActionWrapper):
    """Map agent actions {0,1,2} to intended Nim moves {1,2,3}."""

    def action(self, action):
        return int(action) + 1


def _build_raw_env(gym_env_name, render_mode=None, env_kwargs=None):
    env_kwargs = dict(env_kwargs or {})
    if gym_env_name == "maze-sample-3x3-v0":
        return gym.make(gym_env_name, enable_render=render_mode)
    return gym.make(gym_env_name, render_mode=render_mode, **env_kwargs)


def make_env(env_name, gym_env_name, render_mode=None, env_kwargs=None):
    env = _build_raw_env(gym_env_name, render_mode=render_mode, env_kwargs=env_kwargs)

    if gym_env_name == "maze-sample-3x3-v0":
        env = MazeCoordinateToIndexObservation(env, width=3)
        env = DiscreteToOneHotObservation(env)
    elif gym_env_name == "FrozenLake-v1":
        env = DiscreteToOneHotObservation(env)
    elif gym_env_name == "Nim-v0":
        if env_name == "nim":
            env = NimActionShiftWrapper(env)
        env = DiscreteToOneHotObservation(env)

    return env
