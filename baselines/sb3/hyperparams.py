import ast
import os

import yaml


def _parse_schedule_or_scalar(value):
    if not isinstance(value, str):
        return value
    if value.startswith("lin_"):
        return float(value.split("_", 1)[1])
    return value


def _parse_policy_kwargs(value):
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value.startswith("dict("):
        return value
    safe_globals = {"__builtins__": {}}
    safe_locals = {"dict": dict}
    return eval(value, safe_globals, safe_locals)


def _parse_normalize(value):
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        return parsed
    return value


def _postprocess_hyperparams(hyperparams):
    result = {}
    for key, value in hyperparams.items():
        if key in {"n_timesteps", "policy", "env_wrapper", "frame_stack"}:
            continue
        if key == "policy_kwargs":
            result[key] = _parse_policy_kwargs(value)
        elif key == "normalize":
            result[key] = _parse_normalize(value)
        elif key == "train_freq" and isinstance(value, list):
            result[key] = tuple(value)
        else:
            result[key] = _parse_schedule_or_scalar(value)
    return result


def load_zoo_hyperparams(
    algorithm,
    gym_env_name,
    zoo_root="rl-baselines3-zoo/hyperparams",
):
    algo_file = os.path.join(zoo_root, f"{algorithm.lower()}.yml")
    if not os.path.exists(algo_file):
        return {}, "missing"

    with open(algo_file, "r") as f:
        data = yaml.safe_load(f) or {}

    if gym_env_name in data:
        chosen = data[gym_env_name] or {}
        source = "exact"
    else:
        chosen = data.get("default", {}) or {}
        source = "default"
    return _postprocess_hyperparams(dict(chosen)), source


def get_hyperparams(
    algorithm,
    env_name,
    gym_env_name,
    config_overrides=None,
    zoo_root="rl-baselines3-zoo/hyperparams",
):
    del env_name
    result, source = load_zoo_hyperparams(
        algorithm=algorithm,
        gym_env_name=gym_env_name,
        zoo_root=zoo_root,
    )
    if config_overrides:
        result.update(config_overrides)
    return result, source
