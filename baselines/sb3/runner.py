import argparse
import json
import os
import time
import numpy as np
import yaml
from dotenv import load_dotenv
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from sb3_contrib import TRPO

from baselines.sb3.envs import make_env
from baselines.sb3.hyperparams import get_hyperparams
from baselines.sb3.logging_utils import (
    EpisodeRewardLoggingCallback,
    write_algorithm_summary,
    write_run_summary,
)

load_dotenv()

ALGORITHMS = {
    "TRPO": TRPO,
    "A2C": A2C,
    "DDPG": DDPG,
    "DQN": DQN,
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    
}

DISCRETE_ONLY = {"DQN"}
CONTINUOUS_ONLY = {"SAC", "DDPG", "TD3"}


def _sample_std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _parse_overall_log_steps(logdir):
    path = os.path.join(logdir, "overall_log.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"overall_log.txt not found in {logdir}")
    last_total_steps = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Iteration"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            last_total_steps = int(float(parts[4]))
    if last_total_steps is None:
        raise ValueError(f"No step rows found in {path}")
    return last_total_steps


def _parse_overall_log_episodes(logdir):
    path = os.path.join(logdir, "overall_log.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"overall_log.txt not found in {logdir}")
    last_total_episodes = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Iteration"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            last_total_episodes = int(float(parts[3]))
    if last_total_episodes is None:
        raise ValueError(f"No episode rows found in {path}")
    return last_total_episodes


def extract_budgets(reference_logdir, num_trials, budget_mode, budget_target=None):
    if budget_target is not None:
        return [int(budget_target)] * num_trials

    run_dirs = []
    if os.path.isdir(reference_logdir):
        for entry in os.listdir(reference_logdir):
            full = os.path.join(reference_logdir, entry)
            if entry.startswith("run_") and os.path.isdir(full):
                run_dirs.append(full)
    run_dirs.sort(key=lambda p: int(os.path.basename(p).split("_")[1]))

    if len(run_dirs) < num_trials:
        raise ValueError(
            f"Reference logdir {reference_logdir} has only {len(run_dirs)} runs, "
            f"but {num_trials} trials were requested."
        )

    parser = {
        "steps": _parse_overall_log_steps,
        "episodes": _parse_overall_log_episodes,
    }.get(budget_mode)
    if parser is None:
        raise ValueError(
            f"Unsupported budget_mode={budget_mode!r}. Use 'steps' or 'episodes'."
        )

    return [parser(run_dir) for run_dir in run_dirs[:num_trials]]


def _derive_training_episode_budget(total_episode_budget, eval_every_episodes, eval_episodes):
    total_episode_budget = int(total_episode_budget)
    eval_every_episodes = int(eval_every_episodes)
    eval_episodes = int(eval_episodes)

    if eval_every_episodes <= 0 or eval_episodes <= 0:
        return total_episode_budget

    train_episodes = 0
    while True:
        next_train = train_episodes + 1
        # We log an evaluation every eval_every_episodes training episodes,
        # and we also force a final evaluation when training ends.
        num_evals = (next_train + eval_every_episodes - 1) // eval_every_episodes
        total_consumed = next_train + num_evals * eval_episodes
        if total_consumed > total_episode_budget:
            break
        train_episodes = next_train
    return train_episodes


def _make_vec_env(env_name, gym_env_name, n_envs, render_mode=None, env_kwargs=None):
    env_fns = []
    for _ in range(n_envs):
        def _factory():
            env = make_env(
                env_name=env_name,
                gym_env_name=gym_env_name,
                render_mode=render_mode,
                env_kwargs=env_kwargs,
            )
            env = Monitor(env)
            env.reset()
            return env

        env_fns.append(_factory)

    vec_env = DummyVecEnv(env_fns)
    return VecMonitor(vec_env)


def _check_algorithm_compatibility(algorithm, vec_env):
    is_discrete_action = hasattr(vec_env.action_space, "n")
    if algorithm in DISCRETE_ONLY and not is_discrete_action:
        raise ValueError(f"{algorithm} only supports discrete action spaces.")
    if algorithm in CONTINUOUS_ONLY and is_discrete_action:
        raise ValueError(f"{algorithm} only supports continuous action spaces.")


def _split_hyperparams(all_hyperparams):
    control = {
        "n_envs": int(all_hyperparams.pop("n_envs", 1)),
        "normalize": all_hyperparams.pop("normalize", False),
    }
    return control, all_hyperparams


def _maybe_build_action_noise(vec_env, algo_kwargs):
    noise_type = algo_kwargs.pop("noise_type", None)
    noise_std = algo_kwargs.pop("noise_std", None)
    if noise_type is None or noise_std is None:
        return
    action_dim = int(np.prod(vec_env.action_space.shape))
    sigma = noise_std * np.ones(action_dim, dtype=np.float32)
    mean = np.zeros(action_dim, dtype=np.float32)
    if noise_type == "normal":
        algo_kwargs["action_noise"] = NormalActionNoise(mean=mean, sigma=sigma)
    elif noise_type == "ornstein-uhlenbeck":
        algo_kwargs["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=mean, sigma=sigma
        )


def _build_model(algorithm, vec_env, algo_kwargs):
    algo_cls = ALGORITHMS[algorithm]
    _maybe_build_action_noise(vec_env, algo_kwargs)
    return algo_cls(
        "MlpPolicy",
        vec_env,
        verbose=0,
        **algo_kwargs,
    )


def run_algorithm_trials(config, algorithm):
    env_name = config["env_name"]
    gym_env_name = config["gym_env_name"]
    num_trials = int(config["num_trials"])
    reference_logdir = config.get("reference_logdir")
    render_mode = config.get("render_mode")
    env_kwargs = config.get("env_kwargs", {})
    budget_mode = config.get("budget_mode", "steps")
    budget_target = config.get("budget_target")
    use_periodic_eval = bool(config.get("use_periodic_eval", True))
    eval_every_episodes = int(config.get("eval_every_episodes", 1))
    eval_episodes = int(config.get("eval_episodes", 20))
    eval_deterministic = bool(config.get("eval_deterministic", True))
    algorithm_logdir = os.path.join(config["logdir"], algorithm.lower())
    os.makedirs(algorithm_logdir, exist_ok=True)

    budgets = extract_budgets(
        reference_logdir, num_trials, budget_mode, budget_target=budget_target
    )
    per_run_summaries = []

    print(
        f"[SB3] Starting {algorithm} on {env_name} ({gym_env_name}) "
        f"for {num_trials} trials"
    )
    print(f"[SB3] Budget mode: {budget_mode}")
    print(f"[SB3] Budgets: {budgets}")
    if use_periodic_eval:
        print(
            f"[SB3] Periodic eval enabled: every {eval_every_episodes} training episodes, "
            f"{eval_episodes} eval episodes, deterministic={eval_deterministic}"
        )

    for trial_idx, budget_target in enumerate(budgets, start=1):
        run_dir = os.path.join(algorithm_logdir, f"run_{trial_idx}")
        os.makedirs(run_dir, exist_ok=True)

        hyperparams, hyperparam_source = get_hyperparams(
            algorithm,
            env_name,
            gym_env_name,
            config_overrides=(config.get("algorithm_hyperparams", {}) or {}).get(
                algorithm, {}
            ),
            zoo_root=config.get("zoo_hyperparams_dir", "rl-baselines3-zoo/hyperparams"),
        )
        control, algo_kwargs = _split_hyperparams(dict(hyperparams))
        n_envs = control["n_envs"]
        normalize = control["normalize"]
        print(
            f"[SB3] {algorithm} trial {trial_idx}/{num_trials}: "
            f"{budget_mode}_budget={budget_target}, n_envs={n_envs}, "
            f"normalize={normalize}"
        )
        print(
            f"[SB3] {algorithm} hyperparams source for {gym_env_name}: "
            f"{hyperparam_source}"
        )
        if hyperparams:
            print(f"[SB3] {algorithm} hyperparams: {hyperparams}")
        if budget_mode == "episodes" and n_envs > 1:
            print(
                f"[SB3] {algorithm} note: n_envs={n_envs}, so episode budget is counted "
                f"across all parallel envs. Training may overshoot by at most {n_envs - 1} "
                f"episodes on the final vectorized step."
            )

        vec_env = _make_vec_env(
            env_name=env_name,
            gym_env_name=gym_env_name,
            n_envs=n_envs,
            render_mode=render_mode,
            env_kwargs=env_kwargs,
        )
        if normalize:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

        _check_algorithm_compatibility(algorithm, vec_env)
        model = _build_model(algorithm, vec_env, algo_kwargs=algo_kwargs)
        print(f"[SB3] {algorithm} trial {trial_idx}: model initialized")

        overall_log_path = os.path.join(run_dir, "overall_log.txt")

        eval_env_factory = None
        if use_periodic_eval:
            def eval_env_factory():
                return make_env(
                    env_name=env_name,
                    gym_env_name=gym_env_name,
                    render_mode=render_mode,
                    env_kwargs=env_kwargs,
                )

        if budget_mode == "steps":
            learn_timesteps = int(budget_target)
            logging_callback = EpisodeRewardLoggingCallback(
                overall_log_path,
                total_timesteps_target=learn_timesteps,
                eval_env_factory=eval_env_factory,
                eval_every_episodes=eval_every_episodes if use_periodic_eval else None,
                eval_episodes=eval_episodes,
                eval_deterministic=eval_deterministic,
            )
        else:
            training_episode_budget = int(budget_target)
            if use_periodic_eval:
                training_episode_budget = _derive_training_episode_budget(
                    total_episode_budget=int(budget_target),
                    eval_every_episodes=eval_every_episodes,
                    eval_episodes=eval_episodes,
                )
                print(
                    f"[SB3] {algorithm} trial {trial_idx}: total_episode_budget={budget_target}, "
                    f"derived_training_episode_budget={training_episode_budget}, "
                    f"expected_eval_episodes<={int(budget_target) - training_episode_budget}"
                )
            learn_timesteps = int(
                config.get("episode_budget_max_timesteps", 2_000_000_000)
            )
            logging_callback = EpisodeRewardLoggingCallback(
                overall_log_path,
                max_episodes=training_episode_budget,
                eval_env_factory=eval_env_factory,
                eval_every_episodes=eval_every_episodes if use_periodic_eval else None,
                eval_episodes=eval_episodes,
                eval_deterministic=eval_deterministic,
            )

        train_start = time.time()
        print(f"[SB3] {algorithm} trial {trial_idx}: training started")
        model.learn(total_timesteps=learn_timesteps, callback=logging_callback)
        train_time_s = time.time() - train_start
        print(
            f"[SB3] {algorithm} trial {trial_idx}: training finished in "
            f"{train_time_s:.2f}s, steps_used={model.num_timesteps}, "
            f"episodes_used={logging_callback.total_episodes}"
        )

        if normalize:
            vec_env.save(os.path.join(run_dir, "vecnormalize.pkl"))
        model.save(os.path.join(run_dir, "model.zip"))

        rewards = logging_callback.episode_rewards
        run_summary = {
            "label": f"{algorithm}/run_{trial_idx}",
            "logdir": run_dir,
            "budget_mode": budget_mode,
            "budget_target": int(budget_target),
            "steps_used": int(model.num_timesteps),
            "num_episodes": int(logging_callback.total_episodes),
            "total_env_episodes": int(logging_callback.total_env_episodes),
            "total_eval_episodes": int(logging_callback.total_eval_episodes),
            "n_envs": int(n_envs),
            "hyperparam_source": hyperparam_source,
            "best_reward": float(max(rewards)) if rewards else None,
            "final_reward": float(rewards[-1]) if rewards else None,
            "mean_reward": float(np.mean(rewards)) if rewards else None,
            "std_reward": _sample_std(rewards) if rewards else None,
            "use_periodic_eval": bool(use_periodic_eval),
            "eval_every_episodes": int(eval_every_episodes) if use_periodic_eval else None,
            "eval_episodes": int(eval_episodes) if use_periodic_eval else None,
            "eval_deterministic": bool(eval_deterministic) if use_periodic_eval else None,
            "num_logged_evals": int(len(rewards)),
            "best_train_reward": (
                float(logging_callback.best_train_reward)
                if logging_callback.train_episode_rewards
                else None
            ),
            "final_train_reward": (
                float(logging_callback.train_episode_rewards[-1])
                if logging_callback.train_episode_rewards
                else None
            ),
            "mean_train_reward": (
                float(np.mean(logging_callback.train_episode_rewards))
                if logging_callback.train_episode_rewards
                else None
            ),
            "std_train_reward": (
                _sample_std(logging_callback.train_episode_rewards)
                if logging_callback.train_episode_rewards
                else None
            ),
            "train_time_s": float(train_time_s),
            "hyperparams": hyperparams,
        }
        write_run_summary(os.path.join(run_dir, "summary.json"), run_summary)
        per_run_summaries.append(run_summary)
        print(
            f"[SB3] {algorithm} trial {trial_idx}: "
            f"best_reward={run_summary['best_reward']}, "
            f"final_reward={run_summary['final_reward']}, "
            f"mean_reward={run_summary['mean_reward']}"
        )
        vec_env.close()

    summary = write_algorithm_summary(
        algorithm_logdir,
        algorithm,
        env_name,
        per_run_summaries,
        budget_mode,
        budgets,
    )
    print(
        f"[SB3] {algorithm} finished: "
        f"mean_best_reward={summary['mean_best_reward']:.4f}, "
        f"mean_final_reward={summary['mean_final_reward']:.4f}"
    )
    return summary


def run_baseline_suite(config):
    summaries = []
    for algorithm in config["algorithms"]:
        summaries.append(run_algorithm_trials(config, algorithm))

    comparison = []
    for summary in summaries:
        comparison.append(
            {
                "label": summary["label"],
                "num_runs": summary["num_runs"],
                "mean_best_reward": summary["mean_best_reward"],
                "std_best_reward": summary["std_best_reward"],
                "mean_reward": summary["mean_reward_mean"],
            }
        )

    os.makedirs(config["logdir"], exist_ok=True)
    with open(os.path.join(config["logdir"], "comparison_summary.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"[SB3] Wrote comparison summary to {config['logdir']}/comparison_summary.json")
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Run SB3 baselines from config.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_baseline_suite(config)


if __name__ == "__main__":
    main()
