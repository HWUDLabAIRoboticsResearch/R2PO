import json
import os
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeRewardLoggingCallback(BaseCallback):
    def __init__(
        self,
        overall_log_path,
        max_episodes=None,
        total_timesteps_target=None,
        eval_env_factory=None,
        eval_every_episodes=None,
        eval_episodes=20,
        eval_deterministic=True,
        verbose=0,
    ):
        super().__init__(verbose=verbose)
        self.overall_log_path = overall_log_path
        self.max_episodes = max_episodes
        self.total_timesteps_target = total_timesteps_target
        self.eval_env_factory = eval_env_factory
        self.eval_every_episodes = eval_every_episodes
        self.eval_episodes = int(eval_episodes)
        self.eval_deterministic = eval_deterministic
        self.start_wall_time = None
        self.total_episodes = 0
        self.total_env_episodes = 0
        self.total_eval_episodes = 0
        self.episode_rewards = []
        self.train_episode_rewards = []
        self.best_reward = float("-inf")
        self.best_train_reward = float("-inf")
        self._last_print_time = None
        self._last_logged_eval_episode = 0

    def _on_training_start(self):
        self.start_wall_time = time.time()
        self._last_print_time = self.start_wall_time
        with open(self.overall_log_path, "w") as f:
            f.write(
                "Iteration, CPU Time, API Time, Total Episodes, Total Steps, Total Reward\n"
            )

    def _write_log_row(self, reward):
        elapsed = time.time() - self.start_wall_time
        with open(self.overall_log_path, "a") as f:
            f.write(
                f"{len(self.episode_rewards)}, {elapsed}, 0.0, {self.total_env_episodes}, "
                f"{self.num_timesteps}, {reward}\n"
            )

    def _run_periodic_eval(self):
        if self.eval_env_factory is None:
            return None

        eval_env = self.eval_env_factory()
        rewards = []
        try:
            for _ in range(self.eval_episodes):
                reset_out = eval_env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                done = False
                episode_reward = 0.0
                while not done:
                    action, _ = self.model.predict(
                        obs, deterministic=self.eval_deterministic
                    )
                    if hasattr(eval_env.action_space, "n"):
                        action = int(np.asarray(action).reshape(-1)[0])
                    step_out = eval_env.step(action)
                    if len(step_out) == 5:
                        obs, reward, terminated, truncated, _ = step_out
                        done = bool(terminated or truncated)
                    else:
                        obs, reward, done, _ = step_out
                    episode_reward += float(reward)
                rewards.append(episode_reward)
                self.total_eval_episodes += 1
                self.total_env_episodes += 1
        finally:
            eval_env.close()

        return float(np.mean(rewards)) if rewards else None

    def _maybe_log_evaluation(self, force=False):
        if self.eval_every_episodes is None:
            return
        if self.total_episodes == 0:
            return
        if self._last_logged_eval_episode == self.total_episodes:
            return
        if not force:
            if self.total_episodes - self._last_logged_eval_episode < self.eval_every_episodes:
                return

        eval_reward = self._run_periodic_eval()
        if eval_reward is None:
            return

        self.episode_rewards.append(eval_reward)
        self.best_reward = max(self.best_reward, eval_reward)
        self._last_logged_eval_episode = self.total_episodes
        self._write_log_row(eval_reward)

        episode_progress = str(self.total_episodes)
        if self.max_episodes is not None:
            episode_progress = f"{episode_progress}/{self.max_episodes}"

        step_progress = str(self.num_timesteps)
        if self.total_timesteps_target is not None:
            step_progress = f"{step_progress}/{self.total_timesteps_target}"

        print(
            f"[SB3] eval episodes={episode_progress} "
            f"steps={step_progress} "
            f"mean_reward={eval_reward:.4f} "
            f"best_eval={self.best_reward:.4f}"
        )
        self._last_print_time = time.time()

    def _on_step(self):
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        should_continue = True
        for info in infos:
            episode = info.get("episode")
            if not episode:
                continue
            reward = float(episode["r"])
            self.total_episodes += 1
            self.total_env_episodes += 1
            self.train_episode_rewards.append(reward)
            self.best_train_reward = max(self.best_train_reward, reward)
            episode_progress = str(self.total_episodes)
            if self.max_episodes is not None:
                episode_progress = f"{episode_progress}/{self.max_episodes}"

            step_progress = str(self.num_timesteps)
            if self.total_timesteps_target is not None:
                step_progress = f"{step_progress}/{self.total_timesteps_target}"

            if self.eval_every_episodes is None:
                self.episode_rewards.append(reward)
                self.best_reward = max(self.best_reward, reward)
                self._write_log_row(reward)
                if (
                    self.total_episodes <= 3
                    or reward > self.best_reward - 1e-12
                    or time.time() - self._last_print_time >= 30
                    or (
                        self.max_episodes is not None
                        and self.total_episodes >= self.max_episodes
                    )
                ):
                    print(
                        f"[SB3] episodes={episode_progress} "
                        f"steps={step_progress} "
                        f"reward={reward:.4f} "
                        f"best={self.best_reward:.4f}"
                    )
                    self._last_print_time = time.time()
            else:
                self._maybe_log_evaluation()

            if self.max_episodes is not None and self.total_episodes >= self.max_episodes:
                if self.eval_every_episodes is not None:
                    self._maybe_log_evaluation(force=True)
                should_continue = False

        return should_continue

    def _on_training_end(self):
        if self.eval_every_episodes is not None:
            self._maybe_log_evaluation(force=True)


def write_run_summary(path, summary):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def _sample_std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def write_algorithm_summary(logdir, algorithm, env_name, per_run_summaries, budget_mode, budgets):
    best_rewards = [run["best_reward"] for run in per_run_summaries]
    final_rewards = [run["final_reward"] for run in per_run_summaries]
    mean_rewards = [run["mean_reward"] for run in per_run_summaries]
    train_times = [run["train_time_s"] for run in per_run_summaries]

    summary = {
        "label": algorithm,
        "env_name": env_name,
        "logdir": logdir,
        "num_runs": len(per_run_summaries),
        "budget_mode": budget_mode,
        "budgets": budgets,
        "mean_best_reward": float(np.mean(best_rewards)) if best_rewards else None,
        "std_best_reward": _sample_std(best_rewards) if best_rewards else None,
        "mean_final_reward": float(np.mean(final_rewards)) if final_rewards else None,
        "std_final_reward": _sample_std(final_rewards) if final_rewards else None,
        "mean_reward_mean": float(np.mean(mean_rewards)) if mean_rewards else None,
        "mean_reward_std": _sample_std(mean_rewards) if mean_rewards else None,
        "mean_train_time_s": float(np.mean(train_times)) if train_times else None,
        "std_train_time_s": _sample_std(train_times) if train_times else None,
        "per_run_summaries": per_run_summaries,
    }

    os.makedirs(logdir, exist_ok=True)
    write_run_summary(os.path.join(logdir, "summary.json"), summary)
    with open(os.path.join(logdir, "summary.txt"), "w") as f:
        f.write(
            f"Algorithm: {algorithm}\n"
            f"Environment: {env_name}\n"
            f"Runs: {summary['num_runs']}\n"
            f"Budget mode: {summary['budget_mode']}\n"
            f"Mean best reward: {summary['mean_best_reward']:.4f} +/- {summary['std_best_reward']:.4f}\n"
            f"Mean final reward: {summary['mean_final_reward']:.4f} +/- {summary['std_final_reward']:.4f}\n"
            f"Mean episode reward: {summary['mean_reward_mean']:.4f} +/- {summary['mean_reward_std']:.4f}\n"
            f"Mean train time: {summary['mean_train_time_s']:.2f}s +/- {summary['std_train_time_s']:.2f}s\n"
        )
    return summary
