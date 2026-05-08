import os
import re
import time

import numpy as np


def resolve_run_logdir(logdir, rerun=None, resume_run=None):
    if rerun is not None and resume_run is not None:
        raise ValueError("Use only one of rerun or resume_run.")

    if resume_run is not None:
        run_logdir = os.path.join(logdir, f"run_{resume_run}")
        if not os.path.isdir(run_logdir):
            raise FileNotFoundError(f"Run to resume not found: {run_logdir}")
        print(f"Resuming run: {run_logdir}")
        return run_logdir, "resume"

    if rerun is not None:
        run_logdir = os.path.join(logdir, f"run_{rerun}")
        print(f"Overwriting run: {run_logdir}")
        return run_logdir, "rerun"

    run_idx = 1
    while os.path.exists(os.path.join(logdir, f"run_{run_idx}")):
        run_idx += 1
    run_logdir = os.path.join(logdir, f"run_{run_idx}")
    print(f"Logging to: {run_logdir}")
    return run_logdir, "new"


def parse_overall_log(logdir):
    path = os.path.join(logdir, "overall_log.txt")
    if not os.path.exists(path):
        return []

    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Iteration"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            rows.append(
                {
                    "iteration": int(parts[0]),
                    "cpu_time": float(parts[1]),
                    "api_time": float(parts[2]),
                    "total_episodes": int(parts[3]),
                    "total_steps": int(parts[4]),
                    "total_reward": float(parts[5]),
                }
            )
    return rows


def _sorted_matching_files(folder, prefix):
    if not os.path.isdir(folder):
        return []
    files = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.startswith(prefix)
    ]
    files.sort(key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))
    return files


def _parse_continuous_params_from_rollout(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            if text == "parameter ends":
                break
            if "," in text:
                return np.array([float(x.strip()) for x in text.split(",")], dtype=float)
    return None


def _parse_qtable_params_from_parameters(path, expected_rank):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        text = f.read().strip()
    pattern = re.compile(r":\s*(?:np\.float64\()?([+-]?\d+(?:\.\d+)?)\)?(?=,|\})")
    values = [float(x) for x in pattern.findall(text)]
    if len(values) != expected_rank:
        return None
    return np.array(values, dtype=float)


def _append_first_logged_trajectory(traj_buffer, rollout_path):
    if traj_buffer is None or not os.path.exists(rollout_path):
        return False

    lines = [line.rstrip("\n") for line in open(rollout_path, "r")]
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == "state | action | reward":
            start_idx = idx + 1
            break
    if start_idx is None:
        return False

    traj_buffer.start_new_trajectory()
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Total reward:"):
            return True
        parts = line.split(" | ")
        if len(parts) < 3:
            continue
        state, action, reward = parts[0], parts[1], parts[2]
        try:
            reward_value = float(reward)
        except ValueError:
            continue
        traj_buffer.add_step(state, action, reward_value)
    return traj_buffer.buffer[-1].size() > 0 if traj_buffer.buffer else False


def _restore_warmup_history(agent, warmup_dir, semantic=False, is_qtable=False):
    if not warmup_dir or not os.path.isdir(warmup_dir):
        return

    warmup_files = _sorted_matching_files(warmup_dir, "warmup_rollout_")
    if not warmup_files:
        return

    restored = 0
    skipped = 0
    for rollout_path in warmup_files:
        if is_qtable:
            skipped += 1
            continue
        params = _parse_continuous_params_from_rollout(rollout_path)
        if params is None:
            skipped += 1
            continue

        rewards = []
        with open(rollout_path, "r") as f:
            for line in f:
                if line.startswith("Total reward:"):
                    try:
                        rewards.append(float(line.split(":", 1)[1].strip()))
                    except ValueError:
                        pass
        reward = float(np.mean(rewards)) if rewards else None
        if reward is None:
            skipped += 1
            continue

        agent.replay_buffer.add(params, reward)
        if semantic:
            _append_first_logged_trajectory(agent.traj_buffer, rollout_path)
        restored += 1

    if restored:
        print(f"Restored {restored} warmup entries from {warmup_dir}")
    if skipped:
        print(
            f"Skipped {skipped} warmup entries that could not be reconstructed exactly"
        )


def restore_agent_from_run(
    agent,
    logdir,
    semantic=False,
    is_qtable=False,
    warmup_dir=None,
):
    rows = parse_overall_log(logdir)

    effective_warmup_dir = warmup_dir or os.path.join(logdir, "warmup")
    _restore_warmup_history(
        agent,
        effective_warmup_dir,
        semantic=semantic,
        is_qtable=is_qtable,
    )

    completed = len(rows)
    for episode_idx, row in enumerate(rows):
        ep_dir = os.path.join(logdir, f"episode_{episode_idx}")
        if is_qtable:
            params = _parse_qtable_params_from_parameters(
                os.path.join(ep_dir, "parameters.txt"),
                expected_rank=agent.rank,
            )
        else:
            params = _parse_continuous_params_from_rollout(
                os.path.join(ep_dir, "training_rollout.txt")
            )

        if params is None:
            raise RuntimeError(
                f"Could not reconstruct completed episode_{episode_idx} in {logdir}"
            )

        agent.replay_buffer.add(params, row["total_reward"])
        if semantic:
            _append_first_logged_trajectory(
                agent.traj_buffer,
                os.path.join(ep_dir, "training_rollout.txt"),
            )

    if rows:
        last = rows[-1]
        agent.api_call_time = last["api_time"]
        agent.total_episodes = last["total_episodes"]
        agent.total_steps = last["total_steps"]
        agent.start_time = time.process_time() - last["cpu_time"]
    agent.training_episodes = completed

    print(
        f"Restored {completed} completed training episodes from {logdir}. "
        f"Next episode index: {completed}"
    )
    return completed
