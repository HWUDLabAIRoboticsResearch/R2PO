import numpy as np


def _stringify_buffer_item(item):
    if hasattr(item, "tolist"):
        return item.tolist()
    return item


def _format_number(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3g}"
    return str(value)


def _format_value(value):
    value = _stringify_buffer_item(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_number(v) for v in value) + "]"
    return _format_number(value)


def format_params_inline(parameters):
    flat = np.array(parameters).reshape(-1).tolist()
    return "[" + ",".join(_format_number(v) for v in flat) + "]"


def format_compact_trajectory(traj_buffer):
    if not traj_buffer.buffer:
        return "No trajectory available."

    steps = list(traj_buffer.buffer[0].buffer)
    if not steps:
        return "No trajectory available."

    formatted_steps = []
    for idx, (state, action, reward) in enumerate(steps):
        state_text = _format_value(state)
        action_text = _format_value(action)
        if idx + 1 < len(steps):
            next_state_text = _format_value(steps[idx + 1][0])
            formatted_steps.append(f"{state_text}->{action_text}->{next_state_text}")
        else:
            terminal = (
                "[ROLLOUT_CAP]"
                if len(steps) >= traj_buffer.max_traj_length
                else "[TERMINATED]"
            )
            formatted_steps.append(f"{state_text}->{action_text}->{terminal}")
    return " | ".join(formatted_steps)


def build_history_text(
    replay_buffer,
    warmup_replay_count=0,
):
    entries = []
    for idx, buffer_item in enumerate(replay_buffer.buffer):
        parameters, reward = buffer_item[:2]
        is_warmup = idx < warmup_replay_count
        reward_key = "rollout_reward" if is_warmup else "mean_reward"
        entry = (
            f"Trial {idx}: params={format_params_inline(parameters)} "
            f"{reward_key}={float(reward):.2f}"
        )

        entries.append(entry)

    return "\n\n".join(entries) if entries else "No previous trials yet."


def build_search_llm_history_text(replay_buffer, warmup_replay_count=0):
    entries = []

    for idx, buffer_item in enumerate(replay_buffer.buffer):
        parameters, reward = buffer_item[:2]
        is_warmup = idx < warmup_replay_count
        reward_key = "rollout_reward" if is_warmup else "mean_reward"
        entries.append(
            f"Trial {idx}: params={format_params_inline(parameters)} "
            f"{reward_key}={float(reward):.2f}"
        )

    return "\n".join(entries) if entries else "No previous trials yet."


def select_median_rollout_index(results):
    """Return the evaluated rollout closest to the sample median reward."""
    rewards = np.array(results, dtype=float)
    if len(rewards) == 0:
        raise ValueError("Cannot select a median rollout from no rewards.")
    median_target = float(np.median(rewards))
    return int(np.argmin(np.abs(rewards - median_target)))


def build_median_trajectory_stats_summary(results, traj_buffers, optimum=None):
    rewards = np.array(results, dtype=float)
    if len(rewards) == 0:
        return "No evaluation trajectories available."

    median_idx = select_median_rollout_index(rewards)
    lengths = np.array(
        [
            len(traj_buffer.buffer[0].buffer)
            if traj_buffer.buffer
            else 0
            for traj_buffer in traj_buffers
        ],
        dtype=int,
    )
    max_reward = float(np.max(rewards))
    success_reward = (
        float(optimum) if optimum is not None else max_reward
    )
    atol = max(1e-6, 1e-3 * max(1.0, abs(success_reward)))
    success_mask = np.isclose(rewards, success_reward, atol=atol) | (
        rewards > success_reward
    )
    success_count = int(np.sum(success_mask))
    total_count = len(rewards)
    failure_count = total_count - success_count
    return (
        f"Reward: mean={float(np.mean(rewards)):.2f}, "
        f"min={float(np.min(rewards)):.2f}, max={max_reward:.2f}\n"
        f"Episode length: mean={float(np.mean(lengths)):.1f}, "
        f"min={int(np.min(lengths))}, max={int(np.max(lengths))}\n"
        f"Success rate: {success_count}/{total_count} rollouts reached reward>={success_reward:.2f}\n"
        f"Failure rate: {failure_count}/{total_count} rollouts finished below reward={success_reward:.2f}\n\n"
        f"Median rollout (rollout {median_idx}, reward={rewards[median_idx]:.4f}, length={int(lengths[median_idx])}):\n"
        f"{format_compact_trajectory(traj_buffers[median_idx])}"
    )


def build_critic_llm_reflection_context(
    initial_params,
    initial_reward,
    trajectory_summary,
    replay_buffer,
    step_number,
    env_desc_file,
    max_iterations=None,
    rank=None,
    optimum=None,
    search_step_size=0.1,
    actions=None,
    warmup_replay_count=0,
    conservative_threshold=None,
):
    params_str = "; ".join(
        f"params[{i}]: {_format_number(v)}"
        for i, v in enumerate(initial_params.reshape(-1).tolist())
    )

    return {
        "proposed_params": params_str,
        "proposed_params_list": initial_params.reshape(-1).tolist(),
        "achieved_reward": float(initial_reward),
        "trajectory_summary": trajectory_summary,
        "history_text": build_history_text(
            replay_buffer,
            warmup_replay_count=warmup_replay_count,
        ),
        "step_number": str(step_number),
        "max_iterations": max_iterations,
        "rank": rank,
        "optimum": str(optimum),
        "actions": actions,
        "step_size": str(search_step_size),
        "env_description": env_desc_file,
        "conservative_threshold": (
            None
            if conservative_threshold is None
            else float(conservative_threshold)
        ),
    }
