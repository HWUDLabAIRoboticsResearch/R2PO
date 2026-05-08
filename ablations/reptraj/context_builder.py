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


def _format_params_inline(parameters):
    flat = np.array(parameters).reshape(-1).tolist()
    return "[" + ", ".join(_format_number(v) for v in flat) + "]"


def _trajectory_steps(traj_buffer):
    if traj_buffer is None:
        return []
    if hasattr(traj_buffer, "buffer"):
        buffer = traj_buffer.buffer
        if not buffer:
            return []
        first_item = buffer[0]
        if hasattr(first_item, "buffer"):
            return list(first_item.buffer)
        if isinstance(first_item, tuple) and len(first_item) == 3:
            return list(buffer)
    return []


def _trajectory_capacity_reached(traj_buffer, steps):
    if traj_buffer is None:
        return False
    max_len = getattr(traj_buffer, "max_traj_length", None)
    if max_len is None:
        max_len = getattr(traj_buffer, "buffer_size", None)
    return max_len is not None and len(steps) >= max_len


def _format_compact_transition(step, next_state=None, is_last=False, capped=False):
    state, action, _ = step
    state_text = _format_value(state)
    action_text = _format_value(action)
    if next_state is not None:
        return f"{state_text}->{action_text}->{_format_value(next_state)}"
    terminal = "[ROLLOUT_CAP]" if capped else "[TERMINATED]"
    if not is_last:
        terminal = "[UNKNOWN]"
    return f"{state_text}->{action_text}->{terminal}"


def _format_trajectory_excerpt(traj_buffer, max_steps=12):
    steps = _trajectory_steps(traj_buffer)
    if not steps:
        return "No trajectory available."

    capped = _trajectory_capacity_reached(traj_buffer, steps)
    if len(steps) <= max_steps:
        formatted = []
        for idx, step in enumerate(steps):
            next_state = steps[idx + 1][0] if idx + 1 < len(steps) else None
            formatted.append(
                _format_compact_transition(
                    step,
                    next_state=next_state,
                    is_last=(idx + 1 == len(steps)),
                    capped=capped,
                )
            )
        return " | ".join(formatted)

    head_count = max_steps // 2
    tail_count = max_steps - head_count
    formatted = []
    for idx in range(head_count):
        next_state = steps[idx + 1][0] if idx + 1 < len(steps) else None
        formatted.append(
            _format_compact_transition(
                steps[idx],
                next_state=next_state,
                is_last=False,
                capped=capped,
            )
        )
    formatted.append(f"... {len(steps) - max_steps} omitted steps ...")
    tail_start = len(steps) - tail_count
    for idx in range(tail_start, len(steps)):
        next_state = steps[idx + 1][0] if idx + 1 < len(steps) else None
        formatted.append(
            _format_compact_transition(
                steps[idx],
                next_state=next_state,
                is_last=(idx + 1 == len(steps)),
                capped=capped,
            )
        )
    return " | ".join(formatted)


def _trajectory_total_reward(traj_buffer):
    steps = _trajectory_steps(traj_buffer)
    return float(sum(reward for _, _, reward in steps))


def _build_history_text(replay_buffer, traj_buffer=None, traj_history_last_n=5):
    entries = []
    n_total = len(replay_buffer.buffer)
    for idx, buffer_item in enumerate(replay_buffer.buffer):
        if len(buffer_item) == 2:
            parameters, reward = buffer_item
            entry = (
                f"Trial {idx}: params={_format_params_inline(parameters)} "
                f"mean_reward={float(reward):.2f}"
            )
        elif len(buffer_item) == 3:
            weights, bias, reward = buffer_item
            entry = (
                f"Trial {idx}: weights={_format_params_inline(weights)} "
                f"bias={_format_params_inline(bias)} "
                f"mean_reward={float(reward):.2f}"
            )
        else:
            compact = ", ".join(_format_value(x) for x in buffer_item)
            entry = f"Trial {idx}: values=[{compact}]"

        is_recent = (n_total - idx) <= traj_history_last_n
        if traj_buffer is not None and idx < len(traj_buffer.buffer) and is_recent:
            history_traj = traj_buffer.buffer[idx]
            steps = _trajectory_steps(history_traj)
            if steps:
                entry += (
                    f"\n  rollout_excerpt={_format_trajectory_excerpt(history_traj, max_steps=8)}"
                )
        entries.append(entry)

    return "\n\n".join(entries) if entries else "No previous trials yet."


def serialize_trajectory_buffer(traj_buffer):
    """Serialize trajectory buffer to JSON-friendly format."""
    serialized = []
    for idx, trajectory in enumerate(traj_buffer.buffer):
        steps = []
        for state, action, reward in trajectory.buffer:
            steps.append(
                {
                    "state": _stringify_buffer_item(state),
                    "action": _stringify_buffer_item(action),
                    "reward": float(reward),
                }
            )
        serialized.append({"trajectory_idx": idx, "steps": steps})
    return serialized


def build_critic_llm_reflection_context(
    initial_params,
    initial_reward,
    trajectory_summary,
    stats_summary,
    replay_buffer,
    step_number,
    env_desc_file,
    max_iterations=None,
    rank=None,
    optimum=None,
    search_step_size=0.1,
    actions=None,
    traj_buffer=None,
    traj_history_last_n=5,
):
    """Build context for Critic-LLM (reflector)."""
    params_str = "; ".join(
        f"params[{i}]: {_format_number(v)}"
        for i, v in enumerate(initial_params.reshape(-1).tolist())
    )

    return {
        "proposed_params": params_str,
        "proposed_params_list": initial_params.reshape(-1).tolist(),
        "achieved_reward": float(initial_reward),
        "trajectory_summary": trajectory_summary,
        "stats_json": "N/A" if not stats_summary else str(stats_summary),
        "history_json": _build_history_text(
            replay_buffer,
            traj_buffer=traj_buffer,
            traj_history_last_n=traj_history_last_n,
        ),
        "step_number": str(step_number),
        "max_iterations": max_iterations,
        "rank": rank,
        "optimum": str(optimum),
        "actions": actions,
        "step_size": str(search_step_size),
        "env_description": env_desc_file,
    }


def format_trajectory_summary(traj_buffer, last_n=1):
    """Format recent trajectories in a raw step-by-step style for the reflector."""
    if not traj_buffer.buffer:
        return "No trajectory data available."

    lines = []
    trajectories = list(traj_buffer.buffer)[-last_n:]
    for traj_idx, trajectory in enumerate(trajectories):
        steps = _trajectory_steps(trajectory)
        lines.append(
            f"Trajectory {traj_idx} (length={len(steps)}, "
            f"step_reward_sum={_trajectory_total_reward(trajectory):.2f}):"
        )
        if steps:
            for step_idx, (state, action, reward) in enumerate(steps):
                lines.append(
                    "  "
                    f"step {step_idx}: state={_format_value(state)} "
                    f"-> action={_format_value(action)} "
                    f"-> reward={_format_number(reward)}"
                )
    return "\n".join(lines)
