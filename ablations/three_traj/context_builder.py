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


def _format_step_sequence(steps):
    formatted_steps = []
    for idx, (state, action, reward) in enumerate(steps):
        state_text = _format_value(state)
        action_text = _format_value(action)
        if idx + 1 < len(steps):
            next_state_text = _format_value(steps[idx + 1][0])
            formatted_steps.append(f"{state_text}->{action_text}->{next_state_text}")
        else:
            terminal = "[GOAL]" if float(reward) > 0 else "[END]"
            formatted_steps.append(f"{state_text}->{action_text}->{terminal}")
    return formatted_steps


def format_compact_trajectory(traj_buffer, head_steps=None, tail_steps=None):
    if not traj_buffer.buffer:
        return "No trajectory available."

    steps = list(traj_buffer.buffer[0].buffer)
    if not steps:
        return "No trajectory available."

    if (
        head_steps is not None
        and tail_steps is not None
        and len(steps) > head_steps + tail_steps
    ):
        head = steps[:head_steps]
        tail = steps[-tail_steps:]
        omitted = len(steps) - head_steps - tail_steps
        formatted_steps = _format_step_sequence(head)
        formatted_steps.append(f"... {omitted} omitted steps ...")
        formatted_steps.extend(_format_step_sequence(tail))
        return " | ".join(formatted_steps)

    return " | ".join(_format_step_sequence(steps))


def build_history_text(
    replay_buffer,
    traj_buffer=None,
    traj_history_last_n=5,
    warmup_replay_count=0,
):
    entries = []
    n_total = len(replay_buffer.buffer)

    for idx, buffer_item in enumerate(replay_buffer.buffer):
        parameters, reward = buffer_item[:2]
        is_warmup = idx < warmup_replay_count
        tag = "Warmup" if is_warmup else "Training"
        if is_warmup:
            entry = (
                f"{tag} trial {idx}: params={format_params_inline(parameters)} "
                f"rollout_reward={float(reward):.2f}"
            )
        else:
            entry = (
                f"{tag} trial {idx}: params={format_params_inline(parameters)} "
                f"mean_reward={float(reward):.2f}"
            )

        entries.append(entry)

    return "\n\n".join(entries) if entries else "No previous trials yet."


def build_search_llm_history_text(replay_buffer, warmup_replay_count=0):
    entries = [
        "Important:",
        "- Warmup trial entries come from single random rollouts.",
        "- Training trial entries come from the optimization loop.",
        "- Treat high training mean_reward as stronger evidence than lucky warmup rollouts.",
        "- Do not copy warmup policies blindly; use them mainly as exploration clues.",
        "",
    ]

    for idx, buffer_item in enumerate(replay_buffer.buffer):
        parameters, reward = buffer_item[:2]
        is_warmup = idx < warmup_replay_count
        tag = "Warmup" if is_warmup else "Training"
        reward_key = "rollout_reward" if is_warmup else "mean_reward"
        entries.append(
            f"{tag} trial {idx}: params={format_params_inline(parameters)} "
            f"{reward_key}={float(reward):.2f}"
        )

    return "\n".join(entries) if len(entries) > 6 else "No previous trials yet."


def build_three_trajectory_summary(results, traj_buffers):
    rewards = np.array(results, dtype=float)
    if len(rewards) == 0:
        return "No evaluation trajectories available."

    best_idx = int(np.argmax(rewards))
    worst_idx = int(np.argmin(rewards))
    median_target = float(np.median(rewards))
    median_idx = int(np.argmin(np.abs(rewards - median_target)))

    sections = []
    for label, idx in [
        ("Worst rollout", worst_idx),
        ("Median rollout", median_idx),
        ("Best rollout", best_idx),
    ]:
        sections.append(
            f"{label} (rollout {idx}, reward={rewards[idx]:.4f}):\n"
            f"{format_compact_trajectory(traj_buffers[idx], head_steps=25, tail_steps=25)}"
        )
    return "\n\n".join(sections)


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
    traj_buffer=None,
    traj_history_last_n=5,
    warmup_replay_count=0,
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
            traj_buffer=traj_buffer,
            traj_history_last_n=traj_history_last_n,
            warmup_replay_count=warmup_replay_count,
        ),
        "step_number": str(step_number),
        "max_iterations": max_iterations,
        "rank": rank,
        "optimum": str(optimum),
        "actions": actions,
        "step_size": str(search_step_size),
        "env_description": env_desc_file,
    }
