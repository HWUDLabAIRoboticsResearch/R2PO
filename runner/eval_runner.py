"""
Reads training logs and produces eval_report/ inside each logdir:
  - training_curve.png, reward_distribution.png, api_time.png
  - summary.json + summary.txt

Multi-run mode: if logdir contains run_N/ subdirs, each run is analyzed
individually and an aggregate report (mean ± std) is produced at the top level.
"""
import json
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_overall_log(logdir: str):
    path = os.path.join(logdir, "overall_log.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"overall_log.txt not found in {logdir}")
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Iteration"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                rows.append({
                    "iteration": int(parts[0]),
                    "cpu_time": float(parts[1]),
                    "api_time": float(parts[2]),
                    "total_episodes": int(parts[3]),
                    "total_steps": int(parts[4]),
                    "total_reward": float(parts[5]),
                })
            except ValueError:
                continue
    return rows


def _parse_training_rollout(episode_dir: str):
    rewards = []
    for filename in _selected_rollout_files(episode_dir):
        path = os.path.join(episode_dir, filename)
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            for line in f:
                m = re.search(r"Total reward:\s*([+-]?\d+(?:\.\d+)?)", line)
                if m:
                    rewards.append(float(m.group(1)))
    return rewards


def _read_key_value_file(path: str):
    data = {}
    if not os.path.exists(path):
        return data
    with open(path, "r") as f:
        for line in f:
            if "=" not in line:
                continue
            key, value = line.strip().split("=", 1)
            data[key.strip()] = value.strip()
    return data


def _selected_rollout_files(episode_dir: str):
    """Return rollout file(s) corresponding to the selected policy only."""
    reflection_outcome = _read_key_value_file(
        os.path.join(episode_dir, "reflection_outcome.txt")
    )
    if reflection_outcome:
        winner = reflection_outcome.get("winner")
        if winner == "original_won":
            return ["initial_proposal_rollout.txt"]
        return ["revised_proposal_rollout.txt"]

    selection_outcome = _read_key_value_file(
        os.path.join(episode_dir, "selection_outcome.txt")
    )
    if selection_outcome:
        winner = selection_outcome.get("winner")
        if winner == "proposal_a":
            return ["proposal_a_rollout.txt"]
        if winner == "proposal_b":
            return ["proposal_b_rollout.txt"]

    critic_only_outcome = os.path.join(episode_dir, "critic_only_outcome.txt")
    if os.path.exists(critic_only_outcome):
        return ["training_rollout.txt"]

    return ["training_rollout.txt"]


def _rolling_mean(values, window=5):
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(values[start:i+1])))
    return result


def _sample_std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _find_best_checkpoint_info(logdir: str, log_rows=None):
    best_reward = float("-inf")
    best_ep = None
    for entry in sorted(os.listdir(logdir)):
        ep_dir = os.path.join(logdir, entry)
        if not (os.path.isdir(ep_dir) and entry.startswith("episode_")):
            continue
        ckpt = os.path.join(ep_dir, "checkpoint.json")
        if not os.path.exists(ckpt):
            continue
        with open(ckpt) as f:
            data = json.load(f)
        r = data.get("reward", float("-inf"))
        if r > best_reward:
            best_reward = r
            best_ep = entry

    if best_ep is None and log_rows:
        best_row = max(log_rows, key=lambda r: r["total_reward"])
        best_reward = float(best_row["total_reward"])
        best_ep = f"episode_{best_row['iteration'] - 1}"
    return best_reward, best_ep


def _find_run_dirs(logdir: str, max_runs: int = None):
    """Return numerically-sorted run_N subdirs, or [] if none exist."""
    if not os.path.isdir(logdir):
        return []
    run_dirs = [
        os.path.join(logdir, e)
        for e in os.listdir(logdir)
        if re.match(r"^run_\d+$", e) and os.path.isdir(os.path.join(logdir, e))
    ]
    run_dirs.sort(key=lambda d: int(os.path.basename(d).split("_")[1]))
    if max_runs is not None:
        return run_dirs[:max_runs]
    return run_dirs


def _analyze_single_run(logdir: str, label: str = None):
    """Analyze one run (logdir must contain overall_log.txt). Saves eval_report/."""
    label = label or os.path.basename(logdir.rstrip("/"))
    report_dir = os.path.join(logdir, "eval_report")
    os.makedirs(report_dir, exist_ok=True)

    log_rows = _parse_overall_log(logdir)
    if not log_rows:
        raise ValueError(f"No data rows in {logdir}/overall_log.txt")

    iterations = [r["iteration"] for r in log_rows]
    rewards = [r["total_reward"] for r in log_rows]
    api_times = [r["api_time"] for r in log_rows]
    rolling = _rolling_mean(rewards, window=5)

    per_episode_rewards, episode_labels = [], []
    for entry in sorted(
        os.listdir(logdir),
        key=lambda x: int(x.split("_")[1]) if x.startswith("episode_") and x.split("_")[1].isdigit() else -1,
    ):
        ep_dir = os.path.join(logdir, entry)
        if not (os.path.isdir(ep_dir) and entry.startswith("episode_")):
            continue
        ep_rewards = _parse_training_rollout(ep_dir)
        if ep_rewards:
            per_episode_rewards.append(ep_rewards)
            episode_labels.append(entry.replace("episode_", "ep"))

    best_reward, best_ep = _find_best_checkpoint_info(logdir, log_rows=log_rows)

    summary = {
        "label": label,
        "logdir": logdir,
        "num_iterations": len(log_rows),
        "best_reward": best_reward,
        "best_episode": best_ep,
        "final_reward": rewards[-1] if rewards else None,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": _sample_std(rewards),
        "max_reward": float(np.max(rewards)),
        "min_reward": float(np.min(rewards)),
        "total_api_time_s": float(np.sum(api_times)),
        "mean_api_time_per_iter_s": float(np.mean(api_times)),
        # kept for cross-run aggregation; stripped before saving
        "_iterations": iterations,
        "_rewards": rewards,
        "_api_times": api_times,
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iterations, rewards, alpha=0.4, color="steelblue", label="Per-iteration reward")
    ax.plot(iterations, rolling, color="steelblue", linewidth=2, label="Rolling mean (window=5)")
    if best_reward > float("-inf"):
        ax.axhline(best_reward, color="red", linestyle="--", linewidth=1, label=f"Best: {best_reward:.3f}")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Mean Episodic Reward")
    ax.set_title(f"Training Curve: {label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(report_dir, "training_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    if per_episode_rewards:
        fig, ax = plt.subplots(figsize=(max(8, len(per_episode_rewards) * 0.6), 5))
        ax.boxplot(
            per_episode_rewards,
            labels=episode_labels,
            patch_artist=True,
            boxprops=dict(facecolor="lightsteelblue", color="steelblue"),
            medianprops=dict(color="darkblue", linewidth=2),
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Rollout Reward")
        ax.set_title(f"Reward Distribution per Episode: {label}")
        ax.grid(True, alpha=0.3, axis="y")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        fig.savefig(os.path.join(report_dir, "reward_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(iterations, api_times, color="coral", alpha=0.7)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("API Time (s)")
    ax.set_title(f"API Call Time per Iteration: {label}")
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(os.path.join(report_dir, "api_time.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary_out = {k: v for k, v in summary.items() if not k.startswith("_")}
    with open(os.path.join(report_dir, "summary.json"), "w") as f:
        json.dump(summary_out, f, indent=2)

    lines = [
        f"Evaluation Report: {label}",
        f"Logdir:            {logdir}",
        f"Iterations:        {summary['num_iterations']}",
        f"Best reward:       {summary['best_reward']:.4f} ({best_ep})",
        f"Final reward:      {summary['final_reward']:.4f}",
        f"Mean reward:       {summary['mean_reward']:.4f} +/- {summary['std_reward']:.4f}",
        f"Max reward:        {summary['max_reward']:.4f}",
        f"Total API time:    {summary['total_api_time_s']:.1f}s",
        f"Mean API/iter:     {summary['mean_api_time_per_iter_s']:.2f}s",
        "",
        f"Saved to: {report_dir}",
    ]
    summary_txt = "\n".join(lines)
    with open(os.path.join(report_dir, "summary.txt"), "w") as f:
        f.write(summary_txt)
    print(summary_txt)

    return summary


def _analyze_multi_run(logdir: str, run_dirs: list, label: str = None):
    """Analyze all run_N dirs; saves per-run reports and an aggregate report at logdir."""
    label = label or os.path.basename(logdir.rstrip("/"))
    colors = ["steelblue", "coral", "forestgreen", "mediumpurple", "goldenrod",
              "saddlebrown", "teal", "crimson"]

    per_run_summaries = []
    for run_dir in run_dirs:
        print(f"\n  Analyzing {run_dir}")
        try:
            s = _analyze_single_run(run_dir, label=f"{label}/{os.path.basename(run_dir)}")
            per_run_summaries.append(s)
        except (FileNotFoundError, ValueError) as e:
            print(f"  Warning: skipping {run_dir}: {e}")

    if not per_run_summaries:
        raise ValueError(f"No valid run_N directories found under {logdir}")

    report_dir = os.path.join(logdir, "eval_report")
    os.makedirs(report_dir, exist_ok=True)

    # Align to shortest run
    all_rewards = [s["_rewards"] for s in per_run_summaries]
    min_len = min(len(r) for r in all_rewards)
    aligned = np.array([r[:min_len] for r in all_rewards])
    iterations = per_run_summaries[0]["_iterations"][:min_len]
    mean_r = aligned.mean(axis=0)
    std_r = (
        aligned.std(axis=0, ddof=1)
        if len(per_run_summaries) > 1
        else np.zeros_like(mean_r)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, s in enumerate(per_run_summaries):
        ax.plot(iterations, s["_rewards"][:min_len], alpha=0.2,
                color=colors[i % len(colors)], label=os.path.basename(s["logdir"]))
    ax.fill_between(iterations, mean_r - std_r, mean_r + std_r, alpha=0.15, color="steelblue")
    ax.plot(iterations, _rolling_mean(mean_r.tolist(), window=5),
            color="steelblue", linewidth=2,
            label=f"Mean rolling (n={len(per_run_summaries)} runs)")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Mean Episodic Reward")
    ax.set_title(f"Aggregate Training Curve: {label} ({len(per_run_summaries)} runs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(report_dir, "training_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    mean_rewards_per_run = [s["mean_reward"] for s in per_run_summaries]
    max_rewards_per_run = [s["max_reward"] for s in per_run_summaries]
    # exclude runs with no checkpoints (best_reward == -inf)
    best_rewards_per_run = [s["best_reward"] for s in per_run_summaries if s["best_reward"] > float("-inf")]
    final_rewards_per_run = [s["final_reward"] for s in per_run_summaries if s["final_reward"] is not None]

    agg_summary = {
        "label": label,
        "logdir": logdir,
        "num_runs": len(per_run_summaries),
        "num_iterations_per_run": [s["num_iterations"] for s in per_run_summaries],
        "mean_reward_mean": float(np.mean(mean_rewards_per_run)),
        "mean_reward_std": _sample_std(mean_rewards_per_run),
        "max_reward_mean": float(np.mean(max_rewards_per_run)),
        "max_reward_std": _sample_std(max_rewards_per_run),
        "best_reward_mean": float(np.mean(best_rewards_per_run)) if best_rewards_per_run else None,
        "best_reward_std": _sample_std(best_rewards_per_run) if best_rewards_per_run else None,
        "final_reward_mean": float(np.mean(final_rewards_per_run)) if final_rewards_per_run else None,
        "final_reward_std": _sample_std(final_rewards_per_run) if final_rewards_per_run else None,
        "per_run_summaries": [
            {k: v for k, v in s.items() if not k.startswith("_")}
            for s in per_run_summaries
        ],
    }

    with open(os.path.join(report_dir, "summary.json"), "w") as f:
        json.dump(agg_summary, f, indent=2)

    lines = [
        f"Aggregate Evaluation Report: {label}",
        f"Logdir:         {logdir}",
        f"Runs:           {agg_summary['num_runs']}",
        f"Mean reward:    {agg_summary['mean_reward_mean']:.4f} +/- {agg_summary['mean_reward_std']:.4f}",
        f"Max reward:     {agg_summary['max_reward_mean']:.4f} +/- {agg_summary['max_reward_std']:.4f}",
    ]
    if agg_summary["best_reward_mean"] is not None:
        lines.append(f"Best reward:    {agg_summary['best_reward_mean']:.4f} +/- {agg_summary['best_reward_std']:.4f}")
    if agg_summary["final_reward_mean"] is not None:
        lines.append(f"Final reward:   {agg_summary['final_reward_mean']:.4f} +/- {agg_summary['final_reward_std']:.4f}")
    lines += ["", f"Saved to: {report_dir}"]
    summary_txt = "\n".join(lines)
    with open(os.path.join(report_dir, "summary.txt"), "w") as f:
        f.write(summary_txt)
    print(summary_txt)

    return agg_summary


def analyze(logdir: str, label: str = None, max_runs: int = None):
    """
    Analyze a training experiment. Auto-detects multi-run layout (run_N/ subdirs)
    vs single-run (overall_log.txt directly in logdir).
    """
    run_dirs = _find_run_dirs(logdir, max_runs=max_runs)
    if run_dirs:
        print(f"  Found {len(run_dirs)} run(s): {[os.path.basename(d) for d in run_dirs]}")
        return _analyze_multi_run(logdir, run_dirs, label=label)
    return _analyze_single_run(logdir, label=label)


def compare(
    logdirs: list,
    labels: list = None,
    output_dir: str = None,
    max_runs: int = None,
):
    """
    Overlay multiple experiments on one plot. Multi-run experiments show
    per-run faded lines + mean ± std band; single-run shows a plain curve.
    """
    if labels is None:
        labels = [os.path.basename(d.rstrip("/")) for d in logdirs]
    if output_dir is None:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    colors = ["steelblue", "coral", "forestgreen", "mediumpurple", "goldenrod"]

    fig, ax = plt.subplots(figsize=(12, 6))
    all_summaries = []
    for i, (logdir, label) in enumerate(zip(logdirs, labels)):
        color = colors[i % len(colors)]
        run_dirs = _find_run_dirs(logdir, max_runs=max_runs)

        if run_dirs:
            all_run_rows = []
            for run_dir in run_dirs:
                try:
                    all_run_rows.append(_parse_overall_log(run_dir))
                except FileNotFoundError:
                    print(f"Warning: no overall_log.txt in {run_dir}, skipping")
            if not all_run_rows:
                print(f"Warning: no valid runs in {logdir}, skipping")
                continue
            min_len = min(len(r) for r in all_run_rows)
            # Use actual iteration numbers from the first run
            iterations = [r["iteration"] for r in all_run_rows[0][:min_len]]
            all_run_rewards = [[r["total_reward"] for r in rows[:min_len]] for rows in all_run_rows]
            aligned = np.array(all_run_rewards)
            mean_r = aligned.mean(axis=0)
            std_r = (
                aligned.std(axis=0, ddof=1)
                if len(all_run_rewards) > 1
                else np.zeros_like(mean_r)
            )
            for run_rewards in all_run_rewards:
                ax.plot(iterations, run_rewards, alpha=0.12, color=color)
            ax.fill_between(iterations, mean_r - std_r, mean_r + std_r, alpha=0.15, color=color)
            ax.plot(iterations, _rolling_mean(mean_r.tolist(), window=5),
                    color=color, linewidth=2, label=f"{label} (n={len(all_run_rewards)})")
            all_summaries.append({
                "label": label,
                "num_runs": len(all_run_rewards),
                "mean_reward": float(np.mean(mean_r)),
                "best_reward": float(np.max(aligned)),  # true best across all runs
            })
        else:
            try:
                log_rows = _parse_overall_log(logdir)
            except FileNotFoundError:
                print(f"Warning: no overall_log.txt in {logdir}, skipping")
                continue
            iterations = [r["iteration"] for r in log_rows]
            rewards = [r["total_reward"] for r in log_rows]
            ax.plot(iterations, rewards, alpha=0.25, color=color)
            ax.plot(iterations, _rolling_mean(rewards, window=5),
                    color=color, linewidth=2, label=label)
            all_summaries.append({
                "label": label,
                "num_runs": 1,
                "mean_reward": float(np.mean(rewards)),
                "best_reward": float(np.max(rewards)),
            })

    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Mean Episodic Reward")
    ax.set_title("Training Curve Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    compare_path = os.path.join(output_dir, "comparison_training_curve.png")
    fig.savefig(compare_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved: {compare_path}")

    with open(os.path.join(output_dir, "comparison_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    return all_summaries
