"""
Generate multi-panel reward-vs-episodes figures for paper results.

Each subplot is one environment. For every logged episode budget value, the plot
shows the mean reward across runs with a shaded standard deviation band.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

os_env = __import__("os").environ
os_env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from results_table import DEFAULT_ENVS


METHOD_PRESETS = {
    "core": ["ProPS", "ProPS+", "Ref_Cons", "SB3"],
    "variants": [
        "BestBaseline",
        "PureSearch",
        "ActorSecondPass",
        "CriticOnly",
        "AlwaysCritic",
        "Reflective",
        "ThreeTraj",
        "Ref_Cons",
    ],
}

DISPLAY_NAMES = {
    "Reflective": "RepTraj",
    "Ref_Cons": "R2PO",
}

ENV_DISPLAY_NAMES = {
    "nim": "Nim",
    "pong": "Pong",
    "swimmer": "Swimmer",
    "mountaincarcon": "MountainCarCon",
    "mountaincar": "MountainCar",
    "inverteddoublependulum": "InvDblPend",
    "invertedpendulum": "InvPendulum",
    "frozenlake": "FrozenLake",
    "cartpole": "CartPole",
    "maze": "Maze",
}


CORE_COLORS = {
    "ProPS": "#2563EB",
    "ProPS+": "#EA580C",
    "BestBaseline": "#111827",
    "Reflective": "#2563EB",
    "Ref_Cons": "#059669",
    "SB3": "#8B5CF6",
    "ActorSecondPass": "#DC2626",
    "ThreeTraj": "#0891B2",
    "AlwaysCritic": "#D97706",
    "CriticOnly": "#7C3AED",
    "PureSearch": "#6B7280",
}


def _selected_envs(env_names=None):
    if not env_names:
        return DEFAULT_ENVS

    requested = [name.strip().lower() for name in env_names if name.strip()]
    env_lookup = {env_name.lower(): (env_name, env_prefix) for env_name, env_prefix in DEFAULT_ENVS}
    prefix_lookup = {env_prefix.lower(): (env_name, env_prefix) for env_name, env_prefix in DEFAULT_ENVS}

    selected = []
    missing = []
    for name in requested:
        entry = env_lookup.get(name) or prefix_lookup.get(name)
        if entry is None:
            missing.append(name)
            continue
        if entry not in selected:
            selected.append(entry)

    if missing:
        valid = ", ".join(env_name for env_name, _ in DEFAULT_ENVS)
        raise ValueError(f"Unknown env(s): {', '.join(missing)}. Valid values: {valid}")

    return selected


def _read_overall_log(path: Path):
    xs = []
    ys = []
    with path.open() as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                total_episodes = float(parts[3])
                total_reward = float(parts[5])
            except ValueError:
                continue
            xs.append(total_episodes)
            ys.append(total_reward)
    return xs, ys


def _aggregate_run_curves(logdir: Path, max_runs=10):
    series_by_x = defaultdict(list)
    for run_idx in range(1, max_runs + 1):
        overall_log = logdir / f"run_{run_idx}" / "overall_log.txt"
        if not overall_log.is_file():
            continue
        xs, ys = _read_overall_log(overall_log)
        for x, y in zip(xs, ys):
            series_by_x[x].append(y)

    if not series_by_x:
        return None

    x_values = sorted(series_by_x)
    means = np.array([np.mean(series_by_x[x]) for x in x_values], dtype=float)
    stds = np.array(
        [
            np.std(series_by_x[x], ddof=1) if len(series_by_x[x]) > 1 else 0.0
            for x in x_values
        ],
        dtype=float,
    )
    return np.array(x_values, dtype=float), means, stds


def _read_summary_metric(summary_path: Path, metric_key: str):
    if not summary_path.is_file():
        return None
    data = json.loads(summary_path.read_text())
    return data.get(metric_key)


def _find_best_baseline(logs_root: Path, env_prefix: str):
    candidates = []

    props_dir = logs_root / f"{env_prefix}_props"
    if props_dir.is_dir():
        metric = _read_summary_metric(
            props_dir / "eval_report" / "summary.json", "mean_reward_mean"
        )
        if metric is not None:
            candidates.append((float(metric), "BestBaseline (ProPS)", props_dir))

    propsp_dir = logs_root / f"{env_prefix}_propsp"
    if propsp_dir.is_dir():
        metric = _read_summary_metric(
            propsp_dir / "eval_report" / "summary.json", "mean_reward_mean"
        )
        if metric is not None:
            candidates.append((float(metric), "BestBaseline (ProPS+)", propsp_dir))

    sb3_root = logs_root / f"{env_prefix}_sb3"
    if sb3_root.is_dir():
        for algo_dir in sorted(x for x in sb3_root.iterdir() if x.is_dir()):
            metric = _read_summary_metric(
                algo_dir / "eval_report" / "summary.json", "mean_reward_mean"
            )
            if metric is None:
                metric = _read_summary_metric(algo_dir / "summary.json", "mean_reward_mean")
            if metric is not None:
                algo = algo_dir.name.upper()
                candidates.append((float(metric), f"BestBaseline ({algo})", algo_dir))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, label, path = candidates[0]
    return label, path


def _candidate_methods(logs_root: Path, env_prefix: str, preset: str):
    if preset == "core":
        candidates = [
            ("ProPS", logs_root / f"{env_prefix}_props"),
            ("ProPS+", logs_root / f"{env_prefix}_propsp"),
            (
                "Ref_Cons",
                logs_root
                / f"{env_prefix}_reflective_prompted_policy_optimization",
            ),
        ]

        sb3_root = logs_root / f"{env_prefix}_sb3"
        if sb3_root.is_dir():
            sb3_candidates = []
            for algo_dir in sorted(x for x in sb3_root.iterdir() if x.is_dir()):
                metric = _read_summary_metric(
                    algo_dir / "eval_report" / "summary.json", "mean_reward_mean"
                )
                if metric is None:
                    metric = _read_summary_metric(algo_dir / "summary.json", "mean_reward_mean")
                if metric is not None:
                    sb3_candidates.append((float(metric), algo_dir))
            if sb3_candidates:
                _, best_algo_dir = max(sb3_candidates, key=lambda item: item[0])
                candidates.append((f"SB3 ({best_algo_dir.name.upper()})", best_algo_dir))
    elif preset == "variants":
        candidates = [
            ("Reflective", logs_root / f"{env_prefix}_reflective"),
            (
                "Ref_Cons",
                logs_root
                / f"{env_prefix}_reflective_prompted_policy_optimization",
            ),
            ("ActorSecondPass", logs_root / f"{env_prefix}_actor_second_pass"),
            ("ThreeTraj", logs_root / f"{env_prefix}_three_traj"),
            ("AlwaysCritic", logs_root / f"{env_prefix}_always_critic"),
            ("CriticOnly", logs_root / f"{env_prefix}_critic_only"),
            ("PureSearch", logs_root / f"{env_prefix}_pure_search"),
        ]
        best_baseline = _find_best_baseline(logs_root, env_prefix)
        if best_baseline is not None:
            candidates.insert(0, best_baseline)
    else:
        raise ValueError(f"Unsupported preset: {preset}")

    return [(label, path) for label, path in candidates if path.is_dir()]


def _base_method_name(label: str):
    if label.startswith("BestBaseline"):
        return "BestBaseline"
    if label.startswith("SB3"):
        return "SB3"
    return label


def _display_label(label: str):
    if label.startswith("BestBaseline ("):
        inner = label[len("BestBaseline ("):-1]
        return f"BestBaseline ({DISPLAY_NAMES.get(inner, inner)})"
    return DISPLAY_NAMES.get(label, label)


def _display_env_name(env_name: str):
    return ENV_DISPLAY_NAMES.get(env_name, env_name.title())


def _plot_env_curves(
    ax,
    logs_root: Path,
    env_name: str,
    env_prefix: str,
    preset: str,
    max_runs=10,
    show_legend=True,
):
    candidates = _candidate_methods(logs_root, env_prefix, preset)
    if not candidates:
        ax.axis("off")
        return

    plotted = False
    if preset == "core":
        candidates = sorted(
            candidates,
            key=lambda item: 0 if item[0].startswith("SB3") else 1,
        )
    for label, logdir in candidates:
        aggregated = _aggregate_run_curves(logdir, max_runs=max_runs)
        if aggregated is None:
            continue
        xs, means, stds = aggregated
        is_sb3 = label.startswith("SB3")
        is_best_baseline = label.startswith("BestBaseline")
        color = CORE_COLORS[_base_method_name(label)]
        ax.plot(
            xs,
            means,
            label=_display_label(label),
            color=color,
            linewidth=2.25 if is_sb3 else (2.4 if label == "Ref_Cons" else (2.2 if is_best_baseline else 2.0)),
            linestyle=":" if is_best_baseline else "-",
            zorder=1 if is_sb3 else (4 if label == "Ref_Cons" else (3 if is_best_baseline else 2)),
        )
        ax.fill_between(
            xs,
            means - stds,
            means + stds,
            color=color,
            alpha=0.16 if is_sb3 else (0.08 if is_best_baseline else 0.12),
            zorder=0 if is_sb3 else 1,
        )
        plotted = True

    ax.set_title(_display_env_name(env_name), fontsize=12, fontweight="bold")
    ax.set_xlabel("Total Episodes")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.25, linewidth=0.7)
    if plotted and show_legend:
        ax.legend(fontsize=8, frameon=False, loc="best")
    elif not plotted:
        ax.text(0.5, 0.5, "No curves found", ha="center", va="center")


def write_curves(
    logs_root: Path,
    output_path: Path,
    preset: str,
    max_runs=10,
    envs=None,
):
    env_entries = _selected_envs(envs)
    ncols = 2
    nrows = int(np.ceil(len(env_entries) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.9 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, (env_name, env_prefix) in zip(axes_flat, env_entries):
        _plot_env_curves(ax, logs_root, env_name, env_prefix, preset, max_runs=max_runs)

    for ax in axes_flat[len(env_entries):]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_curves_per_env(logs_root: Path, output_dir: Path, preset: str, max_runs=10, envs=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    for env_name, env_prefix in _selected_envs(envs):
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        _plot_env_curves(ax, logs_root, env_name, env_prefix, preset, max_runs=max_runs)
        fig.suptitle(
            f"{_display_env_name(env_name)}: {preset.title()} Reward vs Episodes (Mean ± Std)",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_dir / f"{env_name}_reward_curve.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--output", default="paper_results/core_reward_curves.png")
    parser.add_argument(
        "--preset",
        choices=sorted(METHOD_PRESETS),
        default="core",
        help="Which method family to plot",
    )
    parser.add_argument(
        "--layout",
        choices=["all-in-one", "per-env"],
        default="all-in-one",
        help="Render one combined figure or one file per environment",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        help="Optional subset of environments to plot, e.g. cartpole swimmer invertedpendulum mountaincar",
    )
    parser.add_argument("--max-runs", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.layout == "all-in-one":
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_curves(
            Path(args.logs_root),
            output_path,
            preset=args.preset,
            max_runs=args.max_runs,
            envs=args.envs,
        )
    else:
        output_dir = Path(args.output)
        write_curves_per_env(
            Path(args.logs_root),
            output_dir,
            preset=args.preset,
            max_runs=args.max_runs,
            envs=args.envs,
        )


if __name__ == "__main__":
    main()
