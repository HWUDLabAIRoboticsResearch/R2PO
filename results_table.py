"""
Generate a cross-method results table from existing aggregate log summaries.

Examples:
  uv run python results_table.py

  uv run python results_table.py --output results_10env_methods_table.md

  uv run python results_table.py --format csv --output results_10env_methods_table.csv

  uv run python results_table.py --format png --output results_10env_methods_table.png
"""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ENVS = [
    ("nim", "nim"),
    ("pong", "pong"),
    ("swimmer", "swimmer"),
    ("mountaincarcon", "mountaincarcontinuous"),
    ("mountaincar", "mountaincar"),
    ("inverteddoublependulum", "inverteddoublependulum"),
    ("invertedpendulum", "invertedpendulum"),
    ("frozenlake", "frozenlake"),
    ("cartpole", "cartpole"),
    ("maze", "maze"),
]

METHOD_PRESETS = {
    "all": None,
    "core": ["ProPS", "ProPS+", "Ref_Cons", "SB3"],
    "variants": [
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


def _display_method(method: str) -> str:
    return DISPLAY_NAMES.get(method, method)


def _load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def _normalize_metrics(data: dict):
    if "best_reward_mean" in data:
        return {
            "mean_reward": data.get("mean_reward_mean"),
            "mean_reward_std": data.get("mean_reward_std"),
            "mean_best_reward": data.get("best_reward_mean"),
            "mean_best_reward_std": data.get("best_reward_std"),
            "mean_final_reward": data.get("final_reward_mean"),
            "mean_final_reward_std": data.get("final_reward_std"),
        }
    if "mean_best_reward" in data:
        return {
            "mean_reward": data.get("mean_reward_mean"),
            "mean_reward_std": data.get("mean_reward_std"),
            "mean_best_reward": data.get("mean_best_reward"),
            "mean_best_reward_std": data.get("std_best_reward"),
            "mean_final_reward": data.get("mean_final_reward"),
            "mean_final_reward_std": data.get("std_final_reward"),
        }
    raise ValueError("Unrecognized summary schema")


def _validate_summary(
    data,
    method_name,
    expected_runs=10,
    expected_episodes=100,
    expected_props_episodes=None,
    expected_critic_only_episodes=None,
    expected_sb3_episodes=None,
):
    issues = []

    expected_episode_count = expected_episodes
    if method_name in {"ProPS", "ProPS+"} and expected_props_episodes is not None:
        expected_episode_count = expected_props_episodes
    if method_name == "CriticOnly" and expected_critic_only_episodes is not None:
        expected_episode_count = expected_critic_only_episodes

    num_runs = data.get("num_runs")
    if num_runs is not None and int(num_runs) != int(expected_runs):
        issues.append(f"num_runs={num_runs} (expected {expected_runs})")

    iteration_list = data.get("num_iterations_per_run")
    if iteration_list is not None:
        bad = [x for x in iteration_list if int(x) != int(expected_episode_count)]
        if bad:
            issues.append(
                f"num_iterations_per_run contains non-{expected_episode_count} values"
            )
        return issues

    per_run = data.get("per_run_summaries") or []
    if not per_run:
        return issues

    first = per_run[0]
    if "num_iterations" in first:
        bad = [
            (idx + 1, run.get("num_iterations"))
            for idx, run in enumerate(per_run)
            if int(run.get("num_iterations", -1)) != int(expected_episode_count)
        ]
        if bad:
            issues.append(
                "invalid num_iterations runs: "
                + ", ".join(
                    f"run_{run_idx}={value}" for run_idx, value in bad
                )
                + f" (expected {expected_episode_count})"
            )
    elif expected_sb3_episodes is not None and "num_episodes" in first:
        bad = [
            (idx + 1, run.get("num_episodes"))
            for idx, run in enumerate(per_run)
            if int(run.get("num_episodes", -1)) != int(expected_sb3_episodes)
        ]
        if bad:
            issues.append(
                "invalid num_episodes runs: "
                + ", ".join(
                    f"run_{run_idx}={value}" for run_idx, value in bad
                )
                + f" (expected {expected_sb3_episodes})"
            )

    return issues


def _maybe_add_row(
    rows,
    env_name,
    method_name,
    summary_path: Path,
    expected_runs=10,
    expected_episodes=100,
    expected_props_episodes=None,
    expected_critic_only_episodes=None,
    strict_validation=False,
    expected_sb3_episodes=None,
):
    if not summary_path.is_file():
        return
    data = _load_json(summary_path)
    issues = _validate_summary(
        data,
        method_name,
        expected_runs=expected_runs,
        expected_episodes=expected_episodes,
        expected_props_episodes=expected_props_episodes,
        expected_critic_only_episodes=expected_critic_only_episodes,
        expected_sb3_episodes=expected_sb3_episodes,
    )
    if issues:
        print(
            f"[validation] {env_name} {method_name}: "
            + "; ".join(issues)
            + f" [{summary_path}]"
        )
        if strict_validation:
            return
    metrics = _normalize_metrics(data)
    rows.append(
        {
            "environment": env_name,
            "method": method_name,
            "mean_reward": metrics["mean_reward"],
            "mean_reward_std": metrics["mean_reward_std"],
            "mean_best_reward": metrics["mean_best_reward"],
            "mean_best_reward_std": metrics["mean_best_reward_std"],
            "mean_final_reward": metrics["mean_final_reward"],
            "mean_final_reward_std": metrics["mean_final_reward_std"],
            "label": data.get("label"),
            "source": str(summary_path),
            "validation_issues": issues,
        }
    )


def collect_rows(
    logs_root: Path,
    env_specs,
    expected_runs=10,
    expected_episodes=100,
    expected_props_episodes=None,
    expected_critic_only_episodes=None,
    strict_validation=False,
    expected_sb3_episodes=None,
):
    rows = []
    for env_name, env_prefix in env_specs:
        _maybe_add_row(
            rows,
            env_name,
            "ProPS",
            logs_root / f"{env_prefix}_props" / "eval_report" / "summary.json",
            expected_runs=expected_runs,
            expected_episodes=expected_episodes,
            expected_props_episodes=expected_props_episodes,
            expected_critic_only_episodes=expected_critic_only_episodes,
            strict_validation=strict_validation,
            expected_sb3_episodes=expected_sb3_episodes,
        )
        _maybe_add_row(
            rows,
            env_name,
            "ProPS+",
            logs_root / f"{env_prefix}_propsp" / "eval_report" / "summary.json",
            expected_runs=expected_runs,
            expected_episodes=expected_episodes,
            expected_props_episodes=expected_props_episodes,
            expected_critic_only_episodes=expected_critic_only_episodes,
            strict_validation=strict_validation,
            expected_sb3_episodes=expected_sb3_episodes,
        )
        _maybe_add_row(
            rows,
            env_name,
            "Reflective",
            logs_root / f"{env_prefix}_reflective" / "eval_report" / "summary.json",
            expected_runs=expected_runs,
            expected_episodes=expected_episodes,
            expected_props_episodes=expected_props_episodes,
            expected_critic_only_episodes=expected_critic_only_episodes,
            strict_validation=strict_validation,
            expected_sb3_episodes=expected_sb3_episodes,
        )
        _maybe_add_row(
            rows,
            env_name,
            "Ref_Cons",
            logs_root
            / f"{env_prefix}_reflective_prompted_policy_optimization"
            / "eval_report"
            / "summary.json",
            expected_runs=expected_runs,
            expected_episodes=expected_episodes,
            expected_props_episodes=expected_props_episodes,
            expected_critic_only_episodes=expected_critic_only_episodes,
            strict_validation=strict_validation,
            expected_sb3_episodes=expected_sb3_episodes,
        )
        _maybe_add_row(
            rows,
            env_name,
            "ActorSecondPass",
            logs_root
            / f"{env_prefix}_actor_second_pass"
            / "eval_report"
            / "summary.json",
            expected_runs=expected_runs,
            expected_episodes=expected_episodes,
            expected_props_episodes=expected_props_episodes,
            expected_critic_only_episodes=expected_critic_only_episodes,
            strict_validation=strict_validation,
            expected_sb3_episodes=expected_sb3_episodes,
        )
        _maybe_add_row(
            rows,
            env_name,
            "ThreeTraj",
            logs_root
            / f"{env_prefix}_three_traj"
            / "eval_report"
            / "summary.json",
            expected_runs=expected_runs,
            expected_episodes=expected_episodes,
            expected_props_episodes=expected_props_episodes,
            expected_critic_only_episodes=expected_critic_only_episodes,
            strict_validation=strict_validation,
            expected_sb3_episodes=expected_sb3_episodes,
        )
        _maybe_add_row(
            rows,
            env_name,
            "PureSearch",
            logs_root / f"{env_prefix}_pure_search" / "eval_report" / "summary.json",
            expected_runs=expected_runs,
            expected_episodes=expected_episodes,
            expected_props_episodes=expected_props_episodes,
            expected_critic_only_episodes=expected_critic_only_episodes,
            strict_validation=strict_validation,
            expected_sb3_episodes=expected_sb3_episodes,
        )
        _maybe_add_row(
            rows,
            env_name,
            "AlwaysCritic",
            logs_root / f"{env_prefix}_always_critic" / "eval_report" / "summary.json",
            expected_runs=expected_runs,
            expected_episodes=expected_episodes,
            expected_props_episodes=expected_props_episodes,
            expected_critic_only_episodes=expected_critic_only_episodes,
            strict_validation=strict_validation,
            expected_sb3_episodes=expected_sb3_episodes,
        )
        _maybe_add_row(
            rows,
            env_name,
            "CriticOnly",
            logs_root / f"{env_prefix}_critic_only" / "eval_report" / "summary.json",
            expected_runs=expected_runs,
            expected_episodes=expected_episodes,
            expected_props_episodes=expected_props_episodes,
            expected_critic_only_episodes=expected_critic_only_episodes,
            strict_validation=strict_validation,
            expected_sb3_episodes=expected_sb3_episodes,
        )

        sb3_dir = logs_root / f"{env_prefix}_sb3"
        if sb3_dir.is_dir():
            for algo_dir in sorted(x for x in sb3_dir.iterdir() if x.is_dir()):
                summary_path = algo_dir / "summary.json"
                if summary_path.is_file():
                    _maybe_add_row(
                        rows,
                        env_name,
                        f"SB3-{algo_dir.name.upper()}",
                        summary_path,
                        expected_runs=expected_runs,
                        expected_episodes=expected_episodes,
                        expected_props_episodes=expected_props_episodes,
                        expected_critic_only_episodes=expected_critic_only_episodes,
                        strict_validation=strict_validation,
                        expected_sb3_episodes=expected_sb3_episodes,
                    )
    return rows


def filter_rows(rows, method_preset="all"):
    allowed = METHOD_PRESETS.get(method_preset)
    if allowed is None:
        filtered = list(rows)
    else:
        filtered = []
        for row in rows:
            method = row["method"]
            if "SB3" in allowed and method.startswith("SB3-"):
                filtered.append(row)
                continue
            if method in allowed:
                filtered.append(row)

    method_rank = {
        "ProPS": 0,
        "ProPS+": 1,
        "PureSearch": 2,
        "ActorSecondPass": 3,
        "CriticOnly": 4,
        "AlwaysCritic": 5,
        "Reflective": 6,
        "ThreeTraj": 7,
        "Ref_Cons": 8,
    }

    def sort_key(row):
        method = row["method"]
        if method.startswith("SB3-"):
            rank = 9
        else:
            rank = method_rank.get(method, 99)
        return (row["environment"], rank, method)

    return sorted(filtered, key=sort_key)


def _format_number(value):
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _format_metric_cell(value, std, line_break="\n"):
    mean_text = _format_number(value)
    if not mean_text:
        return ""
    std_text = _format_number(std)
    if not std_text:
        return mean_text
    return f"{mean_text} \u00b1 {std_text}"


def _compute_best_by_env(rows):
    metric_keys = ["mean_reward", "mean_best_reward", "mean_final_reward"]
    best = {}
    envs = sorted({row["environment"] for row in rows})
    for env in envs:
        env_rows = [row for row in rows if row["environment"] == env]
        best[env] = {}
        for key in metric_keys:
            values = [row[key] for row in env_rows if row[key] is not None]
            best[env][key] = max(values) if values else None
    return best


def _is_best(row, metric_key, best_by_env, tol=1e-12):
    value = row.get(metric_key)
    best_value = best_by_env.get(row["environment"], {}).get(metric_key)
    if value is None or best_value is None:
        return False
    return abs(float(value) - float(best_value)) <= tol


def render_markdown(rows, include_source=False):
    best_by_env = _compute_best_by_env(rows)
    headers = [
        "Environment",
        "Method",
        "Mean Reward",
        "Mean Best Reward",
        "Mean Final Reward",
    ]
    if include_source:
        headers.append("Source")

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for row in rows:
        values = [
            row["environment"],
            _display_method(row["method"]),
            _format_metric_cell(
                row["mean_reward"],
                row["mean_reward_std"],
                line_break="<br>",
            ),
            _format_metric_cell(
                row["mean_best_reward"],
                row["mean_best_reward_std"],
                line_break="<br>",
            ),
            _format_metric_cell(
                row["mean_final_reward"],
                row["mean_final_reward_std"],
                line_break="<br>",
            ),
        ]
        for idx, metric_key in zip([2, 3, 4], ["mean_reward", "mean_best_reward", "mean_final_reward"]):
            if _is_best(row, metric_key, best_by_env):
                values[idx] = f"**{values[idx]}**"
        if include_source:
            values.append(row["source"])
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_csv(rows, output_path: Path, include_source=False):
    fieldnames = [
        "environment",
        "method",
        "mean_reward",
        "mean_reward_std",
        "mean_best_reward",
        "mean_best_reward_std",
        "mean_final_reward",
        "mean_final_reward_std",
    ]
    if include_source:
        fieldnames.append("source")

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {
                "environment": row["environment"],
                "method": _display_method(row["method"]),
                "mean_reward": _format_number(row["mean_reward"]),
                "mean_reward_std": _format_number(row["mean_reward_std"]),
                "mean_best_reward": _format_number(row["mean_best_reward"]),
                "mean_best_reward_std": _format_number(row["mean_best_reward_std"]),
                "mean_final_reward": _format_number(row["mean_final_reward"]),
                "mean_final_reward_std": _format_number(row["mean_final_reward_std"]),
            }
            if include_source:
                out["source"] = row["source"]
            writer.writerow(out)


def write_png(rows, output_path: Path, include_source=False, title=None):
    best_by_env = _compute_best_by_env(rows)
    headers = [
        "Method",
        "Mean Reward",
        "Mean Best Reward",
        "Mean Final Reward",
    ]
    if include_source:
        headers.append("Source")

    grouped = {}
    for row in rows:
        grouped.setdefault(row["environment"], []).append(row)

    env_names = list(grouped.keys())
    height_units = sum(len(grouped[env]) + 1.9 for env in env_names)
    fig_width = 12 if not include_source else 18
    fig_height = max(10, height_units * 0.52)

    fig, axes = plt.subplots(
        nrows=len(env_names),
        ncols=1,
        figsize=(fig_width, fig_height),
        squeeze=False,
        gridspec_kw={"height_ratios": [len(grouped[env]) + 1.5 for env in env_names]},
    )

    col_widths = [0.24, 0.19, 0.19, 0.19]
    if include_source:
        col_widths.append(0.19)

    for ax, env_name in zip(axes.flatten(), env_names):
        ax.axis("off")
        env_rows = grouped[env_name]
        table_rows = []
        for row in env_rows:
            values = [
                _display_method(row["method"]),
                _format_metric_cell(row["mean_reward"], row["mean_reward_std"]),
                _format_metric_cell(
                    row["mean_best_reward"], row["mean_best_reward_std"]
                ),
                _format_metric_cell(
                    row["mean_final_reward"], row["mean_final_reward_std"]
                ),
            ]
            if include_source:
                values.append(row["source"])
            table_rows.append(values)

        tbl = ax.table(
            cellText=table_rows,
            colLabels=headers,
            cellLoc="center",
            colLoc="center",
            colWidths=col_widths,
            bbox=[0.02, 0.02, 0.96, 0.88],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        tbl.scale(1, 1.95)

        for (row_idx, col_idx), cell in tbl.get_celld().items():
            cell.set_edgecolor("#BFC7D5")
            cell.set_linewidth(0.8)
            if row_idx == 0:
                cell.set_facecolor("#DCE6F2")
                cell.set_text_props(weight="bold", color="#1F2937")
            else:
                cell.set_facecolor("#F8FAFC" if row_idx % 2 else "#EEF3F8")
                row = env_rows[row_idx - 1]
                metric_map = {
                    1: "mean_reward",
                    2: "mean_best_reward",
                    3: "mean_final_reward",
                }
                metric_key = metric_map.get(col_idx)
                if metric_key and _is_best(row, metric_key, best_by_env):
                    cell.set_facecolor("#D9F2E3")
                    cell.set_text_props(weight="bold", color="#14532D")

        ax.text(
            0.02,
            0.98,
            env_name,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            color="#111827",
        )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)

    plt.tight_layout(pad=1.4, h_pad=1.2)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root",
        default="logs",
        help="Root directory containing experiment logs",
    )
    parser.add_argument(
        "--format",
        choices=["md", "csv", "png"],
        default="md",
        help="Output table format",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file path. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--include-source",
        action="store_true",
        help="Include source summary path in the output table",
    )
    parser.add_argument(
        "--method-preset",
        choices=sorted(METHOD_PRESETS.keys()),
        default="all",
        help="Which method family to include in the table",
    )
    parser.add_argument(
        "--title",
        default="10-Environment Results Table",
        help="Optional title for PNG output",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=10,
        help="Expected number of runs per experiment for validation",
    )
    parser.add_argument(
        "--expected-episodes",
        type=int,
        default=100,
        help="Expected iterations/episodes per run for validation",
    )
    parser.add_argument(
        "--expected-props-episodes",
        type=int,
        default=None,
        help="Expected iterations per run for ProPS and ProPS+, e.g. 200",
    )
    parser.add_argument(
        "--expected-critic-only-episodes",
        type=int,
        default=None,
        help="Expected iterations per run for CriticOnly, e.g. 200",
    )
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Exclude rows that fail the expected runs/episodes validation",
    )
    parser.add_argument(
        "--expected-sb3-episodes",
        type=int,
        default=None,
        help="Expected SB3 num_episodes per run for validation, e.g. 190",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logs_root = Path(args.logs_root)
    rows = collect_rows(
        logs_root,
        DEFAULT_ENVS,
        expected_runs=args.expected_runs,
        expected_episodes=args.expected_episodes,
        expected_props_episodes=args.expected_props_episodes,
        expected_critic_only_episodes=args.expected_critic_only_episodes,
        strict_validation=args.strict_validation,
        expected_sb3_episodes=args.expected_sb3_episodes,
    )
    rows = filter_rows(rows, method_preset=args.method_preset)

    if args.format == "md":
        rendered = render_markdown(rows, include_source=args.include_source)
        if args.output:
            Path(args.output).write_text(rendered + "\n")
        else:
            print(rendered)
    elif args.format == "csv":
        if args.output is None:
            raise SystemExit("--output is required when --format csv is used")
        write_csv(rows, Path(args.output), include_source=args.include_source)
    else:
        if args.output is None:
            raise SystemExit("--output is required when --format png is used")
        write_png(
            rows,
            Path(args.output),
            include_source=args.include_source,
            title=args.title,
        )


if __name__ == "__main__":
    main()
