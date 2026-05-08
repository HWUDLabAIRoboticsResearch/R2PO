#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from analyze_reptraj_edit_patterns import (
    build_thresholds,
    detect_env_name,
    load_records,
)


METHOD_LABELS = [
    ("reflective_prompted_policy_optimization", "R2PO"),
    ("three_traj", "ThreeTraj"),
    ("always_critic", "AlwaysCritic"),
    ("critic_only", "CriticOnly"),
    ("reflective", "RepTraj"),
]


class _ArgsNamespace:
    distance_mode = "auto"
    epsilon = None
    partial_threshold = None
    high_threshold = None
    near_zero_threshold = None
    meaningful_delta = None
    surgical_max_edit = None
    redesign_min_edit = None
    multi_state_min_edit = None


def infer_method_label(logdir: Path) -> str:
    name = logdir.name.lower()
    for needle, label in METHOD_LABELS:
        if needle in name:
            return label
    return logdir.name


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "# Episode-Level Edit Table",
        "",
        "| Method | Run | Episode | Initial Reward | Revised Reward | Delta | Edit Distance | L2 Distance | Winner | Category |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['run']} | {row['episode']} | "
            f"{row['initial_reward']:.3f} | {row['revised_reward']:.3f} | {row['delta']:.3f} | "
            f"{row['edit_distance']} | {row['l2_distance']:.3f} | {row['winner']} | {row['category']} |"
        )
    lines.append("")
    return "\n".join(lines)


def summarize_by_method(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    summaries: list[dict] = []
    for method, method_rows in sorted(grouped.items()):
        total = len(method_rows)
        winner_counts = Counter(row["winner"] for row in method_rows)
        category_counts = Counter(row["category"] for row in method_rows)
        direction_counts = Counter()
        for row in method_rows:
            delta = row["delta"]
            if delta > 0:
                direction_counts["improvement"] += 1
            elif delta < 0:
                direction_counts["regression"] += 1
            else:
                direction_counts["no_change"] += 1

        summaries.append(
            {
                "method": method,
                "episodes": total,
                "mean_initial_reward": sum(row["initial_reward"] for row in method_rows) / total,
                "mean_revised_reward": sum(row["revised_reward"] for row in method_rows) / total,
                "mean_delta": sum(row["delta"] for row in method_rows) / total,
                "mean_edit_distance": sum(row["edit_distance"] for row in method_rows) / total,
                "revision_won_pct": 100.0
                * (winner_counts.get("revision_won", 0) + winner_counts.get("revision_forced", 0))
                / total,
                "original_won_pct": 100.0 * winner_counts.get("original_won", 0) / total,
                "improvement_pct": 100.0 * direction_counts.get("improvement", 0) / total,
                "regression_pct": 100.0 * direction_counts.get("regression", 0) / total,
                "no_change_pct": 100.0 * direction_counts.get("no_change", 0) / total,
                "pure_stochasticity_pct": 100.0 * category_counts.get("pure_stochasticity", 0) / total,
                "surgical_backfire_pct": 100.0 * category_counts.get("surgical_fix_backfires", 0) / total,
                "near_optimal_break_pct": 100.0
                * category_counts.get("compulsive_revision_of_near_optimal", 0)
                / total,
                "history_guided_rescue_pct": 100.0 * category_counts.get("history_guided_rescue", 0) / total,
                "pinpoint_fix_pct": 100.0 * category_counts.get("pinpoint_bottleneck_fix", 0) / total,
                "fine_tune_near_optimal_pct": 100.0
                * category_counts.get("fine_tuning_near_optimal", 0)
                / total,
            }
        )
    return summaries


def render_summary_markdown(rows: list[dict]) -> str:
    lines = [
        "# Edit Method Summary",
        "",
        "| Method | Episodes | Mean Initial | Mean Revised | Mean Delta | Mean Edit Dist | Revision Won % | Original Won % | Improvement % | Regression % | No Change % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['episodes']} | {row['mean_initial_reward']:.3f} | "
            f"{row['mean_revised_reward']:.3f} | {row['mean_delta']:.3f} | {row['mean_edit_distance']:.3f} | "
            f"{row['revision_won_pct']:.1f} | {row['original_won_pct']:.1f} | {row['improvement_pct']:.1f} | "
            f"{row['regression_pct']:.1f} | {row['no_change_pct']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Pattern Breakdown",
            "",
            "| Method | Pure Stochasticity % | Surgical Backfire % | Near-Optimal Break % | History-Guided Rescue % | Pinpoint Fix % | Fine-Tune Near-Optimal % |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['pure_stochasticity_pct']:.1f} | {row['surgical_backfire_pct']:.1f} | "
            f"{row['near_optimal_break_pct']:.1f} | {row['history_guided_rescue_pct']:.1f} | "
            f"{row['pinpoint_fix_pct']:.1f} | {row['fine_tune_near_optimal_pct']:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge episode-level edit analysis from multiple logdirs into one table."
    )
    parser.add_argument("logdirs", nargs="+", help="Experiment logdirs to merge.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where merged CSV/Markdown files will be written.",
    )
    parser.add_argument(
        "--env",
        help="Optional environment override for threshold selection. Defaults to env inferred from each logdir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict] = []
    threshold_args = _ArgsNamespace()

    for logdir_str in args.logdirs:
        logdir = Path(logdir_str)
        env = (args.env or detect_env_name(logdir)).lower()
        cfg = build_thresholds(threshold_args, env)
        method_label = infer_method_label(logdir)
        records = load_records(logdir, cfg)

        for record in records:
            merged_rows.append(
                {
                    "method": method_label,
                    "run": record.run,
                    "episode": record.episode,
                    "initial_reward": record.initial_reward,
                    "revised_reward": record.revised_reward,
                    "delta": record.delta,
                    "edit_distance": record.edit_distance,
                    "l2_distance": record.l2_distance,
                    "winner": record.winner or "",
                    "category": record.category,
                }
            )

    merged_rows.sort(key=lambda row: (row["method"], row["run"], row["episode"]))
    method_summary_rows = summarize_by_method(merged_rows)

    csv_path = output_dir / "merged_episodes.csv"
    md_path = output_dir / "merged_episodes.md"
    summary_csv_path = output_dir / "method_summary.csv"
    summary_md_path = output_dir / "method_summary.md"

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "method",
                "run",
                "episode",
                "initial_reward",
                "revised_reward",
                "delta",
                "edit_distance",
                "l2_distance",
                "winner",
                "category",
            ],
        )
        writer.writeheader()
        writer.writerows(merged_rows)

    md_path.write_text(render_markdown(merged_rows), encoding="utf-8")
    with summary_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "method",
                "episodes",
                "mean_initial_reward",
                "mean_revised_reward",
                "mean_delta",
                "mean_edit_distance",
                "revision_won_pct",
                "original_won_pct",
                "improvement_pct",
                "regression_pct",
                "no_change_pct",
                "pure_stochasticity_pct",
                "surgical_backfire_pct",
                "near_optimal_break_pct",
                "history_guided_rescue_pct",
                "pinpoint_fix_pct",
                "fine_tune_near_optimal_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(method_summary_rows)
    summary_md_path.write_text(render_summary_markdown(method_summary_rows), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {summary_md_path}")


if __name__ == "__main__":
    main()
