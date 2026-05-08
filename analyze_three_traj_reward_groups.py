#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from analyze_reptraj_edit_patterns import detect_env_name, parse_reflection_outcome


ROLLOUT_RE = re.compile(
    r"^(Worst|Median|Best) rollout \(rollout \d+, reward=([-+]?\d+(?:\.\d+)?)\):$"
)


@dataclass
class EpisodeGroupRecord:
    env: str
    run: str
    episode: int
    worst_reward: float
    median_reward: float
    best_reward: float
    initial_reward: float
    revised_reward: float
    delta: float
    winner: str | None
    reward_group: str


def parse_three_rollout_rewards(reasoning_path: Path) -> tuple[float, float, float]:
    rewards: dict[str, float] = {}
    for line in reasoning_path.read_text(encoding="utf-8").splitlines():
        match = ROLLOUT_RE.match(line.strip())
        if match:
            label, reward = match.groups()
            rewards[label.lower()] = float(reward)
            if len(rewards) == 3:
                break

    if len(rewards) != 3:
        raise ValueError(f"Could not parse worst/median/best rollout rewards from {reasoning_path}")

    return rewards["worst"], rewards["median"], rewards["best"]


def classify_reward_group(worst: float, median: float, best: float, eps: float = 1e-12) -> str:
    worst_eq_median = abs(worst - median) <= eps
    median_eq_best = abs(median - best) <= eps
    worst_eq_best = abs(worst - best) <= eps

    if worst_eq_median and median_eq_best:
        return "identical"
    if worst_eq_median:
        return "worst_eq_median"
    if median_eq_best:
        return "median_eq_best"
    if not worst_eq_median and not median_eq_best and not worst_eq_best:
        return "all_diverse"
    return "other"


def load_records(logdir: Path) -> list[EpisodeGroupRecord]:
    env = detect_env_name(logdir).lower()
    records: list[EpisodeGroupRecord] = []

    for run_dir in sorted(logdir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        for episode_dir in sorted(
            run_dir.glob("episode_*"), key=lambda p: int(p.name.split("_")[1])
        ):
            reasoning_path = episode_dir / "parameters_reasoning.txt"
            reflection_path = episode_dir / "reflection_outcome.txt"
            if not (reasoning_path.exists() and reflection_path.exists()):
                continue

            try:
                worst_reward, median_reward, best_reward = parse_three_rollout_rewards(
                    reasoning_path
                )
                initial_reward, revised_reward, winner = parse_reflection_outcome(
                    reflection_path
                )
            except ValueError:
                continue

            records.append(
                EpisodeGroupRecord(
                    env=env,
                    run=run_dir.name,
                    episode=int(episode_dir.name.split("_")[1]),
                    worst_reward=worst_reward,
                    median_reward=median_reward,
                    best_reward=best_reward,
                    initial_reward=initial_reward,
                    revised_reward=revised_reward,
                    delta=revised_reward - initial_reward,
                    winner=winner,
                    reward_group=classify_reward_group(
                        worst_reward, median_reward, best_reward
                    ),
                )
            )

    return records


def summarize(records: list[EpisodeGroupRecord]) -> list[dict]:
    grouped: dict[str, list[EpisodeGroupRecord]] = defaultdict(list)
    for record in records:
        grouped[record.reward_group].append(record)

    rows = []
    for group_name, group_records in sorted(grouped.items()):
        count = len(group_records)
        rows.append(
            {
                "reward_group": group_name,
                "episodes": count,
                "pct": 100.0 * count / len(records),
                "mean_worst_reward": sum(r.worst_reward for r in group_records) / count,
                "mean_median_reward": sum(r.median_reward for r in group_records) / count,
                "mean_best_reward": sum(r.best_reward for r in group_records) / count,
                "mean_initial_reward": sum(r.initial_reward for r in group_records) / count,
                "mean_revised_reward": sum(r.revised_reward for r in group_records) / count,
                "mean_delta": sum(r.delta for r in group_records) / count,
            }
        )
    return rows


def write_episode_csv(records: list[EpisodeGroupRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "env",
                "run",
                "episode",
                "worst_reward",
                "median_reward",
                "best_reward",
                "initial_reward",
                "revised_reward",
                "delta",
                "winner",
                "reward_group",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def write_summary_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "reward_group",
                "episodes",
                "pct",
                "mean_worst_reward",
                "mean_median_reward",
                "mean_best_reward",
                "mean_initial_reward",
                "mean_revised_reward",
                "mean_delta",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(env: str, rows: list[dict]) -> str:
    lines = [
        f"# {env.title()} ThreeTraj Reward Group Analysis",
        "",
        "| Group | Episodes | % | Mean Worst | Mean Median | Mean Best | Mean Initial | Mean Revised | Mean Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['reward_group']} | {row['episodes']} | {row['pct']:.1f} | "
            f"{row['mean_worst_reward']:.3f} | {row['mean_median_reward']:.3f} | {row['mean_best_reward']:.3f} | "
            f"{row['mean_initial_reward']:.3f} | {row['mean_revised_reward']:.3f} | {row['mean_delta']:.3f} |"
        )
    lines.append("")
    lines.append("Group definitions:")
    lines.append("- `identical`: worst = median = best")
    lines.append("- `worst_eq_median`: worst = median < best")
    lines.append("- `all_diverse`: worst < median < best")
    lines.append("- `median_eq_best`: worst < median = best")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze ThreeTraj iterations by worst/median/best reward pattern and mean delta."
    )
    parser.add_argument("logdirs", nargs="+", help="ThreeTraj experiment logdirs to analyze.")
    parser.add_argument(
        "--output-dir",
        default="paper_results",
        help="Directory where outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for logdir_str in args.logdirs:
        logdir = Path(logdir_str)
        env = detect_env_name(logdir).lower()
        records = load_records(logdir)
        if not records:
            print(f"Skipping {logdir}: no parseable ThreeTraj episodes found.")
            continue

        summary_rows = summarize(records)
        env_dir = output_dir / env
        env_dir.mkdir(parents=True, exist_ok=True)

        episodes_csv = env_dir / "episodes.csv"
        summary_csv = env_dir / "summary.csv"
        summary_md = env_dir / "summary.md"
        summary_json = env_dir / "summary.json"

        write_episode_csv(records, episodes_csv)
        write_summary_csv(summary_rows, summary_csv)
        summary_md.write_text(render_markdown(env, summary_rows), encoding="utf-8")
        summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

        print(f"Wrote {episodes_csv}")
        print(f"Wrote {summary_csv}")
        print(f"Wrote {summary_md}")
        print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
