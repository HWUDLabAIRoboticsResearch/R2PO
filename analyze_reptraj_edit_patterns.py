#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", re.IGNORECASE)


ENV_PRESETS = {
    "cartpole": {
        "distance_mode": "continuous",
        "epsilon": 0.05,
        "partial_threshold": 200.0,
        "high_threshold": 450.0,
        "near_zero_threshold": 20.0,
        "meaningful_delta": 0.1,
        "surgical_max_edit": 3,
        "redesign_min_edit": 8,
        "multi_state_min_edit": 4,
    },
    "pong": {
        "distance_mode": "continuous",
        "epsilon": 0.05,
        "partial_threshold": 1.0,
        "high_threshold": 2.8,
        "near_zero_threshold": 0.2,
        "meaningful_delta": 0.1,
        "surgical_max_edit": 3,
        "redesign_min_edit": 8,
        "multi_state_min_edit": 4,
    },
    "frozenlake": {
        "distance_mode": "discrete",
        "epsilon": 0.0,
        "partial_threshold": 0.3,
        "high_threshold": 0.8,
        "near_zero_threshold": 0.05,
        "meaningful_delta": 0.1,
        "surgical_max_edit": 3,
        "redesign_min_edit": 8,
        "multi_state_min_edit": 4,
    },
}


@dataclass
class ThresholdConfig:
    env: str
    distance_mode: str
    epsilon: float
    partial_threshold: float
    high_threshold: float
    near_zero_threshold: float
    meaningful_delta: float
    surgical_max_edit: int
    redesign_min_edit: int
    multi_state_min_edit: int


@dataclass
class EpisodeRecord:
    run: str
    episode: int
    initial_reward: float
    revised_reward: float
    delta: float
    initial_params: list[float]
    revised_params: list[float]
    edit_distance: int
    l2_distance: float
    abs_max_change: float
    changed_indices: list[int]
    winner: str | None
    direction: str
    category: str


def detect_env_name(logdir: Path) -> str:
    name = logdir.name.lower()
    for candidate in ENV_PRESETS:
        if candidate in name:
            return candidate
    if "_reflective" in name:
        return name.split("_reflective", 1)[0]
    return name


def build_thresholds(args: argparse.Namespace, env: str) -> ThresholdConfig:
    preset = ENV_PRESETS.get(env, ENV_PRESETS["cartpole"]).copy()
    if args.distance_mode != "auto":
        preset["distance_mode"] = args.distance_mode
    if args.epsilon is not None:
        preset["epsilon"] = args.epsilon
    if args.partial_threshold is not None:
        preset["partial_threshold"] = args.partial_threshold
    if args.high_threshold is not None:
        preset["high_threshold"] = args.high_threshold
    if args.near_zero_threshold is not None:
        preset["near_zero_threshold"] = args.near_zero_threshold
    if args.meaningful_delta is not None:
        preset["meaningful_delta"] = args.meaningful_delta
    if args.surgical_max_edit is not None:
        preset["surgical_max_edit"] = args.surgical_max_edit
    if args.redesign_min_edit is not None:
        preset["redesign_min_edit"] = args.redesign_min_edit
    if args.multi_state_min_edit is not None:
        preset["multi_state_min_edit"] = args.multi_state_min_edit
    return ThresholdConfig(env=env, **preset)


def _parse_compact_params_line(text: str) -> list[float] | None:
    value_matches = re.findall(r"params\[\d+\]\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", text, re.IGNORECASE)
    if value_matches:
        params = [float(token) for token in value_matches]
    else:
        params = [float(token) for token in FLOAT_RE.findall(text)]
    return params or None


def _parse_initial_params_from_reasoning(text: str) -> list[float] | None:
    seen_marker = False
    collected: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "LLM:":
            seen_marker = True
            continue
        if not seen_marker:
            continue
        if stripped.startswith("params["):
            collected.append(stripped)
            continue
        if collected:
            break
    return _parse_compact_params_line(" ".join(collected)) if collected else None


def _parse_revised_params_from_reasoning(text: str) -> list[float] | None:
    seen_marker = False
    collected: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "Critic-LLM response:":
            seen_marker = True
            continue
        if not seen_marker:
            continue
        if stripped.startswith("params["):
            collected.append(stripped)
            continue
        if collected:
            break
    return _parse_compact_params_line(" ".join(collected)) if collected else None


def parse_params_for_episode(episode_dir: Path, which: str) -> list[float]:
    rollout_name = "initial_proposal_rollout.txt" if which == "initial" else "revised_proposal_rollout.txt"
    rollout_path = episode_dir / rollout_name
    if rollout_path.exists():
        with rollout_path.open("r", encoding="utf-8") as fh:
            first_line = fh.readline().strip()
        params = _parse_compact_params_line(first_line)
        if params:
            return params

    reasoning_path = episode_dir / "parameters_reasoning.txt"
    if reasoning_path.exists():
        text = reasoning_path.read_text(encoding="utf-8")
        if which == "initial":
            params = _parse_initial_params_from_reasoning(text)
        else:
            params = _parse_revised_params_from_reasoning(text)
        if params:
            return params

    raise ValueError(f"Could not parse {which} params from {episode_dir}")


def parse_reflection_outcome(path: Path) -> tuple[float, float, str | None]:
    initial_reward = None
    revised_reward = None
    winner = None
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("initial_reward="):
            initial_reward = float(line.split("=", 1)[1].strip())
        elif line.startswith("revised_reward="):
            revised_reward = float(line.split("=", 1)[1].strip())
        elif line.startswith("winner="):
            winner = line.split("=", 1)[1].strip()
    if initial_reward is None or revised_reward is None:
        raise ValueError(f"Could not parse rewards from {path}")
    return initial_reward, revised_reward, winner


def compute_edit_metrics(
    initial_params: list[float],
    revised_params: list[float],
    distance_mode: str,
    epsilon: float,
) -> tuple[int, float, float, list[int]]:
    if len(initial_params) != len(revised_params):
        raise ValueError("Initial and revised params have different lengths")

    diffs = [revised - initial for initial, revised in zip(initial_params, revised_params)]
    if distance_mode == "discrete":
        changed_indices = [idx for idx, diff in enumerate(diffs) if abs(diff) > 1e-12]
    else:
        changed_indices = [idx for idx, diff in enumerate(diffs) if abs(diff) > epsilon]

    l2_distance = math.sqrt(sum(diff * diff for diff in diffs))
    abs_max_change = max((abs(diff) for diff in diffs), default=0.0)
    return len(changed_indices), l2_distance, abs_max_change, changed_indices


def classify_regression(record: EpisodeRecord, cfg: ThresholdConfig) -> str:
    if record.edit_distance == 0:
        return "pure_stochasticity"
    if record.initial_reward > cfg.high_threshold and record.delta < -cfg.meaningful_delta:
        return "compulsive_revision_of_near_optimal"
    if (
        record.initial_reward > cfg.partial_threshold
        and record.edit_distance <= cfg.surgical_max_edit
        and record.delta < -cfg.meaningful_delta
    ):
        return "surgical_fix_backfires"
    if (
        abs(record.initial_reward) <= cfg.near_zero_threshold
        and record.edit_distance >= cfg.redesign_min_edit
    ):
        return "full_redesign_on_uninformative_trajectories"
    return "other"


def classify_improvement(record: EpisodeRecord, cfg: ThresholdConfig) -> str:
    if (
        record.initial_reward > cfg.high_threshold
        and record.edit_distance <= cfg.surgical_max_edit
        and record.delta > cfg.meaningful_delta
    ):
        return "fine_tuning_near_optimal"
    if (
        record.initial_reward > cfg.partial_threshold
        and record.edit_distance <= cfg.surgical_max_edit
        and record.delta > cfg.meaningful_delta
    ):
        return "pinpoint_bottleneck_fix"
    if (
        abs(record.initial_reward) <= cfg.near_zero_threshold
        and record.edit_distance >= cfg.redesign_min_edit
        and record.delta > cfg.meaningful_delta
    ):
        return "history_guided_rescue"
    if (
        record.initial_reward > cfg.near_zero_threshold
        and record.edit_distance >= cfg.multi_state_min_edit
        and record.delta > cfg.meaningful_delta
    ):
        return "targeted_multi_state_fix"
    return "other"


def classify(record: EpisodeRecord, cfg: ThresholdConfig) -> tuple[str, str]:
    if record.delta < 0:
        return "regression", classify_regression(record, cfg)
    if record.delta > 0:
        return "improvement", classify_improvement(record, cfg)
    return "no_change", "no_change"


def iter_episode_dirs(logdir: Path) -> Iterable[tuple[str, int, Path]]:
    for run_dir in sorted(logdir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        for episode_dir in sorted(run_dir.glob("episode_*"), key=lambda p: int(p.name.split("_")[1])):
            if episode_dir.is_dir():
                yield run_dir.name, int(episode_dir.name.split("_")[1]), episode_dir


def load_records(logdir: Path, cfg: ThresholdConfig) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for run_name, episode_idx, episode_dir in iter_episode_dirs(logdir):
        initial_rollout = episode_dir / "initial_proposal_rollout.txt"
        revised_rollout = episode_dir / "revised_proposal_rollout.txt"
        reflection = episode_dir / "reflection_outcome.txt"
        if not (initial_rollout.exists() and revised_rollout.exists() and reflection.exists()):
            continue

        try:
            initial_params = parse_params_for_episode(episode_dir, "initial")
            revised_params = parse_params_for_episode(episode_dir, "revised")
            initial_reward, revised_reward, winner = parse_reflection_outcome(reflection)
            edit_distance, l2_distance, abs_max_change, changed_indices = compute_edit_metrics(
                initial_params,
                revised_params,
                cfg.distance_mode,
                cfg.epsilon,
            )
        except ValueError as exc:
            print(f"[warn] skipping {episode_dir}: {exc}", file=sys.stderr)
            continue

        record = EpisodeRecord(
            run=run_name,
            episode=episode_idx,
            initial_reward=initial_reward,
            revised_reward=revised_reward,
            delta=revised_reward - initial_reward,
            initial_params=initial_params,
            revised_params=revised_params,
            edit_distance=edit_distance,
            l2_distance=l2_distance,
            abs_max_change=abs_max_change,
            changed_indices=changed_indices,
            winner=winner,
            direction="",
            category="",
        )
        record.direction, record.category = classify(record, cfg)
        records.append(record)
    return records


def write_csv(records: list[EpisodeRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "run",
                "episode",
                "initial_reward",
                "revised_reward",
                "delta",
                "edit_distance",
                "l2_distance",
                "abs_max_change",
                "changed_indices",
                "winner",
                "direction",
                "category",
                "initial_params",
                "revised_params",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.run,
                    record.episode,
                    f"{record.initial_reward:.6f}",
                    f"{record.revised_reward:.6f}",
                    f"{record.delta:.6f}",
                    record.edit_distance,
                    f"{record.l2_distance:.6f}",
                    f"{record.abs_max_change:.6f}",
                    ",".join(str(idx) for idx in record.changed_indices),
                    record.winner or "",
                    record.direction,
                    record.category,
                    json.dumps(record.initial_params),
                    json.dumps(record.revised_params),
                ]
            )


def summarize(records: list[EpisodeRecord], cfg: ThresholdConfig) -> dict:
    direction_counts = Counter(record.direction for record in records)
    category_counts = Counter((record.direction, record.category) for record in records)
    winner_counts = Counter(record.winner or "unknown" for record in records)

    regressions = [record for record in records if record.direction == "regression"]
    improvements = [record for record in records if record.direction == "improvement"]
    equality_records = [record for record in records if abs(record.delta) <= 1e-12]
    equality_no_edit_records = [record for record in equality_records if record.edit_distance == 0]
    equality_with_edit_records = [record for record in equality_records if record.edit_distance > 0]

    def bucket(rows: list[EpisodeRecord]) -> dict[str, int]:
        counter = Counter(record.category for record in rows)
        return dict(sorted(counter.items()))

    def stats(rows: list[EpisodeRecord]) -> dict[str, float]:
        if not rows:
            return {"count": 0, "mean_delta": 0.0, "mean_edit_distance": 0.0, "mean_l2_distance": 0.0}
        return {
            "count": len(rows),
            "mean_delta": sum(record.delta for record in rows) / len(rows),
            "mean_edit_distance": sum(record.edit_distance for record in rows) / len(rows),
            "mean_l2_distance": sum(record.l2_distance for record in rows) / len(rows),
        }

    return {
        "logdir": str(records[0].run if False else ""),
        "thresholds": asdict(cfg),
        "num_episodes_analyzed": len(records),
        "direction_counts": dict(sorted(direction_counts.items())),
        "winner_counts": dict(sorted(winner_counts.items())),
        "equality_counts": {
            "reward_equal": len(equality_records),
            "reward_equal_and_no_edit": len(equality_no_edit_records),
            "reward_equal_with_edit": len(equality_with_edit_records),
        },
        "regression_category_counts": bucket(regressions),
        "improvement_category_counts": bucket(improvements),
        "regression_stats": stats(regressions),
        "improvement_stats": stats(improvements),
        "category_counts_flat": {
            f"{direction}:{category}": count
            for (direction, category), count in sorted(category_counts.items())
        },
    }


def write_markdown(summary: dict, records: list[EpisodeRecord], path: Path) -> None:
    regressions = [record for record in records if record.direction == "regression"]
    improvements = [record for record in records if record.direction == "improvement"]

    def top_examples(rows: list[EpisodeRecord], reverse: bool, n: int = 8) -> list[EpisodeRecord]:
        return sorted(rows, key=lambda row: row.delta, reverse=reverse)[:n]

    lines = [
        "# Edit Pattern Analysis",
        "",
        f"- Episodes analyzed: {summary['num_episodes_analyzed']}",
        f"- Thresholds: `{json.dumps(summary['thresholds'], sort_keys=True)}`",
        "",
        "## Winner Counts",
        "",
    ]
    for name, count in summary["winner_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Equality Counts", ""])
    for name, count in summary["equality_counts"].items():
        lines.append(f"- `{name}`: {count}")

    lines.extend([
        "",
        "## Regression Counts",
        "",
    ])
    for name, count in summary["regression_category_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Improvement Counts", ""])
    for name, count in summary["improvement_category_counts"].items():
        lines.append(f"- `{name}`: {count}")

    lines.extend(["", "## Largest Regressions", "", "| Run | Episode | Initial | Revised | Delta | Edit Dist | L2 | Category |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in top_examples(regressions, reverse=False):
        lines.append(
            f"| {row.run} | {row.episode} | {row.initial_reward:.3f} | {row.revised_reward:.3f} | "
            f"{row.delta:.3f} | {row.edit_distance} | {row.l2_distance:.3f} | {row.category} |"
        )

    lines.extend(["", "## Largest Improvements", "", "| Run | Episode | Initial | Revised | Delta | Edit Dist | L2 | Category |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in top_examples(improvements, reverse=True):
        lines.append(
            f"| {row.run} | {row.episode} | {row.initial_reward:.3f} | {row.revised_reward:.3f} | "
            f"{row.delta:.3f} | {row.edit_distance} | {row.l2_distance:.3f} | {row.category} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Search-LLM -> Critic-LLM parameter edits and bucket failures/improvements."
    )
    parser.add_argument("logdir", help="Experiment logdir, e.g. logs/cartpole_reflective_prompted_policy_optimization")
    parser.add_argument("--output-dir", help="Directory for CSV/JSON/MD outputs. Defaults to <logdir>/edit_pattern_analysis")
    parser.add_argument("--env", help="Override environment name used for preset thresholds, e.g. cartpole or pong")
    parser.add_argument("--distance-mode", choices=["auto", "continuous", "discrete"], default="auto")
    parser.add_argument("--epsilon", type=float, help="Continuous param change threshold for edit distance.")
    parser.add_argument("--partial-threshold", type=float, help="Threshold for a partially working policy.")
    parser.add_argument("--high-threshold", type=float, help="Threshold for a near-optimal policy.")
    parser.add_argument("--near-zero-threshold", type=float, help="Threshold for an uninformative near-zero policy.")
    parser.add_argument("--meaningful-delta", type=float, help="Minimum absolute delta to treat as meaningful.")
    parser.add_argument("--surgical-max-edit", type=int, help="Maximum changed params for a surgical fix.")
    parser.add_argument("--redesign-min-edit", type=int, help="Minimum changed params for a full redesign.")
    parser.add_argument("--multi-state-min-edit", type=int, help="Minimum changed params for a targeted multi-state fix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logdir = Path(args.logdir)
    env = (args.env or detect_env_name(logdir)).lower()
    cfg = build_thresholds(args, env)
    output_dir = Path(args.output_dir) if args.output_dir else logdir / "edit_pattern_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(logdir, cfg)
    if not records:
        raise SystemExit(f"No reflective/edit episode files found in {logdir}")

    summary = summarize(records, cfg)
    summary["logdir"] = str(logdir)

    csv_path = output_dir / "episodes.csv"
    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"

    write_csv(records, csv_path)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, records, md_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(summary["direction_counts"], indent=2))


if __name__ == "__main__":
    main()
