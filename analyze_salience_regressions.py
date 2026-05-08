#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


DEFAULT_ENVS = ["cartpole", "pong"]
WORST_REFERENCE_RE = re.compile(r"\bworst\s+(?:rollout|trajectory)\b", re.IGNORECASE)


def _mentions_worst(text: str, mode: str) -> bool:
    if mode == "word":
        return "worst" in text.lower()
    if mode == "phrase":
        return bool(WORST_REFERENCE_RE.search(text))
    raise ValueError(f"Unknown mention mode: {mode}")


def _load_episode_rows(
    results_root: Path,
    logs_root: Path,
    envs: list[str],
    mention_mode: str,
) -> list[dict]:
    rows: list[dict] = []
    for env in envs:
        episodes_path = results_root / env / "episodes.csv"
        if not episodes_path.exists():
            raise FileNotFoundError(f"Missing {episodes_path}")

        with episodes_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                reasoning_path = (
                    logs_root
                    / f"{env}_three_traj"
                    / row["run"]
                    / f"episode_{row['episode']}"
                    / "parameters_reasoning.txt"
                )
                if not reasoning_path.exists():
                    raise FileNotFoundError(f"Missing {reasoning_path}")

                text = reasoning_path.read_text(encoding="utf-8")
                critic_llm_text = text.split("Critic-LLM response:", 1)[-1]
                row["env"] = env
                row["reasoning_references_worst"] = _mentions_worst(critic_llm_text, mention_mode)
                row["worst_reward"] = float(row["worst_reward"])
                row["median_reward"] = float(row["median_reward"])
                row["best_reward"] = float(row["best_reward"])
                row["initial_reward"] = float(row["initial_reward"])
                row["revised_reward"] = float(row["revised_reward"])
                row["delta"] = float(row["delta"])
                rows.append(row)
    return rows


def _pct(num: int, denom: int) -> float:
    return 100.0 * num / denom if denom else 0.0


def _summarize(rows: list[dict], mention_mode: str) -> dict:
    regressions = [row for row in rows if row["delta"] < 0]

    old_reward_proxy = [
        row
        for row in regressions
        if row["worst_reward"] <= row["median_reward"]
        and row["worst_reward"] < row["best_reward"]
    ]
    strict_reward_proxy = [
        row
        for row in regressions
        if row["worst_reward"] < row["median_reward"]
    ]
    old_with_reference = [
        row for row in old_reward_proxy if row["reasoning_references_worst"]
    ]
    strict_with_reference = [
        row for row in strict_reward_proxy if row["reasoning_references_worst"]
    ]
    excluded_worst_eq_median = [
        row
        for row in old_reward_proxy
        if row["worst_reward"] == row["median_reward"] < row["best_reward"]
    ]

    per_env = {}
    for env in sorted({row["env"] for row in rows}):
        env_rows = [row for row in rows if row["env"] == env]
        per_env[env] = _summarize_env(env_rows)

    return {
        "episodes": len(rows),
        "mention_mode": mention_mode,
        "regressions": len(regressions),
        "old_reward_proxy": {
            "count": len(old_reward_proxy),
            "pct_of_regressions": _pct(len(old_reward_proxy), len(regressions)),
            "definition": "delta < 0, worst < best, and worst <= median",
        },
        "strict_reward_proxy": {
            "count": len(strict_reward_proxy),
            "pct_of_regressions": _pct(len(strict_reward_proxy), len(regressions)),
            "definition": "delta < 0, worst < median",
        },
        "old_with_worst_reference": {
            "count": len(old_with_reference),
            "pct_of_regressions": _pct(len(old_with_reference), len(regressions)),
            "definition": "old reward proxy plus Critic-LLM explicitly references the worst rollout/trajectory",
        },
        "strict_with_worst_reference": {
            "count": len(strict_with_reference),
            "pct_of_regressions": _pct(len(strict_with_reference), len(regressions)),
            "definition": "strict reward proxy plus Critic-LLM explicitly references the worst rollout/trajectory",
        },
        "excluded_by_strict_change": {
            "count": len(excluded_worst_eq_median),
            "pct_of_old_reward_proxy": _pct(
                len(excluded_worst_eq_median), len(old_reward_proxy)
            ),
            "definition": "old proxy cases where worst = median < best",
        },
        "per_env": per_env,
    }


def _summarize_env(rows: list[dict]) -> dict:
    regressions = [row for row in rows if row["delta"] < 0]
    old = [
        row
        for row in regressions
        if row["worst_reward"] <= row["median_reward"]
        and row["worst_reward"] < row["best_reward"]
    ]
    strict = [row for row in regressions if row["worst_reward"] < row["median_reward"]]
    old_ref = [row for row in old if row["reasoning_references_worst"]]
    strict_ref = [row for row in strict if row["reasoning_references_worst"]]
    return {
        "regressions": len(regressions),
        "old_reward_proxy_count": len(old),
        "old_reward_proxy_pct": _pct(len(old), len(regressions)),
        "strict_reward_proxy_count": len(strict),
        "strict_reward_proxy_pct": _pct(len(strict), len(regressions)),
        "old_with_worst_reference_count": len(old_ref),
        "old_with_worst_reference_pct": _pct(len(old_ref), len(regressions)),
        "strict_with_worst_reference_count": len(strict_ref),
        "strict_with_worst_reference_pct": _pct(len(strict_ref), len(regressions)),
    }


def _write_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# Salience-Problem Regression Proxy Rerun",
        "",
        f"- Episodes analyzed: {summary['episodes']}",
        f"- Worst-reference detector: `{summary['mention_mode']}`",
        f"- Denominator for headline proxy: {summary['regressions']} regressions (`delta < 0`)",
        "",
        "| Criterion | Count | % of Regressions |",
        "| --- | ---: | ---: |",
    ]
    for key in [
        "old_reward_proxy",
        "strict_reward_proxy",
        "old_with_worst_reference",
        "strict_with_worst_reference",
    ]:
        item = summary[key]
        lines.append(
            f"| {item['definition']} | {item['count']} | {item['pct_of_regressions']:.1f}% |"
        )
    lines.extend(
        [
            "",
            f"The strict `worst < median` change excludes {summary['excluded_by_strict_change']['count']} old-proxy cases "
            f"({summary['excluded_by_strict_change']['pct_of_old_reward_proxy']:.1f}% of old-proxy cases) where `worst = median < best`.",
            "",
            "## Per Environment",
            "",
            "| Env | Regressions | Old | Strict | Old + Worst Ref | Strict + Worst Ref |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for env, row in summary["per_env"].items():
        lines.append(
            f"| {env} | {row['regressions']} | "
            f"{row['old_reward_proxy_count']} ({row['old_reward_proxy_pct']:.1f}%) | "
            f"{row['strict_reward_proxy_count']} ({row['strict_reward_proxy_pct']:.1f}%) | "
            f"{row['old_with_worst_reference_count']} ({row['old_with_worst_reference_pct']:.1f}%) | "
            f"{row['strict_with_worst_reference_count']} ({row['strict_with_worst_reference_pct']:.1f}%) |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun salience-problem regression counts with old and strict worst-vs-median criteria."
    )
    parser.add_argument("--results-root", default="paper_results")
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--env", action="append", choices=DEFAULT_ENVS, dest="envs")
    parser.add_argument(
        "--mention-mode",
        choices=["word", "phrase"],
        default="word",
        help="`word` reproduces the original Pong table by matching any `worst` mention; `phrase` only matches `worst rollout`/`worst trajectory`.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_results/edit_pattern_tables",
        help="Directory where salience rerun summary files are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    envs = args.envs or DEFAULT_ENVS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_episode_rows(
        Path(args.results_root),
        Path(args.logs_root),
        envs,
        args.mention_mode,
    )
    summary = _summarize(rows, args.mention_mode)

    json_path = output_dir / "salience_regression_rerun.json"
    md_path = output_dir / "salience_regression_rerun.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, md_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        "Old proxy: "
        f"{summary['old_reward_proxy']['count']}/{summary['regressions']} "
        f"({summary['old_reward_proxy']['pct_of_regressions']:.1f}%)"
    )
    print(
        "Strict proxy: "
        f"{summary['strict_reward_proxy']['count']}/{summary['regressions']} "
        f"({summary['strict_reward_proxy']['pct_of_regressions']:.1f}%)"
    )
    print(
        "Strict + worst-reference proxy: "
        f"{summary['strict_with_worst_reference']['count']}/{summary['regressions']} "
        f"({summary['strict_with_worst_reference']['pct_of_regressions']:.1f}%)"
    )


if __name__ == "__main__":
    main()
