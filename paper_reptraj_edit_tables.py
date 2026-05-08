#!/usr/bin/env python3
"""
Generate per-environment tables for reflective edit-pattern analysis.

Each table compares reflective variants for one environment and highlights
the strongest method per metric. The table is saved as both Markdown and PNG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

os_env = __import__("os").environ
os_env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_reptraj_edit_patterns import build_thresholds, load_records, summarize


METHOD_SPECS = [
    (
        "reptraj",
        "RepTraj",
        "{env}_reflective",
    ),
    (
        "r2po",
        "R2PO",
        "{env}_reflective_prompted_policy_optimization",
    ),
    (
        "three_traj",
        "ThreeTraj",
        "{env}_three_traj",
    ),
    (
        "always_critic",
        "AlwaysCritic",
        "{env}_always_critic",
    ),
]

METHOD_SPEC_MAP = {
    key: (label, pattern) for key, label, pattern in METHOD_SPECS
}

DEFAULT_ENVS = ["frozenlake", "cartpole"]

METRIC_SPECS = [
    ("revision_won", "Revision Applied/Selected", None),
    ("original_won", "Original Kept", None),
    ("reward_equal", "Reward Equal", None),
    ("reward_equal_and_no_edit", "Equal + No Edit", None),
    ("reward_equal_with_edit", "Equal + Edited", "min"),
    ("improvement", "Improvements", "max"),
    ("regression", "Regressions", "min"),
    ("pure_stochasticity", "Pure Stochasticity", "min"),
    ("surgical_fix_backfires", "Surgical Backfire", "min"),
    ("compulsive_revision_of_near_optimal", "Near-Optimal Break", "min"),
    ("full_redesign_on_uninformative_trajectories", "Full Redesign", None),
    ("pinpoint_bottleneck_fix", "Pinpoint Fix", "max"),
    ("history_guided_rescue", "History-Guided Rescue", "max"),
    ("targeted_multi_state_fix", "Targeted Multi-State Fix", "max"),
    ("fine_tuning_near_optimal", "Fine-Tune Near-Optimal", "max"),
    ("other_regression", "Other Regressions", "min"),
    ("other_improvement", "Other Improvements", None),
]

METRIC_PRESETS = {
    "paper": [
        "revision_won",
        "original_won",
        "improvement",
        "regression",
        "compulsive_revision_of_near_optimal",
        "surgical_fix_backfires",
        "fine_tuning_near_optimal",
        "pinpoint_bottleneck_fix",
    ],
    "full": [metric_key for metric_key, _, _ in METRIC_SPECS],
}

METRIC_SPEC_MAP = {metric_key: (label, direction) for metric_key, label, direction in METRIC_SPECS}


def _default_threshold_args() -> SimpleNamespace:
    return SimpleNamespace(
        distance_mode="auto",
        epsilon=None,
        partial_threshold=None,
        high_threshold=None,
        near_zero_threshold=None,
        meaningful_delta=None,
        surgical_max_edit=None,
        redesign_min_edit=None,
        multi_state_min_edit=None,
    )


def _format_cell(value: int | None, total: int | None) -> str:
    if value is None or total in (None, 0):
        return ""
    return f"{value} ({100.0 * value / total:.1f}%)"


def _extract_metric(summary: dict, metric_key: str) -> tuple[int | None, int | None]:
    total = summary["num_episodes_analyzed"]
    if metric_key == "revision_won":
        winner_counts = summary.get("winner_counts", {})
        return int(winner_counts.get("revision_won", 0) + winner_counts.get("revision_forced", 0)), total
    if metric_key == "original_won":
        return int(summary.get("winner_counts", {}).get("original_won", 0)), total
    if metric_key in summary.get("equality_counts", {}):
        return int(summary["equality_counts"].get(metric_key, 0)), total
    if metric_key in summary.get("direction_counts", {}) or metric_key in {"improvement", "regression", "no_change"}:
        return int(summary.get("direction_counts", {}).get(metric_key, 0)), total
    if metric_key in summary.get("regression_category_counts", {}) or metric_key in {
        "pure_stochasticity",
        "surgical_fix_backfires",
        "compulsive_revision_of_near_optimal",
        "full_redesign_on_uninformative_trajectories",
    }:
        return int(summary.get("regression_category_counts", {}).get(metric_key, 0)), total
    if metric_key in summary.get("improvement_category_counts", {}) or metric_key in {
        "pinpoint_bottleneck_fix",
        "history_guided_rescue",
        "targeted_multi_state_fix",
        "fine_tuning_near_optimal",
    }:
        return int(summary.get("improvement_category_counts", {}).get(metric_key, 0)), total
    if metric_key == "other_regression":
        return int(summary["regression_category_counts"].get("other", 0)), total
    if metric_key == "other_improvement":
        return int(summary["improvement_category_counts"].get("other", 0)), total
    return None, total


def _metric_values_for_highlight(method_summaries: dict[str, dict], metric_key: str) -> dict[str, int]:
    values = {}
    for method, summary in method_summaries.items():
        value, _ = _extract_metric(summary, metric_key)
        if value is not None:
            values[method] = value
    return values


def _best_methods(metric_key: str, direction: str | None, method_summaries: dict[str, dict]) -> set[str]:
    if direction is None:
        return set()
    values = _metric_values_for_highlight(method_summaries, metric_key)
    if not values:
        return set()
    target = max(values.values()) if direction == "max" else min(values.values())
    return {method for method, value in values.items() if value == target}


def _load_method_summary(logs_root: Path, env: str, method_label: str, pattern: str) -> dict | None:
    logdir = logs_root / pattern.format(env=env)
    if not logdir.exists():
        return None

    cfg = build_thresholds(_default_threshold_args(), env)
    records = load_records(logdir, cfg)
    if not records:
        return None

    summary = summarize(records, cfg)
    summary["logdir"] = str(logdir)
    summary["method"] = method_label
    return summary


def _collect_env_summaries(logs_root: Path, env: str, method_keys: list[str]) -> dict[str, dict]:
    summaries = {}
    for method_key in method_keys:
        method_label, pattern = METHOD_SPEC_MAP[method_key]
        summary = _load_method_summary(logs_root, env, method_label, pattern)
        if summary is not None:
            summaries[method_label] = summary
    return summaries


def render_markdown(env: str, method_summaries: dict[str, dict], metric_keys: list[str]) -> str:
    methods = list(method_summaries)
    lines = [
        f"## {env.title()} Edit Patterns",
        "",
        "| Metric | " + " | ".join(methods) + " |",
        "|" + "|".join(["---"] * (len(methods) + 1)) + "|",
    ]

    for metric_key in metric_keys:
        label, direction = METRIC_SPEC_MAP[metric_key]
        best = _best_methods(metric_key, direction, method_summaries)
        row = [label]
        for method in methods:
            value, total = _extract_metric(method_summaries[method], metric_key)
            cell = _format_cell(value, total)
            if method in best and cell:
                cell = f"**{cell}**"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("Threshold presets:")
    lines.append(f"`{json.dumps(next(iter(method_summaries.values()))['thresholds'], sort_keys=True)}`")
    lines.append("")
    return "\n".join(lines)


def write_png(env: str, method_summaries: dict[str, dict], metric_keys: list[str], output_path: Path) -> None:
    methods = list(method_summaries)
    headers = ["Metric"] + methods
    table_rows = []
    highlight = []

    for metric_key in metric_keys:
        label, direction = METRIC_SPEC_MAP[metric_key]
        best = _best_methods(metric_key, direction, method_summaries)
        row = [label]
        row_highlight = [False]
        for method in methods:
            value, total = _extract_metric(method_summaries[method], metric_key)
            row.append(_format_cell(value, total))
            row_highlight.append(method in best and bool(row[-1]))
        table_rows.append(row)
        highlight.append(row_highlight)

    fig_width = max(12, 2.4 * len(headers))
    fig_height = max(7, 0.5 * len(table_rows) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        colWidths=[1 / len(headers)] * len(headers),
        bbox=[0.01, 0.01, 0.98, 0.92],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#BFC7D5")
        cell.set_linewidth(0.8)
        if row_idx == 0:
            cell.set_facecolor("#DCE6F2")
            cell.set_text_props(weight="bold", color="#1F2937")
        else:
            cell.set_facecolor("#F8FAFC" if row_idx % 2 else "#EEF3F8")
            if highlight[row_idx - 1][col_idx]:
                cell.set_facecolor("#D9F2E3")
                cell.set_text_props(weight="bold", color="#14532D")

    ax.set_title(f"{env.title()} Edit Patterns", fontsize=15, fontweight="bold", pad=16)
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--output-dir", default="paper_results/edit_pattern_tables")
    parser.add_argument("--env", action="append", dest="envs", help="Environment(s) to render, e.g. frozenlake")
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        choices=sorted(METHOD_SPEC_MAP),
        help="Method(s) to include. Repeat the flag to select multiple variants.",
    )
    parser.add_argument(
        "--metric-preset",
        choices=sorted(METRIC_PRESETS),
        default="paper",
        help="Which metric subset to render.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs_root = Path(args.logs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    envs = args.envs or DEFAULT_ENVS
    metric_keys = METRIC_PRESETS[args.metric_preset]
    method_keys = args.methods or [method_key for method_key, _, _ in METHOD_SPECS]
    for env in envs:
        method_summaries = _collect_env_summaries(logs_root, env, method_keys)
        if not method_summaries:
            print(f"Skipping {env}: no matching edit-pattern logs found.")
            continue

        md_path = output_dir / f"{env}_edit_patterns.md"
        png_path = output_dir / f"{env}_edit_patterns.png"
        md_path.write_text(render_markdown(env, method_summaries, metric_keys), encoding="utf-8")
        write_png(env, method_summaries, metric_keys, png_path)
        print(f"Wrote {md_path}")
        print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
