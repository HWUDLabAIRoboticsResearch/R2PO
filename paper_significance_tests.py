"""
Run per-environment significance tests for R2PO vs. core baselines.

Two comparison modes are supported:
- best: compare R2PO against the strongest observed core baseline
  on each environment.
- all: compare R2PO separately against every available core baseline
  (ProPS, ProPS+, and SB3 when present).

All tests use a two-sided Welch's t-test on the per-run values for the selected
metric. Holm correction scope depends on the comparison mode:
- best: across the 10 environment-level strongest-baseline comparisons
- all: separately within each baseline family across the 10 environments

Examples:
  uv run python paper_significance_tests.py
  uv run python paper_significance_tests.py --comparison-mode all
  uv run python paper_significance_tests.py --metric best_reward
  uv run python paper_significance_tests.py --format latex --output paper_results/core_significance_mean_reward.tex
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import math

from scipy import stats

from results_table import DEFAULT_ENVS


DISPLAY_NAMES = {
    "ProPS": "ProPS",
    "ProPS+": "ProPS+",
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

METRIC_CONFIG = {
    "mean_reward": {
        "row_key": "mean_reward",
        "summary_key": "mean_reward",
        "title": "Mean Reward",
    },
    "best_reward": {
        "row_key": "mean_best_reward",
        "summary_key": "best_reward",
        "title": "Mean Best Reward",
    },
    "final_reward": {
        "row_key": "mean_final_reward",
        "summary_key": "final_reward",
        "title": "Mean Final Reward",
    },
}


@dataclass
class TestRow:
    environment: str
    baseline_method: str
    baseline_mean: float
    rp2o_mean: float
    mean_gap: float
    t_statistic: float
    raw_p: float
    holm_p: float | None = None


def _load_per_run_metric(summary_path: Path, summary_key: str) -> list[float]:
    with summary_path.open() as f:
        data = json.load(f)
    per_run = data.get("per_run_summaries") or []
    values = []
    for run in per_run:
        value = run.get(summary_key)
        if value is None:
            continue
        values.append(float(value))
    return values


def _normalize_metrics(data: dict) -> dict[str, float | None]:
    return {
        "mean_reward": data.get("mean_reward_mean"),
        "best_reward": data.get("best_reward_mean", data.get("mean_best_reward")),
        "final_reward": data.get("final_reward_mean", data.get("mean_final_reward")),
    }


def _load_summary(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _core_method_sources(logs_root: Path, env_prefix: str) -> dict[str, Path]:
    sources = {
        "ProPS": logs_root / f"{env_prefix}_props" / "eval_report" / "summary.json",
        "ProPS+": logs_root / f"{env_prefix}_propsp" / "eval_report" / "summary.json",
        "Ref_Cons": (
            logs_root
            / f"{env_prefix}_reflective_prompted_policy_optimization"
            / "eval_report"
            / "summary.json"
        ),
    }
    sb3_dir = logs_root / f"{env_prefix}_sb3"
    if sb3_dir.is_dir():
        for algo_dir in sorted(x for x in sb3_dir.iterdir() if x.is_dir()):
            sources[f"SB3-{algo_dir.name.upper()}"] = algo_dir / "summary.json"
    return sources


def _holm_correct(p_values: Iterable[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    m = len(indexed)
    adjusted_sorted: list[float] = [0.0] * m
    running_max = 0.0
    for rank, (_, p_value) in enumerate(indexed, start=1):
        adjusted = min(1.0, (m - rank + 1) * p_value)
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank - 1] = running_max

    corrected = [0.0] * m
    for (original_idx, _), adjusted in zip(indexed, adjusted_sorted):
        corrected[original_idx] = adjusted
    return corrected


def _format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _format_p_value(value: float, digits: int = 4) -> str:
    if value == 0:
        return "0"
    threshold = 10 ** (-digits)
    if 0 < abs(value) < threshold:
        return f"{value:.1e}"
    return f"{value:.{digits}f}"


def _format_method_name(method: str) -> str:
    if method.startswith("SB3-"):
        algo = method.split("-", 1)[1]
        return f"SB3 ({algo})"
    return DISPLAY_NAMES.get(method, method)


def _format_environment_name(environment: str) -> str:
    return ENV_DISPLAY_NAMES.get(environment, environment)


def _baseline_family(method: str) -> str:
    if method.startswith("SB3-"):
        return "SB3"
    return method


def compute_tests(logs_root: Path, metric: str, comparison_mode: str = "best") -> list[TestRow]:
    if metric not in METRIC_CONFIG:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {sorted(METRIC_CONFIG)}")
    if comparison_mode not in {"best", "all"}:
        raise ValueError("comparison_mode must be one of {'best', 'all'}")

    summary_key = METRIC_CONFIG[metric]["summary_key"]

    results: list[TestRow] = []
    for environment, env_prefix in DEFAULT_ENVS:
        method_sources = _core_method_sources(logs_root, env_prefix)
        method_summaries = {method: _load_summary(path) for method, path in method_sources.items()}

        rp2o_summary = method_summaries["Ref_Cons"]
        rp2o_mean = float(_normalize_metrics(rp2o_summary)[summary_key])

        baseline_candidates = []
        for method, summary in method_summaries.items():
            if method == "Ref_Cons":
                continue
            metrics = _normalize_metrics(summary)
            value = metrics[summary_key]
            if value is None:
                continue
            baseline_candidates.append((float(value), method, summary))
        if not baseline_candidates:
            raise ValueError(f"No baseline candidates found for environment {environment}")

        rp2o_values = _load_per_run_metric(method_sources["Ref_Cons"], summary_key)
        if comparison_mode == "best":
            selected_candidates = [max(baseline_candidates, key=lambda item: item[0])]
        else:
            selected_candidates = sorted(baseline_candidates, key=lambda item: item[1])

        for baseline_mean, baseline_method, _ in selected_candidates:
            baseline_values = _load_per_run_metric(method_sources[baseline_method], summary_key)
            test_result = stats.ttest_ind(rp2o_values, baseline_values, equal_var=False)
            t_statistic = float(test_result.statistic)
            raw_p = float(test_result.pvalue)
            if math.isnan(t_statistic):
                t_statistic = 0.0
            if math.isnan(raw_p):
                raw_p = 1.0

            results.append(
                TestRow(
                    environment=environment,
                    baseline_method=baseline_method,
                    baseline_mean=baseline_mean,
                    rp2o_mean=rp2o_mean,
                    mean_gap=rp2o_mean - baseline_mean,
                    t_statistic=t_statistic,
                    raw_p=raw_p,
                )
            )

    if comparison_mode == "best":
        holm = _holm_correct([row.raw_p for row in results])
        for row, corrected in zip(results, holm):
            row.holm_p = corrected
    else:
        grouped_rows: dict[str, list[TestRow]] = {}
        for row in results:
            grouped_rows.setdefault(_baseline_family(row.baseline_method), []).append(row)
        for family_rows in grouped_rows.values():
            holm = _holm_correct([row.raw_p for row in family_rows])
            for row, corrected in zip(family_rows, holm):
                row.holm_p = corrected
    return results


def render_markdown(rows: list[TestRow], metric: str, comparison_mode: str) -> str:
    title = METRIC_CONFIG[metric]["title"]
    subtitle = "Best Core Baseline" if comparison_mode == "best" else "All Core Baselines"
    holm_note = (
        "Holm correction over the 10 strongest-baseline comparisons."
        if comparison_mode == "best"
        else "Holm correction applied separately within each baseline family across the 10 environments."
    )
    lines = [
        f"## R2PO vs {subtitle} ({title})",
        "",
        holm_note,
        "",
        "| Environment | Baseline | Baseline mean | R2PO mean | Gap | Welch t | Raw p | Holm p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_environment_name(row.environment),
                    _format_method_name(row.baseline_method),
                    _format_float(row.baseline_mean, 4),
                    _format_float(row.rp2o_mean, 4),
                    _format_float(row.mean_gap, 4),
                    _format_float(row.t_statistic, 4),
                    _format_p_value(row.raw_p, 6),
                    _format_p_value(row.holm_p or 0.0, 6),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def render_latex(rows: list[TestRow], metric: str, comparison_mode: str) -> str:
    title = METRIC_CONFIG[metric]["title"]
    lines = [
        f"% R2PO vs {'best core baseline' if comparison_mode == 'best' else 'all core baselines'} ({title})",
        (
            "% Holm correction over the 10 strongest-baseline comparisons."
            if comparison_mode == "best"
            else "% Holm correction applied separately within each baseline family across the 10 environments."
        ),
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r"Environment & Baseline & Base mean & R2PO mean & Gap & Welch $t$ & Raw $p$ & Holm $p$ \\",
        r"\midrule",
    ]
    for row in rows:
        base_mean = _format_float(row.baseline_mean, 2)
        rp2o_mean = _format_float(row.rp2o_mean, 2)
        if abs(row.baseline_mean - row.rp2o_mean) <= 1e-12:
            base_mean = rf"\cellcolor{{bestcolor}}\textbf{{{base_mean}}}"
            rp2o_mean = rf"\cellcolor{{bestcolor}}\textbf{{{rp2o_mean}}}"
        elif row.rp2o_mean > row.baseline_mean:
            rp2o_mean = rf"\cellcolor{{bestcolor}}\textbf{{{rp2o_mean}}}"
        else:
            base_mean = rf"\cellcolor{{bestcolor}}\textbf{{{base_mean}}}"
        lines.append(
            f"{_format_environment_name(row.environment)} & {_format_method_name(row.baseline_method)}"
            f" & {base_mean}"
            f" & {rp2o_mean}"
            f" & {_format_float(row.mean_gap, 2)}"
            f" & {_format_float(row.t_statistic, 3)}"
            f" & {_format_p_value(row.raw_p, 4)}"
            f" & {_format_p_value(row.holm_p or 0.0, 4)}"
            r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def render_json(rows: list[TestRow]) -> str:
    payload = []
    for row in rows:
        payload.append(
            {
                "environment": row.environment,
                "baseline_method": row.baseline_method,
                "baseline_mean": row.baseline_mean,
                "rp2o_mean": row.rp2o_mean,
                "mean_gap": row.mean_gap,
                "t_statistic": row.t_statistic,
                "raw_p": row.raw_p,
                "holm_p": row.holm_p,
            }
        )
    return json.dumps(payload, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, default=Path("logs"))
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_CONFIG),
        default="mean_reward",
        help="Per-run metric to compare across methods.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "latex", "json"),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--comparison-mode",
        choices=("best", "all"),
        default="best",
        help="Compare R2PO against the strongest observed baseline per environment or against all core baselines.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the rendered results.",
    )
    args = parser.parse_args()

    rows = compute_tests(args.logs_root, args.metric, args.comparison_mode)
    if args.format == "latex":
        rendered = render_latex(rows, args.metric, args.comparison_mode)
    elif args.format == "json":
        rendered = render_json(rows)
    else:
        rendered = render_markdown(rows, args.metric, args.comparison_mode)

    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
