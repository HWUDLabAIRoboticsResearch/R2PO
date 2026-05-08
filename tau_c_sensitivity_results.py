"""
Analyze and summarize tau_c sensitivity experiments.

This script can:
1. run the standard eval analyzer on any existing tau_c sensitivity logdirs, and
2. render a compact table comparing threshold settings across environments.

Examples:
  uv run python tau_c_sensitivity_results.py

  uv run python tau_c_sensitivity_results.py --reanalyze

  uv run python tau_c_sensitivity_results.py --format latex --output paper_results/tau_c_sensitivity_table.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from runner.eval_runner import analyze


EXPERIMENTS = {
    "cartpole": {
        "display": "CartPole",
        "settings": {
            "default": {
                "config": Path("configs/cartpole/cartpole_reflective_prompted_policy_optimization.yaml"),
                "logdir": Path("logs/cartpole_reflective_prompted_policy_optimization"),
            },
            "half": {
                "config": Path("configs/cartpole/cartpole_tau_c_sensitivity_half.yaml"),
                "logdir": Path("logs/cartpole_tau_c_sensitivity_half"),
            },
            "none": {
                "config": Path("configs/cartpole/cartpole_tau_c_sensitivity_none.yaml"),
                "logdir": Path("logs/cartpole_tau_c_sensitivity_none"),
            },
        },
    },
    "pong": {
        "display": "Pong",
        "settings": {
            "default": {
                "config": Path("configs/pong/pong_reflective_prompted_policy_optimization.yaml"),
                "logdir": Path("logs/pong_reflective_prompted_policy_optimization"),
            },
            "half": {
                "config": Path("configs/pong/pong_tau_c_sensitivity_half.yaml"),
                "logdir": Path("logs/pong_tau_c_sensitivity_half"),
            },
            "none": {
                "config": Path("configs/pong/pong_tau_c_sensitivity_none.yaml"),
                "logdir": Path("logs/pong_tau_c_sensitivity_none"),
            },
        },
    },
}

SETTING_DISPLAY = {
    "default": "default",
    "half": "0.5τ_c",
    "none": "None",
}


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _threshold_label(config_path: Path) -> str:
    cfg = _load_yaml(config_path)
    tau = cfg.get("critic_llm_conservative_threshold")
    if tau is None:
        return "None"
    return str(tau)


def _metric_cell(summary: dict, mean_key: str, std_key: str, latex: bool = False) -> str:
    mean = summary.get(mean_key)
    std = summary.get(std_key)
    if mean is None:
        return "--"
    if std is None:
        return f"{float(mean):.2f}"
    pm = r" $\pm$ " if latex else " ± "
    return f"{float(mean):.2f}{pm}{float(std):.2f}"


def _ensure_eval_report(logdir: Path, label: str, reanalyze: bool) -> Path | None:
    if not logdir.is_dir():
        return None
    summary_path = logdir / "eval_report" / "summary.json"
    if reanalyze or not summary_path.is_file():
        analyze(str(logdir), label=label)
    return summary_path if summary_path.is_file() else None


def collect_rows(reanalyze: bool = False) -> list[dict]:
    rows = []
    for env_key, env_info in EXPERIMENTS.items():
        for setting_key, setting_info in env_info["settings"].items():
            label = f"{env_info['display']} / {SETTING_DISPLAY[setting_key]}"
            summary_path = _ensure_eval_report(setting_info["logdir"], label, reanalyze)
            row = {
                "environment": env_info["display"],
                "setting": setting_key,
                "setting_display": SETTING_DISPLAY[setting_key],
                "tau_c": _threshold_label(setting_info["config"]),
                "logdir": str(setting_info["logdir"]),
                "available": summary_path is not None,
                "summary": None,
            }
            if summary_path is not None:
                row["summary"] = _load_json(summary_path)
            rows.append(row)
    return rows


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "## Tau_c Sensitivity",
        "",
        "| Environment | Setting | τ_c | Mean reward | Mean best reward | Mean final reward |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if not row["available"]:
            lines.append(
                f"| {row['environment']} | {row['setting_display']} | {row['tau_c']} | -- | -- | -- |"
            )
            continue
        summary = row["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["environment"],
                    row["setting_display"],
                    row["tau_c"],
                    _metric_cell(summary, "mean_reward_mean", "mean_reward_std"),
                    _metric_cell(summary, "best_reward_mean", "best_reward_std"),
                    _metric_cell(summary, "final_reward_mean", "final_reward_std"),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def render_latex(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{@{}lllrrr@{}}",
        r"\toprule",
        r"Environment & Setting & $\tau_c$ & Mean reward & Mean best reward & Mean final reward \\",
        r"\midrule",
    ]
    for row in rows:
        if not row["available"]:
            lines.append(
                f"{row['environment']} & {row['setting_display']} & {row['tau_c']} & -- & -- & -- \\\\"
            )
            continue
        summary = row["summary"]
        lines.append(
            f"{row['environment']} & {row['setting_display']} & {row['tau_c']}"
            f" & {_metric_cell(summary, 'mean_reward_mean', 'mean_reward_std', latex=True)}"
            f" & {_metric_cell(summary, 'best_reward_mean', 'best_reward_std', latex=True)}"
            f" & {_metric_cell(summary, 'final_reward_mean', 'final_reward_std', latex=True)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def render_json(rows: list[dict]) -> str:
    payload = []
    for row in rows:
        payload.append(
            {
                "environment": row["environment"],
                "setting": row["setting"],
                "tau_c": row["tau_c"],
                "available": row["available"],
                "summary": row["summary"],
            }
        )
    return json.dumps(payload, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reanalyze", action="store_true", help="Rebuild eval_report for available logdirs.")
    parser.add_argument(
        "--format",
        choices=("markdown", "latex", "json"),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional file to write.")
    args = parser.parse_args()

    rows = collect_rows(reanalyze=args.reanalyze)
    if args.format == "latex":
        rendered = render_latex(rows)
    elif args.format == "json":
        rendered = render_json(rows)
    else:
        rendered = render_markdown(rows)

    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
