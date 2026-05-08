"""
Generate paper-style result tables, one metric per table.

Outputs three separate tables for:
  - mean best reward ± std
  - mean reward ± std
  - mean final reward ± std

Rows are environments. Columns depend on the selected preset.
"""

import argparse
from pathlib import Path

os_env = __import__("os").environ
os_env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from results_table import DEFAULT_ENVS, collect_rows, filter_rows


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
METRICS = {
    "mean_best_reward": {
        "title": "Mean Best Reward",
        "filename": "mean_best_reward",
    },
    "mean_reward": {
        "title": "Mean Reward",
        "filename": "mean_reward",
    },
    "mean_final_reward": {
        "title": "Mean Final Reward",
        "filename": "mean_final_reward",
    },
}


def _format_number(value):
    if value is None:
        return ""
    return f"{float(value):.2f}"


def _display_name(method):
    return DISPLAY_NAMES.get(method, method)


def _select_best_row(rows, metric_key):
    if rows is None:
        return None
    if isinstance(rows, dict):
        return rows if rows.get(metric_key) is not None else None
    candidates = [row for row in rows if row.get(metric_key) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row[metric_key]))


def _row_for_display_method(row_map, method, metric_key):
    if method != "BestBaseline":
        return _select_best_row(row_map.get(method), metric_key)

    baseline_candidates = []
    for baseline_method in ["ProPS", "ProPS+", "SB3"]:
        row = _select_best_row(row_map.get(baseline_method), metric_key)
        if row is None:
            continue
        baseline_candidates.append((float(row[metric_key]), baseline_method, row))
    if not baseline_candidates:
        return None
    baseline_candidates.sort(key=lambda item: item[0], reverse=True)
    return baseline_candidates[0][2]


def _format_cell(row, metric_key, display_method=None):
    if row is None:
        return ""
    std_key = f"{metric_key}_std"
    metric_text = f"{_format_number(row[metric_key])} ± {_format_number(row[std_key])}"
    if display_method == "BestBaseline":
        method_name = row["method"]
        if method_name.startswith("SB3-"):
            method_name = method_name.split("-", 1)[1]
        return f"{method_name}: {metric_text}"
    if row["method"].startswith("SB3-"):
        algo = row["method"].split("-", 1)[1]
        return f"{algo}: {metric_text}"
    return metric_text


def _build_matrix(rows, methods):
    env_order = [env_name for env_name, _ in DEFAULT_ENVS]
    matrix = []
    for env_name in env_order:
        env_rows = [row for row in rows if row["environment"] == env_name]
        row_map = {}
        methods_to_populate = list(methods)
        if "BestBaseline" in methods_to_populate:
            methods_to_populate.extend(["ProPS", "ProPS+", "SB3"])
        for method in methods_to_populate:
            if method == "SB3":
                row_map[method] = [
                    row for row in env_rows if row["method"].startswith("SB3-")
                ]
            elif method == "BestBaseline":
                continue
            else:
                row_map[method] = next(
                    (row for row in env_rows if row["method"] == method),
                    None,
                )
        matrix.append((env_name, row_map))
    return matrix


def _best_methods_for_env(row_map, metric_key, methods):
    values = []
    for method in methods:
        row = _row_for_display_method(row_map, method, metric_key)
        if row is None or row.get(metric_key) is None:
            continue
        values.append((method, float(row[metric_key])))
    if not values:
        return set()
    best_value = max(value for _, value in values)
    return {method for method, value in values if abs(value - best_value) <= 1e-12}


def render_markdown(matrix, metric_key, methods):
    title = METRICS[metric_key]["title"]
    headers = ["Environment"] + [_display_name(method) for method in methods]
    lines = [f"## {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for env_name, row_map in matrix:
        best_methods = _best_methods_for_env(row_map, metric_key, methods)
        values = [env_name]
        for method in methods:
            row = _row_for_display_method(row_map, method, metric_key)
            cell = _format_cell(row, metric_key, display_method=method)
            if method in best_methods and cell:
                cell = f"**{cell}**"
            values.append(cell)
        lines.append("| " + " | ".join(values) + " |")

    lines.append("")
    return "\n".join(lines)


def write_csv(matrix, metric_key, methods, output_path: Path):
    import csv

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["environment"] + [_display_name(method) for method in methods])
        for env_name, row_map in matrix:
            row = [env_name]
            for method in methods:
                display_row = _row_for_display_method(row_map, method, metric_key)
                row.append(_format_cell(display_row, metric_key, display_method=method))
            writer.writerow(row)


def write_png(matrix, metric_key, methods, output_path: Path):
    title = METRICS[metric_key]["title"]
    headers = ["Environment"] + [_display_name(method) for method in methods]
    table_rows = []
    highlight = []

    for env_name, row_map in matrix:
        best_methods = _best_methods_for_env(row_map, metric_key, methods)
        row = [env_name]
        row_highlight = [False]
        for method in methods:
            display_row = _row_for_display_method(row_map, method, metric_key)
            row.append(_format_cell(display_row, metric_key, display_method=method))
            row_highlight.append(method in best_methods and bool(row[-1]))
        table_rows.append(row)
        highlight.append(row_highlight)

    fig_width = max(16, 2.5 * len(headers))
    fig, ax = plt.subplots(figsize=(fig_width, max(7, 0.7 * len(table_rows) + 1.8)))
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
    tbl.set_fontsize(11)
    tbl.scale(1, 1.8)

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

    ax.set_title(title, fontsize=16, fontweight="bold", pad=16)
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--output-dir", default="paper_results")
    parser.add_argument(
        "--preset",
        choices=sorted(METHOD_PRESETS),
        default="core",
        help="Which method family to render",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["md", "csv", "png"],
        default=["md", "png"],
        help="Output formats to generate",
    )
    parser.add_argument("--expected-runs", type=int, default=10)
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument("--expected-props-episodes", type=int, default=200)
    parser.add_argument("--expected-critic-only-episodes", type=int, default=200)
    parser.add_argument("--expected-sb3-episodes", type=int, default=190)
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Exclude rows that fail validation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = METHOD_PRESETS[args.preset]

    all_rows = collect_rows(
        Path(args.logs_root),
        DEFAULT_ENVS,
        expected_runs=args.expected_runs,
        expected_episodes=args.expected_episodes,
        expected_props_episodes=args.expected_props_episodes,
        expected_critic_only_episodes=args.expected_critic_only_episodes,
        strict_validation=args.strict_validation,
        expected_sb3_episodes=args.expected_sb3_episodes,
    )
    matrix = _build_matrix(all_rows, methods)

    for metric_key, meta in METRICS.items():
        stem = f"{args.preset}_{meta['filename']}"
        if "md" in args.formats:
            (output_dir / f"{stem}.md").write_text(
                render_markdown(matrix, metric_key, methods) + "\n"
            )
        if "csv" in args.formats:
            write_csv(matrix, metric_key, methods, output_dir / f"{stem}.csv")
        if "png" in args.formats:
            write_png(matrix, metric_key, methods, output_dir / f"{stem}.png")


if __name__ == "__main__":
    main()
