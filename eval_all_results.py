"""
Batch evaluation helper for the 10-environment results table.

It runs the existing log analyzer on all requested environments and on the
methods that are actually present under logs/.

Examples:
  uv run python eval_all_results.py

  uv run python eval_all_results.py --only-swimmer

  uv run python eval_all_results.py --max-runs 10
"""

import argparse
from pathlib import Path

from runner.eval_runner import analyze


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


def _candidate_logdirs(logs_root: Path, env_name: str, env_prefix: str):
    candidates = [
        ("ProPS", logs_root / f"{env_prefix}_props"),
        ("ProPS+", logs_root / f"{env_prefix}_propsp"),
        ("RepTraj", logs_root / f"{env_prefix}_reflective"),
        (
            "R2PO",
            logs_root
            / f"{env_prefix}_reflective_prompted_policy_optimization",
        ),
        (
            "ThreeTraj",
            logs_root / f"{env_prefix}_three_traj",
        ),
        ("PureSearch", logs_root / f"{env_prefix}_pure_search"),
        ("AlwaysCritic", logs_root / f"{env_prefix}_always_critic"),
        ("CriticOnly", logs_root / f"{env_prefix}_critic_only"),
        (
            "ActorSecondPass",
            logs_root / f"{env_prefix}_actor_second_pass",
        ),
    ]

    sb3_root = logs_root / f"{env_prefix}_sb3"
    if sb3_root.is_dir():
        for algo_dir in sorted(x for x in sb3_root.iterdir() if x.is_dir()):
            candidates.append((f"SB3-{algo_dir.name.upper()}", algo_dir))

    return [(label, path) for label, path in candidates if path.is_dir()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root",
        default="logs",
        help="Root directory containing experiment logs",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=10,
        help="Maximum number of run_N directories to analyze per experiment",
    )
    parser.add_argument(
        "--only-env",
        action="append",
        default=[],
        help="Short environment name to analyze, e.g. --only-env swimmer",
    )
    parser.add_argument(
        "--only-method",
        action="append",
        default=[],
        help="Method label to analyze, e.g. --only-method ActorSecondPass",
    )
    return parser.parse_args()


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ModuleNotFoundError:
        pass
    args = parse_args()
    logs_root = Path(args.logs_root)

    env_specs = DEFAULT_ENVS
    if args.only_env:
        wanted = set(args.only_env)
        env_specs = [spec for spec in DEFAULT_ENVS if spec[0] in wanted]

    if not env_specs:
        raise SystemExit("No matching environments selected.")

    wanted_methods = set(args.only_method)

    for env_name, env_prefix in env_specs:
        print(f"\n=== {env_name} ===")
        candidates = _candidate_logdirs(logs_root, env_name, env_prefix)
        if wanted_methods:
            candidates = [
                (label, path) for label, path in candidates if label in wanted_methods
            ]
        if not candidates:
            print("No logdirs found.")
            continue

        for label, logdir in candidates:
            print(f"\nAnalyzing {label}: {logdir}")
            try:
                analyze(str(logdir), label=label, max_runs=args.max_runs)
            except Exception as exc:
                print(f"Warning: failed to analyze {logdir}: {exc}")


if __name__ == "__main__":
    main()
