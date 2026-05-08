"""
Analysis CLI for ProPS experiments.
Reads existing training logs. Does not re-run episodes.

Single run:
  uv run python eval.py --logdir logs/frozenlake_reflective_prompted_policy_optimization

With label:
  uv run python eval.py --logdir logs/frozenlake_reflective_prompted_policy_optimization --labels "R2PO"

Compare multiple runs:
  uv run python eval.py \\
    --logdir logs/frozenlake_props logs/frozenlake_propsp logs/frozenlake_reflective_prompted_policy_optimization \\
    --labels ProPS ProPS+ R2PO \\
    --compare-output logs/comparison
"""
import argparse
from dotenv import load_dotenv
from runner.eval_runner import analyze, compare
import gym_maze
from envs import nim, pong

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze training runs from existing logs. Does not re-run episodes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--logdir", nargs="+", required=True,
        help="One or more training log directories",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Display labels for each logdir (must match --logdir count if provided)",
    )
    parser.add_argument(
        "--compare-output", default=None,
        help="Directory for comparison plots when multiple logdirs are given",
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.logdir):
        parser.error("--labels count must match --logdir count")

    labels = args.labels or [None] * len(args.logdir)

    for logdir, label in zip(args.logdir, labels):
        print(f"\nAnalyzing: {logdir}")
        analyze(logdir, label=label)

    if len(args.logdir) > 1:
        import os
        output_dir = args.compare_output or os.path.dirname(args.logdir[0].rstrip("/"))
        print(f"\nGenerating comparison plots -> {output_dir}")
        compare(args.logdir, labels=args.labels, output_dir=output_dir, max_runs=10)


if __name__ == "__main__":
    main()
