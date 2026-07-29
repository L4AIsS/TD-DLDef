from __future__ import annotations

import argparse
import json

from td_dldef.behavior_analysis import run_behavior_correlation


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure structural/behavioral diversity correlation")
    parser.add_argument("--output", default="results/behavior_correlation")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--task", default="vision")
    parser.add_argument("--models", type=int, default=20)
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.quick:
        args.models = min(args.models, 6)
        args.nodes = min(args.nodes, 5)
        args.max_pairs = min(args.max_pairs, 20)
    result = run_behavior_correlation(output_dir=args.output, seed=args.seed, task_name=args.task, models=args.models, nodes=args.nodes, max_pairs=args.max_pairs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
