from __future__ import annotations

import argparse
import json

from td_dldef.config import load_config
from td_dldef.runner import ExperimentRunner


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one TD-DLDef experiment")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--valid-tests", type=int)
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args(argv)
    overrides = {}
    if args.output:
        overrides["output"] = args.output
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.valid_tests is not None or args.seconds is not None:
        overrides["budget"] = {}
        if args.valid_tests is not None:
            overrides["budget"]["valid_tests"] = args.valid_tests
        if args.seconds is not None:
            overrides["budget"]["seconds"] = args.seconds
    config = load_config(args.config, overrides=overrides)
    summary = ExperimentRunner(config).run()
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
