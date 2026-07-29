from __future__ import annotations

import argparse
from pathlib import Path

from experiments import (
    run_behavior_correlation,
    run_diversity_ablation,
    run_efficiency,
    run_false_positive_audit,
    run_mutation_ablation,
    run_nonvision,
    run_policy_comparison,
    run_scalability,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the complete supplemental experiment suite")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="results/all")
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.output)
    common = ["--config", args.config, "--seeds", args.seeds]
    quick = ["--quick"] if args.quick else []
    run_policy_comparison.main(common + ["--output", str(root / "policy")] + quick)
    run_diversity_ablation.main(common + ["--output", str(root / "diversity")] + quick)
    run_mutation_ablation.main(common + ["--output", str(root / "mutation")] + quick)
    run_efficiency.main(common + ["--output", str(root / "efficiency")] + quick)
    run_scalability.main(common + ["--output", str(root / "scalability")] + quick)
    run_nonvision.main(common + ["--output", str(root / "nonvision")] + quick)
    run_behavior_correlation.main(["--output", str(root / "behavior")] + quick)
    run_false_positive_audit.main(["--output", str(root / "false_positive")])


if __name__ == "__main__":
    main()
