from __future__ import annotations

from experiments.common import matrix_parser, resolve_seeds
from td_dldef.experiment_matrix import run_matrix


def main(argv=None):
    parser = matrix_parser("Compare random/greedy/epsilon-greedy/UCB1/Thompson under identical budgets", "results/policy_comparison")
    args = parser.parse_args(argv)
    variants = {
        "random": {"generation": {"policy": "random", "policy_args": {}}},
        "greedy": {"generation": {"policy": "greedy", "policy_args": {}}},
        "epsilon_greedy": {"generation": {"policy": "epsilon_greedy", "policy_args": {"epsilon": 0.1}}},
        "ucb1": {"generation": {"policy": "ucb1", "policy_args": {}}},
        "thompson": {"generation": {"policy": "thompson", "policy_args": {"alpha": 1.0, "beta": 1.0}}},
    }
    if args.quick:
        for override in variants.values():
            override["budget"] = {"valid_tests": 4, "seconds": 20}
    run_matrix(base_config_path=args.config, variants=variants, seeds=resolve_seeds(args), output_root=args.output)


if __name__ == "__main__":
    main()
