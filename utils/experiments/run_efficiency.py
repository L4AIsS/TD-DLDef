from __future__ import annotations

from experiments.common import matrix_parser, resolve_seeds
from td_dldef.experiment_matrix import run_matrix


def main(argv=None):
    parser = matrix_parser("Efficiency under equal wall-clock and valid-test budgets", "results/efficiency")
    args = parser.parse_args(argv)
    variants = {
        name: {
            "generation": {"policy": name, "policy_args": ({"epsilon": 0.1} if name == "epsilon_greedy" else {})},
            "budget": {"valid_tests": 100 if not args.quick else 5, "seconds": 300 if not args.quick else 20},
        }
        for name in ["random", "greedy", "epsilon_greedy", "ucb1", "thompson"]
    }
    run_matrix(base_config_path=args.config, variants=variants, seeds=resolve_seeds(args), output_root=args.output)


if __name__ == "__main__":
    main()
