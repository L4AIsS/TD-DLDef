from __future__ import annotations

from experiments.common import matrix_parser, resolve_seeds
from td_dldef.experiment_matrix import run_matrix

SPACES = ["layer", "edge", "input_shape", "input_dimension"]


def main(argv=None):
    parser = matrix_parser("Only-one and drop-one ablation of the four diversity spaces", "results/diversity_ablation")
    args = parser.parse_args(argv)
    variants = {"full": {"generation": {"enabled_diversity_spaces": SPACES}}}
    for space in SPACES:
        variants[f"only_{space}"] = {"generation": {"enabled_diversity_spaces": [space]}}
        variants[f"drop_{space}"] = {"generation": {"enabled_diversity_spaces": [x for x in SPACES if x != space]}}
    if args.quick:
        for override in variants.values():
            override["budget"] = {"valid_tests": 3, "seconds": 20}
    run_matrix(base_config_path=args.config, variants=variants, seeds=resolve_seeds(args), output_root=args.output)


if __name__ == "__main__":
    main()
