from __future__ import annotations

from experiments.common import matrix_parser, resolve_seeds
from td_dldef.experiment_matrix import run_matrix

OPS = ["PV", "BV", "IT", "NF", "SW", "WR"]


def main(argv=None):
    parser = matrix_parser("Only-one and drop-one ablation of six mutation operators", "results/mutation_ablation")
    args = parser.parse_args(argv)
    variants = {
        "all": {"mutation": {"enabled": OPS, "max_mutations": 6}},
        "none": {"mutation": {"enabled": [], "max_mutations": 0}},
    }
    for op in OPS:
        variants[f"only_{op}"] = {"mutation": {"enabled": [op], "max_mutations": 1}}
        variants[f"drop_{op}"] = {"mutation": {"enabled": [x for x in OPS if x != op], "max_mutations": 5}}
    if args.quick:
        for override in variants.values():
            override["budget"] = {"valid_tests": 3, "seconds": 20}
    run_matrix(base_config_path=args.config, variants=variants, seeds=resolve_seeds(args), output_root=args.output)


if __name__ == "__main__":
    main()
