from __future__ import annotations

from experiments.common import matrix_parser, resolve_seeds
from td_dldef.experiment_matrix import run_matrix


def main(argv=None):
    parser = matrix_parser("Scalability over model length and candidate-pool size", "results/scalability")
    args = parser.parse_args(argv)
    node_sizes = [4, 8] if args.quick else [4, 8, 16, 32]
    candidate_sizes = [12, 32] if args.quick else [12, 24, 48, 96]
    variants = {}
    for nodes in node_sizes:
        for candidates in candidate_sizes:
            variants[f"nodes{nodes}_candidates{candidates}"] = {
                "generation": {"nodes": [nodes, nodes], "max_candidates": candidates},
                "budget": {"valid_tests": 3 if args.quick else 30, "seconds": 30 if args.quick else 300},
            }
    run_matrix(base_config_path=args.config, variants=variants, seeds=resolve_seeds(args), output_root=args.output)


if __name__ == "__main__":
    main()
