from __future__ import annotations

from experiments.common import matrix_parser, resolve_seeds
from td_dldef.experiment_matrix import run_matrix


def main(argv=None):
    parser = matrix_parser("Vision, transformer-like sequence, text/embedding, and graph-model tasks", "results/nonvision")
    args = parser.parse_args(argv)
    variants = {
        "vision": {"task": "vision"},
        "transformer": {"task": "transformer"},
        "text_embedding": {"task": "text"},
        "graph_gnn": {"task": "graph"},
    }
    if args.quick:
        for override in variants.values():
            override["budget"] = {"valid_tests": 3, "seconds": 20}
    run_matrix(base_config_path=args.config, variants=variants, seeds=resolve_seeds(args), output_root=args.output)


if __name__ == "__main__":
    main()
