from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from td_dldef.backends.numpy_backend import NumpyBackend
from td_dldef.constraints import default_registry
from td_dldef.execution import generate_inputs
from td_dldef.catalog import TaskSpec
from td_dldef.model_ir import LayerCandidate, ModelGraph, TensorSpec
from td_dldef.mutations import MutationEngine
from td_dldef.oracles import OracleEngine, contains_nan
from td_dldef.reporting import atomic_json


def base_graph():
    spec = TensorSpec((1, 8), "float32", "vector")
    graph = ModelGraph(inputs={"input": spec})
    candidate = LayerCandidate(
        op="layer_norm",
        inputs=("input",),
        input_specs=(spec,),
        output_spec=spec,
        params={"epsilon": 1e-5, "axis": -1},
    )
    graph.add_candidate(candidate)
    graph.metadata["contract_valid"] = True
    graph.metadata["expected_rejection"] = False
    return graph


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit oracle false positives on valid and deliberate invalid cases")
    parser.add_argument("--output", default="results/false_positive_audit")
    parser.add_argument("--seed", type=int, default=20260525)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    registry = default_registry()
    backend = NumpyBackend()
    oracle = OracleEngine()
    task = TaskSpec("vector", TensorSpec((1, 8), "float32"))
    inputs = generate_inputs(task, np.random.default_rng(args.seed))

    valid = MutationEngine(registry=registry, rng=np.random.default_rng(args.seed), mode="contract_valid", enabled=["BV"], max_mutations=1).apply(base_graph(), ["BV"])
    robust = MutationEngine(registry=registry, rng=np.random.default_rng(args.seed), mode="robustness", enabled=["BV"], max_mutations=1).apply(base_graph(), ["BV"])

    rows = []
    for label, mutation in [("contract_valid", valid), ("robustness", robust)]:
        result = backend.execute(mutation.graph, inputs, contract_valid=mutation.contract_valid, expected_rejection=mutation.expected_rejection)
        report = oracle.evaluate(mutation.graph, [result])
        rows.append({"label": label, "mutation": mutation.to_dict(), "execution": result.to_dict(), "oracle": report.to_dict()})
    summary = {
        "cases": rows,
        "none_is_nan": contains_nan(None),
        "expected": {
            "contract_valid_case": "Exceptions/NaN are candidates only after contract validation.",
            "robustness_case": "Rejection or acceptance is recorded separately and is not counted as a bug by default.",
        },
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
