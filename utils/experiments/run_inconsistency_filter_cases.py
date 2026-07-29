"""Targeted cases for validating inconsistency-oracle filters.

The cases provide positive and negative controls for nondifferentiable points, datatype
precision, extreme scaling, state-dependent execution, and non-pure operations.  They
do not claim framework defects; they validate whether the oracle retains or excludes the
intended observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from td_dldef.backends.numpy_backend import NumpyBackend
from td_dldef.execution import ExecutionResult, TraceEntry
from td_dldef.model_ir import LayerCandidate, ModelGraph, TensorSpec
from td_dldef.oracles import (
    OracleEngine,
    assess_gradients,
    finite_difference_stability,
    nondifferentiable_input_mask,
)
from td_dldef.reporting import atomic_json


def _unary_graph(
    op: str,
    spec: TensorSpec,
    *,
    output_spec: TensorSpec | None = None,
    params: dict[str, Any] | None = None,
    tags: set[str] | None = None,
) -> ModelGraph:
    graph = ModelGraph(inputs={"input": spec})
    graph.add_candidate(
        LayerCandidate(
            op,
            ("input",),
            (spec,),
            output_spec or spec,
            params=params or {},
            tags=tags or set(),
        )
    )
    return graph


def _result(
    backend: str,
    value: np.ndarray,
    *,
    metadata: dict[str, Any] | None = None,
) -> ExecutionResult:
    arr = np.asarray(value)
    return ExecutionResult(
        backend=backend,
        backend_version="probe",
        status="ok",
        outputs={"n0000": arr},
        traces=[TraceEntry.from_array("n0000", "probe", arr)],
        metadata=metadata or {},
    )


def _record(name: str, expected: str, observed: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": name,
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
        "details": details,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run targeted inconsistency-filter cases")
    parser.add_argument("--output", default="results/inconsistency_filter_cases")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    backend = NumpyBackend()
    cases: list[dict[str, Any]] = []

    # 1. ReLU at zero: the numerical gradient (0.5) and a valid subgradient convention
    # (0.0) differ, but the boundary coordinate must be excluded.
    relu_spec = TensorSpec((1, 3), "float64", "vector")
    relu_graph = _unary_graph("relu", relu_spec, tags={"nondifferentiable"})
    relu_input = {"input": np.array([[-1.0, 0.0, 1.0]], dtype=np.float64)}
    relu_numerical = backend.input_gradient(relu_graph, relu_input, epsilon=1e-5)
    relu_analytical = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    relu_mask = nondifferentiable_input_mask(
        relu_graph, relu_input, epsilon=1e-5, kappa=2.0
    )
    decision = assess_gradients(
        relu_analytical,
        relu_numerical.outputs["input_gradient"],
        dtype="float64",
        nondifferentiable_mask=relu_mask,
    )
    observed = "excluded" if decision.excluded else ("finding" if decision.finding else "consistent")
    cases.append(_record("nondifferentiable_relu_zero", "consistent", observed, decision.to_dict()))

    # 2. A differentiable ReLU control must remain comparable.
    relu_positive = {"input": np.array([[1.0, 2.0, 3.0]], dtype=np.float64)}
    relu_positive_num = backend.input_gradient(relu_graph, relu_positive, epsilon=1e-5)
    positive_mask = nondifferentiable_input_mask(
        relu_graph, relu_positive, epsilon=1e-5, kappa=2.0
    )
    decision = assess_gradients(
        np.ones((1, 3), dtype=np.float64),
        relu_positive_num.outputs["input_gradient"],
        dtype="float64",
        nondifferentiable_mask=positive_mask,
    )
    observed = "excluded" if decision.excluded else ("finding" if decision.finding else "consistent")
    cases.append(_record("differentiable_relu_positive", "consistent", observed, decision.to_dict()))

    # 3. Max-pooling with a tied maximum is nondifferentiable and must be excluded.
    pool_input_spec = TensorSpec((1, 2, 2, 1), "float64", "image")
    pool_output_spec = TensorSpec((1, 1, 1, 1), "float64", "feature_map")
    pool_graph = _unary_graph(
        "max_pool2d",
        pool_input_spec,
        output_spec=pool_output_spec,
        params={"pool_size": [2, 2], "strides": [2, 2], "padding": "valid"},
        tags={"nondifferentiable"},
    )
    pool_input = {
        "input": np.array([[[[1.0], [1.0]], [[0.0], [-1.0]]]], dtype=np.float64)
    }
    pool_num = backend.input_gradient(pool_graph, pool_input, epsilon=1e-5)
    pool_analytical = np.array([[[[1.0], [0.0]], [[0.0], [0.0]]]], dtype=np.float64)
    pool_mask = nondifferentiable_input_mask(
        pool_graph, pool_input, epsilon=1e-5, kappa=2.0
    )
    decision = assess_gradients(
        pool_analytical,
        pool_num.outputs["input_gradient"],
        dtype="float64",
        nondifferentiable_mask=pool_mask,
    )
    observed = "excluded" if decision.excluded else ("finding" if decision.finding else "consistent")
    cases.append(_record("nondifferentiable_maxpool_tie", "excluded", observed, decision.to_dict()))

    # 4. A float16-sized rounding difference must be tolerated.
    decision = assess_gradients(
        np.array([1.0, 2.0], dtype=np.float16),
        np.array([1.002, 1.996], dtype=np.float16),
        dtype="float16",
    )
    observed = "excluded" if decision.excluded else ("finding" if decision.finding else "consistent")
    cases.append(_record("precision_within_float16_tolerance", "consistent", observed, decision.to_dict()))

    # 5. A stable, large discrepancy must remain a finding.
    decision = assess_gradients(
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([1.2, 1.7], dtype=np.float32),
        dtype="float32",
    )
    observed = "excluded" if decision.excluded else ("finding" if decision.finding else "consistent")
    cases.append(_record("precision_beyond_float32_tolerance", "finding", observed, decision.to_dict()))

    # 6. Finite-difference estimates that do not converge are excluded.
    fd = finite_difference_stability(
        {
            1e-2: np.array([1.0, 1.0]),
            1e-3: np.array([0.4, 1.6]),
            1e-4: np.array([1.8, 0.2]),
        },
        threshold=5e-2,
    )
    decision = assess_gradients(
        np.array([1.0, 1.0]),
        np.array([1.8, 0.2]),
        precision_unstable=not bool(fd["stable"]),
    )
    observed = "excluded" if decision.excluded else ("finding" if decision.finding else "consistent")
    cases.append(
        _record(
            "finite_difference_instability",
            "excluded",
            observed,
            {"stability": fd, "decision": decision.to_dict()},
        )
    )

    # 7. Extreme scaling is reported separately rather than as an inconsistency defect.
    extreme_graph = _unary_graph("identity", TensorSpec((1, 2), "float32"))
    extreme_report = OracleEngine().evaluate(
        extreme_graph,
        [
            _result("a", np.array([[3.2e38, 3.0e38]], dtype=np.float32)),
            _result("b", np.array([[2.8e38, 2.5e38]], dtype=np.float32)),
        ],
    )
    observed = "excluded" if any(item.get("reason") == "extreme_scaling" for item in extreme_report.skipped) else ("finding" if extreme_report.bug_candidates else "consistent")
    cases.append(_record("extreme_scaling", "excluded", observed, extreme_report.to_dict()))

    # 8/9. Stateful output differences are excluded until initial states are synchronised.
    state_graph = _unary_graph("identity", TensorSpec((1, 2)), tags={"stateful"})
    state_unsync = OracleEngine().evaluate(
        state_graph,
        [
            _result("a", np.array([[0.0, 1.0]]), metadata={"state_synchronised": False}),
            _result("b", np.array([[1.0, 0.0]]), metadata={"state_synchronised": False}),
        ],
    )
    observed = "excluded" if any(item.get("reason") == "unsynchronised_state" for item in state_unsync.skipped) else ("finding" if state_unsync.bug_candidates else "consistent")
    cases.append(_record("state_unsynchronised", "excluded", observed, state_unsync.to_dict()))

    state_sync = OracleEngine().evaluate(
        state_graph,
        [
            _result("a", np.array([[0.0, 1.0]]), metadata={"state_synchronised": True}),
            _result("b", np.array([[1.0, 0.0]]), metadata={"state_synchronised": True}),
        ],
    )
    observed = "excluded" if state_sync.skipped else ("finding" if state_sync.bug_candidates else "consistent")
    cases.append(_record("state_synchronised_difference", "finding", observed, state_sync.to_dict()))

    # 10/11. Non-pure operations require isolated input/model snapshots.
    non_pure_graph = _unary_graph("identity", TensorSpec((1, 2)), tags={"non_pure"})
    non_pure_unisolated = OracleEngine().evaluate(
        non_pure_graph,
        [
            _result("a", np.array([[0.0, 1.0]]), metadata={"isolated_snapshot": False}),
            _result("b", np.array([[1.0, 0.0]]), metadata={"isolated_snapshot": False}),
        ],
    )
    observed = "excluded" if any(item.get("reason") == "non_pure_operation" for item in non_pure_unisolated.skipped) else ("finding" if non_pure_unisolated.bug_candidates else "consistent")
    cases.append(_record("non_pure_unisolated", "excluded", observed, non_pure_unisolated.to_dict()))

    non_pure_isolated = OracleEngine().evaluate(
        non_pure_graph,
        [
            _result("a", np.array([[0.0, 1.0]]), metadata={"isolated_snapshot": True}),
            _result("b", np.array([[1.0, 0.0]]), metadata={"isolated_snapshot": True}),
        ],
    )
    observed = "excluded" if non_pure_isolated.skipped else ("finding" if non_pure_isolated.bug_candidates else "consistent")
    cases.append(_record("non_pure_isolated_difference", "finding", observed, non_pure_isolated.to_dict()))

    summary = {
        "case_count": len(cases),
        "passed": sum(bool(case["passed"]) for case in cases),
        "failed": sum(not bool(case["passed"]) for case in cases),
        "cases": cases,
    }
    atomic_json(output / "filter_cases.json", summary)
    print(json.dumps(summary, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
