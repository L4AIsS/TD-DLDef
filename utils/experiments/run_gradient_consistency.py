"""Cross-backend gradient consistency with oracle-validity filtering.

The experiment compares gradients only after excluding finite-difference instability,
provable nondifferentiable input locations, unsynchronised state, non-pure execution,
and extreme scaling.  Each exclusion is logged separately from retained defect
candidates so that false-positive filtering can be reported in the paper.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from td_dldef.backends import create_backend
from td_dldef.backends.base import BackendUnavailable
from td_dldef.bandits import create_policy
from td_dldef.catalog import LayerCatalog, built_in_task
from td_dldef.constraints import default_registry
from td_dldef.diversity import DiversityTracker
from td_dldef.execution import generate_inputs
from td_dldef.generator import ModelGenerator
from td_dldef.oracles import (
    assess_gradients,
    finite_difference_stability,
    nondifferentiable_input_mask,
)
from td_dldef.reporting import append_jsonl, atomic_json


def _parse_epsilons(raw: str) -> list[float]:
    values = sorted({float(item.strip()) for item in raw.split(",") if item.strip()}, reverse=True)
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("--epsilons must contain one or more positive values")
    return values


def _graph_tags(graph) -> set[str]:
    return set().union(*(node.tags for node in graph.nodes)) if graph.nodes else set()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cross-backend input-gradient consistency")
    parser.add_argument("--backends", default="numpy,tensorflow,pytorch")
    parser.add_argument("--output", default="results/gradient_consistency")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--max-elements", type=int, default=64)
    parser.add_argument("--epsilons", default="1e-2,1e-3,1e-4,1e-5")
    parser.add_argument("--fd-stability-threshold", type=float, default=5e-2)
    parser.add_argument("--nondiff-kappa", type=float, default=2.0)
    args = parser.parse_args(argv)

    epsilons = _parse_epsilons(args.epsilons)
    selected_epsilon = min(epsilons)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    backends = []
    unavailable = []
    for name in [item.strip() for item in args.backends.split(",") if item.strip()]:
        try:
            backends.append(create_backend(name))
        except BackendUnavailable as exc:
            unavailable.append({"backend": name, "error": str(exc)})
    if len(backends) < 2:
        raise RuntimeError("Gradient differential testing requires at least two available backends")

    rng = np.random.default_rng(args.seed)
    task = built_in_task("vector")
    # Keep the input under the finite-difference limit and use float64 to minimise the
    # reference-gradient rounding error.
    task = type(task)(
        task.name,
        type(task.input_spec)((1, min(32, args.max_elements)), "float64", "vector"),
        task.distribution,
        task.metadata,
    )
    catalog = LayerCatalog(
        task=task,
        enabled_ops=[
            "identity",
            "dense",
            "relu",
            "tanh",
            "sigmoid",
            "layer_norm",
            "reshape",
            "flatten",
            "add",
            "concat",
        ],
    )
    registry = default_registry()
    tracker = DiversityTracker()
    policy = create_policy("thompson")

    records = []
    findings = []
    exclusion_counts: Counter[str] = Counter()
    consistent_count = 0
    path = output / "cases.jsonl"
    path.unlink(missing_ok=True)

    for case_index in range(args.cases):
        graph = ModelGenerator(
            task=task,
            catalog=catalog,
            registry=registry,
            tracker=tracker,
            policy=policy,
            rng=rng,
            target_nodes=args.nodes,
        ).generate().graph
        inputs = generate_inputs(task, rng)
        tags = _graph_tags(graph)

        canonical_results = []
        all_gradient_results = []
        finite_difference_gradients: dict[float, np.ndarray] = {}

        for backend in backends:
            if backend.name == "numpy":
                successful = []
                for epsilon in epsilons:
                    result = backend.input_gradient(
                        graph,
                        inputs,
                        epsilon=epsilon,
                        max_elements=args.max_elements,
                    )
                    result.metadata["requested_epsilon"] = epsilon
                    all_gradient_results.append(result)
                    if result.status == "ok":
                        finite_difference_gradients[epsilon] = np.asarray(
                            result.outputs["input_gradient"]
                        )
                        successful.append((epsilon, result))
                if successful:
                    # Use the smallest finite-difference step for the cross-backend
                    # comparison; convergence is assessed from all successful steps.
                    canonical_results.append(min(successful, key=lambda item: item[0])[1])
            else:
                result = backend.input_gradient(
                    graph,
                    inputs,
                    epsilon=selected_epsilon,
                    max_elements=args.max_elements,
                )
                all_gradient_results.append(result)
                if result.status == "ok":
                    canonical_results.append(result)

        fd_assessment = finite_difference_stability(
            finite_difference_gradients,
            threshold=args.fd_stability_threshold,
        )
        nondiff_mask = nondifferentiable_input_mask(
            graph,
            inputs,
            epsilon=float(fd_assessment.get("selected_epsilon") or selected_epsilon),
            kappa=args.nondiff_kappa,
        )

        case_findings = []
        decisions = []
        for left_index in range(len(canonical_results)):
            for right_index in range(left_index + 1, len(canonical_results)):
                left = canonical_results[left_index]
                right = canonical_results[right_index]
                left_gradient = np.asarray(left.outputs["input_gradient"])
                right_gradient = np.asarray(right.outputs["input_gradient"])
                includes_finite_difference = (
                    left.metadata.get("method") == "central_finite_difference"
                    or right.metadata.get("method") == "central_finite_difference"
                )
                stateful = "stateful" in tags or "stateful_formula" in tags
                non_pure = "non_pure" in tags
                decision = assess_gradients(
                    left_gradient,
                    right_gradient,
                    dtype=task.input_spec.dtype,
                    nondifferentiable_mask=nondiff_mask,
                    stateful=stateful,
                    state_synchronised=(
                        not stateful
                        or (
                            bool(left.metadata.get("state_synchronised", False))
                            and bool(right.metadata.get("state_synchronised", False))
                        )
                    ),
                    non_pure=non_pure,
                    isolated_snapshot=(
                        not non_pure
                        or (
                            bool(left.metadata.get("isolated_snapshot", False))
                            and bool(right.metadata.get("isolated_snapshot", False))
                        )
                    ),
                    precision_unstable=(
                        includes_finite_difference and not bool(fd_assessment["stable"])
                    ),
                    extreme_scaling=bool(graph.metadata.get("extreme_scaling", False)),
                )
                if decision.finding is not None:
                    decision.finding.backends = (left.backend, right.backend)
                    serialised = decision.finding.to_dict()
                    case_findings.append(serialised)
                    findings.append(serialised)
                elif decision.excluded:
                    exclusion_counts[str(decision.reason)] += 1
                else:
                    consistent_count += 1
                decisions.append(
                    {
                        "backends": [left.backend, right.backend],
                        **decision.to_dict(),
                    }
                )

        record = {
            "case_id": case_index,
            "model": graph.to_dict(include_arrays=True),
            "graph_tags": sorted(tags),
            "input": np.asarray(inputs["input"]).tolist(),
            "finite_difference_stability": fd_assessment,
            "nondifferentiable_mask_count": (
                int(np.asarray(nondiff_mask, dtype=bool).sum()) if nondiff_mask is not None else 0
            ),
            "gradients": [result.to_dict(include_outputs=True) for result in all_gradient_results],
            "decisions": decisions,
            "findings": case_findings,
        }
        records.append(record)
        append_jsonl(path, record)

    summary = {
        "seed": args.seed,
        "cases": args.cases,
        "backends": [backend.environment() for backend in backends],
        "unavailable": unavailable,
        "epsilons": epsilons,
        "fd_stability_threshold": args.fd_stability_threshold,
        "nondiff_kappa": args.nondiff_kappa,
        "finding_count": len(findings),
        "consistent_comparison_count": consistent_count,
        "excluded_comparison_count": int(sum(exclusion_counts.values())),
        "exclusions_by_reason": dict(sorted(exclusion_counts.items())),
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
