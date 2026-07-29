"""Coverage, diversity, throughput, and efficiency measurements."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence
import math

import numpy as np

from .constraints import ConstraintRegistry
from .execution import ExecutionResult
from .model_ir import LayerCandidate, ModelGraph


def _freeze(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return repr((tuple(value.shape), str(value.dtype), value.tolist()))
    return repr(value)


@dataclass(slots=True)
class CoveragePoint:
    elapsed_seconds: float
    valid_tests: int
    composite: float
    lic: float
    lpc: float
    lsc: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "valid_tests": self.valid_tests,
            "composite": self.composite,
            "lic": self.lic,
            "lpc": self.lpc,
            "lsc": self.lsc,
        }


class CoverageAccumulator:
    """COMET-style structural coverage over the realised, constraint-valid universe.

    The denominator is the set of contract-valid candidates exposed by the catalogue
    during a run, not invalid Cartesian products. This makes the valid-space assumption
    explicit and serialisable.
    """

    def __init__(self, registry: ConstraintRegistry) -> None:
        self.registry = registry
        self.universe_types: dict[str, set[str]] = defaultdict(set)
        self.universe_dims: dict[str, set[int]] = defaultdict(set)
        self.universe_shapes: dict[str, set[tuple[int, ...]]] = defaultdict(set)
        self.universe_params: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.universe_edges: set[tuple[str, str]] = set()

        self.covered_types: dict[str, set[str]] = defaultdict(set)
        self.covered_dims: dict[str, set[int]] = defaultdict(set)
        self.covered_shapes: dict[str, set[tuple[int, ...]]] = defaultdict(set)
        self.covered_params: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.covered_edges: set[tuple[str, str]] = set()
        self.covered_layers: set[str] = set()
        self.behavior_signatures: set[tuple[str, tuple[int, int, int]]] = set()
        self.trace_digests: set[tuple[str, str]] = set()
        self.points: list[CoveragePoint] = []

    def register_candidates(self, graph: ModelGraph, candidates: Sequence[LayerCandidate]) -> None:
        for candidate in candidates:
            op = candidate.op
            for spec in candidate.input_specs:
                self.universe_types[op].add(spec.dtype)
                self.universe_dims[op].add(spec.ndim)
                self.universe_shapes[op].add(tuple(spec.shape))
            for name, value in candidate.params.items():
                self.universe_params[op][name].add(_freeze(value))
                constraint = self.registry.get(op, name)
                if constraint and constraint.choices:
                    self.universe_params[op][name].update(_freeze(v) for v in constraint.choices)
            for ref in candidate.inputs:
                self.universe_edges.add((graph.source_op(ref), op))

    def observe_graph(self, graph: ModelGraph) -> None:
        for node in graph.nodes:
            op = node.op
            self.covered_layers.add(op)
            for ref in node.inputs:
                spec = graph.get_spec(ref)
                self.covered_types[op].add(spec.dtype)
                self.covered_dims[op].add(spec.ndim)
                self.covered_shapes[op].add(tuple(spec.shape))
                self.covered_edges.add((graph.source_op(ref), op))
            for name, value in node.params.items():
                self.covered_params[op][name].add(_freeze(value))

    def observe_execution(self, results: Sequence[ExecutionResult]) -> None:
        for result in results:
            if result.status != "ok":
                continue
            for trace in result.traces:
                self.behavior_signatures.add((trace.op, trace.branch_signature))
                self.trace_digests.add((trace.op, trace.digest))

    @staticmethod
    def _ratio(covered: set[Any], universe: set[Any]) -> float:
        if not universe:
            return 0.0
        return min(1.0, len(covered & universe) / len(universe))

    def lic(self) -> float:
        ops = sorted(set(self.universe_types) | set(self.covered_types))
        if not ops:
            return 0.0
        per_op = []
        for op in ops:
            ratios = [
                self._ratio(self.covered_types[op], self.universe_types[op]),
                self._ratio(self.covered_dims[op], self.universe_dims[op]),
                self._ratio(self.covered_shapes[op], self.universe_shapes[op]),
            ]
            per_op.append(sum(ratios) / 3.0)
        return float(np.mean(per_op))

    def lpc(self) -> float:
        values: list[float] = []
        ops = set(self.universe_params) | set(self.covered_params)
        for op in ops:
            params = set(self.universe_params[op]) | set(self.covered_params[op])
            for param in params:
                universe = self.universe_params[op][param]
                covered = self.covered_params[op][param]
                if universe:
                    values.append(self._ratio(covered, universe))
        return float(np.mean(values)) if values else 0.0

    def lsc(self) -> float:
        return self._ratio(self.covered_edges, self.universe_edges)

    def layer_coverage(self, enabled_layers: Iterable[str]) -> float:
        universe = set(enabled_layers)
        return self._ratio(self.covered_layers, universe)

    def composite(self) -> float:
        return float(np.mean([self.lic(), self.lpc(), self.lsc()]))

    def checkpoint(self, elapsed_seconds: float, valid_tests: int) -> CoveragePoint:
        point = CoveragePoint(
            elapsed_seconds=float(elapsed_seconds),
            valid_tests=int(valid_tests),
            composite=self.composite(),
            lic=self.lic(),
            lpc=self.lpc(),
            lsc=self.lsc(),
        )
        self.points.append(point)
        return point

    def coverage_auc(self, *, budget_seconds: float | None = None) -> float:
        if not self.points:
            return 0.0
        xs = np.array([p.elapsed_seconds for p in self.points], dtype=np.float64)
        ys = np.array([p.composite for p in self.points], dtype=np.float64)
        if len(xs) == 1:
            duration = budget_seconds or max(xs[0], 1e-12)
            return float(ys[0] * min(xs[0], duration) / duration)
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        end = float(budget_seconds or xs[-1])
        if end <= 0:
            return 0.0
        if xs[0] > 0:
            xs = np.insert(xs, 0, 0.0)
            ys = np.insert(ys, 0, 0.0)
        if xs[-1] < end:
            xs = np.append(xs, end)
            ys = np.append(ys, ys[-1])
        mask = xs <= end
        xs, ys = xs[mask], ys[mask]
        if xs[-1] < end:
            xs = np.append(xs, end)
            ys = np.append(ys, ys[-1])
        return float(np.trapezoid(ys, xs) / end)

    def summary(self, *, enabled_layers: Iterable[str], budget_seconds: float | None = None) -> dict[str, Any]:
        return {
            "LIC": self.lic(),
            "LPC": self.lpc(),
            "LSC": self.lsc(),
            "layer_coverage": self.layer_coverage(enabled_layers),
            "composite_structural_coverage": self.composite(),
            "coverage_auc": self.coverage_auc(budget_seconds=budget_seconds),
            "unique_behavior_signatures": len(self.behavior_signatures),
            "unique_execution_traces": len(self.trace_digests),
            "covered_layer_types": len(self.covered_layers),
            "covered_edges": len(self.covered_edges),
            "universe_edges": len(self.universe_edges),
            "coverage_points": [point.to_dict() for point in self.points],
            "denominator_policy": "realised contract-valid candidate universe",
        }
