"""Four-space diversity evaluation used by TD-DLDef.

The paper defines layer, edge, input-shape, and input-dimension spaces.  This
implementation keeps the spaces separately observable, supports full/only-one/drop-one
ablations, records the magnitude of gain, and exposes the paper-compatible binary reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .model_ir import LayerCandidate, ModelGraph

SPACE_NAMES = ("layer", "edge", "input_shape", "input_dimension")


@dataclass(frozen=True, slots=True)
class DiversityGain:
    layer: frozenset[Any] = frozenset()
    edge: frozenset[Any] = frozenset()
    input_shape: frozenset[Any] = frozenset()
    input_dimension: frozenset[Any] = frozenset()

    @property
    def total(self) -> int:
        return sum(len(getattr(self, name)) for name in SPACE_NAMES)

    @property
    def binary_reward(self) -> int:
        return int(self.total > 0)

    def weighted_reward(self, weights: dict[str, float] | None = None) -> float:
        weights = weights or {name: 1.0 for name in SPACE_NAMES}
        return float(sum(weights.get(name, 0.0) * len(getattr(self, name)) for name in SPACE_NAMES))

    def to_dict(self) -> dict[str, Any]:
        return {
            name: sorted(repr(x) for x in getattr(self, name))
            for name in SPACE_NAMES
        } | {"total": self.total, "binary_reward": self.binary_reward}


@dataclass(slots=True)
class DiversitySnapshot:
    layer: set[Any] = field(default_factory=set)
    edge: set[Any] = field(default_factory=set)
    input_shape: set[Any] = field(default_factory=set)
    input_dimension: set[Any] = field(default_factory=set)

    def copy(self) -> "DiversitySnapshot":
        return DiversitySnapshot(**{name: set(getattr(self, name)) for name in SPACE_NAMES})

    def counts(self) -> dict[str, int]:
        return {name: len(getattr(self, name)) for name in SPACE_NAMES}

    def to_dict(self) -> dict[str, list[str]]:
        return {name: sorted(repr(x) for x in getattr(self, name)) for name in SPACE_NAMES}


class DiversityTracker:
    """Corpus-level four-space diversity state.

    `enabled_spaces` controls ablations. Disabled spaces are still derivable from a graph,
    but do not contribute to acceptance or bandit reward.
    """

    def __init__(self, enabled_spaces: Iterable[str] = SPACE_NAMES) -> None:
        enabled = tuple(dict.fromkeys(str(x) for x in enabled_spaces))
        unknown = set(enabled) - set(SPACE_NAMES)
        if unknown:
            raise ValueError(f"Unknown diversity spaces: {sorted(unknown)}")
        if not enabled:
            raise ValueError("At least one diversity space must be enabled")
        self.enabled_spaces = enabled
        self.state = DiversitySnapshot()

    @staticmethod
    def candidate_elements(graph: ModelGraph, candidate: LayerCandidate) -> DiversitySnapshot:
        layer = {candidate.op}
        edge = {(graph.source_op(ref), candidate.op) for ref in candidate.inputs}
        input_shape = {
            (candidate.op, tuple(spec.shape), spec.dtype)
            for spec in candidate.input_specs
        }
        input_dimension = {
            (candidate.op, spec.ndim)
            for spec in candidate.input_specs
        }
        return DiversitySnapshot(layer=layer, edge=edge, input_shape=input_shape, input_dimension=input_dimension)

    def preview(self, graph: ModelGraph, candidate: LayerCandidate) -> DiversityGain:
        elements = self.candidate_elements(graph, candidate)
        gains: dict[str, frozenset[Any]] = {}
        for name in SPACE_NAMES:
            if name in self.enabled_spaces:
                gains[name] = frozenset(getattr(elements, name) - getattr(self.state, name))
            else:
                gains[name] = frozenset()
        return DiversityGain(**gains)

    def commit(self, graph: ModelGraph, candidate: LayerCandidate) -> DiversityGain:
        gain = self.preview(graph, candidate)
        elements = self.candidate_elements(graph, candidate)
        for name in self.enabled_spaces:
            getattr(self.state, name).update(getattr(elements, name))
        return gain

    def observe_graph(self, graph: ModelGraph) -> None:
        empty = ModelGraph(inputs=dict(graph.inputs))
        for node in graph.nodes:
            candidate = LayerCandidate(
                op=node.op,
                inputs=node.inputs,
                input_specs=tuple(empty.get_spec(ref) for ref in node.inputs),
                output_spec=node.output_spec,
                params=node.params,
                weights=node.weights,
                tags=node.tags,
                candidate_id=node.node_id,
            )
            self.commit(empty, candidate)
            empty.add_candidate(candidate, node.node_id)

    def counts(self) -> dict[str, int]:
        return self.state.counts()

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled_spaces": list(self.enabled_spaces),
            "counts": self.counts(),
            "elements": self.state.to_dict(),
        }
