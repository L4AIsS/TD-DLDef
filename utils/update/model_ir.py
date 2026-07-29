"""Framework-neutral model intermediate representation.

The original repository mixed model generation, framework execution, and experiment
bookkeeping.  This module separates those concerns so that every generated case has a
serialisable model, concrete tensor specifications, mutation provenance, and an oracle
contract label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from typing import Any, Iterable, Mapping
import copy
import hashlib
import json

import numpy as np


JSONScalar = str | int | float | bool | None


def _jsonable(value: Any, *, include_arrays: bool = False) -> Any:
    """Convert nested values into deterministic JSON-compatible objects."""
    if isinstance(value, np.ndarray):
        if include_arrays:
            return value.tolist()
        digest = hashlib.sha256(value.tobytes()).hexdigest()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": digest,
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v, include_arrays=include_arrays) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, include_arrays=include_arrays) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v, include_arrays=include_arrays) for v in value)
    return value


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """A concrete tensor contract used by generation and execution."""

    shape: tuple[int, ...]
    dtype: str = "float32"
    semantic: str = "generic"

    def __post_init__(self) -> None:
        if not self.shape:
            raise ValueError("TensorSpec.shape must contain at least one dimension")
        if any(not isinstance(d, int) or d <= 0 for d in self.shape):
            raise ValueError(f"Tensor dimensions must be positive integers: {self.shape!r}")
        object.__setattr__(self, "dtype", normalize_dtype(self.dtype))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return reduce(mul, self.shape, 1)

    def to_dict(self) -> dict[str, Any]:
        return {"shape": list(self.shape), "dtype": self.dtype, "semantic": self.semantic}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TensorSpec":
        return cls(tuple(int(x) for x in raw["shape"]), str(raw.get("dtype", "float32")), str(raw.get("semantic", "generic")))


_DTYPE_ALIASES = {
    "half": "float16",
    "float": "float32",
    "double": "float64",
    "long": "int64",
    "int": "int32",
    "bool_": "bool",
}


def normalize_dtype(dtype: str) -> str:
    name = str(dtype).lower().replace("torch.", "").replace("tf.", "").replace("numpy.", "")
    return _DTYPE_ALIASES.get(name, name)


@dataclass(slots=True)
class LayerNode:
    """One executable node in a generated DAG."""

    node_id: str
    op: str
    inputs: tuple[str, ...]
    output_spec: TensorSpec
    params: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, np.ndarray] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)
    provenance: list[dict[str, Any]] = field(default_factory=list)

    def clone(self) -> "LayerNode":
        return LayerNode(
            node_id=self.node_id,
            op=self.op,
            inputs=tuple(self.inputs),
            output_spec=self.output_spec,
            params=copy.deepcopy(self.params),
            weights={k: np.array(v, copy=True) for k, v in self.weights.items()},
            tags=set(self.tags),
            provenance=copy.deepcopy(self.provenance),
        )

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "op": self.op,
            "inputs": list(self.inputs),
            "output_spec": self.output_spec.to_dict(),
            "params": _jsonable(self.params, include_arrays=include_arrays),
            "weights": _jsonable(self.weights, include_arrays=include_arrays),
            "tags": sorted(self.tags),
            "provenance": _jsonable(self.provenance, include_arrays=include_arrays),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LayerNode":
        return cls(
            node_id=str(raw["node_id"]),
            op=str(raw["op"]),
            inputs=tuple(str(x) for x in raw.get("inputs", [])),
            output_spec=TensorSpec.from_dict(raw["output_spec"]),
            params=copy.deepcopy(dict(raw.get("params", {}))),
            weights={k: np.asarray(v, dtype=np.float64) for k, v in dict(raw.get("weights", {})).items()},
            tags=set(str(x) for x in raw.get("tags", [])),
            provenance=copy.deepcopy(list(raw.get("provenance", []))),
        )


@dataclass(slots=True)
class LayerCandidate:
    """A contract-valid candidate considered by a layer-selection policy."""

    op: str
    inputs: tuple[str, ...]
    input_specs: tuple[TensorSpec, ...]
    output_spec: TensorSpec
    params: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, np.ndarray] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)
    candidate_id: str = ""
    arm_key: str = ""

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("LayerCandidate requires at least one input reference")
        if len(self.inputs) != len(self.input_specs):
            raise ValueError("inputs and input_specs must have the same length")
        if not self.candidate_id:
            payload = {
                "op": self.op,
                "inputs": self.inputs,
                "input_specs": [s.to_dict() for s in self.input_specs],
                "output_spec": self.output_spec.to_dict(),
                "params": _jsonable(self.params),
            }
            self.candidate_id = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:16]
        if not self.arm_key:
            self.arm_key = self.op

    def to_node(self, node_id: str) -> LayerNode:
        return LayerNode(
            node_id=node_id,
            op=self.op,
            inputs=tuple(self.inputs),
            output_spec=self.output_spec,
            params=copy.deepcopy(self.params),
            weights={k: np.array(v, copy=True) for k, v in self.weights.items()},
            tags=set(self.tags),
        )


@dataclass(slots=True)
class ModelGraph:
    """A topologically ordered directed acyclic model graph."""

    inputs: dict[str, TensorSpec]
    nodes: list[LayerNode] = field(default_factory=list)
    outputs: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "ModelGraph":
        return ModelGraph(
            inputs=dict(self.inputs),
            nodes=[node.clone() for node in self.nodes],
            outputs=tuple(self.outputs),
            metadata=copy.deepcopy(self.metadata),
        )

    @property
    def values(self) -> tuple[str, ...]:
        return tuple(self.inputs) + tuple(node.node_id for node in self.nodes)

    def get_spec(self, ref: str) -> TensorSpec:
        if ref in self.inputs:
            return self.inputs[ref]
        for node in self.nodes:
            if node.node_id == ref:
                return node.output_spec
        raise KeyError(f"Unknown graph value: {ref}")

    def get_node(self, ref: str) -> LayerNode | None:
        for node in self.nodes:
            if node.node_id == ref:
                return node
        return None

    def source_op(self, ref: str) -> str:
        node = self.get_node(ref)
        return node.op if node is not None else "__input__"

    def add_candidate(self, candidate: LayerCandidate, node_id: str | None = None) -> LayerNode:
        known = set(self.values)
        missing = [ref for ref in candidate.inputs if ref not in known]
        if missing:
            raise ValueError(f"Candidate references missing values: {missing}")
        node_id = node_id or f"n{len(self.nodes):04d}"
        if node_id in known:
            raise ValueError(f"Duplicate graph value: {node_id}")
        node = candidate.to_node(node_id)
        self.nodes.append(node)
        self.outputs = (node_id,)
        return node

    def replace_node(self, index: int, node: LayerNode) -> None:
        old = self.nodes[index]
        if old.node_id != node.node_id:
            raise ValueError("Replacement node must preserve node_id")
        self.nodes[index] = node

    def validate(self) -> list[str]:
        errors: list[str] = []
        known: dict[str, TensorSpec] = dict(self.inputs)
        for node in self.nodes:
            if node.node_id in known:
                errors.append(f"duplicate value id: {node.node_id}")
            for ref in node.inputs:
                if ref not in known:
                    errors.append(f"{node.node_id} references unknown/non-topological input {ref}")
            known[node.node_id] = node.output_spec
        for ref in self.outputs:
            if ref not in known:
                errors.append(f"unknown output reference: {ref}")
        return errors

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        return {
            "inputs": {k: v.to_dict() for k, v in sorted(self.inputs.items())},
            "nodes": [n.to_dict(include_arrays=include_arrays) for n in self.nodes],
            "outputs": list(self.outputs),
            "metadata": _jsonable(self.metadata, include_arrays=include_arrays),
        }


    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ModelGraph":
        graph = cls(
            inputs={str(k): TensorSpec.from_dict(v) for k, v in dict(raw.get("inputs", {})).items()},
            nodes=[LayerNode.from_dict(item) for item in raw.get("nodes", [])],
            outputs=tuple(str(x) for x in raw.get("outputs", [])),
            metadata=copy.deepcopy(dict(raw.get("metadata", {}))),
        )
        errors = graph.validate()
        if errors:
            raise ValueError("Invalid serialized graph: " + "; ".join(errors))
        return graph

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(include_arrays=False), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def iter_edges(self) -> Iterable[tuple[str, str]]:
        for node in self.nodes:
            for source in node.inputs:
                yield self.source_op(source), node.op
