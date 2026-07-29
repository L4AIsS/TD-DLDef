"""Constraint-aware candidate layer catalogue.

The catalogue produces only shape- and dtype-compatible candidates.  This removes API
contract violations from the default search path and makes robustness violations an
explicit, separately labelled mutation mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import prod
from typing import Any, Iterable, Sequence

import numpy as np

from .model_ir import LayerCandidate, ModelGraph, TensorSpec, normalize_dtype

FLOAT_DTYPES = {"float16", "bfloat16", "float32", "float64"}
INT_DTYPES = {"int8", "int16", "int32", "int64", "uint8"}


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    input_spec: TensorSpec
    distribution: str = "normal"
    metadata: dict[str, Any] = field(default_factory=dict)


def built_in_task(name: str, dtype: str | None = None) -> TaskSpec:
    key = name.strip().lower()
    if key in {"vision", "image", "cnn"}:
        return TaskSpec("vision", TensorSpec((1, 8, 8, 3), dtype or "float32", "image"), "normal")
    if key in {"vector", "mlp"}:
        return TaskSpec("vector", TensorSpec((1, 32), dtype or "float32", "vector"), "normal")
    if key in {"sequence", "transformer", "nonvision", "nlp"}:
        return TaskSpec("transformer", TensorSpec((1, 8, 16), dtype or "float32", "sequence"), "normal")
    if key in {"text", "embedding"}:
        return TaskSpec("text", TensorSpec((1, 8), dtype or "int32", "token_ids"), "integers", {"vocab_size": 64})
    if key in {"graph", "gnn"}:
        return TaskSpec("graph", TensorSpec((1, 8, 16), dtype or "float32", "node_features"), "normal", {"nodes": 8})
    raise ValueError(f"Unknown task {name!r}")


def _normal(rng: np.random.Generator, shape: Sequence[int], scale: float = 0.1) -> np.ndarray:
    return rng.normal(0.0, scale, size=tuple(shape)).astype(np.float64)


def _arm_key(op: str, graph: ModelGraph, inputs: tuple[str, ...], granularity: str) -> str:
    if granularity == "layer_type":
        return op
    if granularity == "contextual":
        sources = "+".join(graph.source_op(ref) for ref in inputs)
        ndims = "+".join(str(graph.get_spec(ref).ndim) for ref in inputs)
        return f"{op}|{sources}|d{ndims}"
    raise ValueError(f"Unknown arm granularity: {granularity}")


class LayerCatalog:
    """Generate concrete candidates from a model prefix."""

    def __init__(
        self,
        *,
        task: TaskSpec,
        enabled_ops: Iterable[str] | None = None,
        candidate_window: int = 5,
        max_candidates: int = 48,
        arm_granularity: str = "layer_type",
    ) -> None:
        self.task = task
        self.enabled_ops = set(enabled_ops or {
            "identity",
            "dense",
            "relu",
            "tanh",
            "sigmoid",
            "leaky_relu",
            "softmax",
            "layer_norm",
            "flatten",
            "reshape",
            "transpose",
            "conv2d_1x1",
            "max_pool2d",
            "global_avg_pool2d",
            "simple_rnn",
            "attention",
            "embedding",
            "graph_conv",
            "add",
            "concat",
        })
        self.candidate_window = max(1, int(candidate_window))
        self.max_candidates = max(1, int(max_candidates))
        self.arm_granularity = arm_granularity

    def enumerate(self, graph: ModelGraph, rng: np.random.Generator) -> list[LayerCandidate]:
        refs = list(graph.values)[-self.candidate_window :]
        candidates: list[LayerCandidate] = []
        for ref in refs:
            candidates.extend(self._unary_candidates(graph, ref, rng))
        candidates.extend(self._multi_input_candidates(graph, refs, rng))

        # Deduplicate exact candidates, then cap with a deterministic RNG permutation.
        unique: dict[str, LayerCandidate] = {c.candidate_id: c for c in candidates}
        result = list(unique.values())
        if len(result) > self.max_candidates:
            indices = rng.choice(len(result), size=self.max_candidates, replace=False)
            result = [result[int(i)] for i in sorted(indices)]
        return result

    def _candidate(
        self,
        graph: ModelGraph,
        op: str,
        inputs: tuple[str, ...],
        output: TensorSpec,
        *,
        params: dict[str, Any] | None = None,
        weights: dict[str, np.ndarray] | None = None,
        tags: set[str] | None = None,
    ) -> LayerCandidate:
        return LayerCandidate(
            op=op,
            inputs=inputs,
            input_specs=tuple(graph.get_spec(ref) for ref in inputs),
            output_spec=output,
            params=params or {},
            weights=weights or {},
            tags=tags or set(),
            arm_key=_arm_key(op, graph, inputs, self.arm_granularity),
        )

    def _unary_candidates(self, graph: ModelGraph, ref: str, rng: np.random.Generator) -> list[LayerCandidate]:
        spec = graph.get_spec(ref)
        dtype = normalize_dtype(spec.dtype)
        out: list[LayerCandidate] = []

        if "identity" in self.enabled_ops:
            out.append(self._candidate(graph, "identity", (ref,), spec))

        if dtype in FLOAT_DTYPES:
            for op in ("relu", "tanh", "sigmoid", "softmax"):
                if op in self.enabled_ops:
                    params = {"axis": -1} if op == "softmax" else {}
                    tags = {"nondifferentiable"} if op == "relu" else set()
                    out.append(self._candidate(graph, op, (ref,), spec, params=params, tags=tags))
            if "leaky_relu" in self.enabled_ops:
                alpha = float(rng.choice([0.01, 0.1, 0.2]))
                out.append(self._candidate(graph, "leaky_relu", (ref,), spec, params={"negative_slope": alpha}, tags={"nondifferentiable"}))
            if "layer_norm" in self.enabled_ops:
                epsilon = float(rng.choice([1e-5, 1e-4, 1e-3]))
                out.append(self._candidate(graph, "layer_norm", (ref,), spec, params={"epsilon": epsilon, "axis": -1}))

        if dtype in FLOAT_DTYPES and spec.ndim >= 2 and "dense" in self.enabled_ops:
            in_features = spec.shape[-1]
            units = int(rng.choice([4, 8, 16, 32]))
            output = TensorSpec(spec.shape[:-1] + (units,), spec.dtype, spec.semantic)
            weights = {
                "kernel": _normal(rng, (in_features, units)),
                "bias": np.zeros((units,), dtype=np.float64),
            }
            params = {
                "units": units,
                "use_bias": True,
                "activation": str(rng.choice(["linear", "relu", "tanh"])),
            }
            out.append(self._candidate(graph, "dense", (ref,), output, params=params, weights=weights))

        if spec.ndim > 2 and "flatten" in self.enabled_ops:
            output = TensorSpec((spec.shape[0], prod(spec.shape[1:])), spec.dtype, "vector")
            out.append(self._candidate(graph, "flatten", (ref,), output, params={"preserve_batch": True}))

        if spec.ndim == 2 and spec.shape[1] % 2 == 0 and "reshape" in self.enabled_ops:
            output = TensorSpec((spec.shape[0], 2, spec.shape[1] // 2), spec.dtype, spec.semantic)
            out.append(self._candidate(graph, "reshape", (ref,), output, params={"target_shape": list(output.shape)}))
        elif spec.ndim >= 3 and "reshape" in self.enabled_ops:
            output = TensorSpec((spec.shape[0], prod(spec.shape[1:])), spec.dtype, "vector")
            out.append(self._candidate(graph, "reshape", (ref,), output, params={"target_shape": list(output.shape)}))

        if spec.ndim >= 3 and "transpose" in self.enabled_ops:
            perm = list(range(spec.ndim))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            output_shape = tuple(spec.shape[i] for i in perm)
            out.append(self._candidate(graph, "transpose", (ref,), TensorSpec(output_shape, spec.dtype, spec.semantic), params={"perm": perm}))

        if dtype in FLOAT_DTYPES and spec.ndim == 4:
            batch, height, width, channels = spec.shape
            if "conv2d_1x1" in self.enabled_ops:
                filters = int(rng.choice([4, 8, 16]))
                output = TensorSpec((batch, height, width, filters), spec.dtype, "feature_map")
                weights = {
                    "kernel": _normal(rng, (channels, filters)),
                    "bias": np.zeros((filters,), dtype=np.float64),
                }
                params = {"filters": filters, "kernel_size": [1, 1], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "use_bias": True}
                out.append(self._candidate(graph, "conv2d_1x1", (ref,), output, params=params, weights=weights))
            if height >= 2 and width >= 2 and "max_pool2d" in self.enabled_ops:
                output = TensorSpec((batch, height // 2, width // 2, channels), spec.dtype, "feature_map")
                params = {"pool_size": [2, 2], "strides": [2, 2], "padding": "valid", "data_format": "channels_last"}
                out.append(self._candidate(graph, "max_pool2d", (ref,), output, params=params, tags={"nondifferentiable"}))
            if "global_avg_pool2d" in self.enabled_ops:
                output = TensorSpec((batch, channels), spec.dtype, "vector")
                out.append(self._candidate(graph, "global_avg_pool2d", (ref,), output, params={"data_format": "channels_last"}))

        if dtype in FLOAT_DTYPES and spec.ndim == 3:
            batch, steps, features = spec.shape
            if "simple_rnn" in self.enabled_ops:
                units = int(rng.choice([4, 8, 16]))
                output = TensorSpec((batch, steps, units), spec.dtype, "sequence")
                weights = {
                    "kernel": _normal(rng, (features, units)),
                    "recurrent_kernel": _normal(rng, (units, units)),
                    "bias": np.zeros((units,), dtype=np.float64),
                }
                params = {"units": units, "activation": str(rng.choice(["tanh", "relu"])), "return_sequences": True, "recurrent_dropout": 0.0}
                out.append(self._candidate(graph, "simple_rnn", (ref,), output, params=params, weights=weights, tags={"stateful_formula"}))
            if "attention" in self.enabled_ops and features <= 64:
                weights = {
                    "wq": _normal(rng, (features, features)),
                    "wk": _normal(rng, (features, features)),
                    "wv": _normal(rng, (features, features)),
                    "wo": _normal(rng, (features, features)),
                }
                params = {"num_heads": 1, "key_dim": features, "scale": True}
                out.append(self._candidate(graph, "attention", (ref,), spec, params=params, weights=weights))
            if self.task.name == "graph" and "graph_conv" in self.enabled_ops:
                units = int(rng.choice([4, 8, 16]))
                output = TensorSpec((batch, steps, units), spec.dtype, "node_features")
                adjacency = np.eye(steps, dtype=np.float64)
                for i in range(steps):
                    adjacency[i, (i + 1) % steps] = 1.0
                    adjacency[(i + 1) % steps, i] = 1.0
                degree = adjacency.sum(axis=1, keepdims=True)
                adjacency /= np.maximum(degree, 1.0)
                weights = {"kernel": _normal(rng, (features, units)), "adjacency": adjacency}
                out.append(self._candidate(graph, "graph_conv", (ref,), output, params={"units": units, "activation": "relu"}, weights=weights))

        if dtype in INT_DTYPES and spec.ndim == 2 and "embedding" in self.enabled_ops:
            vocab = int(self.task.metadata.get("vocab_size", 64))
            dim = int(rng.choice([8, 16, 32]))
            output = TensorSpec(spec.shape + (dim,), "float32", "sequence")
            weights = {"embeddings": _normal(rng, (vocab, dim))}
            out.append(self._candidate(graph, "embedding", (ref,), output, params={"input_dim": vocab, "output_dim": dim}, weights=weights))

        return out

    def _multi_input_candidates(self, graph: ModelGraph, refs: list[str], rng: np.random.Generator) -> list[LayerCandidate]:
        del rng  # currently all multi-input candidates are deterministic
        out: list[LayerCandidate] = []
        for left, right in combinations(refs, 2):
            a, b = graph.get_spec(left), graph.get_spec(right)
            if a.dtype != b.dtype:
                continue
            if "add" in self.enabled_ops and a.shape == b.shape:
                out.append(self._candidate(graph, "add", (left, right), a))
            if "concat" in self.enabled_ops and a.ndim == b.ndim and a.shape[:-1] == b.shape[:-1]:
                output = TensorSpec(a.shape[:-1] + (a.shape[-1] + b.shape[-1],), a.dtype, a.semantic)
                out.append(self._candidate(graph, "concat", (left, right), output, params={"axis": -1}))
        return out

    def theoretical_layer_types(self) -> set[str]:
        return set(self.enabled_ops)
