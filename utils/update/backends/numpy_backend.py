"""Executable NumPy reference backend used for smoke tests and algorithm ablations."""

from __future__ import annotations

from math import sqrt
from time import perf_counter
from typing import Mapping

import numpy as np

from .base import Backend
from ..execution import ExecutionResult, TraceEntry, numpy_dtype
from ..model_ir import LayerNode, ModelGraph


def _activation(name: str, value: np.ndarray) -> np.ndarray:
    if name in {"linear", "identity", "none", "None", ""}:
        return value
    if name == "relu":
        return np.maximum(value, 0)
    if name == "tanh":
        return np.tanh(value)
    if name == "sigmoid":
        clipped = np.clip(value, -80.0, 80.0)
        return 1.0 / (1.0 + np.exp(-clipped))
    raise ValueError(f"Unsupported activation {name!r}")


def _softmax(value: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = value - np.max(value, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


class NumpyBackend(Backend):
    name = "numpy"

    @property
    def version(self) -> str:
        return np.__version__

    def execute(
        self,
        graph: ModelGraph,
        inputs: Mapping[str, np.ndarray],
        *,
        contract_valid: bool = True,
        expected_rejection: bool = False,
    ) -> ExecutionResult:
        started = perf_counter()
        try:
            values = {name: np.asarray(value) for name, value in inputs.items()}
            traces: list[TraceEntry] = []
            for node in graph.nodes:
                args = [values[ref] for ref in node.inputs]
                output = self._execute_node(node, args)
                expected_shape = tuple(node.output_spec.shape)
                if tuple(output.shape) != expected_shape:
                    raise ValueError(f"{node.node_id}/{node.op}: expected shape {expected_shape}, got {tuple(output.shape)}")
                target_dtype = numpy_dtype(node.output_spec.dtype)
                if output.dtype != target_dtype:
                    output = output.astype(target_dtype)
                values[node.node_id] = output
                traces.append(TraceEntry.from_array(node.node_id, node.op, output))
            output_refs = graph.outputs or ((graph.nodes[-1].node_id,) if graph.nodes else tuple(graph.inputs))
            outputs = {ref: values[ref] for ref in output_refs}
            return ExecutionResult(
                backend=self.name,
                backend_version=self.version,
                status="ok",
                outputs=outputs,
                traces=traces,
                elapsed_seconds=perf_counter() - started,
                contract_valid=contract_valid,
                expected_rejection=expected_rejection,
            )
        except BaseException as exc:
            return ExecutionResult.from_exception(
                self.name,
                self.version,
                exc,
                elapsed_seconds=perf_counter() - started,
                contract_valid=contract_valid,
                expected_rejection=expected_rejection,
            )


    def input_gradient(
        self,
        graph: ModelGraph,
        inputs: Mapping[str, np.ndarray],
        *,
        epsilon: float = 1e-4,
        max_elements: int = 256,
    ) -> ExecutionResult:
        started = perf_counter()
        try:
            if set(inputs) != {"input"}:
                raise ValueError("NumPy finite-difference gradient currently supports one input named 'input'")
            original = np.asarray(inputs["input"], dtype=np.float64)
            if original.size > max_elements:
                raise ValueError(f"input has {original.size} elements, exceeds max_elements={max_elements}")
            gradient = np.empty_like(original, dtype=np.float64)
            for flat_index in range(original.size):
                plus = original.copy().reshape(-1)
                minus = original.copy().reshape(-1)
                plus[flat_index] += epsilon
                minus[flat_index] -= epsilon
                plus_result = self.execute(graph, {"input": plus.reshape(original.shape)})
                minus_result = self.execute(graph, {"input": minus.reshape(original.shape)})
                if plus_result.status != "ok" or minus_result.status != "ok":
                    raise RuntimeError("forward execution failed during finite differences")
                plus_scalar = sum(float(np.sum(v, dtype=np.float64)) for v in plus_result.outputs.values())
                minus_scalar = sum(float(np.sum(v, dtype=np.float64)) for v in minus_result.outputs.values())
                gradient.reshape(-1)[flat_index] = (plus_scalar - minus_scalar) / (2.0 * epsilon)
            return ExecutionResult(
                backend=self.name,
                backend_version=self.version,
                status="ok",
                outputs={"input_gradient": gradient},
                traces=[TraceEntry.from_array("input_gradient", "finite_difference", gradient)],
                elapsed_seconds=perf_counter() - started,
                metadata={"mode": "input_gradient", "method": "central_finite_difference", "epsilon": epsilon},
            )
        except BaseException as exc:
            return ExecutionResult.from_exception(
                self.name, self.version, exc, elapsed_seconds=perf_counter() - started,
                contract_valid=True, expected_rejection=False,
            )

    def _execute_node(self, node: LayerNode, args: list[np.ndarray]) -> np.ndarray:
        op = node.op
        x = args[0]
        if op == "identity":
            return np.array(x, copy=True)
        if op == "dense":
            y = np.tensordot(x, node.weights["kernel"], axes=([-1], [0]))
            if node.params.get("use_bias", True):
                y = y + node.weights["bias"]
            return _activation(str(node.params.get("activation", "linear")), y)
        if op == "relu":
            return np.maximum(x, 0)
        if op == "leaky_relu":
            slope = float(node.params.get("negative_slope", 0.2))
            return np.where(x >= 0, x, slope * x)
        if op == "tanh":
            return np.tanh(x)
        if op == "sigmoid":
            return _activation("sigmoid", x)
        if op == "softmax":
            return _softmax(x, int(node.params.get("axis", -1)))
        if op == "layer_norm":
            epsilon = float(node.params.get("epsilon", 1e-5))
            mean = np.mean(x, axis=-1, keepdims=True)
            variance = np.var(x, axis=-1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                return (x - mean) / np.sqrt(variance + epsilon)
        if op == "flatten":
            return x.reshape((x.shape[0], -1))
        if op == "reshape":
            return x.reshape(tuple(int(v) for v in node.params["target_shape"]))
        if op == "transpose":
            return np.transpose(x, axes=tuple(int(v) for v in node.params["perm"]))
        if op == "conv2d_1x1":
            y = np.tensordot(x, node.weights["kernel"], axes=([-1], [0]))
            if node.params.get("use_bias", True):
                y = y + node.weights["bias"]
            return y
        if op == "max_pool2d":
            n, h, w, c = x.shape
            if h % 2 or w % 2:
                x = x[:, : h - (h % 2), : w - (w % 2), :]
                n, h, w, c = x.shape
            return x.reshape(n, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))
        if op == "global_avg_pool2d":
            return np.mean(x, axis=(1, 2))
        if op == "simple_rnn":
            kernel = node.weights["kernel"]
            recurrent = node.weights["recurrent_kernel"]
            bias = node.weights["bias"]
            state = np.zeros((x.shape[0], recurrent.shape[0]), dtype=np.float64)
            outputs = []
            activation = str(node.params.get("activation", "tanh"))
            for step in range(x.shape[1]):
                state = _activation(activation, x[:, step, :] @ kernel + state @ recurrent + bias)
                outputs.append(state)
            return np.stack(outputs, axis=1)
        if op == "attention":
            q = x @ node.weights["wq"]
            k = x @ node.weights["wk"]
            v = x @ node.weights["wv"]
            scores = q @ np.swapaxes(k, -1, -2)
            if node.params.get("scale", True):
                scores = scores / sqrt(max(1, q.shape[-1]))
            attention = _softmax(scores, axis=-1)
            return (attention @ v) @ node.weights["wo"]
        if op == "embedding":
            indices = np.asarray(x, dtype=np.int64)
            table = node.weights["embeddings"]
            if np.any(indices < 0) or np.any(indices >= table.shape[0]):
                raise ValueError("embedding index outside [0, input_dim)")
            return table[indices]
        if op == "graph_conv":
            adjacency = node.weights["adjacency"]
            aggregated = np.einsum("ij,bjf->bif", adjacency, x)
            y = aggregated @ node.weights["kernel"]
            return _activation(str(node.params.get("activation", "linear")), y)
        if op == "add":
            return args[0] + args[1]
        if op == "concat":
            return np.concatenate(args, axis=int(node.params.get("axis", -1)))
        raise ValueError(f"Unsupported operation: {op}")
