"""Optional TensorFlow adapter for same-case differential execution."""

from __future__ import annotations

from math import sqrt
from time import perf_counter
from typing import Mapping

import numpy as np

from .base import Backend, BackendUnavailable
from ..execution import ExecutionResult, TraceEntry
from ..model_ir import LayerNode, ModelGraph, normalize_dtype


class TensorFlowBackend(Backend):
    name = "tensorflow"

    def __init__(self, *, device: str = "/CPU:0", deterministic: bool = True) -> None:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise BackendUnavailable("TensorFlow is not installed; install requirements-tensorflow.txt") from exc
        self.tf = tf
        self.device = device
        if deterministic:
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                pass

    @property
    def version(self) -> str:
        return str(self.tf.__version__)

    def _dtype(self, dtype: str):
        tf = self.tf
        mapping = {
            "float16": tf.float16,
            "bfloat16": tf.bfloat16,
            "float32": tf.float32,
            "float64": tf.float64,
            "int32": tf.int32,
            "int64": tf.int64,
            "bool": tf.bool,
        }
        key = normalize_dtype(dtype)
        if key not in mapping:
            raise ValueError(f"Unsupported TensorFlow dtype {dtype}")
        return mapping[key]

    def _weight(self, node: LayerNode, name: str, dtype):
        return self.tf.convert_to_tensor(node.weights[name], dtype=dtype)

    def execute(
        self,
        graph: ModelGraph,
        inputs: Mapping[str, np.ndarray],
        *,
        contract_valid: bool = True,
        expected_rejection: bool = False,
    ) -> ExecutionResult:
        tf = self.tf
        started = perf_counter()
        try:
            with tf.device(self.device):
                values = {
                    name: tf.convert_to_tensor(value, dtype=self._dtype(graph.inputs[name].dtype))
                    for name, value in inputs.items()
                }
                traces: list[TraceEntry] = []
                for node in graph.nodes:
                    args = [values[ref] for ref in node.inputs]
                    output = self._execute_node(node, args)
                    if tuple(int(x) for x in output.shape) != tuple(node.output_spec.shape):
                        raise ValueError(f"{node.node_id}/{node.op}: expected {node.output_spec.shape}, got {tuple(output.shape)}")
                    output = tf.cast(output, self._dtype(node.output_spec.dtype))
                    values[node.node_id] = output
                    traces.append(TraceEntry.from_array(node.node_id, node.op, output.numpy()))
                refs = graph.outputs or ((graph.nodes[-1].node_id,) if graph.nodes else tuple(graph.inputs))
                outputs = {ref: values[ref].numpy() for ref in refs}
            return ExecutionResult(
                backend=self.name,
                backend_version=self.version,
                status="ok",
                outputs=outputs,
                traces=traces,
                elapsed_seconds=perf_counter() - started,
                contract_valid=contract_valid,
                expected_rejection=expected_rejection,
                metadata={"device": self.device},
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
        del epsilon
        tf = self.tf
        started = perf_counter()
        try:
            if set(inputs) != {"input"}:
                raise ValueError("TensorFlow gradient adapter supports one input named 'input'")
            spec = graph.inputs["input"]
            if not normalize_dtype(spec.dtype).startswith("float") and normalize_dtype(spec.dtype) != "bfloat16":
                raise ValueError("input gradients require a floating input")
            if np.asarray(inputs["input"]).size > max_elements:
                raise ValueError("input exceeds max_elements")
            with tf.device(self.device):
                x = tf.Variable(tf.convert_to_tensor(inputs["input"], dtype=self._dtype(spec.dtype)))
                with tf.GradientTape() as tape:
                    values = {"input": x}
                    for node in graph.nodes:
                        values[node.node_id] = self._execute_node(node, [values[ref] for ref in node.inputs])
                    refs = graph.outputs or ((graph.nodes[-1].node_id,) if graph.nodes else ("input",))
                    scalar = tf.add_n([tf.reduce_sum(values[ref]) for ref in refs])
                gradient_tensor = tape.gradient(scalar, x)
                if gradient_tensor is None:
                    raise RuntimeError("TensorFlow returned no input gradient")
                gradient = gradient_tensor.numpy()
            return ExecutionResult(
                backend=self.name, backend_version=self.version, status="ok",
                outputs={"input_gradient": gradient},
                traces=[TraceEntry.from_array("input_gradient", "autodiff", gradient)],
                elapsed_seconds=perf_counter() - started,
                metadata={"mode": "input_gradient", "method": "autodiff", "device": self.device},
            )
        except BaseException as exc:
            return ExecutionResult.from_exception(
                self.name, self.version, exc, elapsed_seconds=perf_counter() - started,
                contract_valid=True, expected_rejection=False,
            )

    def _activation(self, name: str, x):
        tf = self.tf
        if name in {"linear", "identity", "none", "None", ""}:
            return x
        if name == "relu":
            return tf.nn.relu(x)
        if name == "tanh":
            return tf.math.tanh(x)
        if name == "sigmoid":
            return tf.math.sigmoid(x)
        raise ValueError(f"Unsupported activation {name}")

    def _execute_node(self, node: LayerNode, args):
        tf = self.tf
        op = node.op
        x = args[0]
        if op == "identity":
            return tf.identity(x)
        if op == "dense":
            y = tf.tensordot(x, self._weight(node, "kernel", x.dtype), axes=[[-1], [0]])
            if node.params.get("use_bias", True):
                y = y + self._weight(node, "bias", x.dtype)
            return self._activation(str(node.params.get("activation", "linear")), y)
        if op == "relu":
            return tf.nn.relu(x)
        if op == "leaky_relu":
            return tf.nn.leaky_relu(x, alpha=float(node.params.get("negative_slope", 0.2)))
        if op == "tanh":
            return tf.math.tanh(x)
        if op == "sigmoid":
            return tf.math.sigmoid(x)
        if op == "softmax":
            return tf.nn.softmax(x, axis=int(node.params.get("axis", -1)))
        if op == "layer_norm":
            epsilon = float(node.params.get("epsilon", 1e-5))
            mean, variance = tf.nn.moments(x, axes=[-1], keepdims=True)
            return (x - mean) / tf.sqrt(variance + tf.cast(epsilon, x.dtype))
        if op == "flatten":
            return tf.reshape(x, (tf.shape(x)[0], -1))
        if op == "reshape":
            return tf.reshape(x, tuple(int(v) for v in node.params["target_shape"]))
        if op == "transpose":
            return tf.transpose(x, perm=tuple(int(v) for v in node.params["perm"]))
        if op == "conv2d_1x1":
            kernel = self._weight(node, "kernel", x.dtype)
            kernel = tf.reshape(kernel, (1, 1, kernel.shape[0], kernel.shape[1]))
            y = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME")
            if node.params.get("use_bias", True):
                y = y + self._weight(node, "bias", x.dtype)
            return y
        if op == "max_pool2d":
            return tf.nn.max_pool2d(x, ksize=2, strides=2, padding="VALID", data_format="NHWC")
        if op == "global_avg_pool2d":
            return tf.reduce_mean(x, axis=(1, 2))
        if op == "simple_rnn":
            kernel = self._weight(node, "kernel", x.dtype)
            recurrent = self._weight(node, "recurrent_kernel", x.dtype)
            bias = self._weight(node, "bias", x.dtype)
            state = tf.zeros((tf.shape(x)[0], recurrent.shape[0]), dtype=x.dtype)
            outputs = []
            activation = str(node.params.get("activation", "tanh"))
            for step in range(int(x.shape[1])):
                state = self._activation(activation, x[:, step, :] @ kernel + state @ recurrent + bias)
                outputs.append(state)
            return tf.stack(outputs, axis=1)
        if op == "attention":
            q = x @ self._weight(node, "wq", x.dtype)
            k = x @ self._weight(node, "wk", x.dtype)
            v = x @ self._weight(node, "wv", x.dtype)
            scores = tf.matmul(q, k, transpose_b=True)
            if node.params.get("scale", True):
                scores = scores / tf.cast(sqrt(max(1, int(q.shape[-1]))), x.dtype)
            attention = tf.nn.softmax(scores, axis=-1)
            return (attention @ v) @ self._weight(node, "wo", x.dtype)
        if op == "embedding":
            table = self._weight(node, "embeddings", self._dtype("float32"))
            return tf.gather(table, tf.cast(x, tf.int64))
        if op == "graph_conv":
            adjacency = self._weight(node, "adjacency", x.dtype)
            aggregated = tf.einsum("ij,bjf->bif", adjacency, x)
            y = aggregated @ self._weight(node, "kernel", x.dtype)
            return self._activation(str(node.params.get("activation", "linear")), y)
        if op == "add":
            return tf.add(args[0], args[1])
        if op == "concat":
            return tf.concat(args, axis=int(node.params.get("axis", -1)))
        raise ValueError(f"Unsupported operation {op}")
