"""Optional PyTorch adapter for same-case differential execution."""

from __future__ import annotations

from math import sqrt
from time import perf_counter
from typing import Mapping

import numpy as np

from .base import Backend, BackendUnavailable
from ..execution import ExecutionResult, TraceEntry
from ..model_ir import LayerNode, ModelGraph, normalize_dtype


class TorchBackend(Backend):
    name = "pytorch"

    def __init__(self, *, device: str = "cpu", deterministic: bool = True) -> None:
        try:
            import torch
        except ImportError as exc:
            raise BackendUnavailable("PyTorch is not installed; install requirements-pytorch.txt") from exc
        self.torch = torch
        self.device = torch.device(device)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

    @property
    def version(self) -> str:
        return str(self.torch.__version__)

    def _dtype(self, dtype: str):
        torch = self.torch
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool,
        }
        key = normalize_dtype(dtype)
        if key not in mapping:
            raise ValueError(f"Unsupported PyTorch dtype {dtype}")
        return mapping[key]

    def _weight(self, node: LayerNode, name: str, dtype):
        return self.torch.as_tensor(node.weights[name], dtype=dtype, device=self.device)

    def execute(
        self,
        graph: ModelGraph,
        inputs: Mapping[str, np.ndarray],
        *,
        contract_valid: bool = True,
        expected_rejection: bool = False,
    ) -> ExecutionResult:
        torch = self.torch
        started = perf_counter()
        try:
            values = {
                name: torch.as_tensor(value, dtype=self._dtype(graph.inputs[name].dtype), device=self.device)
                for name, value in inputs.items()
            }
            traces: list[TraceEntry] = []
            with torch.no_grad():
                for node in graph.nodes:
                    args = [values[ref] for ref in node.inputs]
                    output = self._execute_node(node, args)
                    if tuple(output.shape) != tuple(node.output_spec.shape):
                        raise ValueError(f"{node.node_id}/{node.op}: expected {node.output_spec.shape}, got {tuple(output.shape)}")
                    output = output.to(self._dtype(node.output_spec.dtype))
                    values[node.node_id] = output
                    traces.append(TraceEntry.from_array(node.node_id, node.op, output.detach().cpu().numpy()))
            refs = graph.outputs or ((graph.nodes[-1].node_id,) if graph.nodes else tuple(graph.inputs))
            outputs = {ref: values[ref].detach().cpu().numpy() for ref in refs}
            return ExecutionResult(
                backend=self.name,
                backend_version=self.version,
                status="ok",
                outputs=outputs,
                traces=traces,
                elapsed_seconds=perf_counter() - started,
                contract_valid=contract_valid,
                expected_rejection=expected_rejection,
                metadata={"device": str(self.device)},
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
        torch = self.torch
        started = perf_counter()
        try:
            if set(inputs) != {"input"}:
                raise ValueError("PyTorch gradient adapter supports one input named 'input'")
            spec = graph.inputs["input"]
            if not normalize_dtype(spec.dtype).startswith("float") and normalize_dtype(spec.dtype) != "bfloat16":
                raise ValueError("input gradients require a floating input")
            if np.asarray(inputs["input"]).size > max_elements:
                raise ValueError("input exceeds max_elements")
            x = torch.as_tensor(inputs["input"], dtype=self._dtype(spec.dtype), device=self.device).clone().detach().requires_grad_(True)
            values = {"input": x}
            for node in graph.nodes:
                values[node.node_id] = self._execute_node(node, [values[ref] for ref in node.inputs])
            refs = graph.outputs or ((graph.nodes[-1].node_id,) if graph.nodes else ("input",))
            scalar = sum(values[ref].sum() for ref in refs)
            scalar.backward()
            gradient = x.grad.detach().cpu().numpy()
            return ExecutionResult(
                backend=self.name, backend_version=self.version, status="ok",
                outputs={"input_gradient": gradient},
                traces=[TraceEntry.from_array("input_gradient", "autodiff", gradient)],
                elapsed_seconds=perf_counter() - started,
                metadata={"mode": "input_gradient", "method": "autodiff", "device": str(self.device)},
            )
        except BaseException as exc:
            return ExecutionResult.from_exception(
                self.name, self.version, exc, elapsed_seconds=perf_counter() - started,
                contract_valid=True, expected_rejection=False,
            )

    def _activation(self, name: str, x):
        torch = self.torch
        if name in {"linear", "identity", "none", "None", ""}:
            return x
        if name == "relu":
            return torch.relu(x)
        if name == "tanh":
            return torch.tanh(x)
        if name == "sigmoid":
            return torch.sigmoid(x)
        raise ValueError(f"Unsupported activation {name}")

    def _execute_node(self, node: LayerNode, args):
        torch = self.torch
        import torch.nn.functional as F

        op = node.op
        x = args[0]
        if op == "identity":
            return x.clone()
        if op == "dense":
            kernel = self._weight(node, "kernel", x.dtype)
            y = torch.matmul(x, kernel)
            if node.params.get("use_bias", True):
                y = y + self._weight(node, "bias", x.dtype)
            return self._activation(str(node.params.get("activation", "linear")), y)
        if op == "relu":
            return torch.relu(x)
        if op == "leaky_relu":
            return F.leaky_relu(x, negative_slope=float(node.params.get("negative_slope", 0.2)))
        if op == "tanh":
            return torch.tanh(x)
        if op == "sigmoid":
            return torch.sigmoid(x)
        if op == "softmax":
            return torch.softmax(x, dim=int(node.params.get("axis", -1)))
        if op == "layer_norm":
            return F.layer_norm(x, (x.shape[-1],), eps=float(node.params.get("epsilon", 1e-5)))
        if op == "flatten":
            return torch.flatten(x, start_dim=1)
        if op == "reshape":
            return torch.reshape(x, tuple(int(v) for v in node.params["target_shape"]))
        if op == "transpose":
            return x.permute(tuple(int(v) for v in node.params["perm"]))
        if op == "conv2d_1x1":
            kernel = self._weight(node, "kernel", x.dtype).transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
            y = F.conv2d(x.permute(0, 3, 1, 2), kernel, bias=None, stride=1, padding=0)
            y = y.permute(0, 2, 3, 1)
            if node.params.get("use_bias", True):
                y = y + self._weight(node, "bias", x.dtype)
            return y
        if op == "max_pool2d":
            y = F.max_pool2d(x.permute(0, 3, 1, 2), kernel_size=2, stride=2)
            return y.permute(0, 2, 3, 1)
        if op == "global_avg_pool2d":
            return x.mean(dim=(1, 2))
        if op == "simple_rnn":
            kernel = self._weight(node, "kernel", x.dtype)
            recurrent = self._weight(node, "recurrent_kernel", x.dtype)
            bias = self._weight(node, "bias", x.dtype)
            state = torch.zeros((x.shape[0], recurrent.shape[0]), dtype=x.dtype, device=x.device)
            outputs = []
            activation = str(node.params.get("activation", "tanh"))
            for step in range(x.shape[1]):
                state = self._activation(activation, x[:, step, :] @ kernel + state @ recurrent + bias)
                outputs.append(state)
            return torch.stack(outputs, dim=1)
        if op == "attention":
            q = x @ self._weight(node, "wq", x.dtype)
            k = x @ self._weight(node, "wk", x.dtype)
            v = x @ self._weight(node, "wv", x.dtype)
            scores = q @ k.transpose(-1, -2)
            if node.params.get("scale", True):
                scores = scores / sqrt(max(1, q.shape[-1]))
            attention = torch.softmax(scores, dim=-1)
            return (attention @ v) @ self._weight(node, "wo", x.dtype)
        if op == "embedding":
            table = self._weight(node, "embeddings", self._dtype("float32"))
            return F.embedding(x.to(torch.int64), table)
        if op == "graph_conv":
            adjacency = self._weight(node, "adjacency", x.dtype)
            aggregated = torch.einsum("ij,bjf->bif", adjacency, x)
            y = aggregated @ self._weight(node, "kernel", x.dtype)
            return self._activation(str(node.params.get("activation", "linear")), y)
        if op == "add":
            return args[0] + args[1]
        if op == "concat":
            return torch.cat(args, dim=int(node.params.get("axis", -1)))
        raise ValueError(f"Unsupported operation {op}")
