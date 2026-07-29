"""Execution result types and deterministic input generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import hashlib
import traceback

import numpy as np

from .catalog import TaskSpec
from .model_ir import TensorSpec, normalize_dtype


def numpy_dtype(dtype: str) -> np.dtype:
    name = normalize_dtype(dtype)
    # NumPy builds without bfloat16 execute it as float32 while retaining the logical
    # dtype in the TensorSpec. Framework adapters use native bfloat16 where available.
    if name == "bfloat16":
        return np.dtype("float32")
    try:
        return np.dtype(name)
    except TypeError as exc:
        raise ValueError(f"Unsupported NumPy dtype: {dtype}") from exc


def generate_inputs(task: TaskSpec, rng: np.random.Generator) -> dict[str, np.ndarray]:
    spec = task.input_spec
    dtype = numpy_dtype(spec.dtype)
    if task.distribution == "integers":
        high = int(task.metadata.get("vocab_size", 64))
        value = rng.integers(0, high, size=spec.shape, dtype=np.int64).astype(dtype)
    elif task.distribution == "uniform":
        value = rng.uniform(-1.0, 1.0, size=spec.shape).astype(dtype)
    else:
        value = rng.normal(0.0, 1.0, size=spec.shape).astype(dtype)
    return {"input": value}


def array_digest(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


@dataclass(slots=True)
class TraceEntry:
    node_id: str
    op: str
    shape: tuple[int, ...]
    dtype: str
    minimum: float | None
    maximum: float | None
    mean: float | None
    std: float | None
    nan_count: int
    inf_count: int
    digest: str
    branch_signature: tuple[int, int, int]

    @classmethod
    def from_array(cls, node_id: str, op: str, value: np.ndarray) -> "TraceEntry":
        arr = np.asarray(value)
        finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.number) else np.array([])
        if finite.size:
            minimum = float(np.min(finite))
            maximum = float(np.max(finite))
            mean = float(np.mean(finite))
            std = float(np.std(finite))
        else:
            minimum = maximum = mean = std = None
        if np.issubdtype(arr.dtype, np.number):
            nan_count = int(np.isnan(arr).sum())
            inf_count = int(np.isinf(arr).sum())
            branch_signature = (
                int((arr < 0).sum()),
                int((arr == 0).sum()),
                int((arr > 0).sum()),
            )
        else:
            nan_count = inf_count = 0
            branch_signature = (0, 0, int(arr.size))
        return cls(
            node_id=node_id,
            op=op,
            shape=tuple(int(x) for x in arr.shape),
            dtype=str(arr.dtype),
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            std=std,
            nan_count=nan_count,
            inf_count=inf_count,
            digest=array_digest(arr),
            branch_signature=branch_signature,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "op": self.op,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "std": self.std,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "digest": self.digest,
            "branch_signature": list(self.branch_signature),
        }


@dataclass(slots=True)
class ExecutionResult:
    backend: str
    backend_version: str
    status: str
    outputs: dict[str, np.ndarray] = field(default_factory=dict)
    traces: list[TraceEntry] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    exception_type: str | None = None
    exception_message: str | None = None
    traceback_text: str | None = None
    contract_valid: bool = True
    expected_rejection: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_exception(
        cls,
        backend: str,
        backend_version: str,
        exc: BaseException,
        *,
        elapsed_seconds: float,
        contract_valid: bool,
        expected_rejection: bool,
    ) -> "ExecutionResult":
        return cls(
            backend=backend,
            backend_version=backend_version,
            status="exception",
            elapsed_seconds=elapsed_seconds,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback_text="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            contract_valid=contract_valid,
            expected_rejection=expected_rejection,
        )

    def to_dict(self, *, include_outputs: bool = False) -> dict[str, Any]:
        output_summary: dict[str, Any] = {}
        for name, value in self.outputs.items():
            arr = np.asarray(value)
            output_summary[name] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "digest": array_digest(arr),
                "values": arr.tolist() if include_outputs and arr.size <= 2048 else None,
            }
        return {
            "backend": self.backend,
            "backend_version": self.backend_version,
            "status": self.status,
            "outputs": output_summary,
            "traces": [trace.to_dict() for trace in self.traces],
            "elapsed_seconds": self.elapsed_seconds,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "contract_valid": self.contract_valid,
            "expected_rejection": self.expected_rejection,
            "metadata": self.metadata,
        }
