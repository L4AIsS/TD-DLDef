"""Versioned parameter and dtype constraints for contract-aware mutation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping
import copy
import json
from pathlib import Path

from .model_ir import ModelGraph, normalize_dtype


@dataclass(frozen=True, slots=True)
class ParamConstraint:
    kind: str
    choices: tuple[Any, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    min_inclusive: bool = True
    max_inclusive: bool = True
    structural: bool = False
    description: str = ""

    def accepts(self, value: Any) -> bool:
        if self.choices and value not in self.choices:
            return False
        if self.kind == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            return False
        if self.kind == "float" and not isinstance(value, (int, float)):
            return False
        if self.minimum is not None:
            if self.min_inclusive and value < self.minimum:
                return False
            if not self.min_inclusive and value <= self.minimum:
                return False
        if self.maximum is not None:
            if self.max_inclusive and value > self.maximum:
                return False
            if not self.max_inclusive and value >= self.maximum:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "choices": list(self.choices),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "min_inclusive": self.min_inclusive,
            "max_inclusive": self.max_inclusive,
            "structural": self.structural,
            "description": self.description,
        }


@dataclass(slots=True)
class ConstraintRegistry:
    version: str
    framework: str = "generic"
    framework_version: str = "unspecified"
    params: dict[str, dict[str, ParamConstraint]] = field(default_factory=dict)
    supported_dtypes: set[str] = field(default_factory=lambda: {"float16", "float32", "float64", "int32", "int64"})
    layer_dtype_overrides: dict[str, set[str]] = field(default_factory=dict)

    def get(self, op: str, param: str) -> ParamConstraint | None:
        return self.params.get(op, {}).get(param)

    def legal_values(self, op: str, param: str) -> tuple[Any, ...]:
        constraint = self.get(op, param)
        return constraint.choices if constraint else ()

    def supports_dtype(self, op: str, dtype: str) -> bool:
        dtype = normalize_dtype(dtype)
        allowed = self.layer_dtype_overrides.get(op, self.supported_dtypes)
        return dtype in allowed

    def validate_param(self, op: str, param: str, value: Any) -> bool:
        constraint = self.get(op, param)
        return True if constraint is None else constraint.accepts(value)

    def validate_graph(self, graph: ModelGraph) -> list[str]:
        errors = graph.validate()
        for input_name, spec in graph.inputs.items():
            if normalize_dtype(spec.dtype) not in self.supported_dtypes:
                errors.append(f"input {input_name}: unsupported dtype {spec.dtype}")
        for node in graph.nodes:
            for name, value in node.params.items():
                constraint = self.get(node.op, name)
                if constraint and not constraint.accepts(value):
                    errors.append(f"{node.node_id}/{node.op}.{name}: invalid value {value!r}")
            for ref in node.inputs:
                dtype = graph.get_spec(ref).dtype
                if not self.supports_dtype(node.op, dtype):
                    errors.append(f"{node.node_id}/{node.op}: unsupported input dtype {dtype}")
        return errors

    def clone(self) -> "ConstraintRegistry":
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "supported_dtypes": sorted(self.supported_dtypes),
            "layer_dtype_overrides": {k: sorted(v) for k, v in sorted(self.layer_dtype_overrides.items())},
            "params": {
                op: {name: c.to_dict() for name, c in sorted(mapping.items())}
                for op, mapping in sorted(self.params.items())
            },
        }

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def default_registry(framework: str = "generic", framework_version: str = "unspecified") -> ConstraintRegistry:
    """Built-in constraints used by the executable reference implementation.

    Framework-specific artifact runs should export and review a registry for the exact
    version under test.  The version fields are recorded in every experiment result.
    """

    p: dict[str, dict[str, ParamConstraint]] = {
        "dense": {
            "units": ParamConstraint("integer", minimum=1, maximum=256, structural=True),
            "use_bias": ParamConstraint("enum", choices=(True, False)),
            "activation": ParamConstraint("enum", choices=("linear", "relu", "tanh", "sigmoid")),
        },
        "conv2d_1x1": {
            "filters": ParamConstraint("integer", minimum=1, maximum=128, structural=True),
            "kernel_size": ParamConstraint("enum", choices=([1, 1],), structural=True),
            "padding": ParamConstraint("enum", choices=("same", "valid"), structural=True),
            "data_format": ParamConstraint("enum", choices=("channels_last",), structural=True),
            "use_bias": ParamConstraint("enum", choices=(True, False)),
        },
        "max_pool2d": {
            "pool_size": ParamConstraint("enum", choices=([2, 2],), structural=True),
            "strides": ParamConstraint("enum", choices=([2, 2],), structural=True),
            "padding": ParamConstraint("enum", choices=("valid",), structural=True),
            "data_format": ParamConstraint("enum", choices=("channels_last",), structural=True),
        },
        "leaky_relu": {
            "negative_slope": ParamConstraint("float", minimum=0.0, maximum=1.0),
        },
        "layer_norm": {
            "epsilon": ParamConstraint("float", minimum=1e-12, maximum=1.0, min_inclusive=False),
            "axis": ParamConstraint("enum", choices=(-1,)),
        },
        "softmax": {"axis": ParamConstraint("enum", choices=(-1,))},
        "simple_rnn": {
            "units": ParamConstraint("integer", minimum=1, maximum=128, structural=True),
            "activation": ParamConstraint("enum", choices=("tanh", "relu")),
            "return_sequences": ParamConstraint("enum", choices=(True,)),
            "recurrent_dropout": ParamConstraint("float", minimum=0.0, maximum=1.0, max_inclusive=False),
        },
        "attention": {
            "num_heads": ParamConstraint("integer", choices=(1,), minimum=1),
            "key_dim": ParamConstraint("integer", minimum=1, maximum=64, structural=True),
            "scale": ParamConstraint("enum", choices=(True, False)),
        },
        "embedding": {
            "input_dim": ParamConstraint("integer", minimum=2, maximum=100000, structural=True),
            "output_dim": ParamConstraint("integer", minimum=1, maximum=256, structural=True),
        },
        "graph_conv": {
            "units": ParamConstraint("integer", minimum=1, maximum=128, structural=True),
            "activation": ParamConstraint("enum", choices=("linear", "relu", "tanh")),
        },
        "reshape": {"target_shape": ParamConstraint("shape", structural=True)},
        "transpose": {"perm": ParamConstraint("permutation", structural=True)},
        "concat": {"axis": ParamConstraint("enum", choices=(-1,))},
    }
    supported = {"float16", "float32", "float64", "int32", "int64"}
    overrides = {
        "embedding": {"int32", "int64"},
        "dense": {"float16", "float32", "float64"},
        "conv2d_1x1": {"float16", "float32", "float64"},
        "max_pool2d": {"float16", "float32", "float64"},
        "simple_rnn": {"float16", "float32", "float64"},
        "attention": {"float16", "float32", "float64"},
        "graph_conv": {"float16", "float32", "float64"},
    }
    return ConstraintRegistry(
        version="td-dldef-constraints-2.0",
        framework=framework,
        framework_version=framework_version,
        params=p,
        supported_dtypes=supported,
        layer_dtype_overrides=overrides,
    )
