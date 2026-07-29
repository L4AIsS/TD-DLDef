"""Constraint-aware value and neuron mutations described in the paper.

Value mutations:
- PV: least-used legal enumerated parameter value;
- BV: power-of-two boundary values, filtered by parameter-specific contracts;
- IT: canonicalised, backend-supported input dtypes.

Neuron mutations:
- NF: Laplace, centred exponential, or sinusoidal noise;
- SW: norm-capped weight scaling;
- WR: shape-preserving orthogonal rotation (or signed permutation for large axes).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Iterable, Sequence
import copy

import numpy as np

from .constraints import ConstraintRegistry, ParamConstraint
from .model_ir import LayerNode, ModelGraph, TensorSpec, normalize_dtype

MUTATION_OPERATORS = ("PV", "BV", "IT", "NF", "SW", "WR")
VALUE_MUTATIONS = ("PV", "BV", "IT")
NEURON_MUTATIONS = ("NF", "SW", "WR")


@dataclass(slots=True)
class MutationRecord:
    operator: str
    node_id: str | None
    target: str
    before: Any
    after: Any
    valid: bool
    expected_rejection: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def compact(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "min": float(np.nanmin(value)) if value.size else None,
                    "max": float(np.nanmax(value)) if value.size else None,
                    "norm": float(np.linalg.norm(value)) if value.size else 0.0,
                }
            if isinstance(value, np.generic):
                return value.item()
            return copy.deepcopy(value)

        return {
            "operator": self.operator,
            "node_id": self.node_id,
            "target": self.target,
            "before": compact(self.before),
            "after": compact(self.after),
            "valid": self.valid,
            "expected_rejection": self.expected_rejection,
            "details": copy.deepcopy(self.details),
        }


@dataclass(slots=True)
class MutationResult:
    graph: ModelGraph
    records: list[MutationRecord]
    contract_valid: bool
    expected_rejection: bool
    validation_errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_fingerprint": self.graph.fingerprint(),
            "contract_valid": self.contract_valid,
            "expected_rejection": self.expected_rejection,
            "validation_errors": list(self.validation_errors),
            "records": [record.to_dict() for record in self.records],
        }


class MutationEngine:
    def __init__(
        self,
        *,
        registry: ConstraintRegistry,
        rng: np.random.Generator,
        mode: str = "contract_valid",
        enabled: Iterable[str] = MUTATION_OPERATORS,
        max_mutations: int = 4,
        boundary_max_exponent: int = 64,
        noise_ratio: float = 0.05,
        scale_factors: Sequence[float] = (0.5, 0.9, 1.1, 2.0, -1.0),
    ) -> None:
        if mode not in {"contract_valid", "robustness"}:
            raise ValueError("mode must be 'contract_valid' or 'robustness'")
        enabled_tuple = tuple(dict.fromkeys(x.upper() for x in enabled))
        unknown = set(enabled_tuple) - set(MUTATION_OPERATORS)
        if unknown:
            raise ValueError(f"Unknown mutation operators: {sorted(unknown)}")
        self.registry = registry
        self.rng = rng
        self.mode = mode
        self.enabled = enabled_tuple
        self.max_mutations = max(0, int(max_mutations))
        self.boundary_max_exponent = max(0, int(boundary_max_exponent))
        self.noise_ratio = max(0.0, float(noise_ratio))
        self.scale_factors = tuple(float(x) for x in scale_factors)
        self.value_usage: Counter[tuple[str, str, str]] = Counter()

    def apply(self, graph: ModelGraph, operators: Iterable[str] | None = None) -> MutationResult:
        mutated = graph.clone()
        explicit = operators is not None
        selected = tuple(dict.fromkeys(x.upper() for x in (operators or self.enabled)))
        selected = tuple(x for x in selected if x in self.enabled)
        if not explicit and self.max_mutations == 4:
            # Paper-aligned default: apply the three value mutations plus one randomly
            # selected neuron mutation. Ablation scripts pass explicit operator sets.
            values = tuple(x for x in VALUE_MUTATIONS if x in selected)
            neurons = tuple(x for x in NEURON_MUTATIONS if x in selected)
            chosen_neuron = (str(self.rng.choice(neurons)),) if neurons else ()
            selected = values + chosen_neuron
        elif self.max_mutations:
            selected = selected[: self.max_mutations]
        else:
            selected = ()

        records: list[MutationRecord] = []
        for operator in selected:
            before_graph = mutated.clone()
            record = self._apply_one(mutated, operator)
            if record is None:
                continue
            errors = self._validate(mutated)
            record.valid = not errors
            if errors and self.mode == "contract_valid":
                mutated = before_graph
                record.details["reverted_errors"] = errors
                record.valid = False
            elif errors and self.mode == "robustness":
                record.expected_rejection = True
                record.details["contract_errors"] = errors
            records.append(record)

        validation_errors = self._validate(mutated)
        contract_valid = not validation_errors
        expected_rejection = any(record.expected_rejection for record in records)
        mutated.metadata.setdefault("mutations", []).extend(record.to_dict() for record in records)
        mutated.metadata["contract_valid"] = contract_valid
        mutated.metadata["expected_rejection"] = expected_rejection
        return MutationResult(mutated, records, contract_valid, expected_rejection, validation_errors)

    def _apply_one(self, graph: ModelGraph, operator: str) -> MutationRecord | None:
        return {
            "PV": self._parameter_value,
            "BV": self._boundary_value,
            "IT": self._input_type,
            "NF": self._noise_fuzzing,
            "SW": self._scale_weights,
            "WR": self._rotate_weights,
        }[operator](graph)

    def _parameter_value(self, graph: ModelGraph) -> MutationRecord | None:
        eligible: list[tuple[int, str, ParamConstraint, list[Any]]] = []
        for index, node in enumerate(graph.nodes):
            for param, current in node.params.items():
                constraint = self.registry.get(node.op, param)
                if not constraint or not constraint.choices or constraint.structural:
                    continue
                alternatives = [value for value in constraint.choices if value != current and constraint.accepts(value)]
                if alternatives:
                    eligible.append((index, param, constraint, alternatives))
        if not eligible:
            return None
        index, param, constraint, alternatives = eligible[int(self.rng.integers(0, len(eligible)))]
        node = graph.nodes[index]
        minimum = min(self.value_usage[(node.op, param, repr(value))] for value in alternatives)
        least_used = [value for value in alternatives if self.value_usage[(node.op, param, repr(value))] == minimum]
        value = copy.deepcopy(least_used[int(self.rng.integers(0, len(least_used)))])
        before = copy.deepcopy(node.params[param])
        node.params[param] = value
        self.value_usage[(node.op, param, repr(value))] += 1
        return MutationRecord("PV", node.node_id, f"params.{param}", before, copy.deepcopy(value), True, details={"selection": "least_used_legal_value"})

    def _boundary_candidates(self, constraint: ParamConstraint) -> tuple[list[Any], list[Any]]:
        values: set[int | float] = set()
        for n in range(self.boundary_max_exponent + 1):
            base_int = 2**n
            values.update({base_int, base_int - 1, base_int + 1, -base_int, -(base_int - 1), -(base_int + 1)})
        if constraint.kind == "float":
            values = {float(x) for x in values if isfinite(float(x))}
        elif constraint.kind == "integer":
            values = {int(x) for x in values}
        else:
            return [], []
        valid = sorted((x for x in values if constraint.accepts(x)), key=lambda x: (abs(float(x)), float(x)))
        invalid = sorted((x for x in values if not constraint.accepts(x)), key=lambda x: (abs(float(x)), float(x)))
        return valid, invalid

    def _boundary_value(self, graph: ModelGraph) -> MutationRecord | None:
        valid_targets: list[tuple[int, str, list[Any], list[Any]]] = []
        for index, node in enumerate(graph.nodes):
            for param, current in node.params.items():
                constraint = self.registry.get(node.op, param)
                if not constraint or constraint.structural or constraint.kind not in {"integer", "float"}:
                    continue
                valid, invalid = self._boundary_candidates(constraint)
                valid = [x for x in valid if x != current]
                invalid = [x for x in invalid if x != current]
                if self.mode == "contract_valid" and valid:
                    valid_targets.append((index, param, valid, invalid))
                elif self.mode == "robustness" and invalid:
                    valid_targets.append((index, param, valid, invalid))
        if not valid_targets:
            return None
        index, param, valid, invalid = valid_targets[int(self.rng.integers(0, len(valid_targets)))]
        node = graph.nodes[index]
        before = copy.deepcopy(node.params[param])
        expected = self.mode == "robustness"
        pool = invalid if expected else valid
        # Boundary values closest to the current value are useful and avoid immediate
        # floating overflow. Randomise among the first few equally plausible values.
        pool = sorted(pool, key=lambda x: abs(float(x) - float(before)))
        top = pool[: min(8, len(pool))]
        value = top[int(self.rng.integers(0, len(top)))]
        node.params[param] = value
        return MutationRecord(
            "BV",
            node.node_id,
            f"params.{param}",
            before,
            value,
            not expected,
            expected_rejection=expected,
            details={"mode": self.mode, "candidate_rule": "2^n and 2^n±1, n∈[0,max_exponent]", "max_exponent": self.boundary_max_exponent},
        )

    def _input_type(self, graph: ModelGraph) -> MutationRecord | None:
        if not graph.inputs:
            return None
        input_name = sorted(graph.inputs)[0]
        spec = graph.inputs[input_name]
        current = normalize_dtype(spec.dtype)
        current_is_float = current.startswith("float") or current == "bfloat16"
        current_is_int = current.startswith("int") or current.startswith("uint")
        candidates = sorted(self.registry.supported_dtypes - {current})
        if current_is_float:
            candidates = [x for x in candidates if x.startswith("float") or x == "bfloat16"]
        elif current_is_int:
            candidates = [x for x in candidates if x.startswith("int") or x.startswith("uint")]
        consumers = [node for node in graph.nodes if input_name in node.inputs]
        candidates = [dtype for dtype in candidates if all(self.registry.supports_dtype(node.op, dtype) for node in consumers)]
        if not candidates:
            return None
        dtype = str(self.rng.choice(candidates))
        graph.inputs[input_name] = TensorSpec(spec.shape, dtype, spec.semantic)
        self._propagate_dtypes(graph)
        return MutationRecord("IT", None, f"inputs.{input_name}.dtype", current, dtype, True, details={"aliases_normalised": {"half": "float16", "double": "float64"}})

    def _weight_targets(self, graph: ModelGraph) -> list[tuple[int, str]]:
        return [
            (i, name)
            for i, node in enumerate(graph.nodes)
            for name, value in node.weights.items()
            if name != "adjacency" and isinstance(value, np.ndarray) and value.size > 0 and np.issubdtype(value.dtype, np.number)
        ]

    def _noise_fuzzing(self, graph: ModelGraph) -> MutationRecord | None:
        targets = self._weight_targets(graph)
        if not targets:
            return None
        index, name = targets[int(self.rng.integers(0, len(targets)))]
        node = graph.nodes[index]
        before = np.array(node.weights[name], copy=True, dtype=np.float64)
        base_scale = max(float(np.std(before)), float(np.mean(np.abs(before))), 1e-12)
        scale = self.noise_ratio * base_scale
        noise_type = str(self.rng.choice(["laplace", "exponential", "sinusoidal"]))
        if noise_type == "laplace":
            noise = self.rng.laplace(0.0, scale, size=before.shape)
        elif noise_type == "exponential":
            noise = self.rng.exponential(scale, size=before.shape) - scale
        else:
            phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
            noise = scale * np.sin(np.arange(before.size, dtype=np.float64).reshape(before.shape) + phase)
        after = before + noise
        node.weights[name] = after
        return MutationRecord("NF", node.node_id, f"weights.{name}", before, after, True, details={"distribution": noise_type, "scale": scale, "noise_ratio": self.noise_ratio})

    def _scale_weights(self, graph: ModelGraph) -> MutationRecord | None:
        targets = self._weight_targets(graph)
        if not targets or not self.scale_factors:
            return None
        index, name = targets[int(self.rng.integers(0, len(targets)))]
        node = graph.nodes[index]
        before = np.array(node.weights[name], copy=True, dtype=np.float64)
        factor = float(self.rng.choice(self.scale_factors))
        after = before * factor
        original_norm = float(np.linalg.norm(before))
        max_norm = max(original_norm * 4.0, 1e-12)
        after_norm = float(np.linalg.norm(after))
        if after_norm > max_norm:
            after *= max_norm / after_norm
        node.weights[name] = after
        return MutationRecord("SW", node.node_id, f"weights.{name}", before, after, True, details={"factor": factor, "norm_cap": max_norm})

    def _rotate_weights(self, graph: ModelGraph) -> MutationRecord | None:
        targets = self._weight_targets(graph)
        if not targets:
            return None
        index, name = targets[int(self.rng.integers(0, len(targets)))]
        node = graph.nodes[index]
        before = np.array(node.weights[name], copy=True, dtype=np.float64)
        if before.ndim == 0:
            return None
        if before.ndim == 1:
            after = before[::-1].copy()
            if bool(self.rng.integers(0, 2)):
                after *= -1.0
            method = "signed_reversal"
        else:
            matrix = before.reshape((-1, before.shape[-1]))
            width = matrix.shape[1]
            if width <= 64:
                random_matrix = self.rng.normal(size=(width, width))
                q, _ = np.linalg.qr(random_matrix)
                after = (matrix @ q).reshape(before.shape)
                method = "orthogonal_qr"
            else:
                permutation = self.rng.permutation(width)
                signs = self.rng.choice([-1.0, 1.0], size=width)
                after = (matrix[:, permutation] * signs).reshape(before.shape)
                method = "signed_permutation"
        node.weights[name] = after
        return MutationRecord("WR", node.node_id, f"weights.{name}", before, after, True, details={"method": method, "shape_preserved": list(before.shape) == list(after.shape)})

    def _propagate_dtypes(self, graph: ModelGraph) -> None:
        known = dict(graph.inputs)
        for node in graph.nodes:
            input_specs = [known[ref] for ref in node.inputs]
            if node.op == "embedding":
                dtype = "float32"
            else:
                dtype = input_specs[0].dtype
            node.output_spec = TensorSpec(node.output_spec.shape, dtype, node.output_spec.semantic)
            known[node.node_id] = node.output_spec

    def _validate(self, graph: ModelGraph) -> list[str]:
        errors = self.registry.validate_graph(graph)
        for node in graph.nodes:
            input_specs = [graph.get_spec(ref) for ref in node.inputs]
            try:
                if node.op == "dense":
                    kernel = node.weights["kernel"]
                    if kernel.shape != (input_specs[0].shape[-1], node.output_spec.shape[-1]):
                        errors.append(f"{node.node_id}: dense kernel shape mismatch")
                elif node.op == "conv2d_1x1":
                    kernel = node.weights["kernel"]
                    if kernel.shape != (input_specs[0].shape[-1], node.output_spec.shape[-1]):
                        errors.append(f"{node.node_id}: conv kernel shape mismatch")
                elif node.op == "simple_rnn":
                    units = node.output_spec.shape[-1]
                    if node.weights["kernel"].shape != (input_specs[0].shape[-1], units):
                        errors.append(f"{node.node_id}: RNN input-kernel shape mismatch")
                    if node.weights["recurrent_kernel"].shape != (units, units):
                        errors.append(f"{node.node_id}: RNN recurrent-kernel shape mismatch")
                elif node.op == "attention":
                    features = input_specs[0].shape[-1]
                    for name in ("wq", "wk", "wv", "wo"):
                        if node.weights[name].shape != (features, features):
                            errors.append(f"{node.node_id}: attention {name} shape mismatch")
                elif node.op == "embedding":
                    table = node.weights["embeddings"]
                    if table.shape != (node.params["input_dim"], node.params["output_dim"]):
                        errors.append(f"{node.node_id}: embedding table shape mismatch")
                elif node.op == "graph_conv":
                    if node.weights["kernel"].shape != (input_specs[0].shape[-1], node.output_spec.shape[-1]):
                        errors.append(f"{node.node_id}: graph-conv kernel shape mismatch")
            except KeyError as exc:
                errors.append(f"{node.node_id}: missing weight/parameter {exc}")
            for name, value in node.weights.items():
                if not np.all(np.isfinite(value)):
                    errors.append(f"{node.node_id}: non-finite weight values in {name}")
        return errors
