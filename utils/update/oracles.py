"""Contract-aware crash, non-finite, differential, and gradient oracles.

The oracle keeps contract-invalid robustness cases separate from contract-valid defect
candidates.  Differential findings are filtered for semantic mismatch, unsynchronised
state, non-pure execution, nondeterminism, datatype precision, finite-difference
instability, nondifferentiable points, and extreme numerical scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import hashlib
import json

import numpy as np

from .execution import ExecutionResult
from .model_ir import ModelGraph, normalize_dtype


@dataclass(slots=True)
class Finding:
    kind: str
    message: str
    backends: tuple[str, ...]
    bug_candidate: bool
    security_relevant: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            payload = {
                "kind": self.kind,
                "backends": sorted(self.backends),
                "message": self.message,
                "exception_types": self.details.get("exception_types"),
                "output": self.details.get("output"),
            }
            self.signature = hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:20]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "backends": list(self.backends),
            "bug_candidate": self.bug_candidate,
            "security_relevant": self.security_relevant,
            "details": self.details,
            "signature": self.signature,
        }


@dataclass(slots=True)
class OracleReport:
    findings: list[Finding]
    skipped: list[dict[str, Any]] = field(default_factory=list)

    @property
    def bug_candidates(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.bug_candidate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [finding.to_dict() for finding in self.findings],
            "bug_candidate_count": len(self.bug_candidates),
            "skipped": self.skipped,
        }


@dataclass(slots=True)
class OracleDecision:
    """Detailed decision for one comparison.

    ``excluded`` means that an observed difference was not eligible for defect
    classification because an oracle-validity condition was violated.  A consistent
    comparison has ``excluded=False`` and ``finding=None``.
    """

    finding: Finding | None = None
    excluded: bool = False
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding": self.finding.to_dict() if self.finding is not None else None,
            "excluded": self.excluded,
            "reason": self.reason,
            "details": self.details,
        }


def contains_nan(value: Any) -> bool:
    """Return true only for numeric NaN values; ``None`` is explicitly false."""
    if value is None:
        return False
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    if not np.issubdtype(arr.dtype, np.inexact):
        return False
    return bool(np.isnan(arr).any())


def contains_inf(value: Any) -> bool:
    if value is None:
        return False
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    if not np.issubdtype(arr.dtype, np.inexact):
        return False
    return bool(np.isinf(arr).any())


def dtype_tolerance(dtype: str) -> tuple[float, float]:
    """Return absolute and relative tolerances for a logical datatype."""
    name = normalize_dtype(dtype)
    if name in {"float16", "bfloat16"}:
        return 5e-3, 5e-2
    if name == "float32":
        return 1e-5, 1e-4
    if name == "float64":
        return 1e-8, 1e-7
    return 0.0, 0.0


# Backward-compatible private alias used by earlier experiment code.
_dtype_tolerance = dtype_tolerance


def dtype_limits(dtype: str) -> tuple[float, float] | None:
    """Return ``(maximum finite value, minimum normal value)`` for a float dtype."""
    name = normalize_dtype(dtype)
    if name == "bfloat16":
        # IEEE bfloat16 has the float32 exponent range and a shorter mantissa.
        return 3.3895313892515355e38, 1.1754943508222875e-38
    if name not in {"float16", "float32", "float64"}:
        return None
    info = np.finfo(np.dtype(name))
    return float(info.max), float(info.tiny)


def robust_difference(a: np.ndarray, b: np.ndarray) -> dict[str, float | bool]:
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    delta = np.abs(a64 - b64)
    scale = np.median(np.abs(a64)) + np.median(np.abs(b64)) + 1e-12
    normalised_mad = float(np.median(delta) / scale) if delta.size else 0.0
    max_abs = float(np.max(delta)) if delta.size else 0.0
    mean_abs = float(np.mean(delta)) if delta.size else 0.0
    prediction_changed = False
    if a64.ndim >= 1 and a64.shape[-1] > 1:
        prediction_changed = bool(
            np.any(np.argmax(a64, axis=-1) != np.argmax(b64, axis=-1))
        )
    return {
        "normalised_mad": normalised_mad,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "prediction_changed": prediction_changed,
    }


def assess_array_scale(
    value: Any,
    *,
    dtype: str | None = None,
    overflow_margin: float = 0.95,
    absolute_threshold: float | None = None,
) -> dict[str, Any]:
    """Detect overflow/underflow-dominated arrays using dtype-specific limits."""
    arr = np.asarray(value)
    logical_dtype = normalize_dtype(dtype or str(arr.dtype))
    if arr.size == 0 or not np.issubdtype(arr.dtype, np.inexact):
        return {
            "extreme": False,
            "reason": None,
            "dtype": logical_dtype,
            "max_abs": 0.0,
            "min_nonzero_abs": None,
        }
    finite = np.abs(arr[np.isfinite(arr)].astype(np.float64, copy=False))
    max_abs = float(np.max(finite)) if finite.size else float("inf")
    nonzero = finite[finite > 0.0]
    min_nonzero_abs = float(np.min(nonzero)) if nonzero.size else None
    limits = dtype_limits(logical_dtype)
    reason: str | None = None
    if limits is not None:
        maximum, minimum_normal = limits
        if max_abs >= overflow_margin * maximum:
            reason = "near_dtype_overflow"
        elif min_nonzero_abs is not None and min_nonzero_abs < minimum_normal:
            reason = "dtype_underflow_or_subnormal"
    if reason is None and absolute_threshold is not None and max_abs > absolute_threshold:
        reason = "absolute_extreme_value"
    return {
        "extreme": reason is not None,
        "reason": reason,
        "dtype": logical_dtype,
        "max_abs": max_abs,
        "min_nonzero_abs": min_nonzero_abs,
        "limits": limits,
    }


def assess_result_scale(
    result: ExecutionResult,
    output_name: str,
    *,
    absolute_threshold: float | None,
    overflow_margin: float = 0.95,
) -> dict[str, Any]:
    """Inspect an output and all intermediate traces for extreme scaling."""
    output_assessment = assess_array_scale(
        result.outputs[output_name],
        overflow_margin=overflow_margin,
        absolute_threshold=absolute_threshold,
    )
    if output_assessment["extreme"]:
        return {"extreme": True, "location": f"output:{output_name}", **output_assessment}
    for trace in result.traces:
        dtype = trace.dtype
        limits = dtype_limits(dtype)
        largest_abs = getattr(trace, "largest_abs", None)
        if largest_abs is None:
            candidates = [abs(v) for v in (trace.minimum, trace.maximum) if v is not None]
            largest_abs = max(candidates, default=0.0)
        smallest_nonzero = getattr(trace, "smallest_nonzero_abs", None)
        reason: str | None = None
        if limits is not None:
            maximum, minimum_normal = limits
            if largest_abs >= overflow_margin * maximum:
                reason = "near_dtype_overflow"
            elif smallest_nonzero is not None and 0.0 < smallest_nonzero < minimum_normal:
                reason = "dtype_underflow_or_subnormal"
        if reason is None and absolute_threshold is not None and largest_abs > absolute_threshold:
            reason = "absolute_extreme_value"
        if reason is not None:
            return {
                "extreme": True,
                "location": f"trace:{trace.node_id}",
                "reason": reason,
                "dtype": normalize_dtype(dtype),
                "max_abs": float(largest_abs),
                "min_nonzero_abs": smallest_nonzero,
                "limits": limits,
            }
    return {"extreme": False, "reason": None}


def finite_difference_stability(
    gradients: Mapping[float, np.ndarray],
    *,
    threshold: float = 5e-2,
) -> dict[str, Any]:
    """Assess convergence of central finite differences across decreasing steps."""
    if len(gradients) < 2:
        return {
            "stable": True,
            "reason": "single_step_only",
            "relative_changes": [],
            "selected_epsilon": min(gradients) if gradients else None,
        }
    epsilons = sorted((float(eps) for eps in gradients), reverse=True)
    relative_changes: list[dict[str, float]] = []
    for coarse, fine in zip(epsilons, epsilons[1:]):
        a = np.asarray(gradients[coarse], dtype=np.float64)
        b = np.asarray(gradients[fine], dtype=np.float64)
        if a.shape != b.shape or contains_nan(a) or contains_nan(b) or contains_inf(a) or contains_inf(b):
            return {
                "stable": False,
                "reason": "nonfinite_or_shape_change",
                "relative_changes": relative_changes,
                "selected_epsilon": fine,
            }
        denominator = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-12)
        change = float(np.linalg.norm(a - b) / denominator)
        relative_changes.append({"coarse": coarse, "fine": fine, "relative_change": change})
    final_change = relative_changes[-1]["relative_change"]
    return {
        "stable": final_change <= threshold,
        "reason": "converged" if final_change <= threshold else "finite_difference_instability",
        "relative_changes": relative_changes,
        "selected_epsilon": epsilons[-1],
        "threshold": float(threshold),
    }


def nondifferentiable_input_mask(
    graph: ModelGraph,
    inputs: Mapping[str, np.ndarray],
    *,
    epsilon: float,
    kappa: float = 2.0,
) -> np.ndarray | None:
    """Return a conservative mask for directly observable nondifferentiable inputs.

    The function marks input coordinates at ReLU/LeakyReLU branch boundaries and tied
    maxima in directly connected 2-D max-pooling windows.  For a dense layer with a
    ReLU activation, all input coordinates of a sample are masked when any pre-activation
    is within the boundary neighbourhood because the exact input-coordinate attribution
    is generally non-local.
    """
    if "input" not in inputs:
        return None
    x = np.asarray(inputs["input"], dtype=np.float64)
    mask = np.zeros(x.shape, dtype=bool)
    boundary = abs(float(kappa) * float(epsilon))
    observed = False

    for node in graph.nodes:
        if tuple(node.inputs) != ("input",):
            continue
        if node.op in {"relu", "leaky_relu"}:
            mask |= np.abs(x) <= boundary
            observed = True
        elif node.op == "dense" and str(node.params.get("activation", "linear")) == "relu":
            kernel = np.asarray(node.weights.get("kernel"), dtype=np.float64)
            if kernel.size:
                preactivation = np.tensordot(x, kernel, axes=([-1], [0]))
                if node.params.get("use_bias", True) and "bias" in node.weights:
                    preactivation = preactivation + np.asarray(node.weights["bias"], dtype=np.float64)
                near = np.any(np.abs(preactivation) <= boundary, axis=-1)
                if np.any(near):
                    expanded = near
                    while expanded.ndim < x.ndim:
                        expanded = np.expand_dims(expanded, axis=-1)
                    mask |= np.broadcast_to(expanded, x.shape)
                observed = True
        elif node.op == "max_pool2d" and x.ndim == 4:
            pool = tuple(int(v) for v in node.params.get("pool_size", (2, 2)))
            strides = tuple(int(v) for v in node.params.get("strides", pool))
            if pool == (2, 2) and strides == (2, 2):
                _, height, width, _ = x.shape
                for row in range(0, height - 1, 2):
                    for col in range(0, width - 1, 2):
                        window = x[:, row : row + 2, col : col + 2, :]
                        flat = window.reshape(window.shape[0], 4, window.shape[-1])
                        ordered = np.sort(flat, axis=1)
                        margin = ordered[:, -1, :] - ordered[:, -2, :]
                        tied = margin <= boundary
                        if np.any(tied):
                            for batch_index in range(x.shape[0]):
                                for channel in range(x.shape[-1]):
                                    if tied[batch_index, channel]:
                                        mask[batch_index, row : row + 2, col : col + 2, channel] = True
                observed = True
    if not observed or not np.any(mask):
        return None
    return mask


class OracleEngine:
    def __init__(
        self,
        *,
        normalised_mad_threshold: float = 1e-3,
        require_prediction_change: bool = False,
        extreme_value_threshold: float = 1e12,
        report_inf: bool = True,
        overflow_margin: float = 0.95,
    ) -> None:
        self.normalised_mad_threshold = float(normalised_mad_threshold)
        self.require_prediction_change = bool(require_prediction_change)
        self.extreme_value_threshold = float(extreme_value_threshold)
        self.report_inf = bool(report_inf)
        self.overflow_margin = float(overflow_margin)

    def evaluate(self, graph: ModelGraph, results: Sequence[ExecutionResult]) -> OracleReport:
        findings: list[Finding] = []
        skipped: list[dict[str, Any]] = []
        if not results:
            return OracleReport([], [{"reason": "no_execution_results"}])

        contract_valid = bool(
            graph.metadata.get("contract_valid", all(r.contract_valid for r in results))
        )
        expected_rejection = bool(
            graph.metadata.get("expected_rejection", any(r.expected_rejection for r in results))
        )
        ok = [result for result in results if result.status == "ok"]
        failed = [result for result in results if result.status != "ok"]

        if not contract_valid or expected_rejection:
            if failed:
                findings.append(
                    Finding(
                        kind="expected_rejection",
                        message="A contract-invalid robustness case was rejected as expected.",
                        backends=tuple(result.backend for result in failed),
                        bug_candidate=False,
                        details={
                            "exception_types": {
                                result.backend: result.exception_type for result in failed
                            }
                        },
                    )
                )
            if ok:
                findings.append(
                    Finding(
                        kind="contract_violation_accepted",
                        message=(
                            "A deliberately contract-invalid case executed successfully; "
                            "review framework validation behavior."
                        ),
                        backends=tuple(result.backend for result in ok),
                        bug_candidate=False,
                        details={"robustness_observation": True},
                    )
                )
            return OracleReport(findings, skipped)

        if failed:
            if ok:
                findings.append(
                    Finding(
                        kind="crash",
                        message="A contract-valid case failed on a subset of backends.",
                        backends=tuple(result.backend for result in failed),
                        bug_candidate=True,
                        details={
                            "exception_types": {
                                result.backend: result.exception_type for result in failed
                            },
                            "successful_backends": [result.backend for result in ok],
                        },
                    )
                )
            else:
                findings.append(
                    Finding(
                        kind="crash",
                        message="A contract-valid case failed on every configured backend.",
                        backends=tuple(result.backend for result in failed),
                        bug_candidate=True,
                        details={
                            "exception_types": {
                                result.backend: result.exception_type for result in failed
                            }
                        },
                    )
                )

        for result in ok:
            for output_name, value in result.outputs.items():
                if contains_nan(value):
                    findings.append(
                        Finding(
                            kind="nan",
                            message=f"Backend {result.backend} produced NaN for a contract-valid case.",
                            backends=(result.backend,),
                            bug_candidate=True,
                            details={"output": output_name, "inf_is_separate": True},
                        )
                    )
                if self.report_inf and contains_inf(value):
                    findings.append(
                        Finding(
                            kind="inf",
                            message=f"Backend {result.backend} produced infinity for a contract-valid case.",
                            backends=(result.backend,),
                            bug_candidate=True,
                            details={"output": output_name},
                        )
                    )

        if len(ok) >= 2:
            findings.extend(self._differential(graph, ok, skipped))
        elif len(results) == 1:
            skipped.append(
                {"oracle": "differential", "reason": "requires_at_least_two_backends"}
            )
        return OracleReport(findings, skipped)

    def _differential(
        self,
        graph: ModelGraph,
        results: Sequence[ExecutionResult],
        skipped: list[dict[str, Any]],
    ) -> list[Finding]:
        findings: list[Finding] = []
        reference = results[0]
        graph_tags = set().union(*(node.tags for node in graph.nodes)) if graph.nodes else set()

        if any(result.metadata.get("semantic_equivalent") is False for result in results):
            skipped.append({"oracle": "differential", "reason": "unsupported_semantic_mapping"})
            return findings
        if "non_pure" in graph_tags and not all(
            bool(result.metadata.get("isolated_snapshot", False)) for result in results
        ):
            skipped.append({"oracle": "differential", "reason": "non_pure_operation"})
            return findings
        if {"stateful", "stateful_formula"} & graph_tags and not all(
            bool(result.metadata.get("state_synchronised", False)) for result in results
        ):
            skipped.append({"oracle": "differential", "reason": "unsynchronised_state"})
            return findings
        if "nondeterministic" in graph_tags and not all(
            bool(result.metadata.get("repeat_stable", False)) for result in results
        ):
            skipped.append({"oracle": "differential", "reason": "nondeterministic_result"})
            return findings

        for other in results[1:]:
            common = sorted(set(reference.outputs) & set(other.outputs))
            if not common:
                findings.append(
                    Finding(
                        kind="inconsistency",
                        message="Backends returned disjoint output names.",
                        backends=(reference.backend, other.backend),
                        bug_candidate=True,
                        details={
                            "reference_outputs": sorted(reference.outputs),
                            "other_outputs": sorted(other.outputs),
                        },
                    )
                )
                continue
            for output_name in common:
                a = np.asarray(reference.outputs[output_name])
                b = np.asarray(other.outputs[output_name])
                if a.shape != b.shape:
                    findings.append(
                        Finding(
                            kind="inconsistency",
                            message="Backends returned different output shapes.",
                            backends=(reference.backend, other.backend),
                            bug_candidate=True,
                            details={
                                "output": output_name,
                                "shape_a": list(a.shape),
                                "shape_b": list(b.shape),
                            },
                        )
                    )
                    continue
                if contains_nan(a) or contains_nan(b) or contains_inf(a) or contains_inf(b):
                    skipped.append(
                        {
                            "oracle": "differential",
                            "output": output_name,
                            "reason": "nonfinite_output_handled_by_nan_inf_oracle",
                        }
                    )
                    continue
                scale_a = assess_result_scale(
                    reference,
                    output_name,
                    absolute_threshold=self.extreme_value_threshold,
                    overflow_margin=self.overflow_margin,
                )
                scale_b = assess_result_scale(
                    other,
                    output_name,
                    absolute_threshold=self.extreme_value_threshold,
                    overflow_margin=self.overflow_margin,
                )
                if scale_a["extreme"] or scale_b["extreme"]:
                    skipped.append(
                        {
                            "oracle": "differential",
                            "output": output_name,
                            "reason": "extreme_scaling",
                            "reference": scale_a,
                            "other": scale_b,
                        }
                    )
                    continue
                atol_a, rtol_a = dtype_tolerance(str(a.dtype))
                atol_b, rtol_b = dtype_tolerance(str(b.dtype))
                atol, rtol = max(atol_a, atol_b), max(rtol_a, rtol_b)
                close = bool(
                    np.allclose(
                        a.astype(np.float64),
                        b.astype(np.float64),
                        atol=atol,
                        rtol=rtol,
                        equal_nan=False,
                    )
                )
                difference = robust_difference(a, b)
                significant = difference["normalised_mad"] > self.normalised_mad_threshold
                prediction_ok = bool(difference["prediction_changed"]) or not self.require_prediction_change
                if not close and significant and prediction_ok:
                    findings.append(
                        Finding(
                            kind="inconsistency",
                            message=(
                                "Contract-valid outputs exceed dtype-aware and "
                                "robust-difference thresholds."
                            ),
                            backends=(reference.backend, other.backend),
                            bug_candidate=True,
                            details={"output": output_name, "atol": atol, "rtol": rtol, **difference},
                        )
                    )
        return findings


def assess_gradients(
    analytical: np.ndarray,
    numerical: np.ndarray,
    *,
    dtype: str = "float64",
    nondifferentiable_mask: np.ndarray | None = None,
    stateful: bool = False,
    state_synchronised: bool = False,
    non_pure: bool = False,
    isolated_snapshot: bool = False,
    precision_unstable: bool = False,
    extreme_scaling: bool = False,
    extreme_scale: float = 1e12,
    normalised_mad_threshold: float = 1e-3,
) -> OracleDecision:
    """Compare gradients while recording explicit oracle-validity exclusions."""
    if stateful and not state_synchronised:
        return OracleDecision(excluded=True, reason="unsynchronised_state")
    if non_pure and not isolated_snapshot:
        return OracleDecision(excluded=True, reason="non_pure_operation")
    if precision_unstable:
        return OracleDecision(excluded=True, reason="finite_difference_instability")

    a, b = np.asarray(analytical), np.asarray(numerical)
    if a.shape != b.shape:
        return OracleDecision(
            finding=Finding(
                kind="gradient_inconsistency",
                message="Analytical and numerical gradients have incompatible shapes.",
                backends=("analytical", "numerical"),
                bug_candidate=True,
                details={"shape_a": list(a.shape), "shape_b": list(b.shape)},
            ),
            reason="shape_mismatch",
        )
    if contains_nan(a) or contains_nan(b) or contains_inf(a) or contains_inf(b):
        return OracleDecision(
            finding=Finding(
                kind="gradient_inconsistency",
                message="Analytical or numerical gradient contains non-finite values.",
                backends=("analytical", "numerical"),
                bug_candidate=True,
                details={"shape_a": list(a.shape), "shape_b": list(b.shape)},
            ),
            reason="nonfinite_gradient",
        )

    mask_count = 0
    if nondifferentiable_mask is not None:
        mask = np.asarray(nondifferentiable_mask, dtype=bool)
        if mask.shape != a.shape:
            raise ValueError(
                f"nondifferentiable_mask shape {mask.shape} does not match gradient shape {a.shape}"
            )
        mask_count = int(mask.sum())
        valid = ~mask
        if not np.any(valid):
            return OracleDecision(
                excluded=True,
                reason="nondifferentiable_point",
                details={"excluded_elements": mask_count, "total_elements": int(mask.size)},
            )
        a = a[valid]
        b = b[valid]

    scale_a = assess_array_scale(a, dtype=dtype, absolute_threshold=extreme_scale)
    scale_b = assess_array_scale(b, dtype=dtype, absolute_threshold=extreme_scale)
    if extreme_scaling or scale_a["extreme"] or scale_b["extreme"]:
        return OracleDecision(
            excluded=True,
            reason="extreme_scaling",
            details={"analytical": scale_a, "numerical": scale_b},
        )

    atol, rtol = dtype_tolerance(dtype)
    difference = robust_difference(a, b)
    close = bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False))
    significant = difference["normalised_mad"] > normalised_mad_threshold
    details = {
        "atol": atol,
        "rtol": rtol,
        "excluded_nondifferentiable_elements": mask_count,
        **difference,
    }
    if not close and significant:
        return OracleDecision(
            finding=Finding(
                kind="gradient_inconsistency",
                message="Analytical and numerical gradients differ beyond tolerance.",
                backends=("analytical", "numerical"),
                bug_candidate=True,
                details=details,
            ),
            reason="difference_exceeds_threshold",
            details=details,
        )
    return OracleDecision(excluded=False, reason="consistent_within_tolerance", details=details)


def compare_gradients(
    analytical: np.ndarray,
    numerical: np.ndarray,
    *,
    dtype: str = "float64",
    nondifferentiable: bool = False,
    nondifferentiable_mask: np.ndarray | None = None,
    stateful: bool = False,
    state_synchronised: bool = False,
    non_pure: bool = False,
    isolated_snapshot: bool = False,
    precision_unstable: bool = False,
    extreme_scaling: bool = False,
    extreme_scale: float = 1e12,
) -> Finding | None:
    """Backward-compatible gradient oracle returning only a retained finding."""
    if nondifferentiable and nondifferentiable_mask is None:
        nondifferentiable_mask = np.ones(np.asarray(analytical).shape, dtype=bool)
    return assess_gradients(
        analytical,
        numerical,
        dtype=dtype,
        nondifferentiable_mask=nondifferentiable_mask,
        stateful=stateful,
        state_synchronised=state_synchronised,
        non_pure=non_pure,
        isolated_snapshot=isolated_snapshot,
        precision_unstable=precision_unstable,
        extreme_scaling=extreme_scaling,
        extreme_scale=extreme_scale,
    ).finding
