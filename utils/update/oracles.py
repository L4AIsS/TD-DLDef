"""Contract-aware crash, NaN/Inf, and differential oracles.

Invalid API-contract cases are never mixed with contract-valid bug candidates.  `None`
is not treated as NaN, infinities are reported separately, and large numerical
inconsistencies require dtype-aware tolerances plus a robust normalised difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence
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
            self.signature = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:20]

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


def contains_nan(value: Any) -> bool:
    """Return true only for numeric NaN values; `None` is explicitly false."""
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


def _dtype_tolerance(dtype: str) -> tuple[float, float]:
    name = normalize_dtype(dtype)
    if name in {"float16", "bfloat16"}:
        return 5e-3, 5e-2
    if name == "float32":
        return 1e-5, 1e-4
    if name == "float64":
        return 1e-8, 1e-7
    return 0.0, 0.0


def robust_difference(a: np.ndarray, b: np.ndarray) -> dict[str, float | bool]:
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    delta = np.abs(a64 - b64)
    scale = np.median(np.abs(a64)) + np.median(np.abs(b64)) + 1e-12
    normalised_mad = float(np.median(delta) / scale)
    max_abs = float(np.max(delta)) if delta.size else 0.0
    mean_abs = float(np.mean(delta)) if delta.size else 0.0
    prediction_changed = False
    if a64.ndim >= 1 and a64.shape[-1] > 1:
        prediction_changed = bool(np.any(np.argmax(a64, axis=-1) != np.argmax(b64, axis=-1)))
    return {
        "normalised_mad": normalised_mad,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "prediction_changed": prediction_changed,
    }


class OracleEngine:
    def __init__(
        self,
        *,
        normalised_mad_threshold: float = 1e-3,
        require_prediction_change: bool = False,
        extreme_value_threshold: float = 1e12,
        report_inf: bool = True,
    ) -> None:
        self.normalised_mad_threshold = float(normalised_mad_threshold)
        self.require_prediction_change = bool(require_prediction_change)
        self.extreme_value_threshold = float(extreme_value_threshold)
        self.report_inf = bool(report_inf)

    def evaluate(self, graph: ModelGraph, results: Sequence[ExecutionResult]) -> OracleReport:
        findings: list[Finding] = []
        skipped: list[dict[str, Any]] = []
        if not results:
            return OracleReport([], [{"reason": "no_execution_results"}])

        contract_valid = bool(graph.metadata.get("contract_valid", all(r.contract_valid for r in results)))
        expected_rejection = bool(graph.metadata.get("expected_rejection", any(r.expected_rejection for r in results)))
        ok = [result for result in results if result.status == "ok"]
        failed = [result for result in results if result.status != "ok"]

        if not contract_valid or expected_rejection:
            if failed:
                findings.append(Finding(
                    kind="expected_rejection",
                    message="A contract-invalid robustness case was rejected as expected.",
                    backends=tuple(result.backend for result in failed),
                    bug_candidate=False,
                    details={"exception_types": {result.backend: result.exception_type for result in failed}},
                ))
            if ok:
                findings.append(Finding(
                    kind="contract_violation_accepted",
                    message="A deliberately contract-invalid case executed successfully; review framework validation behavior.",
                    backends=tuple(result.backend for result in ok),
                    bug_candidate=False,
                    details={"robustness_observation": True},
                ))
            return OracleReport(findings, skipped)

        if failed:
            if ok:
                findings.append(Finding(
                    kind="crash",
                    message="A contract-valid case failed on a subset of backends.",
                    backends=tuple(result.backend for result in failed),
                    bug_candidate=True,
                    details={
                        "exception_types": {result.backend: result.exception_type for result in failed},
                        "successful_backends": [result.backend for result in ok],
                    },
                ))
            else:
                findings.append(Finding(
                    kind="crash",
                    message="A contract-valid case failed on every configured backend.",
                    backends=tuple(result.backend for result in failed),
                    bug_candidate=True,
                    details={"exception_types": {result.backend: result.exception_type for result in failed}},
                ))

        for result in ok:
            for output_name, value in result.outputs.items():
                if contains_nan(value):
                    findings.append(Finding(
                        kind="nan",
                        message=f"Backend {result.backend} produced NaN for a contract-valid case.",
                        backends=(result.backend,),
                        bug_candidate=True,
                        details={"output": output_name, "inf_is_separate": True},
                    ))
                if self.report_inf and contains_inf(value):
                    findings.append(Finding(
                        kind="inf",
                        message=f"Backend {result.backend} produced infinity for a contract-valid case.",
                        backends=(result.backend,),
                        bug_candidate=True,
                        details={"output": output_name},
                    ))

        if len(ok) >= 2:
            findings.extend(self._differential(graph, ok, skipped))
        elif len(results) == 1:
            skipped.append({"oracle": "differential", "reason": "requires_at_least_two_backends"})
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
        if "nondeterministic" in graph_tags or "stateful" in graph_tags:
            skipped.append({"oracle": "differential", "reason": "stateful_or_nondeterministic_graph"})
            return findings

        for other in results[1:]:
            common = sorted(set(reference.outputs) & set(other.outputs))
            if not common:
                findings.append(Finding(
                    kind="inconsistency",
                    message="Backends returned disjoint output names.",
                    backends=(reference.backend, other.backend),
                    bug_candidate=True,
                    details={"reference_outputs": sorted(reference.outputs), "other_outputs": sorted(other.outputs)},
                ))
                continue
            for output_name in common:
                a = np.asarray(reference.outputs[output_name])
                b = np.asarray(other.outputs[output_name])
                if a.shape != b.shape:
                    findings.append(Finding(
                        kind="inconsistency",
                        message="Backends returned different output shapes.",
                        backends=(reference.backend, other.backend),
                        bug_candidate=True,
                        details={"output": output_name, "shape_a": list(a.shape), "shape_b": list(b.shape)},
                    ))
                    continue
                if contains_nan(a) or contains_nan(b) or contains_inf(a) or contains_inf(b):
                    skipped.append({"oracle": "differential", "output": output_name, "reason": "nonfinite_output_handled_by_nan_inf_oracle"})
                    continue
                if a.size and max(float(np.max(np.abs(a))), float(np.max(np.abs(b)))) > self.extreme_value_threshold:
                    skipped.append({"oracle": "differential", "output": output_name, "reason": "extreme_scaling"})
                    continue
                atol_a, rtol_a = _dtype_tolerance(str(a.dtype))
                atol_b, rtol_b = _dtype_tolerance(str(b.dtype))
                atol, rtol = max(atol_a, atol_b), max(rtol_a, rtol_b)
                close = bool(np.allclose(a.astype(np.float64), b.astype(np.float64), atol=atol, rtol=rtol, equal_nan=False))
                difference = robust_difference(a, b)
                significant = difference["normalised_mad"] > self.normalised_mad_threshold
                prediction_ok = bool(difference["prediction_changed"]) or not self.require_prediction_change
                if not close and significant and prediction_ok:
                    findings.append(Finding(
                        kind="inconsistency",
                        message="Contract-valid outputs exceed dtype-aware and robust-difference thresholds.",
                        backends=(reference.backend, other.backend),
                        bug_candidate=True,
                        details={"output": output_name, "atol": atol, "rtol": rtol, **difference},
                    ))
        return findings


def compare_gradients(
    analytical: np.ndarray,
    numerical: np.ndarray,
    *,
    dtype: str = "float64",
    nondifferentiable: bool = False,
    stateful: bool = False,
    extreme_scale: float = 1e12,
) -> Finding | None:
    """Standalone gradient oracle with reviewer-requested exclusion rules."""
    if nondifferentiable or stateful:
        return None
    a, b = np.asarray(analytical), np.asarray(numerical)
    if a.shape != b.shape or contains_nan(a) or contains_nan(b) or contains_inf(a) or contains_inf(b):
        return Finding(
            kind="gradient_inconsistency",
            message="Analytical and numerical gradients have incompatible shapes or non-finite values.",
            backends=("analytical", "numerical"),
            bug_candidate=True,
            details={"shape_a": list(a.shape), "shape_b": list(b.shape)},
        )
    if a.size and max(float(np.max(np.abs(a))), float(np.max(np.abs(b)))) > extreme_scale:
        return None
    atol, rtol = _dtype_tolerance(dtype)
    difference = robust_difference(a, b)
    if not np.allclose(a, b, atol=atol, rtol=rtol) and difference["normalised_mad"] > 1e-3:
        return Finding(
            kind="gradient_inconsistency",
            message="Analytical and numerical gradients differ beyond tolerance.",
            backends=("analytical", "numerical"),
            bug_candidate=True,
            details={"atol": atol, "rtol": rtol, **difference},
        )
    return None
