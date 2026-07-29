"""End-to-end experiment runner with fixed budgets and reproducible outputs."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping
import os

import numpy as np
import psutil

from .backends import create_backend
from .backends.base import Backend, BackendUnavailable
from .bandits import create_policy
from .catalog import LayerCatalog, TaskSpec, built_in_task
from .config import dump_config, load_config
from .constraints import default_registry
from .diversity import DiversityTracker
from .datasets import load_dataset
from .execution import ExecutionResult, generate_inputs
from .generator import GenerationResult, ModelGenerator
from .metrics import CoverageAccumulator
from .mutations import MutationEngine, MutationResult
from .oracles import Finding, OracleEngine
from .reporting import append_jsonl, atomic_json, environment_manifest


@dataclass(slots=True)
class ExperimentSummary:
    run_name: str
    seed: int
    output_dir: str
    elapsed_seconds: float
    total_cases: int
    valid_cases: int
    invalid_cases: int
    generated_models: int
    execution_statuses: dict[str, int]
    findings_by_kind: dict[str, int]
    unique_finding_signatures: int
    time_to_first_finding: float | None
    valid_ratio: float
    cases_per_second: float
    valid_tests_per_second: float
    peak_rss_mb: float
    generation_attempts: int
    diversity_evaluations: int
    fallback_count: int
    diversity_counts: dict[str, int]
    coverage: dict[str, Any]
    backends: list[dict[str, Any]]
    stopped_by: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentRunner:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.seed = int(self.config["seed"])
        self.rng = np.random.default_rng(self.seed)
        dataset_config = dict(self.config.get("dataset") or {})
        dataset_name = str(dataset_config.get("name", "synthetic"))
        if dataset_name.lower() in {"synthetic", "random"}:
            self.dataset = None
            self.task = built_in_task(str(self.config["task"]))
        else:
            self.dataset = load_dataset(
                dataset_name,
                split=str(dataset_config.get("split", "test")),
                local_path=dataset_config.get("local_path"),
                limit=dataset_config.get("limit"),
                seed=self.seed,
                sine_length=int(dataset_config.get("sine_length", 50)),
                sine_samples=int(dataset_config.get("sine_samples", 2048)),
            )
            self.task = self.dataset.task
        self.output_dir = Path(str(self.config["output"])).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.registry = default_registry("multi-backend", "see environment.json")
        generation = self.config["generation"]
        self.tracker = DiversityTracker(generation["enabled_diversity_spaces"])
        policy_args = dict(generation.get("policy_args") or {})
        self.policy = create_policy(str(generation["policy"]), **policy_args)
        self.catalog = LayerCatalog(
            task=self.task,
            enabled_ops=generation.get("enabled_ops"),
            candidate_window=int(generation["candidate_window"]),
            max_candidates=int(generation["max_candidates"]),
            arm_granularity=str(generation["arm_granularity"]),
        )
        mutation = self.config["mutation"]
        self.mutator = MutationEngine(
            registry=self.registry,
            rng=self.rng,
            mode=str(mutation["mode"]),
            enabled=mutation["enabled"],
            max_mutations=int(mutation["max_mutations"]),
            boundary_max_exponent=int(mutation["boundary_max_exponent"]),
            noise_ratio=float(mutation["noise_ratio"]),
            scale_factors=mutation["scale_factors"],
        )
        self.oracle = OracleEngine(**dict(self.config["oracle"]))
        self.coverage = CoverageAccumulator(self.registry)
        self.backends = self._load_backends()

    def _load_backends(self) -> list[Backend]:
        backends: list[Backend] = []
        unavailable: list[dict[str, str]] = []
        for raw in self.config["backends"]:
            if isinstance(raw, str):
                name, kwargs = raw, {}
            else:
                name = str(raw["name"])
                kwargs = dict(raw.get("args") or {})
            try:
                backends.append(create_backend(name, **kwargs))
            except BackendUnavailable as exc:
                unavailable.append({"backend": name, "error": str(exc)})
                if not self.config.get("skip_unavailable_backends", True):
                    raise
        if unavailable:
            atomic_json(self.output_dir / "unavailable_backends.json", unavailable)
        if not backends:
            raise RuntimeError("No configured backend is available")
        return backends

    def _target_nodes(self) -> int:
        low, high = (int(x) for x in self.config["generation"]["nodes"])
        return int(self.rng.integers(low, high + 1))

    def _generate(self) -> GenerationResult:
        generation = self.config["generation"]
        generator = ModelGenerator(
            task=self.task,
            catalog=self.catalog,
            registry=self.registry,
            tracker=self.tracker,
            policy=self.policy,
            rng=self.rng,
            target_nodes=self._target_nodes(),
            max_attempts_per_node=int(generation["max_attempts_per_node"]),
            saturation_accept_probability=float(generation["saturation_accept_probability"]),
            reward_mode=str(generation["reward_mode"]),
            reward_weights=dict(generation["reward_weights"]),
            reward_scale=float(generation["reward_scale"]),
            candidate_observer=self.coverage.register_candidates,
        )
        return generator.generate()

    def _case_graphs(self, generation: GenerationResult) -> list[tuple[str, Any, MutationResult | None]]:
        base = generation.graph
        base.metadata["contract_valid"] = True
        base.metadata["expected_rejection"] = False
        cases: list[tuple[str, Any, MutationResult | None]] = []
        if bool(self.config["mutation"].get("include_unmutated", True)):
            cases.append(("unmutated", base, None))
        mutation = self.mutator.apply(base)
        if mutation.records and (not cases or mutation.graph.fingerprint() != base.fingerprint()):
            cases.append(("mutated", mutation.graph, mutation))
        if not cases:
            cases.append(("unmutated", base, None))
        return cases

    def run(self) -> ExperimentSummary:
        started = perf_counter()
        process = psutil.Process(os.getpid())
        peak_rss = process.memory_info().rss
        budget = self.config["budget"]
        valid_limit = int(budget.get("valid_tests", 0) or 0)
        time_limit = float(budget.get("seconds", 0.0) or 0.0)
        total_limit = int(budget.get("total_cases", 0) or max(valid_limit * 5, 100))

        total_cases = valid_cases = invalid_cases = generated_models = 0
        generation_attempts = diversity_evaluations = fallback_count = 0
        execution_statuses: Counter[str] = Counter()
        findings_by_kind: Counter[str] = Counter()
        finding_signatures: set[str] = set()
        first_finding_time: float | None = None
        stopped_by = "unknown"

        cases_path = self.output_dir / "cases.jsonl"
        findings_path = self.output_dir / "findings.jsonl"
        generation_path = self.output_dir / "generation.jsonl"
        for path in (cases_path, findings_path, generation_path):
            path.unlink(missing_ok=True)

        while True:
            elapsed = perf_counter() - started
            if time_limit > 0 and elapsed >= time_limit:
                stopped_by = "time_budget"
                break
            if valid_limit > 0 and valid_cases >= valid_limit:
                stopped_by = "valid_test_budget"
                break
            if total_cases >= total_limit:
                stopped_by = "total_case_safety_budget"
                break

            generation = self._generate()
            generated_models += 1
            generation_attempts += generation.attempts
            diversity_evaluations += generation.diversity_evaluations
            fallback_count += generation.fallback_count
            append_jsonl(generation_path, generation.to_dict())

            for variant, graph, mutation in self._case_graphs(generation):
                elapsed = perf_counter() - started
                if time_limit > 0 and elapsed >= time_limit:
                    stopped_by = "time_budget"
                    break
                if valid_limit > 0 and valid_cases >= valid_limit:
                    stopped_by = "valid_test_budget"
                    break
                if total_cases >= total_limit:
                    stopped_by = "total_case_safety_budget"
                    break

                case_id = f"case-{total_cases:06d}"
                contract_valid = bool(graph.metadata.get("contract_valid", True))
                expected_rejection = bool(graph.metadata.get("expected_rejection", False))
                case_task = TaskSpec(
                    self.task.name,
                    graph.inputs["input"],
                    self.task.distribution,
                    dict(self.task.metadata),
                )
                inputs = (
                    self.dataset.sample(self.rng, graph.inputs["input"])
                    if self.dataset is not None
                    else generate_inputs(case_task, self.rng)
                )
                results: list[ExecutionResult] = [
                    backend.execute(
                        graph,
                        inputs,
                        contract_valid=contract_valid,
                        expected_rejection=expected_rejection,
                    )
                    for backend in self.backends
                ]
                report = self.oracle.evaluate(graph, results)

                total_cases += 1
                if contract_valid:
                    valid_cases += 1
                    self.coverage.observe_graph(graph)
                    self.coverage.observe_execution(results)
                    self.coverage.checkpoint(perf_counter() - started, valid_cases)
                else:
                    invalid_cases += 1
                for result in results:
                    execution_statuses[f"{result.backend}:{result.status}"] += 1
                for finding in report.findings:
                    findings_by_kind[finding.kind] += 1
                    finding_signatures.add(finding.signature)
                    if finding.bug_candidate and first_finding_time is None:
                        first_finding_time = perf_counter() - started
                    append_jsonl(findings_path, {"case_id": case_id, **finding.to_dict()})

                append_jsonl(
                    cases_path,
                    {
                        "case_id": case_id,
                        "variant": variant,
                        "model": graph.to_dict(include_arrays=True),
                        "model_fingerprint": graph.fingerprint(),
                        "input_summary": {
                            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                            for name, value in inputs.items()
                        },
                        "mutation": mutation.to_dict() if mutation else None,
                        "executions": [result.to_dict(include_outputs=False) for result in results],
                        "oracle": report.to_dict(),
                        "elapsed_seconds": perf_counter() - started,
                    },
                )
                peak_rss = max(peak_rss, process.memory_info().rss)

            if stopped_by != "unknown":
                break

        elapsed = perf_counter() - started
        coverage_summary = self.coverage.summary(
            enabled_layers=self.catalog.theoretical_layer_types(),
            budget_seconds=time_limit if time_limit > 0 else None,
        )
        summary = ExperimentSummary(
            run_name=str(self.config["name"]),
            seed=self.seed,
            output_dir=str(self.output_dir),
            elapsed_seconds=elapsed,
            total_cases=total_cases,
            valid_cases=valid_cases,
            invalid_cases=invalid_cases,
            generated_models=generated_models,
            execution_statuses=dict(execution_statuses),
            findings_by_kind=dict(findings_by_kind),
            unique_finding_signatures=len(finding_signatures),
            time_to_first_finding=first_finding_time,
            valid_ratio=(valid_cases / total_cases) if total_cases else 0.0,
            cases_per_second=(total_cases / elapsed) if elapsed else 0.0,
            valid_tests_per_second=(valid_cases / elapsed) if elapsed else 0.0,
            peak_rss_mb=peak_rss / (1024 * 1024),
            generation_attempts=generation_attempts,
            diversity_evaluations=diversity_evaluations,
            fallback_count=fallback_count,
            diversity_counts=self.tracker.counts(),
            coverage=coverage_summary,
            backends=[backend.environment() for backend in self.backends],
            stopped_by=stopped_by,
        )
        dump_config(self.config, self.output_dir / "config.resolved.yaml")
        atomic_json(self.output_dir / "environment.json", environment_manifest(Path(__file__).resolve().parents[1]))
        atomic_json(self.output_dir / "constraint_registry.json", self.registry.to_dict())
        atomic_json(self.output_dir / "diversity_spaces.json", self.tracker.to_dict())
        atomic_json(self.output_dir / "policy_state.json", self.policy.snapshot())
        atomic_json(self.output_dir / "summary.json", summary.to_dict())
        return summary


def run_experiment(config_path: str | Path | None = None, *, overrides: Mapping[str, Any] | None = None) -> ExperimentSummary:
    config = load_config(config_path, overrides=overrides)
    return ExperimentRunner(config).run()
