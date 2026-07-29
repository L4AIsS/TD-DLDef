"""DAG model generation with diversity-guided, terminating layer selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time

import numpy as np

from .bandits import BasePolicy, SelectionDecision
from .catalog import LayerCatalog, TaskSpec
from .constraints import ConstraintRegistry
from .diversity import DiversityGain, DiversityTracker
from .model_ir import LayerCandidate, ModelGraph


@dataclass(slots=True)
class GenerationEvent:
    node_index: int
    attempt: int
    candidate_id: str
    arm_key: str
    op: str
    accepted: bool
    fallback: bool
    reward: float
    gain: dict[str, Any]
    policy_score: float
    reason: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_index": self.node_index,
            "attempt": self.attempt,
            "candidate_id": self.candidate_id,
            "arm_key": self.arm_key,
            "op": self.op,
            "accepted": self.accepted,
            "fallback": self.fallback,
            "reward": self.reward,
            "gain": self.gain,
            "policy_score": self.policy_score,
            "reason": self.reason,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(slots=True)
class GenerationResult:
    graph: ModelGraph
    events: list[GenerationEvent]
    attempts: int
    diversity_evaluations: int
    fallback_count: int
    elapsed_seconds: float
    policy_snapshot: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_fingerprint": self.graph.fingerprint(),
            "node_count": len(self.graph.nodes),
            "attempts": self.attempts,
            "diversity_evaluations": self.diversity_evaluations,
            "fallback_count": self.fallback_count,
            "elapsed_seconds": self.elapsed_seconds,
            "events": [event.to_dict() for event in self.events],
            "policy_snapshot": self.policy_snapshot,
        }


class ModelGenerator:
    """Generate one model while preserving constraints and guaranteeing termination."""

    def __init__(
        self,
        *,
        task: TaskSpec,
        catalog: LayerCatalog,
        registry: ConstraintRegistry,
        tracker: DiversityTracker,
        policy: BasePolicy,
        rng: np.random.Generator,
        target_nodes: int,
        max_attempts_per_node: int = 24,
        saturation_accept_probability: float = 0.02,
        reward_mode: str = "binary",
        reward_weights: dict[str, float] | None = None,
        reward_scale: float = 4.0,
        candidate_observer: Any | None = None,
    ) -> None:
        if target_nodes <= 0:
            raise ValueError("target_nodes must be positive")
        if max_attempts_per_node <= 0:
            raise ValueError("max_attempts_per_node must be positive")
        if not 0.0 <= saturation_accept_probability <= 1.0:
            raise ValueError("saturation_accept_probability must be in [0, 1]")
        self.task = task
        self.catalog = catalog
        self.registry = registry
        self.tracker = tracker
        self.policy = policy
        self.rng = rng
        self.target_nodes = int(target_nodes)
        self.max_attempts_per_node = int(max_attempts_per_node)
        self.saturation_accept_probability = float(saturation_accept_probability)
        self.reward_mode = reward_mode
        self.reward_weights = reward_weights
        self.reward_scale = max(float(reward_scale), 1e-12)
        self.candidate_observer = candidate_observer

    def _reward(self, gain: DiversityGain) -> float:
        if self.reward_mode == "binary":
            return float(gain.binary_reward)
        if self.reward_mode == "magnitude":
            return min(1.0, gain.weighted_reward(self.reward_weights) / self.reward_scale)
        raise ValueError(f"Unknown reward mode: {self.reward_mode}")

    def _valid_candidates(self, graph: ModelGraph) -> list[LayerCandidate]:
        candidates = self.catalog.enumerate(graph, self.rng)
        valid: list[LayerCandidate] = []
        for candidate in candidates:
            params_ok = all(self.registry.validate_param(candidate.op, k, v) for k, v in candidate.params.items())
            dtypes_ok = all(self.registry.supports_dtype(candidate.op, spec.dtype) for spec in candidate.input_specs)
            if params_ok and dtypes_ok:
                valid.append(candidate)
        if self.candidate_observer is not None:
            self.candidate_observer(graph, valid)
        return valid

    def generate(self) -> GenerationResult:
        started = time.perf_counter()
        graph = ModelGraph(
            inputs={"input": self.task.input_spec},
            metadata={"task": self.task.name, "constraint_registry": self.registry.version},
        )
        events: list[GenerationEvent] = []
        attempts_total = 0
        diversity_evaluations = 0
        fallback_count = 0

        for node_index in range(self.target_nodes):
            candidates = self._valid_candidates(graph)
            if not candidates:
                raise RuntimeError(f"No contract-valid candidate at node {node_index}")
            remaining = list(candidates)
            accepted = False
            attempts_limit = min(self.max_attempts_per_node, max(1, len(remaining)))

            for attempt in range(attempts_limit):
                event_started = time.perf_counter()
                decision: SelectionDecision = self.policy.select(remaining, self.rng)
                candidate = decision.candidate
                gain = self.tracker.preview(graph, candidate)
                diversity_evaluations += 1
                reward = self._reward(gain)
                self.policy.update(candidate.arm_key, reward)
                attempts_total += 1

                diversity_accept = gain.binary_reward == 1
                saturated_exploration = (
                    not diversity_accept and self.rng.random() < self.saturation_accept_probability
                )
                accepted = diversity_accept or saturated_exploration
                reason = "diversity_gain" if diversity_accept else ("saturation_exploration" if saturated_exploration else "no_new_diversity")
                events.append(
                    GenerationEvent(
                        node_index=node_index,
                        attempt=attempt,
                        candidate_id=candidate.candidate_id,
                        arm_key=candidate.arm_key,
                        op=candidate.op,
                        accepted=accepted,
                        fallback=False,
                        reward=reward,
                        gain=gain.to_dict(),
                        policy_score=decision.score,
                        reason=reason,
                        elapsed_seconds=time.perf_counter() - event_started,
                    )
                )
                if accepted:
                    self.tracker.commit(graph, candidate)
                    graph.add_candidate(candidate)
                    break
                remaining = [c for c in remaining if c.candidate_id != candidate.candidate_id]
                if not remaining:
                    break

            if accepted:
                continue

            # Explicit fallback guarantees progress when all four sets are saturated or
            # repeated selections fail to add diversity.  The fallback chooses the valid
            # candidate with maximum observed marginal gain, then randomises ties.
            fallback_count += 1
            fallback_started = time.perf_counter()
            gains = [self.tracker.preview(graph, candidate) for candidate in candidates]
            diversity_evaluations += len(candidates)
            scores = [gain.weighted_reward(self.reward_weights) for gain in gains]
            max_score = max(scores)
            best = [i for i, score in enumerate(scores) if np.isclose(score, max_score)]
            index = int(self.rng.choice(best))
            candidate = candidates[index]
            gain = gains[index]
            fallback_reward = self._reward(gain)
            self.policy.update(candidate.arm_key, fallback_reward)
            attempts_total += 1
            prefix = graph.clone()
            self.tracker.commit(prefix, candidate)
            graph.add_candidate(candidate)
            events.append(
                GenerationEvent(
                    node_index=node_index,
                    attempt=attempts_limit,
                    candidate_id=candidate.candidate_id,
                    arm_key=candidate.arm_key,
                    op=candidate.op,
                    accepted=True,
                    fallback=True,
                    reward=fallback_reward,
                    gain=gain.to_dict(),
                    policy_score=float(max_score),
                    reason="termination_fallback",
                    elapsed_seconds=time.perf_counter() - fallback_started,
                )
            )

        errors = self.registry.validate_graph(graph)
        if errors:
            raise RuntimeError("Generated graph failed validation: " + "; ".join(errors))
        graph.metadata["generation"] = {
            "policy": self.policy.name,
            "reward_mode": self.reward_mode,
            "enabled_diversity_spaces": list(self.tracker.enabled_spaces),
            "attempts": attempts_total,
            "fallback_count": fallback_count,
        }
        return GenerationResult(
            graph=graph,
            events=events,
            attempts=attempts_total,
            diversity_evaluations=diversity_evaluations,
            fallback_count=fallback_count,
            elapsed_seconds=time.perf_counter() - started,
            policy_snapshot=self.policy.snapshot(),
        )
