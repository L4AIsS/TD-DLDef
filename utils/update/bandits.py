"""Layer-selection policies, including a corrected Thompson sampler.

Critical repairs relative to the public repository:
1. Thompson selection is the argmax of one Beta draw per candidate arm; it is not
   overwritten by a subsequent uniform-random choice.
2. A success increments alpha and a failure increments beta.
3. Rewards are supplied by the four-space diversity evaluator rather than by historical
   selection frequency.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

from .model_ir import LayerCandidate


@dataclass(slots=True)
class ArmState:
    successes: float = 1.0
    failures: float = 1.0
    pulls: int = 0
    reward_sum: float = 0.0

    @property
    def mean(self) -> float:
        if self.pulls == 0:
            return 0.5
        return self.reward_sum / self.pulls

    def update(self, reward: float) -> None:
        if not 0.0 <= reward <= 1.0:
            raise ValueError(f"Bandit reward must be in [0, 1], got {reward}")
        self.successes += reward
        self.failures += 1.0 - reward
        self.pulls += 1
        self.reward_sum += reward


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    candidate: LayerCandidate
    policy: str
    score: float
    arm_key: str
    diagnostics: dict[str, Any]


class LayerSelectionPolicy(Protocol):
    name: str

    def select(self, candidates: Sequence[LayerCandidate], rng: np.random.Generator) -> SelectionDecision: ...
    def update(self, arm_key: str, reward: float) -> None: ...
    def snapshot(self) -> dict[str, Any]: ...


class BasePolicy:
    name = "base"

    def __init__(self, *, alpha: float = 1.0, beta: float = 1.0) -> None:
        if alpha <= 0 or beta <= 0:
            raise ValueError("Beta-prior parameters must be positive")
        self.alpha0 = float(alpha)
        self.beta0 = float(beta)
        self._states: dict[str, ArmState] = {}
        self.total_pulls = 0

    def state(self, key: str) -> ArmState:
        if key not in self._states:
            self._states[key] = ArmState(self.alpha0, self.beta0)
        return self._states[key]

    def update(self, arm_key: str, reward: float) -> None:
        self.state(arm_key).update(float(reward))
        self.total_pulls += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "policy": self.name,
            "total_pulls": self.total_pulls,
            "arms": {
                key: {
                    "alpha": state.successes,
                    "beta": state.failures,
                    "pulls": state.pulls,
                    "mean": state.mean,
                    "reward_sum": state.reward_sum,
                }
                for key, state in sorted(self._states.items())
            },
        }

    @staticmethod
    def _check(candidates: Sequence[LayerCandidate]) -> None:
        if not candidates:
            raise ValueError("Cannot select from an empty candidate set")

    @staticmethod
    def _argmax_with_random_tie(scores: Sequence[float], rng: np.random.Generator) -> int:
        max_score = max(scores)
        indices = [i for i, score in enumerate(scores) if np.isclose(score, max_score)]
        return int(rng.choice(indices))


class RandomPolicy(BasePolicy):
    name = "random"

    def select(self, candidates: Sequence[LayerCandidate], rng: np.random.Generator) -> SelectionDecision:
        self._check(candidates)
        index = int(rng.integers(0, len(candidates)))
        candidate = candidates[index]
        return SelectionDecision(candidate, self.name, 0.0, candidate.arm_key, {"index": index})


class GreedyPolicy(BasePolicy):
    """Greedy empirical-mean policy with random tie breaking."""

    name = "greedy"

    def select(self, candidates: Sequence[LayerCandidate], rng: np.random.Generator) -> SelectionDecision:
        self._check(candidates)
        scores = [self.state(c.arm_key).mean for c in candidates]
        index = self._argmax_with_random_tie(scores, rng)
        candidate = candidates[index]
        return SelectionDecision(candidate, self.name, float(scores[index]), candidate.arm_key, {"scores": scores})


class EpsilonGreedyPolicy(GreedyPolicy):
    name = "epsilon_greedy"

    def __init__(self, *, epsilon: float = 0.1, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__(alpha=alpha, beta=beta)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = float(epsilon)

    def select(self, candidates: Sequence[LayerCandidate], rng: np.random.Generator) -> SelectionDecision:
        self._check(candidates)
        explore = bool(rng.random() < self.epsilon)
        if explore:
            index = int(rng.integers(0, len(candidates)))
            candidate = candidates[index]
            return SelectionDecision(candidate, self.name, 0.0, candidate.arm_key, {"explore": True, "epsilon": self.epsilon})
        decision = super().select(candidates, rng)
        return SelectionDecision(decision.candidate, self.name, decision.score, decision.arm_key, decision.diagnostics | {"explore": False, "epsilon": self.epsilon})


class UCB1Policy(BasePolicy):
    name = "ucb1"

    def select(self, candidates: Sequence[LayerCandidate], rng: np.random.Generator) -> SelectionDecision:
        self._check(candidates)
        scores: list[float] = []
        for candidate in candidates:
            state = self.state(candidate.arm_key)
            if state.pulls == 0:
                score = float("inf")
            else:
                score = state.mean + sqrt(2.0 * log(max(1, self.total_pulls)) / state.pulls)
            scores.append(score)
        if any(np.isinf(scores)):
            unseen = [i for i, score in enumerate(scores) if np.isinf(score)]
            index = int(rng.choice(unseen))
        else:
            index = self._argmax_with_random_tie(scores, rng)
        candidate = candidates[index]
        return SelectionDecision(candidate, self.name, float(scores[index]), candidate.arm_key, {"scores": scores})


class ThompsonSamplingPolicy(BasePolicy):
    name = "thompson"

    def select(self, candidates: Sequence[LayerCandidate], rng: np.random.Generator) -> SelectionDecision:
        self._check(candidates)
        # One posterior sample per arm key. Multiple concrete candidates of the same layer
        # type share the learned posterior but receive independent, stable tie breaking.
        samples_by_arm: dict[str, float] = {}
        for candidate in candidates:
            if candidate.arm_key not in samples_by_arm:
                state = self.state(candidate.arm_key)
                samples_by_arm[candidate.arm_key] = float(rng.beta(state.successes, state.failures))
        scores = [samples_by_arm[c.arm_key] for c in candidates]
        index = self._argmax_with_random_tie(scores, rng)
        candidate = candidates[index]
        return SelectionDecision(
            candidate,
            self.name,
            float(scores[index]),
            candidate.arm_key,
            {"posterior_samples": samples_by_arm, "scores": scores},
        )


POLICIES = {
    "random": RandomPolicy,
    "greedy": GreedyPolicy,
    "epsilon_greedy": EpsilonGreedyPolicy,
    "epsilon-greedy": EpsilonGreedyPolicy,
    "ucb1": UCB1Policy,
    "thompson": ThompsonSamplingPolicy,
    "ts": ThompsonSamplingPolicy,
}


def create_policy(name: str, **kwargs: Any) -> BasePolicy:
    key = name.strip().lower()
    if key not in POLICIES:
        raise ValueError(f"Unknown policy {name!r}; choose from {sorted(POLICIES)}")
    cls = POLICIES[key]
    return cls(**kwargs)
