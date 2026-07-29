"""Statistical summaries required by the revised evaluation protocol."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class SummaryStatistics:
    n: int
    mean: float
    std: float
    ci95_low: float
    ci95_high: float
    median: float
    minimum: float
    maximum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "ci95_low": self.ci95_low,
            "ci95_high": self.ci95_high,
            "median": self.median,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


def summarise(values: Iterable[float]) -> SummaryStatistics:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return SummaryStatistics(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    # Student-t critical values for small common sample sizes; asymptotic 1.96 beyond 30.
    t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}
    df = max(1, arr.size - 1)
    nearest = min(t_table, key=lambda x: abs(x - df)) if df <= 30 else None
    critical = t_table[nearest] if nearest is not None else 1.96
    half = critical * std / sqrt(arr.size) if arr.size > 1 else 0.0
    return SummaryStatistics(
        int(arr.size), mean, std, mean - half, mean + half,
        float(np.median(arr)), float(np.min(arr)), float(np.max(arr)),
    )


def bootstrap_ci(values: Sequence[float], *, confidence: float = 0.95, resamples: int = 5000, seed: int = 0) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        means[i] = np.mean(rng.choice(arr, size=arr.size, replace=True))
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    greater = 0
    less = 0
    for value in x:
        greater += int(np.sum(value > y))
        less += int(np.sum(value < y))
    return float((greater - less) / (x.size * y.size))


def rankdata(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(arr, dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        average = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = average
        i = j
    return ranks


def spearman_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])
