"""Run controlled experiment matrices and aggregate repeated trials."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import json

from .config import deep_merge, load_config
from .reporting import atomic_json, write_csv
from .runner import ExperimentRunner
from .statistics import cliffs_delta, summarise


def run_matrix(
    *,
    base_config_path: str | Path | None,
    variants: Mapping[str, Mapping[str, Any]],
    seeds: Sequence[int],
    output_root: str | Path,
) -> list[dict[str, Any]]:
    base = load_config(base_config_path)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for variant_name, override in variants.items():
        for seed in seeds:
            output = root / variant_name / f"seed-{seed}"
            config = deep_merge(base, override)
            config["name"] = variant_name
            config["seed"] = int(seed)
            config["output"] = str(output)
            summary = ExperimentRunner(config).run().to_dict()
            row = {
                "variant": variant_name,
                "seed": seed,
                "output": str(output),
                "valid_cases": summary["valid_cases"],
                "valid_ratio": summary["valid_ratio"],
                "valid_tests_per_second": summary["valid_tests_per_second"],
                "time_to_first_finding": summary["time_to_first_finding"],
                "unique_finding_signatures": summary["unique_finding_signatures"],
                "peak_rss_mb": summary["peak_rss_mb"],
                "LIC": summary["coverage"]["LIC"],
                "LPC": summary["coverage"]["LPC"],
                "LSC": summary["coverage"]["LSC"],
                "coverage_auc": summary["coverage"]["coverage_auc"],
                "behavior_signatures": summary["coverage"]["unique_behavior_signatures"],
                "fallback_count": summary["fallback_count"],
            }
            rows.append(row)
    write_csv(root / "runs.csv", rows)
    aggregate = aggregate_rows(rows)
    atomic_json(root / "aggregate.json", aggregate)
    return rows


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [
        "valid_ratio",
        "valid_tests_per_second",
        "time_to_first_finding",
        "unique_finding_signatures",
        "peak_rss_mb",
        "LIC",
        "LPC",
        "LSC",
        "coverage_auc",
        "behavior_signatures",
        "fallback_count",
    ]
    variants = sorted({str(row["variant"]) for row in rows})
    result: dict[str, Any] = {"variants": {}}
    for variant in variants:
        subset = [row for row in rows if row["variant"] == variant]
        result["variants"][variant] = {}
        for metric in metrics:
            values = [float(row[metric]) for row in subset if row.get(metric) is not None]
            result["variants"][variant][metric] = summarise(values).to_dict()
    if "thompson" in variants:
        reference = [row for row in rows if row["variant"] == "thompson"]
        result["effect_sizes_vs_thompson"] = {}
        for variant in variants:
            if variant == "thompson":
                continue
            subset = [row for row in rows if row["variant"] == variant]
            result["effect_sizes_vs_thompson"][variant] = {
                metric: cliffs_delta(
                    [float(row[metric]) for row in reference if row.get(metric) is not None],
                    [float(row[metric]) for row in subset if row.get(metric) is not None],
                )
                for metric in metrics
            }
    return result
