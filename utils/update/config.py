"""Configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "name": "td_dldef_default",
    "seed": 20260525,
    "task": "vision",
    "dataset": {"name": "synthetic", "split": "test", "local_path": None, "limit": 256},
    "backends": ["numpy"],
    "skip_unavailable_backends": True,
    "generation": {
        "policy": "thompson",
        "policy_args": {"alpha": 1.0, "beta": 1.0},
        "nodes": [6, 10],
        "candidate_window": 5,
        "max_candidates": 48,
        "max_attempts_per_node": 24,
        "arm_granularity": "layer_type",
        "reward_mode": "binary",
        "reward_weights": {"layer": 1.0, "edge": 1.0, "input_shape": 1.0, "input_dimension": 1.0},
        "reward_scale": 4.0,
        "enabled_diversity_spaces": ["layer", "edge", "input_shape", "input_dimension"],
        "saturation_accept_probability": 0.02,
        "enabled_ops": None,
    },
    "mutation": {
        "enabled": ["PV", "BV", "IT", "NF", "SW", "WR"],
        "mode": "contract_valid",
        "max_mutations": 4,
        "boundary_max_exponent": 64,
        "noise_ratio": 0.05,
        "scale_factors": [0.5, 0.9, 1.1, 2.0, -1.0],
        "include_unmutated": True,
    },
    "oracle": {
        "normalised_mad_threshold": 1e-3,
        "require_prediction_change": False,
        "extreme_value_threshold": 1e12,
        "report_inf": True,
    },
    "budget": {
        "valid_tests": 20,
        "seconds": 60.0,
    },
    "output": "results/default",
}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            raise ValueError("Configuration root must be a mapping")
        config = deep_merge(config, raw)
    if overrides:
        config = deep_merge(config, overrides)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    generation = config["generation"]
    nodes = generation["nodes"]
    if not isinstance(nodes, (list, tuple)) or len(nodes) != 2 or int(nodes[0]) <= 0 or int(nodes[1]) < int(nodes[0]):
        raise ValueError("generation.nodes must be [positive_min, max>=min]")
    budget = config["budget"]
    if int(budget.get("valid_tests", 0)) <= 0 and float(budget.get("seconds", 0.0)) <= 0:
        raise ValueError("At least one positive budget must be provided")
    if not config.get("backends"):
        raise ValueError("At least one backend must be configured")


def dump_config(config: Mapping[str, Any], path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True), encoding="utf-8")
