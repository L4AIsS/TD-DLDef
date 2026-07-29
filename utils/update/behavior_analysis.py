"""Structural-versus-behavioural diversity correlation experiment."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .backends.numpy_backend import NumpyBackend
from .bandits import create_policy
from .catalog import LayerCatalog, built_in_task
from .constraints import default_registry
from .diversity import DiversityTracker
from .execution import generate_inputs
from .generator import ModelGenerator
from .model_ir import ModelGraph
from .reporting import atomic_json, write_csv
from .statistics import spearman_correlation


def _jaccard_distance(a: set[Any], b: set[Any]) -> float:
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


def structural_distance(a: ModelGraph, b: ModelGraph) -> float:
    def spaces(graph: ModelGraph):
        layers = {node.op for node in graph.nodes}
        edges = set(graph.iter_edges())
        shapes = {(node.op, tuple(graph.get_spec(ref).shape)) for node in graph.nodes for ref in node.inputs}
        dims = {(node.op, graph.get_spec(ref).ndim) for node in graph.nodes for ref in node.inputs}
        return layers, edges, shapes, dims
    return float(np.mean([_jaccard_distance(x, y) for x, y in zip(spaces(a), spaces(b))]))


def behavioural_distance(a: np.ndarray, b: np.ndarray) -> float:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        return 1.0
    denominator = np.linalg.norm(x) + np.linalg.norm(y) + 1e-12
    return float(min(1.0, np.linalg.norm(x - y) / denominator))


def run_behavior_correlation(
    *,
    output_dir: str | Path,
    seed: int = 20260525,
    task_name: str = "vision",
    models: int = 20,
    nodes: int = 8,
    max_pairs: int = 500,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    task = built_in_task(task_name)
    registry = default_registry()
    tracker = DiversityTracker()
    policy = create_policy("thompson")
    catalog = LayerCatalog(task=task)
    backend = NumpyBackend()
    common_inputs = generate_inputs(task, np.random.default_rng(seed + 1))

    graphs: list[ModelGraph] = []
    outputs: list[np.ndarray] = []
    for _ in range(models):
        generator = ModelGenerator(
            task=task,
            catalog=catalog,
            registry=registry,
            tracker=tracker,
            policy=policy,
            rng=rng,
            target_nodes=nodes,
            max_attempts_per_node=24,
        )
        graph = generator.generate().graph
        result = backend.execute(graph, common_inputs)
        if result.status != "ok":
            continue
        graphs.append(graph)
        outputs.append(next(iter(result.outputs.values())))

    all_pairs = list(combinations(range(len(graphs)), 2))
    if len(all_pairs) > max_pairs:
        selected = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        all_pairs = [all_pairs[int(i)] for i in selected]
    rows = []
    structure_values = []
    behavior_values = []
    for left, right in all_pairs:
        sd = structural_distance(graphs[left], graphs[right])
        bd = behavioural_distance(outputs[left], outputs[right])
        structure_values.append(sd)
        behavior_values.append(bd)
        rows.append({"left": left, "right": right, "structural_distance": sd, "behavioural_distance": bd})
    correlation = spearman_correlation(structure_values, behavior_values)
    summary = {
        "seed": seed,
        "task": task_name,
        "requested_models": models,
        "executable_models": len(graphs),
        "pairs": len(rows),
        "spearman_correlation": correlation,
        "interpretation": "A high positive value supports association, not equivalence or causality.",
    }
    write_csv(output / "pairs.csv", rows)
    atomic_json(output / "summary.json", summary)
    return summary
