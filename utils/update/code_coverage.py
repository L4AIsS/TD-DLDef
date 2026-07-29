"""Import line/branch coverage produced by coverage.py or llvm-cov.

TensorFlow/PyTorch binary wheels do not expose meaningful C/C++ branch coverage. For
backend-internal coverage, build the framework with instrumentation and export llvm-cov
JSON. The loader keeps this evidence separate from structural model coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def load_coverage_py_json(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    totals = raw.get("totals", {})
    return {
        "source": "coverage.py",
        "line_coverage_percent": totals.get("percent_covered"),
        "covered_lines": totals.get("covered_lines"),
        "num_statements": totals.get("num_statements"),
        "covered_branches": totals.get("covered_branches"),
        "num_branches": totals.get("num_branches"),
        "branch_coverage_percent": (
            100.0 * totals.get("covered_branches", 0) / totals.get("num_branches", 1)
            if totals.get("num_branches", 0) else None
        ),
    }


def load_llvm_cov_json(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    data = raw.get("data", [])
    line_count = line_covered = branch_count = branch_covered = 0
    for unit in data:
        totals = unit.get("totals", {})
        lines = totals.get("lines", {})
        branches = totals.get("branches", {})
        line_count += int(lines.get("count", 0))
        line_covered += int(lines.get("covered", 0))
        branch_count += int(branches.get("count", 0))
        branch_covered += int(branches.get("covered", 0))
    return {
        "source": "llvm-cov",
        "covered_lines": line_covered,
        "num_statements": line_count,
        "line_coverage_percent": 100.0 * line_covered / line_count if line_count else None,
        "covered_branches": branch_covered,
        "num_branches": branch_count,
        "branch_coverage_percent": 100.0 * branch_covered / branch_count if branch_count else None,
    }
