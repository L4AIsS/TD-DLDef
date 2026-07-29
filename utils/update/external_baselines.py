"""Controlled adapter for external baseline repositories.

The adapter does not silently reimplement third-party methods. It runs each checked-out
baseline under the same declared framework image, wall-clock/valid-test budget, seeds,
and output schema. This avoids introducing reimplementation bias into the main results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json
import os
import shlex
import subprocess
import time

import yaml

from .reporting import atomic_json


@dataclass(frozen=True, slots=True)
class BaselineRun:
    name: str
    command: list[str]
    workdir: Path
    seed: int
    seconds: float
    valid_tests: int
    framework_track: str
    output_dir: Path


def load_baselines(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("baseline configuration must be a mapping")
    return raw


def run_external(run: BaselineRun) -> dict[str, Any]:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "TD_DLDEF_SEED": str(run.seed),
        "TD_DLDEF_TIME_BUDGET_SECONDS": str(run.seconds),
        "TD_DLDEF_VALID_TEST_BUDGET": str(run.valid_tests),
        "TD_DLDEF_FRAMEWORK_TRACK": run.framework_track,
        "TD_DLDEF_OUTPUT_DIR": str(run.output_dir),
    })
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            run.command,
            cwd=run.workdir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=run.seconds + max(30.0, run.seconds * 0.1),
            check=False,
        )
        status = "ok" if completed.returncode == 0 else "nonzero_exit"
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        returncode = None
        completed = exc
    result = {
        "name": run.name,
        "seed": run.seed,
        "framework_track": run.framework_track,
        "command": run.command,
        "workdir": str(run.workdir),
        "seconds_budget": run.seconds,
        "valid_tests_budget": run.valid_tests,
        "elapsed_seconds": time.perf_counter() - started,
        "status": status,
        "returncode": returncode,
        "stdout": (completed.stdout or "")[-200000:] if isinstance(completed.stdout, str) else "",
        "stderr": (completed.stderr or "")[-200000:] if isinstance(completed.stderr, str) else "",
    }
    atomic_json(run.output_dir / "external_run.json", result)
    expected = run.output_dir / "summary.json"
    if expected.exists():
        try:
            result["summary"] = json.loads(expected.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result["summary_parse_error"] = True
    return result
