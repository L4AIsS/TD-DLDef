"""Artifact output helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
import csv
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile


def atomic_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        temp = Path(handle.name)
    temp.replace(target)


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str) + "\n")


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        target.write_text("", encoding="utf-8")
        return
    fieldnames = fieldnames or sorted({key for row in rows for key in row})
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git_metadata(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def environment_manifest(root: str | Path | None = None) -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "PyYAML", "pandas", "scipy", "psutil", "pytest", "tensorflow", "torch", "keras"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "git": _git_metadata(Path(root or Path.cwd())),
    }
