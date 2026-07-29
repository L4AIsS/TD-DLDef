"""Single-source bug ledger and conservative duplicate clustering."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping
import argparse
import csv
import hashlib
import json
import re

from .reporting import write_csv, atomic_json

LEDGER_FIELDS = [
    "finding_id", "framework", "framework_version", "backend", "symptom",
    "test_mode", "layer_or_api", "exception_type", "top_stack_frames",
    "minimal_reproducer", "issue_url", "developer_status", "root_cause_id",
    "known_issue", "security_class", "cve", "notes",
]


def normalise_stack(text: str) -> str:
    text = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text or "")
    text = re.sub(r"(?:/[^\s:]+)+", "/PATH", text)
    text = re.sub(r"line \d+", "line N", text)
    return "\n".join(line.strip() for line in text.splitlines()[:8] if line.strip())


def candidate_signature(row: Mapping[str, Any]) -> str:
    payload = {
        "framework": row.get("framework", ""),
        "symptom": row.get("symptom", ""),
        "layer_or_api": row.get("layer_or_api", ""),
        "exception_type": row.get("exception_type", ""),
        "stack": normalise_stack(str(row.get("top_stack_frames", ""))),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def read_ledger(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_ledger(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    candidate_clusters: dict[str, list[str]] = {}
    status_counts: dict[str, int] = {}
    confirmed_root_causes: set[str] = set()
    for row in rows:
        signature = candidate_signature(row)
        candidate_clusters.setdefault(signature, []).append(str(row.get("finding_id", "")))
        status = str(row.get("developer_status", "unclassified") or "unclassified")
        status_counts[status] = status_counts.get(status, 0) + 1
        root = str(row.get("root_cause_id", "")).strip()
        if root and status.lower() in {"confirmed", "fixed", "accepted"}:
            confirmed_root_causes.add(root)
    return {
        "finding_instances": len(rows),
        "candidate_signature_clusters": len(candidate_clusters),
        "developer_status_counts": status_counts,
        "developer_confirmed_unique_root_causes": len(confirmed_root_causes),
        "candidate_clusters": candidate_clusters,
        "warning": "Candidate signatures assist triage but never replace developer/root-cause confirmation.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger")
    parser.add_argument("--output", default="bug_ledger_audit.json")
    args = parser.parse_args(argv)
    audit = audit_ledger(read_ledger(args.ledger))
    atomic_json(args.output, audit)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
