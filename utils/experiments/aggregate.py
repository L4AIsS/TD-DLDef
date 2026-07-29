from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from td_dldef.experiment_matrix import aggregate_rows
from td_dldef.reporting import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate experiment runs.csv")
    parser.add_argument("runs_csv")
    parser.add_argument("--output", default="aggregate.json")
    args = parser.parse_args(argv)
    with Path(args.runs_csv).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = aggregate_rows(rows)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
