from __future__ import annotations

import argparse
import json
from pathlib import Path

from td_dldef.external_baselines import BaselineRun, load_baselines, run_external
from td_dldef.reporting import atomic_json, write_csv


def parse_seeds(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run checked-out external baselines under one controlled protocol")
    parser.add_argument("--baselines", default="artifacts/external_baselines.example.yaml")
    parser.add_argument("--output", default="results/external_baselines")
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--seconds", type=float, default=21600)
    parser.add_argument("--valid-tests", type=int, default=1000)
    args = parser.parse_args(argv)
    config = load_baselines(args.baselines)
    track = str(config.get("framework_track", "unspecified"))
    root = Path(args.output).resolve()
    results = []
    for name, item in dict(config.get("baselines", {})).items():
        for seed in parse_seeds(args.seeds):
            output = root / name / f"seed-{seed}"
            result = run_external(BaselineRun(
                name=name,
                command=[str(x) for x in item["command"]],
                workdir=Path(item["workdir"]),
                seed=seed,
                seconds=args.seconds,
                valid_tests=args.valid_tests,
                framework_track=track,
                output_dir=output,
            ))
            results.append(result)
    atomic_json(root / "runs.json", results)
    write_csv(root / "runs.csv", [
        {
            "name": r["name"], "seed": r["seed"], "status": r["status"],
            "elapsed_seconds": r["elapsed_seconds"], "framework_track": r["framework_track"],
            "output": str(root / r["name"] / f"seed-{r['seed']}"),
        }
        for r in results
    ])
    print(json.dumps({"runs": len(results), "output": str(root)}, indent=2))


if __name__ == "__main__":
    main()
