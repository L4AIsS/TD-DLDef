from __future__ import annotations

import argparse
import json

from td_dldef.code_coverage import load_coverage_py_json, load_llvm_cov_json
from td_dldef.reporting import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Normalise coverage.py or llvm-cov JSON")
    parser.add_argument("input")
    parser.add_argument("--format", choices=["coverage.py", "llvm-cov"], required=True)
    parser.add_argument("--output", default="normalised_code_coverage.json")
    args = parser.parse_args(argv)
    result = load_coverage_py_json(args.input) if args.format == "coverage.py" else load_llvm_cov_json(args.input)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
