"""Shared command-line helpers for experiment scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def seeds_from_text(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def matrix_parser(description: str, default_output: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--quick", action="store_true", help="Use one seed and a small valid-test budget")
    return parser


def resolve_seeds(args) -> list[int]:
    seeds = seeds_from_text(args.seeds)
    return seeds[:1] if args.quick else seeds
