"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from .config import load_config
from .runner import BenchmarkRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run automated prefill/decode vLLM workload sweeps in Docker."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-container", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config, expanded = load_config(args.config, args.run_name)
        runner = BenchmarkRunner(
            config,
            expanded,
            dry_run=args.dry_run,
            resume=args.resume,
            keep_container=args.keep_container,
        )
        return runner.run()
    except (OSError, ValueError, RuntimeError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
