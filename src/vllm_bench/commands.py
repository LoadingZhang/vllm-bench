"""Command construction with literal CLI argument preservation."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from .config import ArgValue, RawArgs


def merge_args(*arg_maps: RawArgs) -> RawArgs:
    """Merge argument maps while preserving first insertion order."""

    merged: RawArgs = {}
    for arg_map in arg_maps:
        for key, value in arg_map.items():
            merged[key] = value
    return merged


def append_raw_args(command: list[str], args: RawArgs) -> list[str]:
    """Append literal CLI args without normalizing names or values."""

    result = list(command)
    for key, value in args.items():
        if value is None:
            result.append(key)
        elif "." in key.lstrip("-") and not isinstance(value, list):
            result.append(f"{key}={value}")
        elif isinstance(value, list):
            result.append(key)
            result.extend(str(item) for item in value)
        else:
            result.extend((key, str(value)))
    return result


def format_command(command: list[str]) -> str:
    return shlex.join(command)


def build_serve_command(model: str, fixed: RawArgs, variant: RawArgs) -> list[str]:
    return append_raw_args(["vllm", "serve", model], merge_args(fixed, variant))


def build_bench_base_command(wrapper_path: str, profile_path: str) -> list[str]:
    return ["python3", wrapper_path, "--profiles", profile_path]


def write_profiles(path: Path, fixed_args: RawArgs, stages: dict[str, RawArgs]) -> None:
    payload = {
        stage: {"args": merge_args(fixed_args, args)} for stage, args in stages.items()
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_bench_params(path: Path, stage_names: list[str]) -> None:
    path.write_text(
        json.dumps(
            [
                {"_benchmark_name": stage, "vllm_bench_profile": stage}
                for stage in stage_names
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


def build_sweep_command(
    *,
    serve_command: list[str],
    bench_command: list[str],
    bench_params_path: str,
    sweep_args: RawArgs,
    output_dir: str,
    experiment_name: str,
    resume: bool,
) -> list[str]:
    command = [
        "vllm",
        "bench",
        "sweep",
        "serve_workload",
        "--serve-cmd",
        format_command(serve_command),
        "--bench-cmd",
        format_command(bench_command),
        "--bench-params",
        bench_params_path,
        "--output-dir",
        output_dir,
        "--experiment-name",
        experiment_name,
    ]
    command = append_raw_args(command, sweep_args)
    if resume and "--resume" not in sweep_args:
        command.append("--resume")
    return command


def literal_value(value: Any) -> ArgValue:
    """Narrow a decoded JSON value for type checkers."""

    return value
