"""Container-side wrapper that expands a literal benchmark profile."""

from __future__ import annotations

import argparse
import json
import os


def append_args(command: list[str], args: dict[str, object]) -> None:
    for key, value in args.items():
        if value is None:
            command.append(key)
        elif "." in key.lstrip("-") and not isinstance(value, list):
            command.append(f"{key}={value}")
        elif isinstance(value, list):
            command.append(key)
            command.extend(str(item) for item in value)
        else:
            command.extend((key, str(value)))


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--vllm-bench-profile", required=True)
    known, remaining = parser.parse_known_args()

    with open(known.profiles, encoding="utf-8") as file:
        profiles = json.load(file)
    profile = profiles[known.vllm_bench_profile]

    command = ["vllm", "bench", "serve"]
    append_args(command, profile["args"])
    command.extend(remaining)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
