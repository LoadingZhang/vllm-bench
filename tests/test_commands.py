import json
from pathlib import Path

from vllm_bench.commands import (
    append_raw_args,
    build_serve_command,
    build_sweep_command,
    write_bench_params,
    write_profiles,
)


def test_literal_argument_rendering() -> None:
    command = append_raw_args(
        ["vllm"],
        {
            "-tp": 8,
            "-ep": None,
            "--kernel-config.enable_flashinfer_autotune": "False",
            "--header": ["a=1", "b=2"],
        },
    )
    assert command == [
        "vllm",
        "-tp",
        "8",
        "-ep",
        "--kernel-config.enable_flashinfer_autotune=False",
        "--header",
        "a=1",
        "b=2",
    ]


def test_variant_overrides_exact_same_key() -> None:
    command = build_serve_command(
        "/models/model",
        {"-tp": 4, "--tensor-parallel-size": 2},
        {"-tp": 8},
    )
    assert command.count("-tp") == 1
    assert command[command.index("-tp") + 1] == "8"
    assert "--tensor-parallel-size" in command


def test_bench_params_use_wrapper_profile(tmp_path: Path) -> None:
    path = tmp_path / "params.json"
    write_bench_params(path, ["prefill", "decode"])
    assert json.loads(path.read_text()) == [
        {"_benchmark_name": "prefill", "vllm_bench_profile": "prefill"},
        {"_benchmark_name": "decode", "vllm_bench_profile": "decode"},
    ]


def test_sweep_command_preserves_literal_sweep_args() -> None:
    command = build_sweep_command(
        serve_command=["vllm", "serve", "/models/model", "-tp", "8"],
        bench_command=["python3", "/tmp/wrapper.py"],
        bench_params_path="/tmp/params.json",
        sweep_args={"--workload-var": "max_concurrency", "--show-stdout": None},
        output_dir="/tmp/results",
        experiment_name="workload",
        resume=True,
    )
    assert command[-1] == "--resume"
    assert "--show-stdout" in command
    assert "vllm serve /models/model -tp 8" in command


def test_stage_args_override_fixed_args_in_profile(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"
    write_profiles(
        path,
        {"--num-prompts": 64, "--model": "model"},
        {"prefill": {"--num-prompts": 128}},
    )
    profile = json.loads(path.read_text())["prefill"]["args"]
    assert profile == {"--num-prompts": 128, "--model": "model"}
