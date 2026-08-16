import subprocess
from pathlib import Path

from vllm_bench.docker import DockerClient


def test_create_container_command() -> None:
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    client = DockerClient(run_fn=fake_run, printer=lambda _line: None)
    client.create_container(
        name="vllm-0.20.2",
        image="vllm/vllm-openai:v0.20.2",
        model_path=Path("/data/models/DeepSeek-V3.2"),
        container_model_path="/models/DeepSeek-V3.2",
    )
    assert commands[0] == [
        "docker",
        "run",
        "--name",
        "vllm-0.20.2",
        "--gpus",
        "all",
        "--privileged",
        "--ipc=host",
        "--net=host",
        "-it",
        "--entrypoint",
        "bash",
        "-v",
        "/data/models/DeepSeek-V3.2:/models/DeepSeek-V3.2",
        "-d",
        "vllm/vllm-openai:v0.20.2",
    ]


def test_exec_environment_is_literal() -> None:
    client = DockerClient(dry_run=True, printer=lambda _line: None)
    command = client.exec_command(
        "container",
        ["vllm", "bench"],
        environment={"VLLM_ENGINE_READY_TIMEOUT_S": "1200"},
    )
    assert command == [
        "docker",
        "exec",
        "--env",
        "VLLM_ENGINE_READY_TIMEOUT_S=1200",
        "container",
        "vllm",
        "bench",
    ]


def test_create_container_mounts_compile_cache() -> None:
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    client = DockerClient(run_fn=fake_run, printer=lambda _line: None)
    client.create_container(
        name="container",
        image="image:tag",
        model_path=Path("/data/models/DeepSeek-V3.2"),
        container_model_path="/models/DeepSeek-V3.2",
        compile_cache_path=Path("/data/vllm-cache"),
    )

    command = commands[0]
    assert "/data/models/DeepSeek-V3.2:/models/DeepSeek-V3.2" in command
    assert "/data/vllm-cache:/root/.cache/vllm" in command
