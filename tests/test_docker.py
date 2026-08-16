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
        models_path=Path("/data/models"),
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
        "/data/models:/models",
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
