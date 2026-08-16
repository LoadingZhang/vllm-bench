"""Small, testable Docker CLI adapter."""

from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


class CommandError(RuntimeError):
    """Raised when a Docker command exits unsuccessfully."""

    def __init__(self, command: Sequence[str], returncode: int, output: str = ""):
        self.command = list(command)
        self.returncode = returncode
        self.output = output
        suffix = f"\n{output.rstrip()}" if output else ""
        super().__init__(
            f"command failed with exit code {returncode}: {shlex.join(self.command)}"
            f"{suffix}"
        )


@dataclass(frozen=True)
class ImageMetadata:
    configured_image: str
    image_id: str
    repo_digests: list[str]


RunFunction = Callable[..., subprocess.CompletedProcess[str]]
PopenFunction = Callable[..., subprocess.Popen[str]]


class DockerClient:
    """Invoke Docker without a shell."""

    def __init__(
        self,
        *,
        dry_run: bool = False,
        printer: Callable[[str], None] = print,
        run_fn: RunFunction = subprocess.run,
        popen_fn: PopenFunction = subprocess.Popen,
    ) -> None:
        self.dry_run = dry_run
        self.printer = printer
        self._run_fn = run_fn
        self._popen_fn = popen_fn

    def _print(self, command: Sequence[str]) -> None:
        self.printer(f"$ {shlex.join(command)}")

    def run(
        self,
        command: Sequence[str],
        *,
        check: bool = True,
        capture: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = list(command)
        self._print(command)
        if self.dry_run:
            return subprocess.CompletedProcess(command, 0, "", "")
        result = self._run_fn(
            command,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None,
            check=False,
        )
        if check and result.returncode != 0:
            raise CommandError(command, result.returncode, result.stdout or "")
        return result

    def stream(self, command: Sequence[str], log_file: TextIO) -> int:
        command = list(command)
        self._print(command)
        if self.dry_run:
            return 0
        process = self._popen_fn(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return process.wait()

    def daemon_available(self) -> bool:
        result = self.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"], check=False
        )
        return result.returncode == 0

    def container_exists(self, name: str) -> bool:
        result = self.run(["docker", "inspect", name], check=False)
        return result.returncode == 0

    def inspect_container(self, name: str) -> dict[str, Any]:
        result = self.run(["docker", "inspect", name])
        return json.loads(result.stdout)[0]

    def start_container(self, name: str) -> None:
        self.run(["docker", "start", name])

    def create_container(
        self,
        *,
        name: str,
        image: str,
        model_path: Path,
        container_model_path: str,
        compile_cache_path: Path | None = None,
        container_cache_root: str = "/root/.cache/vllm",
    ) -> None:
        command = [
            "docker",
            "run",
            "--name",
            name,
            "--gpus",
            "all",
            "--privileged",
            "--ipc=host",
            "--net=host",
            "-it",
            "--entrypoint",
            "bash",
            "-v",
            f"{model_path}:{container_model_path}",
        ]
        if compile_cache_path is not None:
            command.extend(("-v", f"{compile_cache_path}:{container_cache_root}"))
        command.extend(("-d", image))
        self.run(command, capture=False)

    def remove_container(self, name: str) -> None:
        self.run(["docker", "rm", "-f", name], check=False)

    def inspect_image(self, image: str) -> ImageMetadata:
        result = self.run(["docker", "image", "inspect", image])
        data = json.loads(result.stdout)[0]
        return ImageMetadata(
            configured_image=image,
            image_id=str(data["Id"]),
            repo_digests=list(data.get("RepoDigests") or []),
        )

    def exec_command(
        self,
        name: str,
        command: Sequence[str],
        *,
        environment: dict[str, str] | None = None,
    ) -> list[str]:
        result = ["docker", "exec"]
        for key, value in (environment or {}).items():
            result.extend(("--env", f"{key}={value}"))
        result.append(name)
        result.extend(command)
        return result

    def exec(
        self,
        name: str,
        command: Sequence[str],
        *,
        environment: dict[str, str] | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return self.run(
            self.exec_command(name, command, environment=environment),
            check=check,
        )

    def exec_stream(
        self,
        name: str,
        command: Sequence[str],
        log_file: TextIO,
        *,
        environment: dict[str, str] | None = None,
    ) -> int:
        return self.stream(
            self.exec_command(name, command, environment=environment), log_file
        )

    def copy_to(self, source: Path, name: str, destination: str) -> None:
        self.run(["docker", "cp", str(source), f"{name}:{destination}"])

    def copy_from(self, name: str, source: str, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        self.run(["docker", "cp", f"{name}:{source}", str(destination)])
