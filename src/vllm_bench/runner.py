"""Top-level orchestration for automated vLLM serving benchmarks."""

from __future__ import annotations

import hashlib
import json
import re
import signal
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any

import yaml

from .commands import (
    build_bench_base_command,
    build_serve_command,
    build_sweep_command,
    format_command,
    write_bench_params,
    write_profiles,
)
from .config import VLLM_CONTAINER_CACHE_ROOT, AppConfig, config_digest
from .docker import CommandError, DockerClient, ImageMetadata
from .report import build_reports

_CONTAINER_ROOT = "/tmp/vllm-bench"
_RESERVED_SWEEP_ARGS = {
    "--serve-cmd",
    "--bench-cmd",
    "--serve-params",
    "--bench-params",
    "--output-dir",
    "--experiment-name",
    "--resume",
}


class BenchmarkRunner:
    """Run all configured jobs and produce reports."""

    def __init__(
        self,
        config: AppConfig,
        expanded_config: dict[str, Any],
        *,
        dry_run: bool = False,
        resume: bool = False,
        keep_container: bool = False,
        docker: DockerClient | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.config = config
        self.expanded_config = expanded_config
        self.dry_run = dry_run
        self.resume = resume
        self.keep_container = keep_container
        self.docker = docker or DockerClient(dry_run=dry_run)
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self.output_dir = config.output_dir.expanduser().resolve() / config.run_name
        self.container_root = f"{_CONTAINER_ROOT}/{config.run_name}"
        self.container_name = self._initial_container_name()
        self.failures: list[dict[str, Any]] = []
        self.variant_metadata: dict[tuple[str, str], dict[str, str]] = {}
        self.expected_runs: dict[tuple[str, str], int] = {}
        self.image_metadata: ImageMetadata | None = None
        self._container_ready = False
        self._old_signal_handlers: dict[int, Any] = {}

    def run(self) -> int:
        """Execute the configured benchmark suite."""

        self._validate_runtime_config()
        if self.dry_run:
            return self._run_dry()

        self._prepare_output_dir()
        self._write_manifest(status="initializing")
        self._install_signal_handlers()
        try:
            self._prepare_container()
            self._write_manifest(status="running")
            self._install_container_files()
            self._run_variants()
            assert self.image_metadata is not None
            build_reports(
                output_dir=self.output_dir,
                run_name=self.config.run_name,
                image=self.image_metadata,
                variant_metadata=self.variant_metadata,
                expected_runs=self.expected_runs,
                selection_limits=self._selection_limits(),
            )
            self._write_failures()
            status = "failed" if self.failures else "completed"
            self._write_manifest(status=status)
            return 1 if self.failures else 0
        except BaseException as exc:
            self.failures.append({"job": None, "variant": None, "error": str(exc)})
            self._write_failures()
            self._write_manifest(status="failed")
            raise
        finally:
            self._restore_signal_handlers()
            if self._container_ready and not self.keep_container:
                self.docker.remove_container(self.container_name)

    def _validate_runtime_config(self) -> None:
        if not self.config.docker.model_path.is_dir():
            raise ValueError(
                "docker.model_path does not exist or is not a directory: "
                f"{self.config.docker.model_path}"
            )
        compile_cache_path = self.config.docker.compile_cache_path
        if compile_cache_path is not None:
            compile_cache_path.mkdir(parents=True, exist_ok=True)
            if not compile_cache_path.is_dir():
                raise ValueError(
                    "docker.compile_cache_path is not a directory: "
                    f"{compile_cache_path}"
                )
        for job in self.config.jobs:
            reserved = _RESERVED_SWEEP_ARGS.intersection(job.sweep.args)
            if reserved:
                rendered = ", ".join(sorted(reserved))
                raise ValueError(
                    f"job {job.name} uses tool-managed sweep args: {rendered}"
                )
            workload_var = job.sweep.args.get("--workload-var")
            if workload_var != "max_concurrency":
                raise ValueError(
                    f"job {job.name} must set "
                    "--workload-var: max_concurrency for report aggregation"
                )

    def _run_dry(self) -> int:
        self.docker.create_container(
            name=self.container_name,
            image=self.config.docker.image,
            model_path=self.config.docker.model_path,
            container_model_path=self.config.docker.container_model_path,
            compile_cache_path=self.config.docker.compile_cache_path,
            container_cache_root=VLLM_CONTAINER_CACHE_ROOT,
        )
        self.docker.exec(
            self.container_name,
            ["vllm", "bench", "sweep", "serve_workload", "--help"],
        )
        for job in self.config.jobs:
            for variant in job.serve.variants:
                _, _, _, sweep_command = self._prepare_variant_commands(
                    job.name, job, variant.name, variant.args, resume_variant=False
                )
                with _NullLog() as log:
                    self.docker.exec_stream(
                        self.container_name,
                        sweep_command,
                        log,
                        environment=self._environment(),
                    )
        self.docker.remove_container(self.container_name)
        return 0

    def _prepare_output_dir(self) -> None:
        digest = self._digest()
        manifest_path = self.output_dir / "manifest.json"
        if self.output_dir.exists():
            if not self.resume:
                raise ValueError(
                    f"output directory already exists: {self.output_dir}; use --resume"
                )
            if not manifest_path.exists():
                raise ValueError(f"cannot resume without manifest: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("config_sha256") != digest:
                raise ValueError("resume configuration does not match the original run")
        else:
            if self.resume:
                raise ValueError(
                    f"cannot resume missing run directory: {self.output_dir}"
                )
            self.output_dir.mkdir(parents=True)

        for name in ("generated", "logs", "raw"):
            (self.output_dir / name).mkdir(exist_ok=True)
        (self.output_dir / "expanded_config.yaml").write_text(
            yaml.safe_dump(self.expanded_config, sort_keys=False), encoding="utf-8"
        )

    def _prepare_container(self) -> None:
        if not self.docker.daemon_available():
            raise RuntimeError("Docker daemon is unavailable")

        name = self.container_name
        if self.docker.container_exists(name):
            inspected = self.docker.inspect_container(name)
            expected = self._existing_image_id()
            actual = str(inspected.get("Image", ""))
            reusable = self.resume and expected is not None and actual == expected
            if reusable:
                if not inspected.get("State", {}).get("Running", False):
                    self.docker.start_container(name)
            else:
                self.container_name = self._new_available_container_name()
                name = self.container_name

        if not self.docker.container_exists(name):
            self.docker.create_container(
                name=name,
                image=self.config.docker.image,
                model_path=self.config.docker.model_path,
                container_model_path=self.config.docker.container_model_path,
                compile_cache_path=self.config.docker.compile_cache_path,
                container_cache_root=VLLM_CONTAINER_CACHE_ROOT,
            )

        self._container_ready = True
        self.image_metadata = self.docker.inspect_image(self.config.docker.image)
        inspected = self.docker.inspect_container(name)
        if str(inspected.get("Image", "")) != self.image_metadata.image_id:
            raise ValueError("container image ID does not match configured image")
        self.docker.exec(name, ["vllm", "bench", "sweep", "serve_workload", "--help"])

    def _install_container_files(self) -> None:
        name = self.container_name
        self.docker.exec(name, ["mkdir", "-p", self.container_root])
        wrapper = self.project_root / "src" / "vllm_bench" / "bench_wrapper.py"
        self.docker.copy_to(wrapper, name, f"{self.container_root}/bench_wrapper.py")
        if self.resume and (self.output_dir / "raw").exists():
            self.docker.exec(name, ["mkdir", "-p", f"{self.container_root}/raw"])
            self.docker.copy_to(
                self.output_dir / "raw", name, f"{self.container_root}/"
            )

    def _run_variants(self) -> None:
        for job in self.config.jobs:
            for variant in job.serve.variants:
                key = (job.name, variant.name)
                self.expected_runs[key] = self._expected_num_runs(job.sweep.args)
                resume_variant = self.resume and self._variant_has_results(*key)
                _, _, _, sweep_command = self._prepare_variant_commands(
                    job.name,
                    job,
                    variant.name,
                    variant.args,
                    resume_variant=resume_variant,
                )
                log_path = self.output_dir / "logs" / job.name / f"{variant.name}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    log_mode = "a" if resume_variant else "w"
                    with log_path.open(log_mode, encoding="utf-8") as log:
                        returncode = self.docker.exec_stream(
                            self.container_name,
                            sweep_command,
                            log,
                            environment=self._environment(),
                        )
                    if returncode != 0:
                        raise CommandError(sweep_command, returncode)
                except Exception as exc:
                    self.failures.append(
                        {
                            "job": job.name,
                            "variant": variant.name,
                            "error": str(exc),
                            "log": str(log_path),
                        }
                    )
                finally:
                    self._copy_variant_results(job.name, variant.name)

    def _prepare_variant_commands(
        self,
        job_name: str,
        job: Any,
        variant_name: str,
        variant_args: dict[str, Any],
        *,
        resume_variant: bool,
    ) -> tuple[list[str], list[str], Path, list[str]]:
        host_generated = self.output_dir / "generated" / job_name / variant_name
        if not self.dry_run:
            host_generated.mkdir(parents=True, exist_ok=True)
        container_generated = (
            f"{self.container_root}/generated/{job_name}/{variant_name}"
        )
        profiles_path = host_generated / "profiles.json"
        bench_params_path = host_generated / "bench_params.json"
        stages = {
            name: stage.args
            for name, stage in (
                ("prefill", job.bench.stages.prefill),
                ("decode", job.bench.stages.decode),
            )
            if stage.enabled
        }
        if not self.dry_run:
            write_profiles(profiles_path, job.bench.fixed_args, stages)
            write_bench_params(bench_params_path, list(stages))
            self.docker.exec(
                self.container_name,
                ["mkdir", "-p", container_generated],
            )
            container_generated_parent = container_generated.rsplit("/", 1)[0]
            self.docker.copy_to(
                host_generated,
                self.container_name,
                container_generated_parent,
            )

        serve_command = build_serve_command(
            job.serve.model, job.serve.fixed_args, variant_args
        )
        bench_command = build_bench_base_command(
            f"{self.container_root}/bench_wrapper.py",
            f"{container_generated}/profiles.json",
        )
        raw_parent = f"{self.container_root}/raw/{job_name}/{variant_name}"
        sweep_command = build_sweep_command(
            serve_command=serve_command,
            bench_command=bench_command,
            bench_params_path=f"{container_generated}/bench_params.json",
            sweep_args=job.sweep.args,
            output_dir=raw_parent,
            experiment_name="workload",
            resume=resume_variant,
        )
        self.variant_metadata[(job_name, variant_name)] = {
            "serve_command": format_command(serve_command),
            "bench_command": format_command(bench_command),
        }
        return serve_command, bench_command, host_generated, sweep_command

    def _copy_variant_results(self, job: str, variant: str) -> None:
        host_parent = self.output_dir / "raw" / job / variant
        container_path = f"{self.container_root}/raw/{job}/{variant}/workload"
        try:
            self.docker.copy_from(self.container_name, container_path, host_parent)
        except CommandError as exc:
            if not any(
                failure["job"] == job and failure["variant"] == variant
                for failure in self.failures
            ):
                self.failures.append(
                    {"job": job, "variant": variant, "error": str(exc)}
                )

    def _variant_has_results(self, job: str, variant: str) -> bool:
        return (self.output_dir / "raw" / job / variant / "workload").exists()

    def _expected_num_runs(self, sweep_args: dict[str, Any]) -> int:
        value = sweep_args.get("--num-runs", 3)
        return int(value)

    def _selection_limits(self) -> dict[tuple[str, str], float]:
        limits: dict[tuple[str, str], float] = {}
        for job in self.config.jobs:
            for stage_name, stage in (
                ("prefill", job.bench.stages.prefill),
                ("decode", job.bench.stages.decode),
            ):
                limit = stage.selection.mean_tpot_ms_lt
                if stage.enabled and limit is not None:
                    limits[(job.name, stage_name)] = limit
        return limits

    def _environment(self) -> dict[str, str]:
        return {
            key: str(value) for key, value in self.config.docker.environment.items()
        }

    def _digest(self) -> str:
        return hashlib.sha256(config_digest(self.expanded_config).encode()).hexdigest()

    def _existing_image_id(self) -> str | None:
        path = self.output_dir / "manifest.json"
        if not path.exists():
            return None
        return (
            json.loads(path.read_text(encoding="utf-8"))
            .get("docker", {})
            .get("image_id")
        )

    def _write_manifest(self, *, status: str) -> None:
        image_data = (
            asdict(self.image_metadata)
            if self.image_metadata is not None
            else {
                "configured_image": self.config.docker.image,
                "image_id": None,
                "repo_digests": [],
            }
        )
        payload = {
            "run_name": self.config.run_name,
            "status": status,
            "config_sha256": self._digest(),
            "docker": {
                **image_data,
                "container_name": self.container_name,
                "model_path": str(self.config.docker.model_path),
                "container_model_path": self.config.docker.container_model_path,
                "compile_cache_path": (
                    str(self.config.docker.compile_cache_path)
                    if self.config.docker.compile_cache_path is not None
                    else None
                ),
                "container_cache_root": (
                    VLLM_CONTAINER_CACHE_ROOT
                    if self.config.docker.compile_cache_path is not None
                    else None
                ),
            },
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def _write_failures(self) -> None:
        (self.output_dir / "failures.json").write_text(
            json.dumps(self.failures, indent=2), encoding="utf-8"
        )

    def _install_signal_handlers(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._old_signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)

    def _restore_signal_handlers(self) -> None:
        for signum, handler in self._old_signal_handlers.items():
            signal.signal(signum, handler)
        self._old_signal_handlers.clear()

    def _handle_signal(self, signum: int, _frame: FrameType | None) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    def _initial_container_name(self) -> str:
        manifest_path = self.output_dir / "manifest.json"
        if self.resume and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            container_name = manifest.get("docker", {}).get("container_name")
            if container_name:
                return str(container_name)
        return self._generate_container_name()

    def _new_available_container_name(self) -> str:
        for _ in range(100):
            candidate = self._generate_container_name()
            if not self.docker.container_exists(candidate):
                return candidate
        raise RuntimeError("failed to generate an available Docker container name")

    def _generate_container_name(self) -> str:
        run_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", self.config.run_name).strip("-.")
        run_slug = run_slug[:32] or "run"
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        return f"vllm-bench-{run_slug}-{timestamp}-{suffix}"


class _NullLog:
    def __enter__(self) -> _NullLog:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def write(self, _value: str) -> int:
        return 0

    def flush(self) -> None:
        return None
