"""Configuration loading and validation."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ArgValue = str | int | float | None | list[str | int | float]
RawArgs = dict[str, ArgValue]
_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class StrictModel(BaseModel):
    """Base class that rejects unknown configuration fields."""

    model_config = ConfigDict(extra="forbid")


class DockerConfig(StrictModel):
    image: str
    container_name: str
    host_models_path: Path

    @model_validator(mode="after")
    def validate_models_path(self) -> DockerConfig:
        self.host_models_path = self.host_models_path.expanduser()
        if not self.host_models_path.is_absolute():
            raise ValueError("docker.host_models_path must be an absolute path")
        return self


class VariantConfig(StrictModel):
    name: str
    args: RawArgs = Field(default_factory=dict)


class ServeConfig(StrictModel):
    model: str
    fixed_args: RawArgs = Field(default_factory=dict)
    variants: list[VariantConfig]

    @model_validator(mode="after")
    def validate_variants(self) -> ServeConfig:
        _ensure_unique((item.name for item in self.variants), "serve variant")
        if not self.variants:
            raise ValueError("serve.variants must contain at least one variant")
        return self


class StageConfig(StrictModel):
    enabled: bool = True
    args: RawArgs = Field(default_factory=dict)


class StagesConfig(StrictModel):
    prefill: StageConfig
    decode: StageConfig

    @model_validator(mode="after")
    def validate_enabled_stage(self) -> StagesConfig:
        if not self.prefill.enabled and not self.decode.enabled:
            raise ValueError("at least one benchmark stage must be enabled")
        return self


class BenchConfig(StrictModel):
    fixed_args: RawArgs = Field(default_factory=dict)
    stages: StagesConfig


class SweepConfig(StrictModel):
    args: RawArgs = Field(default_factory=dict)


class JobConfig(StrictModel):
    name: str
    serve: ServeConfig
    bench: BenchConfig
    sweep: SweepConfig


class AppConfig(StrictModel):
    version: int = 1
    run_name: str
    output_dir: Path = Path("./results")
    variables: dict[str, str | int | float] = Field(default_factory=dict)
    environment: dict[str, str | int | float] = Field(default_factory=dict)
    docker: DockerConfig
    jobs: list[JobConfig]

    @model_validator(mode="after")
    def validate_config(self) -> AppConfig:
        if self.version != 1:
            raise ValueError(f"unsupported config version: {self.version}")
        _ensure_unique((job.name for job in self.jobs), "job")
        if not self.jobs:
            raise ValueError("jobs must contain at least one job")
        return self


def _ensure_unique(values: Any, label: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {label} name: {value}")
        seen.add(value)


def _validate_arg_map(value: Any, path: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    for key, item in value.items():
        if not isinstance(key, str) or not key.startswith("-"):
            raise ValueError(f"{path} key must be an original CLI option: {key!r}")
        if isinstance(item, bool):
            raise ValueError(
                f"{path}.{key} uses a YAML boolean; use null for a flag or a quoted "
                'string such as "True"/"False" for a literal value'
            )
        if isinstance(item, list):
            if not item:
                raise ValueError(f"{path}.{key} list must not be empty")
            invalid = any(
                isinstance(element, (bool, list, dict)) or element is None
                for element in item
            )
            if invalid:
                raise ValueError(f"{path}.{key} contains an unsupported list value")
        elif isinstance(item, (dict, tuple, set)):
            raise ValueError(f"{path}.{key} contains an unsupported value")


def _walk_arg_maps(data: dict[str, Any]) -> None:
    jobs = data.get("jobs", [])
    if not isinstance(jobs, list):
        return
    for job_index, job in enumerate(jobs):
        if not isinstance(job, dict):
            continue
        serve = job.get("serve", {})
        bench = job.get("bench", {})
        sweep = job.get("sweep", {})
        if isinstance(serve, dict):
            if "fixed_args" in serve:
                _validate_arg_map(
                    serve["fixed_args"], f"jobs[{job_index}].serve.fixed_args"
                )
            for variant_index, variant in enumerate(serve.get("variants", [])):
                if isinstance(variant, dict) and "args" in variant:
                    _validate_arg_map(
                        variant["args"],
                        f"jobs[{job_index}].serve.variants[{variant_index}].args",
                    )
        if isinstance(bench, dict):
            if "fixed_args" in bench:
                _validate_arg_map(
                    bench["fixed_args"], f"jobs[{job_index}].bench.fixed_args"
                )
            stages = bench.get("stages", {})
            if isinstance(stages, dict):
                for stage_name, stage in stages.items():
                    if isinstance(stage, dict) and "args" in stage:
                        _validate_arg_map(
                            stage["args"],
                            f"jobs[{job_index}].bench.stages.{stage_name}.args",
                        )
        if isinstance(sweep, dict) and "args" in sweep:
            _validate_arg_map(sweep["args"], f"jobs[{job_index}].sweep.args")


def _resolve_string(
    value: str,
    variables: Mapping[str, str],
    environment: Mapping[str, str],
    stack: tuple[str, ...] = (),
) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in stack:
            chain = " -> ".join((*stack, name))
            raise ValueError(f"variable reference cycle: {chain}")
        if name in variables:
            raw = variables[name]
        elif name in environment:
            raw = environment[name]
        else:
            raise ValueError(f"undefined variable: {name}")
        return _resolve_string(raw, variables, environment, (*stack, name))

    previous = None
    resolved = value
    while previous != resolved:
        previous = resolved
        resolved = _VAR_PATTERN.sub(replace, resolved)
    return resolved


def _resolve_tree(
    value: Any, variables: Mapping[str, str], environment: Mapping[str, str]
) -> Any:
    if isinstance(value, str):
        return _resolve_string(value, variables, environment)
    if isinstance(value, list):
        return [_resolve_tree(item, variables, environment) for item in value]
    if isinstance(value, dict):
        return {
            key: _resolve_tree(item, variables, environment)
            for key, item in value.items()
        }
    return value


def load_config(
    path: Path, run_name: str | None = None
) -> tuple[AppConfig, dict[str, Any]]:
    """Load, expand, and validate a YAML configuration file."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be an object")
    variables_raw = raw.get("variables", {})
    if not isinstance(variables_raw, dict):
        raise ValueError("variables must be an object")
    variables = {str(key): str(value) for key, value in variables_raw.items()}
    expanded = _resolve_tree(raw, variables, os.environ)
    if run_name is not None:
        expanded["run_name"] = run_name
    _walk_arg_maps(expanded)
    config = AppConfig.model_validate(expanded)
    return config, expanded


def config_digest(expanded: dict[str, Any]) -> str:
    """Return a stable JSON representation for hashing by the runner."""

    return json.dumps(expanded, sort_keys=True, separators=(",", ":"), default=str)
