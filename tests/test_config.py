from pathlib import Path

import pytest

from vllm_bench.config import load_config

BASE = """
version: 1
run_name: test
output_dir: ./results
variables:
  MODEL: DeepSeek-V3.2
docker:
  image: vllm/vllm-openai:v0.20.2
  container_name: test-vllm
  host_models_path: {models}
jobs:
  - name: model
    serve:
      model: /models/${{MODEL}}
      fixed_args:
        --trust-remote-code: null
      variants:
        - name: tp8
          args:
            -tp: 8
    bench:
      fixed_args:
        --model: /models/${{MODEL}}
      stages:
        prefill:
          args:
            --random-input-len: 8192
            --random-output-len: 1
        decode:
          args:
            --random-input-len: 1
            --random-output-len: 8192
    sweep:
      args:
        --workload-var: max_concurrency
        --num-runs: 1
"""


def write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_preserves_literal_args_and_expands_variables(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    config, _ = load_config(write_config(tmp_path, BASE.format(models=models)))
    job = config.jobs[0]
    assert job.serve.model == "/models/DeepSeek-V3.2"
    assert job.serve.fixed_args == {"--trust-remote-code": None}
    assert job.serve.variants[0].args == {"-tp": 8}


def test_environment_variable_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setenv("IMAGE_TAG", "v0.20.2")
    content = BASE.replace("vllm-openai:v0.20.2", "vllm-openai:${{IMAGE_TAG}}")
    config, _ = load_config(write_config(tmp_path, content.format(models=models)))
    assert config.docker.image.endswith(":v0.20.2")


def test_rejects_yaml_boolean_command_value(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    content = BASE.replace("--trust-remote-code: null", "--trust-remote-code: true")
    with pytest.raises(ValueError, match="YAML boolean"):
        load_config(write_config(tmp_path, content.format(models=models)))


def test_rejects_non_cli_argument_key(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    content = BASE.replace("-tp: 8", "tensor_parallel_size: 8")
    with pytest.raises(ValueError, match="original CLI option"):
        load_config(write_config(tmp_path, content.format(models=models)))


def test_rejects_variable_cycle(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    content = BASE.replace(
        "MODEL: DeepSeek-V3.2", "MODEL: ${{OTHER}}\n  OTHER: ${{MODEL}}"
    )
    with pytest.raises(ValueError, match="variable reference cycle"):
        load_config(write_config(tmp_path, content.format(models=models)))
