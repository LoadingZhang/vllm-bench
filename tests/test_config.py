from pathlib import Path

import pytest

from vllm_bench.config import load_config

BASE = """
version: 1
run_name: test
output_dir: ./results
environment:
  MODEL_PATH: /models/DeepSeek-V3.2
  VLLM_ENGINE_READY_TIMEOUT_S: "1200"
docker:
  image: vllm/vllm-openai:v0.20.2
  host_models_path: {models}
jobs:
  - name: model
    serve:
      fixed_args:
        --trust-remote-code: null
      variants:
        - name: tp8
          args:
            -tp: 8
    bench:
      fixed_args: {{}}
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
    assert job.serve.fixed_args == {
        "--trust-remote-code": None,
        "--served-model-name": "DeepSeek-V3.2",
        "--port": 8000,
    }
    assert job.serve.variants[0].args == {"-tp": 8}
    assert config.environment == {
        "MODEL_PATH": "/models/DeepSeek-V3.2",
        "VLLM_ENGINE_READY_TIMEOUT_S": "1200",
    }
    assert job.bench.fixed_args == {
        "--host": "127.0.0.1",
        "--port": 8000,
        "--model": "/models/DeepSeek-V3.2",
        "--served-model-name": "DeepSeek-V3.2",
    }


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
        "MODEL_PATH: /models/DeepSeek-V3.2",
        "MODEL_PATH: ${{OTHER}}\n  OTHER: ${{MODEL_PATH}}",
    )
    with pytest.raises(ValueError, match="variable reference cycle"):
        load_config(write_config(tmp_path, content.format(models=models)))


def test_environment_values_can_reference_each_other(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    content = BASE.replace(
        "MODEL_PATH: /models/DeepSeek-V3.2",
        "MODEL_ROOT: /models/deepseek-ai\n"
        "  MODEL_NAME: DeepSeek-V3.2\n"
        "  MODEL_PATH: ${{MODEL_ROOT}}/${{MODEL_NAME}}",
    )
    config, _ = load_config(write_config(tmp_path, content.format(models=models)))
    assert config.environment["MODEL_PATH"] == "/models/deepseek-ai/DeepSeek-V3.2"
    assert config.jobs[0].serve.model == "/models/deepseek-ai/DeepSeek-V3.2"


def test_legacy_variables_field_is_rejected(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    content = BASE.replace(
        "environment:\n",
        "variables:\n  UNUSED: value\nenvironment:\n",
        1,
    )
    with pytest.raises(ValueError, match="variables"):
        load_config(write_config(tmp_path, content.format(models=models)))


def test_explicit_connection_args_override_defaults(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    content = BASE.replace(
        "--trust-remote-code: null",
        "--trust-remote-code: null\n"
        "        --served-model-name: custom\n"
        "        --port: 9000",
    ).replace(
        "fixed_args: {{}}",
        "fixed_args:\n"
        "        --host: 0.0.0.0\n"
        "        --port: 9001\n"
        "        --model: /models/other\n"
        "        --served-model-name: other",
    )
    config, _ = load_config(write_config(tmp_path, content.format(models=models)))
    job = config.jobs[0]
    assert job.serve.fixed_args["--served-model-name"] == "custom"
    assert job.serve.fixed_args["--port"] == 9000
    assert job.bench.fixed_args["--host"] == "0.0.0.0"
    assert job.bench.fixed_args["--port"] == 9001
    assert job.bench.fixed_args["--model"] == "/models/other"
    assert job.bench.fixed_args["--served-model-name"] == "other"


def test_environment_can_override_served_name_and_port(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    content = BASE.replace(
        "MODEL_PATH: /models/DeepSeek-V3.2",
        "MODEL_PATH: /models/DeepSeek-V3.2\n"
        "  SERVED_MODEL_NAME: ds\n"
        '  VLLM_PORT: "8081"',
    )
    config, _ = load_config(write_config(tmp_path, content.format(models=models)))
    job = config.jobs[0]
    assert job.serve.fixed_args["--served-model-name"] == "ds"
    assert job.serve.fixed_args["--port"] == "8081"
    assert job.bench.fixed_args["--host"] == "127.0.0.1"
    assert job.bench.fixed_args["--port"] == "8081"
    assert job.bench.fixed_args["--served-model-name"] == "ds"


def test_missing_model_sources_are_rejected(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    content = BASE.replace("  MODEL_PATH: /models/DeepSeek-V3.2\n", "")
    with pytest.raises(ValueError, match="serve.model is omitted"):
        load_config(write_config(tmp_path, content.format(models=models)))
