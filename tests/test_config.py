from pathlib import Path

import pytest

from vllm_bench.config import load_config

BASE = """
version: 1
run_name: test
output_dir: ./results
docker:
  image: vllm/vllm-openai:v0.20.2
  model_path: {model}
  environment:
    VLLM_ENGINE_READY_TIMEOUT_S: "1200"
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


def model_dir(tmp_path: Path) -> Path:
    path = tmp_path / "DeepSeek-V3.2"
    path.mkdir()
    return path


def test_model_path_generates_container_model_and_name(tmp_path: Path) -> None:
    model = model_dir(tmp_path)
    config, _ = load_config(write_config(tmp_path, BASE.format(model=model)))
    job = config.jobs[0]

    assert config.docker.model_path == model
    assert config.docker.container_model_path == "/models/DeepSeek-V3.2"
    assert config.docker.model_name == "DeepSeek-V3.2"
    assert config.docker.environment == {"VLLM_ENGINE_READY_TIMEOUT_S": "1200"}
    assert job.serve.model == "/models/DeepSeek-V3.2"
    assert job.serve.fixed_args == {
        "--trust-remote-code": None,
        "--served-model-name": "DeepSeek-V3.2",
        "--port": 8000,
    }
    assert job.serve.variants[0].args == {"-tp": 8}
    assert job.bench.fixed_args == {
        "--host": "127.0.0.1",
        "--port": 8000,
        "--model": "/models/DeepSeek-V3.2",
        "--served-model-name": "DeepSeek-V3.2",
    }


def test_environment_variable_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = model_dir(tmp_path)
    monkeypatch.setenv("IMAGE_TAG", "v0.20.2")
    content = BASE.replace("vllm-openai:v0.20.2", "vllm-openai:${{IMAGE_TAG}}")
    config, _ = load_config(write_config(tmp_path, content.format(model=model)))
    assert config.docker.image.endswith(":v0.20.2")


def test_rejects_yaml_boolean_command_value(tmp_path: Path) -> None:
    model = model_dir(tmp_path)
    content = BASE.replace("--trust-remote-code: null", "--trust-remote-code: true")
    with pytest.raises(ValueError, match="YAML boolean"):
        load_config(write_config(tmp_path, content.format(model=model)))


def test_rejects_non_cli_argument_key(tmp_path: Path) -> None:
    model = model_dir(tmp_path)
    content = BASE.replace("-tp: 8", "tensor_parallel_size: 8")
    with pytest.raises(ValueError, match="original CLI option"):
        load_config(write_config(tmp_path, content.format(model=model)))


def test_rejects_environment_variable_cycle(tmp_path: Path) -> None:
    model = model_dir(tmp_path)
    content = BASE.replace(
        '    VLLM_ENGINE_READY_TIMEOUT_S: "1200"',
        "    FIRST: ${{SECOND}}\n    SECOND: ${{FIRST}}",
    )
    with pytest.raises(ValueError, match="variable reference cycle"):
        load_config(write_config(tmp_path, content.format(model=model)))


def test_docker_environment_values_can_reference_each_other(
    tmp_path: Path,
) -> None:
    model = model_dir(tmp_path)
    content = BASE.replace(
        '    VLLM_ENGINE_READY_TIMEOUT_S: "1200"',
        '    TIMEOUT: "1200"\n    VLLM_ENGINE_READY_TIMEOUT_S: ${{TIMEOUT}}',
    )
    config, _ = load_config(write_config(tmp_path, content.format(model=model)))
    assert config.docker.environment == {
        "TIMEOUT": "1200",
        "VLLM_ENGINE_READY_TIMEOUT_S": "1200",
    }


def test_legacy_top_level_variables_are_rejected(tmp_path: Path) -> None:
    model = model_dir(tmp_path)
    content = BASE.replace("docker:\n", "variables:\n  UNUSED: value\ndocker:\n", 1)
    with pytest.raises(ValueError, match="variables"):
        load_config(write_config(tmp_path, content.format(model=model)))


def test_legacy_top_level_environment_is_rejected(tmp_path: Path) -> None:
    model = model_dir(tmp_path)
    content = BASE.replace("docker:\n", "environment:\n  UNUSED: value\ndocker:\n", 1)
    with pytest.raises(ValueError, match="environment"):
        load_config(write_config(tmp_path, content.format(model=model)))


def test_explicit_connection_args_override_defaults(tmp_path: Path) -> None:
    model = model_dir(tmp_path)
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
    config, _ = load_config(write_config(tmp_path, content.format(model=model)))
    job = config.jobs[0]
    assert job.serve.fixed_args["--served-model-name"] == "custom"
    assert job.serve.fixed_args["--port"] == 9000
    assert job.bench.fixed_args["--host"] == "0.0.0.0"
    assert job.bench.fixed_args["--port"] == 9001
    assert job.bench.fixed_args["--model"] == "/models/other"
    assert job.bench.fixed_args["--served-model-name"] == "other"


def test_docker_environment_can_override_served_name_and_port(
    tmp_path: Path,
) -> None:
    model = model_dir(tmp_path)
    content = BASE.replace(
        '    VLLM_ENGINE_READY_TIMEOUT_S: "1200"',
        '    VLLM_ENGINE_READY_TIMEOUT_S: "1200"\n'
        "    SERVED_MODEL_NAME: ds\n"
        '    VLLM_PORT: "8081"',
    )
    config, _ = load_config(write_config(tmp_path, content.format(model=model)))
    job = config.jobs[0]
    assert job.serve.fixed_args["--served-model-name"] == "ds"
    assert job.serve.fixed_args["--port"] == "8081"
    assert job.bench.fixed_args["--host"] == "127.0.0.1"
    assert job.bench.fixed_args["--port"] == "8081"
    assert job.bench.fixed_args["--served-model-name"] == "ds"


def test_model_path_is_required(tmp_path: Path) -> None:
    model = model_dir(tmp_path)
    content = BASE.replace("  model_path: {model}\n", "")
    with pytest.raises(ValueError, match="model_path"):
        load_config(write_config(tmp_path, content.format(model=model)))


def test_model_path_must_be_absolute(tmp_path: Path) -> None:
    content = BASE.replace("  model_path: {model}", "  model_path: relative/model")
    with pytest.raises(ValueError, match="model_path"):
        load_config(write_config(tmp_path, content.format(model="unused")))


def test_compile_cache_injects_vllm_cache_root(tmp_path: Path) -> None:
    model = model_dir(tmp_path)
    cache = tmp_path / "compile-cache"
    content = BASE.replace(
        "  model_path: {model}",
        f"  model_path: {{model}}\n  compile_cache_path: {cache}",
    )
    config, expanded = load_config(write_config(tmp_path, content.format(model=model)))

    assert config.docker.compile_cache_path == cache
    assert config.docker.environment["VLLM_CACHE_ROOT"] == "/root/.cache/vllm"
    assert expanded["docker"]["environment"]["VLLM_CACHE_ROOT"] == "/root/.cache/vllm"


def test_compile_cache_path_must_be_absolute(tmp_path: Path) -> None:
    model = model_dir(tmp_path)
    content = BASE.replace(
        "  model_path: {model}",
        "  model_path: {model}\n  compile_cache_path: relative/cache",
    )
    with pytest.raises(ValueError, match="compile_cache_path"):
        load_config(write_config(tmp_path, content.format(model=model)))
