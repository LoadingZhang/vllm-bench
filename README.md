# vllm-bench

`vllm-bench` automates Docker-based `vllm bench sweep serve_workload`
experiments. It starts one vLLM container, tests every configured server
variant, and writes separate Prefill and Decode reports.

## Features

- Literal CLI options: `-tp`, `--served-model-name`, and dotted options are
  never renamed.
- `null` represents a flag with no value.
- Prefill and Decode profiles share one model start for each server variant.
- Failed variants are recorded while later variants continue.
- Raw sweep artifacts are copied from the container after every variant.
- Reports record the configured Docker tag, image ID, and RepoDigest.
- Interrupted runs can be resumed with `--resume`.

## Install

```bash
cd ~/Codes/vllm-bench
uv sync --extra dev
```

The host requires Docker with NVIDIA GPU support. The configured image must
contain `vllm bench sweep serve_workload`.

## Configure

Copy `examples/deepseek-v3.2.yaml` and set an absolute model directory:

```yaml
docker:
  image: vllm/vllm-openai:v0.20.2
  host_models_path: /data/models
```

The tool generates a unique container name for every new run, for example:

```bash
docker run --name vllm-bench-dsv32-b200-20260816123000-a1b2c3d4 \
  --gpus all --privileged \
  --ipc=host --net=host -it --entrypoint bash \
  -v /data/models:/models -d vllm/vllm-openai:v0.20.2
```

The generated name has the form:

```text
vllm-bench-<run-name>-<UTC timestamp>-<random suffix>
```

### Literal argument rules

```yaml
fixed_args:
  --trust-remote-code: null       # --trust-remote-code
  -tp: 8                          # -tp 8
  --kernel-config.enable_flashinfer_autotune: "False"
                                  # --kernel-config.enable_flashinfer_autotune=False
```

Argument keys must already be valid CLI options. No underscore/hyphen or alias
conversion is performed. Use quoted strings for literal boolean values; YAML
booleans are reserved for tool fields such as `enabled`.

Configuration variables and container environment variables use the same
top-level `environment` object:

```yaml
environment:
  MODEL_PATH: /models/deepseek-ai/DeepSeek-V3.2
  VLLM_ENGINE_READY_TIMEOUT_S: "1200"
```

`${NAME}` first resolves from this object, then from the host process
environment. Every configured entry is also passed through
`docker exec --env` to the sweep, server, and benchmark processes.

### Automatic serve and benchmark connection arguments

The following values do not need to be repeated under `jobs[].serve` and
`jobs[].bench`:

```text
serve model
--served-model-name
--port
bench --host
bench --model
bench --served-model-name
bench --port
```

The defaults are derived as follows:

- Model: `environment.MODEL_PATH`, which is the complete model path.
  `environment.MODEL` remains supported as an explicit override.
- Served model name: `environment.SERVED_MODEL_NAME`, then
  `environment.MODEL_NAME`, then the final component of the model path.
- Port: `environment.VLLM_PORT`, defaulting to `8000`.
- Benchmark host: `127.0.0.1`.

For example, this is sufficient:

```yaml
environment:
  MODEL_PATH: /models/deepseek-ai/DeepSeek-V3.2

jobs:
  - name: dsv32
    serve:
      fixed_args:
        --trust-remote-code: null
      variants:
        - name: tp8
          args:
            -tp: 8
    bench:
      fixed_args:
        --backend: vllm
        --endpoint: /v1/completions
      stages:
        prefill:
          args:
            --random-input-len: 8192
            --random-output-len: 1
        decode:
          args:
            --random-input-len: 128
            --random-output-len: 8192
```

Any explicitly configured value still takes precedence. For example, an
explicit serve `--port: 9000` is automatically propagated to the benchmark
unless the benchmark also explicitly sets its own `--port`.

## Run

```bash
uv run vllm-bench --config examples/deepseek-v3.2.yaml
```

Useful options:

```text
--dry-run         Validate and print commands without invoking Docker
--resume          Continue an existing run with the same expanded config
--run-name NAME   Override run_name
--keep-container  Do not remove the container on exit
```

New runs always use dynamically generated container names, so unrelated old
containers do not block execution. During resume, the tool first tries the
container recorded in the original manifest. It reuses that container only
when its image ID matches; otherwise it generates a new name, creates a new
container, and restores the prior raw results into it.

## Results

Results are written to `<output_dir>/<run_name>/`:

```text
expanded_config.yaml
manifest.json
failures.json
prefill_results.csv
decode_results.csv
best_results.json
generated/
logs/
raw/
```

Prefill throughput is computed as:

```text
total_input_tokens / duration
```

Decode throughput uses the benchmark's `output_throughput`. Only workload
points with the configured number of runs and zero failed requests are eligible
for `best_results.json`. Repeated runs are ranked by mean throughput, then by
P99 TTFT for Prefill or P99 TPOT for Decode, then by lower concurrency.

The first version requires `--workload-var: max_concurrency`. The tool selects
the best workload point that `serve_workload` actually tested; it does not
claim a theoretical optimum outside those points.

## Development

```bash
uv run pytest -v
uv run ruff check .
```
