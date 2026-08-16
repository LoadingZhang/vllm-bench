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
  container_name: vllm-0.20.2
  host_models_path: /data/models
```

The container is created as:

```bash
docker run --name vllm-0.20.2 --gpus all --privileged \
  --ipc=host --net=host -it --entrypoint bash \
  -v /data/models:/models -d vllm/vllm-openai:v0.20.2
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

`${NAME}` first resolves from top-level `variables`, then from the host process
environment. `environment` entries are passed through `docker exec --env` to
the sweep, server, and benchmark processes.

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

A same-named existing container is rejected for a new run. During resume it is
accepted only when its image ID matches the original manifest. Otherwise a new
container is created and the prior raw results are restored into it.

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

The first version requires `--workload-var: max_concurrency`. The tool selects the best workload point that `serve_workload` actually tested;
it does not claim a theoretical optimum outside those points.

## Development

```bash
uv run pytest -v
uv run ruff check .
```
