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
  model_path: /data/models/deepseek-ai/DeepSeek-V3.2
  environment:
    VLLM_ENGINE_READY_TIMEOUT_S: "1200"
  # Optional persistent vLLM compile cache:
  compile_cache_path: /data/vllm-compile-cache
```

The tool generates a unique container name for every new run, for example:

```bash
docker run --name vllm-bench-dsv32-b200-20260816123000-a1b2c3d4 \
  --gpus all --privileged \
  --ipc=host --net=host -it --entrypoint bash \
  -v /data/models/deepseek-ai/DeepSeek-V3.2:/models/DeepSeek-V3.2 \
  -v /data/vllm-compile-cache:/root/.cache/vllm \
  -d vllm/vllm-openai:v0.20.2
```

The generated name has the form:

```text
vllm-bench-<run-name>-<UTC timestamp>-<random suffix>
```

### Persistent model compilation cache

Set `docker.compile_cache_path` to reuse vLLM compilation artifacts
between containers and benchmark runs:

```yaml
docker:
  image: vllm/vllm-openai:v0.20.2
  model_path: /data/models/deepseek-ai/DeepSeek-V3.2
  compile_cache_path: /data/vllm-compile-cache
```

The host path must be absolute. The tool creates it when it does not exist,
mounts it at:

```text
/root/.cache/vllm
```

and automatically passes:

```text
VLLM_CACHE_ROOT=/root/.cache/vllm
```

to the sweep, server, and benchmark processes. This persists vLLM's
torch.compile/Inductor/Triton cache, DeepGEMM cache, and other cache entries
that inherit from `VLLM_CACHE_ROOT`. Omit the field to keep the container's
normal ephemeral cache behavior.

The configured host cache path and container cache root are recorded in
`manifest.json`.

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

Configuration variables and container environment variables are configured
under `docker.environment`:

```yaml
docker:
  environment:
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

- `docker.model_path` is the complete host model directory.
- The directory is mounted to `/models/<final directory name>` in the
  container, and this generated container path is used as the serve/bench
  model.
- Served model name: `docker.environment.SERVED_MODEL_NAME`, then
  `docker.environment.MODEL_NAME`, then the final component of
  `docker.model_path`.
- Port: `docker.environment.VLLM_PORT`, defaulting to `8000`.
- Benchmark host: `127.0.0.1`.

For example, this is sufficient:

```yaml
docker:
  image: vllm/vllm-openai:v0.20.2
  model_path: /data/models/deepseek-ai/DeepSeek-V3.2

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
            --random-input-len: 1
            --ignore-eos: null
            --random-output-len: 512
            --num-prompts: 8
            --num-warmups: 2
```

Any explicitly configured value still takes precedence. For example, an
explicit serve `--port: 9000` is automatically propagated to the benchmark
unless the benchmark also explicitly sets its own `--port`.

For Decode, the example intentionally overrides the shared prompt and warmup
counts. It also minimizes the input to one token so that the measured workload
is dominated by Decode rather than Prefill. `--ignore-eos` prevents generation
from stopping when the model emits EOS, so each request continues until the
configured `--random-output-len` limit is reached. It does not create an
actually unbounded request.

With 8 measured requests, 2 warmups, and 512 requested output tokens, the
slowest `max_concurrency=1` workload point generates 5,120 output tokens. This
is intended to keep one Decode workload point around five minutes on the target
B200 setup. Actual time still depends on the model, hardware, and observed
decode throughput; reduce `--random-output-len` or `--num-prompts` further if a
point exceeds five minutes.

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
