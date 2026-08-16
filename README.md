# vllm-bench

`vllm-bench` 自动化执行基于 Docker 的 `vllm bench sweep serve_workload` 评测实验。它会启动一个 vLLM 容器，测试配置中的每个服务端变体（server variant），并分别输出 Prefill 和 Decode 报告。

## 功能特性

- **字面量 CLI 参数**：`-tp`、`--served-model-name` 以及带点号的参数均保持原样，不做重命名。
- **开关参数支持**：`null` 表示不带参数值的布尔开关标志（flag）。
- **共享模型启动**：每个服务端变体的 Prefill 和 Decode 评测共用同一次模型启动。
- **容错运行**：失败的变体会被记录，后续变体继续执行不受影响。
- **自动保存原始数据**：每个变体运行完毕后，容器内的原始 sweep 制品均会被自动复制到宿主机。
- **环境元数据记录**：报告中会完整记录配置的 Docker 标签、镜像 ID 以及 RepoDigest。
- **断点续跑**：中断的运行任务可通过 `--resume` 继续执行。

## 安装

```bash
cd ~/Codes/vllm-bench
uv sync --extra dev
```

宿主机需要安装支持 NVIDIA GPU 的 Docker。配置的镜像必须内置 `vllm bench sweep serve_workload` 命令。

## 配置

复制 `examples/deepseek-v3.2.yaml` 并配置宿主机上模型的绝对路径：

```yaml
docker:
  image: vllm/vllm-openai:v0.20.2
  model_path: /data/models/deepseek-ai/DeepSeek-V3.2
  environment:
    VLLM_ENGINE_READY_TIMEOUT_S: "1200"
  # 可选的持久化 vLLM 编译缓存目录：
  compile_cache_path: /data/vllm-compile-cache
```

工具每次新运行时都会生成一个唯一的容器名称，例如：

```bash
docker run --name vllm-bench-dsv32-b200-20260816123000-a1b2c3d4 \
  --gpus all --privileged \
  --ipc=host --net=host -it --entrypoint bash \
  -v /data/models/deepseek-ai/DeepSeek-V3.2:/models/DeepSeek-V3.2 \
  -v /data/vllm-compile-cache:/root/.cache/vllm \
  -d vllm/vllm-openai:v0.20.2
```

生成的容器名称格式如下：

```text
vllm-bench-<run-name>-<UTC timestamp>-<random suffix>
```

### 持久化模型编译缓存

设置 `docker.compile_cache_path` 可以在不同容器和评测运行之间复用 vLLM 编译产物：

```yaml
docker:
  image: vllm/vllm-openai:v0.20.2
  model_path: /data/models/deepseek-ai/DeepSeek-V3.2
  compile_cache_path: /data/vllm-compile-cache
```

宿主机路径必须是绝对路径。当目录不存在时，工具会自动创建该目录，并挂载到容器内的：

```text
/root/.cache/vllm
```

同时自动将以下环境变量注入到 sweep、server 和 benchmark 进程中：

```text
VLLM_CACHE_ROOT=/root/.cache/vllm
```

这能够持久化保存 vLLM 的 torch.compile / Inductor / Triton 缓存、DeepGEMM 缓存以及其他继承自 `VLLM_CACHE_ROOT` 的缓存。如果省略该字段，则保持容器默认的临时缓存行为（容器销毁即丢弃）。

配置的宿主机缓存路径与容器缓存根目录都会记录在 `manifest.json` 中。

### 字面量参数规则

```yaml
fixed_args:
  --trust-remote-code: null       # --trust-remote-code
  -tp: 8                          # -tp 8
  --kernel-config.enable_flashinfer_autotune: "False"
                                  # --kernel-config.enable_flashinfer_autotune=False
```

参数的键名（key）必须已经是合法的 CLI 参数。工具不会执行下划线/连字符互转或别名转换。字面量布尔值请使用带引号的字符串（如 `"False"`）；YAML 原生布尔值保留用于工具内部字段（如 `enabled`）。

配置变量和容器环境变量配置在 `docker.environment` 下：

```yaml
docker:
  environment:
    VLLM_ENGINE_READY_TIMEOUT_S: "1200"
```

`${NAME}` 变量解析时，会先从该对象中查找，未命中再从宿主机进程的环境变量中查找。配置的每一项都会通过 `docker exec --env` 传递给 sweep、server 和 benchmark 进程。

### 自动推导 serve 与 benchmark 连接参数

以下参数无需在 `jobs[].serve` 和 `jobs[].bench` 中重复手动配置：

```text
serve model
--served-model-name
--port
bench --host
bench --model
bench --served-model-name
bench --port
```

默认值推导规则如下：

- `docker.model_path` 为宿主机上完整的模型目录路径。
- 该目录会被挂载到容器内的 `/models/<目录名末尾部分>`，生成的容器路径将自动用作 serve/bench 的模型路径。
- 服务模型名称（Served model name）：优先取 `docker.environment.SERVED_MODEL_NAME`，其次取 `docker.environment.MODEL_NAME`，最后取 `docker.model_path` 的最后一级目录名。
- 端口号：取 `docker.environment.VLLM_PORT`，默认值为 `8000`。
- 评测目标主机（Benchmark host）：默认为 `127.0.0.1`。

例如，只需编写以下配置即可：

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

任何显式配置的值仍然具有最高优先级。例如，如果在 serve 中显式配置了 `--port: 9000`，该端口会自动传递给 benchmark，除非 benchmark 也显式配置了自己的 `--port`。

对于 Decode 阶段，上述示例特意覆盖了通用的 prompt 数量与预热（warmup）次数，并将输入长度压缩为 1 个 token，以确保所测负载主要由 Decode 主导而非 Prefill。`--ignore-eos` 防止模型在生成 EOS 时提前终止，使每个请求都持续生成直至达到指定的 `--random-output-len` 限制，但不会产生真正无上限的请求。

在 8 个正式请求、2 个预热请求和 512 个输出 token 的配置下，最慢的 `max_concurrency=1` 负载点总共生成 5,120 个输出 token。在目标 B200 环境下，这样设计的单个 Decode 负载点耗时大约在 5 分钟左右。实际耗时仍取决于具体模型、硬件及观测到的解码吞吐量；若单个负载点耗时超过 5 分钟，可进一步降低 `--random-output-len` 或 `--num-prompts`。

## 运行

```bash
uv run vllm-bench --config examples/deepseek-v3.2.yaml
```

常用命令行选项：

```text
--dry-run         校验并打印生成的命令，不实际调用 Docker
--resume          使用相同的展开后配置继续之前未完成的运行
--run-name NAME   覆盖配置中的 run_name
--keep-container  执行退出时不删除 Docker 容器
```

新启动的运行任务始终使用动态生成的容器名称，因此遗留的旧容器不会阻塞当前执行。在断点续跑（`--resume`）期间，工具会优先尝试使用原始清单（manifest）中记录的容器；仅当容器镜像 ID 完全匹配时才会复用该容器，否则会生成新名称、创建新容器并将先前的原始评测结果恢复到其中。

## 评测结果

结果文件会输出至 `<output_dir>/<run_name>/`：

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

Prefill 吞吐量计算公式：

```text
total_input_tokens / duration
```

Decode 吞吐量使用 benchmark 输出的 `output_throughput`。只有完成了指定运行次数且失败请求数为 0 的负载点才有资格参与 `best_results.json` 的优选。重复运行的数据首先按平均吞吐量排序，其次按 Prefill 的 P99 TTFT 或 Decode 的 P99 TPOT 排序，最后按较低并发数排序。

当前版本要求使用 `--workload-var: max_concurrency`。工具会在 `serve_workload` 实际测试过的负载点中选出最优解，并不假设测试范围之外存在理论最优值。

### 混合负载 1600/150 限制 TPOT 的对比实验

`examples/deepseek-v3.2-mixed-1600-150.yaml` 比较了 TP8 与 DP8+EP 在固定 1,600 输入 token 和 150 输出 token 下的表现。每个请求都包含 Prefill 和 Decode 负载。Decode 阶段配置了：

```yaml
selection:
  mean_tpot_ms_lt: 50
```

实测平均 TPOT 不严格小于 50 ms 的负载点仍会保留在 `decode_results.csv` 中，但会被标记为 `slo_met=false` 和 `eligible=false`。`best_results.json` 会为每个变体选出符合条件（eligible）且输出吞吐量最高的负载点，并在以下路径选出 TP8 和 DP8+EP 之间的胜出者：

```text
jobs.mixed-1600-150.decode
```

若该字段不存在，则说明两个变体均未产生满足“平均 TPOT < 50 ms 且无失败请求”的完整负载点数据。

## 开发

```bash
uv run pytest -v
uv run ruff check .
```
