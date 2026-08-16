import json
from pathlib import Path

from vllm_bench.config import AppConfig
from vllm_bench.docker import DockerClient, ImageMetadata
from vllm_bench.runner import BenchmarkRunner


class FakeDocker(DockerClient):
    def __init__(self) -> None:
        super().__init__(printer=lambda _line: None)
        self.commands = []
        self.removed = False

    def daemon_available(self):
        return True

    def container_exists(self, _name):
        return False

    def create_container(self, **kwargs):
        self.commands.append(("create", kwargs))

    def inspect_image(self, image):
        return ImageMetadata(image, "sha256:id", ["image@sha256:digest"])

    def inspect_container(self, _name):
        return {"Image": "sha256:id", "State": {"Running": True}}

    def exec(self, name, command, **_kwargs):
        self.commands.append(("exec", name, list(command)))

    def copy_to(self, source, name, destination):
        self.commands.append(("copy_to", str(source), name, destination))

    def exec_stream(self, name, command, log_file, **_kwargs):
        self.commands.append(("stream", name, list(command)))
        log_file.write("ok\n")
        return 0

    def copy_from(self, _name, _source, destination):
        destination.mkdir(parents=True, exist_ok=True)

    def remove_container(self, _name):
        self.removed = True


class CollisionDocker(FakeDocker):
    def container_exists(self, name):
        return name == "occupied"

    def inspect_container(self, name):
        image = "sha256:other" if name == "occupied" else "sha256:id"
        return {"Image": image, "State": {"Running": True}}


def config(tmp_path: Path) -> tuple[AppConfig, dict]:
    raw = {
        "version": 1,
        "run_name": "run",
        "output_dir": str(tmp_path / "results"),
        "environment": {
            "MODEL_PATH": "/models/model",
            "VLLM_ENGINE_READY_TIMEOUT_S": "1200",
        },
        "docker": {
            "image": "image:tag",
            "host_models_path": str(tmp_path / "models"),
        },
        "jobs": [
            {
                "name": "job",
                "serve": {
                    "fixed_args": {},
                    "variants": [{"name": "tp8", "args": {"-tp": 8}}],
                },
                "bench": {
                    "fixed_args": {},
                    "stages": {
                        "prefill": {
                            "args": {
                                "--random-input-len": 8192,
                                "--random-output-len": 1,
                            }
                        },
                        "decode": {
                            "args": {
                                "--random-input-len": 1,
                                "--random-output-len": 8192,
                            }
                        },
                    },
                },
                "sweep": {
                    "args": {
                        "--workload-var": "max_concurrency",
                        "--num-runs": 1,
                    }
                },
            }
        ],
    }
    (tmp_path / "models").mkdir()
    return AppConfig.model_validate(raw), raw


def test_runner_uses_one_sweep_for_both_stages_and_cleans_up(tmp_path: Path) -> None:
    app_config, expanded = config(tmp_path)
    docker = FakeDocker()
    runner = BenchmarkRunner(
        app_config,
        expanded,
        docker=docker,
        project_root=Path(__file__).parents[1],
    )
    assert runner.run() == 0
    streams = [command for command in docker.commands if command[0] == "stream"]
    assert len(streams) == 1
    assert "--bench-params" in streams[0][2]
    assert docker.removed
    creates = [command for command in docker.commands if command[0] == "create"]
    container_name = creates[0][1]["name"]
    assert container_name.startswith("vllm-bench-run-")
    manifest = json.loads((tmp_path / "results" / "run" / "manifest.json").read_text())
    assert manifest["docker"]["image_id"] == "sha256:id"
    assert manifest["docker"]["container_name"] == container_name


def test_runner_generates_another_name_on_collision(tmp_path: Path) -> None:
    app_config, expanded = config(tmp_path)
    docker = CollisionDocker()
    runner = BenchmarkRunner(
        app_config,
        expanded,
        docker=docker,
        project_root=Path(__file__).parents[1],
    )
    runner.container_name = "occupied"

    assert runner.run() == 0
    creates = [command for command in docker.commands if command[0] == "create"]
    assert len(creates) == 1
    assert creates[0][1]["name"] != "occupied"
    assert creates[0][1]["name"].startswith("vllm-bench-run-")
