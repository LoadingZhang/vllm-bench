import csv
import json
from pathlib import Path

from vllm_bench.docker import ImageMetadata
from vllm_bench.report import build_reports


def write_summary(
    raw: Path,
    job: str,
    variant: str,
    stage: str,
    concurrency: int,
    records: list[dict],
) -> None:
    path = (
        raw
        / job
        / variant
        / "workload"
        / f"BENCH-{stage}-WL-max_concurrency={concurrency}"
    )
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps(records), encoding="utf-8")


def record(stage: str, concurrency: int, throughput: float, latency: float) -> dict:
    duration = 10.0
    base = {
        "vllm_bench_profile": stage,
        "max_concurrency": concurrency,
        "duration": duration,
        "completed": 10,
        "failed": 0,
        "total_input_tokens": throughput * duration,
        "output_throughput": throughput,
        "request_throughput": 1.0,
        "mean_tpot_ms": latency,
        "p99_ttft_ms": latency,
        "p99_tpot_ms": latency,
    }
    return base


def test_reports_split_stages_and_select_best(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write_summary(raw, "job", "tp8", "prefill", 1, [record("prefill", 1, 10, 2)])
    write_summary(raw, "job", "tp8", "prefill", 4, [record("prefill", 4, 20, 5)])
    write_summary(raw, "job", "tp8", "decode", 1, [record("decode", 1, 30, 4)])
    write_summary(raw, "job", "tp8", "decode", 4, [record("decode", 4, 40, 8)])

    best = build_reports(
        output_dir=tmp_path,
        run_name="run",
        image=ImageMetadata("image:tag", "sha256:id", ["image@sha256:digest"]),
        variant_metadata={
            ("job", "tp8"): {"serve_command": "serve", "bench_command": "bench"}
        },
        expected_runs={("job", "tp8"): 1},
    )

    assert best["variants"]["job"]["tp8"]["prefill"]["max_concurrency"] == 4
    assert best["variants"]["job"]["tp8"]["decode"]["max_concurrency"] == 4
    with (tmp_path / "prefill_results.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[0]["docker_image_id"] == "sha256:id"


def test_incomplete_or_failed_points_are_not_eligible(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    failed = record("prefill", 8, 100, 1)
    failed["failed"] = 1
    write_summary(raw, "job", "tp8", "prefill", 8, [failed])
    write_summary(raw, "job", "tp8", "prefill", 4, [record("prefill", 4, 20, 3)])

    best = build_reports(
        output_dir=tmp_path,
        run_name="run",
        image=ImageMetadata("image", "id", []),
        variant_metadata={
            ("job", "tp8"): {"serve_command": "serve", "bench_command": "bench"}
        },
        expected_runs={("job", "tp8"): 1},
    )
    assert best["jobs"]["job"]["prefill"]["max_concurrency"] == 4


def test_decode_selection_requires_mean_tpot_below_limit(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    write_summary(raw, "job", "tp8", "decode", 4, [record("decode", 4, 40, 40)])
    write_summary(raw, "job", "tp8", "decode", 8, [record("decode", 8, 80, 50)])
    write_summary(
        raw, "job", "dp8-ep", "decode", 16, [record("decode", 16, 70, 49)]
    )

    best = build_reports(
        output_dir=tmp_path,
        run_name="run",
        image=ImageMetadata("image", "id", []),
        variant_metadata={
            ("job", "tp8"): {"serve_command": "serve-tp8", "bench_command": "bench"},
            ("job", "dp8-ep"): {
                "serve_command": "serve-dp8-ep",
                "bench_command": "bench",
            },
        },
        expected_runs={("job", "tp8"): 1, ("job", "dp8-ep"): 1},
        selection_limits={("job", "decode"): 50},
    )

    assert best["variants"]["job"]["tp8"]["decode"]["max_concurrency"] == 4
    assert best["jobs"]["job"]["decode"]["variant"] == "dp8-ep"
    with (tmp_path / "decode_results.csv").open() as file:
        rows = list(csv.DictReader(file))
    failed_slo = next(row for row in rows if row["max_concurrency"] == "8")
    assert failed_slo["slo_met"] == "False"
    assert failed_slo["eligible"] == "False"
