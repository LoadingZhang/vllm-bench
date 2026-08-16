"""Aggregate raw sweep output into stage-specific reports."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .docker import ImageMetadata

_RESULT_COLUMNS = [
    "run_name",
    "job",
    "variant",
    "stage",
    "docker_image",
    "docker_image_id",
    "docker_repo_digests",
    "max_concurrency",
    "run_count",
    "expected_run_count",
    "eligible",
    "completed_mean",
    "failed_total",
    "duration_mean",
    "request_throughput_mean",
    "request_throughput_std",
    "stage_throughput_mean",
    "stage_throughput_std",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p99_e2el_ms",
    "serve_command",
    "bench_command",
]
_LATENCY_FIELDS = [
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p99_e2el_ms",
]


def _mean(records: list[dict[str, Any]], field: str) -> float | None:
    values = [
        float(record[field]) for record in records if record.get(field) is not None
    ]
    return statistics.fmean(values) if values else None


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _stage_throughput(record: dict[str, Any], stage: str) -> float:
    if stage == "prefill":
        return float(record["total_input_tokens"]) / float(record["duration"])
    if stage == "decode":
        return float(record["output_throughput"])
    raise ValueError(f"unknown benchmark stage: {stage}")


def _load_run_records(raw_dir: Path) -> list[tuple[str, str, dict[str, Any]]]:
    records: list[tuple[str, str, dict[str, Any]]] = []
    if not raw_dir.exists():
        return records
    for path in raw_dir.rglob("summary.json"):
        relative = path.relative_to(raw_dir)
        if len(relative.parts) < 3:
            continue
        job, variant = relative.parts[0], relative.parts[1]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records.extend((job, variant, record) for record in payload)
    return records


def build_reports(
    *,
    output_dir: Path,
    run_name: str,
    image: ImageMetadata,
    variant_metadata: dict[tuple[str, str], dict[str, str]],
    expected_runs: dict[tuple[str, str], int],
) -> dict[str, Any]:
    """Write prefill/decode CSV files and a best-results JSON file."""

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for job, variant, record in _load_run_records(output_dir / "raw"):
        stage = record.get("vllm_bench_profile")
        concurrency = record.get("max_concurrency")
        if stage in {"prefill", "decode"} and concurrency is not None:
            grouped[(job, variant, stage, int(concurrency))].append(record)

    stage_rows: dict[str, list[dict[str, Any]]] = {"prefill": [], "decode": []}
    for (job, variant, stage, concurrency), records in sorted(grouped.items()):
        throughput_values = [_stage_throughput(record, stage) for record in records]
        expected = expected_runs[(job, variant)]
        failed_total = sum(int(record.get("failed") or 0) for record in records)
        metadata = variant_metadata[(job, variant)]
        row: dict[str, Any] = {
            "run_name": run_name,
            "job": job,
            "variant": variant,
            "stage": stage,
            "docker_image": image.configured_image,
            "docker_image_id": image.image_id,
            "docker_repo_digests": json.dumps(image.repo_digests),
            "max_concurrency": concurrency,
            "run_count": len(records),
            "expected_run_count": expected,
            "eligible": len(records) >= expected and failed_total == 0,
            "completed_mean": _mean(records, "completed"),
            "failed_total": failed_total,
            "duration_mean": _mean(records, "duration"),
            "request_throughput_mean": _mean(records, "request_throughput"),
            "request_throughput_std": _std(
                [float(record["request_throughput"]) for record in records]
            ),
            "stage_throughput_mean": statistics.fmean(throughput_values),
            "stage_throughput_std": _std(throughput_values),
            "serve_command": metadata["serve_command"],
            "bench_command": metadata["bench_command"],
        }
        row.update({field: _mean(records, field) for field in _LATENCY_FIELDS})
        stage_rows[stage].append(row)

    for stage, rows in stage_rows.items():
        with (output_dir / f"{stage}_results.csv").open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=_RESULT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    best = _select_best(stage_rows)
    (output_dir / "best_results.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )
    return best


def _select_best(stage_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {"variants": {}, "jobs": {}}
    for stage, rows in stage_rows.items():
        eligible = [row for row in rows if row["eligible"]]
        tie_field = "p99_ttft_ms" if stage == "prefill" else "p99_tpot_ms"

        def rank(
            row: dict[str, Any], latency_field: str = tie_field
        ) -> tuple[float, float, int]:
            latency = row[latency_field]
            return (
                -float(row["stage_throughput_mean"]),
                float("inf") if latency is None else float(latency),
                int(row["max_concurrency"]),
            )

        per_variant: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        per_job: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in eligible:
            per_variant[(row["job"], row["variant"])].append(row)

        for (job, variant), candidates in per_variant.items():
            winner = min(candidates, key=rank)
            variant_result = (
                result["variants"].setdefault(job, {}).setdefault(variant, {})
            )
            variant_result[stage] = winner
            per_job[job].append(winner)

        for job, candidates in per_job.items():
            result["jobs"].setdefault(job, {})[stage] = min(candidates, key=rank)
    return result
