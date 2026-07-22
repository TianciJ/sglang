#!/usr/bin/env python3
"""Validate and report one clean-upstream Qwen80B baseline run."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


EXPECTED_IMAGE_ID = "sha256:7dd92779d739364d79af34af65815ddc14e567728e5256f65ac922367161213e"
EXPECTED_TRACE_SHA256 = "c5dbbf75c997dfc5d67a18251082f2f246d6c055eb4af5040fbe147f49f4ce5d"
EXPECTED_TPOT_SOURCE = "client_first_last_output_over_usage_completion_tokens"
PROVENANCE_FIELDS = (
    "run_id",
    "image",
    "image_id",
    "trace_sha256",
    "model_id",
    "model_fingerprint",
    "router_sha256",
    "topology",
    "gpu_ids",
    "tp_size",
    "dp_size",
    "output_contract",
    "tpot_metric_source",
)


JsonDict = dict[str, Any]


def read_json(path: Path) -> JsonDict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        raise ValueError(f"missing required file: {path}")
    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def stats(values: Iterable[float]) -> JsonDict:
    values = [float(value) for value in values]
    if not values:
        return {"count": 0, "mean_s": None, "median_s": None, "p95_s": None, "max_s": None}
    return {
        "count": len(values),
        "mean_s": statistics.fmean(values),
        "median_s": statistics.median(values),
        "p95_s": percentile(values, 0.95),
        "max_s": max(values),
    }


def _validate_manifest(manifest: JsonDict) -> None:
    missing = [field for field in PROVENANCE_FIELDS if manifest.get(field) in (None, "")]
    if missing:
        raise ValueError("missing provenance fields: " + ", ".join(missing))
    if manifest["image_id"] != EXPECTED_IMAGE_ID:
        raise ValueError(f"unexpected image_id: {manifest['image_id']}")
    if manifest["trace_sha256"] != EXPECTED_TRACE_SHA256:
        raise ValueError(f"unexpected trace_sha256: {manifest['trace_sha256']}")
    if manifest["topology"] != "1P3D" or manifest["tp_size"] != 2 or manifest["dp_size"] != 1:
        raise ValueError("manifest topology must be 1P3D with TP=2 and DP=1")
    if manifest["output_contract"] != "natural":
        raise ValueError("manifest output_contract must be natural")
    if manifest["tpot_metric_source"] != EXPECTED_TPOT_SOURCE:
        raise ValueError(f"manifest tpot_metric_source must be {EXPECTED_TPOT_SOURCE}")


def _validate_requests(rows: list[JsonDict], expected_requests: int, expected_tokens: int) -> None:
    if len(rows) != expected_requests:
        raise ValueError(f"request count: expected {expected_requests}, got {len(rows)}")
    request_ids = [str(row.get("request_id") or "") for row in rows]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("duplicate request IDs")
    if any(not request_id for request_id in request_ids):
        raise ValueError("empty request ID")
    if any(row.get("status") != "completed" for row in rows):
        raise ValueError("one or more requests are not completed")
    if any(int(row.get("completion_tokens") or -1) != expected_tokens for row in rows):
        raise ValueError(f"completion tokens must equal {expected_tokens}")
    if any(row.get("completion_token_match") is not True for row in rows):
        raise ValueError("completion token count does not match")
    if any(row.get("finish_reason") != "length" for row in rows):
        raise ValueError("finish reason must be length")
    if any(row.get("tpot_metric_source") != EXPECTED_TPOT_SOURCE for row in rows):
        raise ValueError(f"every request must use tpot_metric_source={EXPECTED_TPOT_SOURCE}")
    if any(row.get("avg_tpot_s") is None or row.get("ttft_s") is None for row in rows):
        raise ValueError("every request must contain client-observed TTFT and token-normalized TPOT")


def _count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        raise ValueError(f"missing required file: {path}")
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def _read_tpot(path: Path) -> list[JsonDict]:
    if not path.exists():
        raise ValueError(f"missing required file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _ttft_summary(rows: list[JsonDict]) -> JsonDict:
    result: JsonDict = {}
    for kind in ("all", "short", "long"):
        selected = rows if kind == "all" else [row for row in rows if row.get("prompt_kind") == kind]
        item = stats(float(row["ttft_s"]) for row in selected)
        item["attainment"] = (
            sum(bool(row.get("ttft_met")) for row in selected) / len(selected) if selected else None
        )
        result[kind] = item
    return result


def _stream_event_gap_summary(tpot_rows: list[JsonDict]) -> JsonDict:
    values = [float(row["interval_s"]) for row in tpot_rows]
    good = sum(str(row.get("met", "")).lower() == "true" for row in tpot_rows)
    return {
        "count": len(values),
        "p50_s": percentile(values, 0.50),
        "p95_s": percentile(values, 0.95),
        "p99_s": percentile(values, 0.99),
        "max_s": max(values) if values else None,
        "attainment": good / len(values) if values else None,
    }


def _write_csv(path: Path, rows: list[JsonDict]) -> None:
    fields = [
        "request_id", "prompt_kind", "arrival_offset_s", "prompt_tokens", "ttft_s", "ttft_slo_s",
        "ttft_met", "avg_tpot_s", "p50_tpot_s", "p95_tpot_s", "max_tpot_s",
        "good_tpot_intervals", "total_tpot_intervals", "latency_s", "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_scatter(path: Path, rows: list[JsonDict], value_field: str, title: str, y_label: str) -> None:
    width, height, margin = 1000, 420, 55
    values = [float(row.get(value_field) or 0.0) for row in rows]
    max_value = max(values + [0.001]) * 1.08
    plot_width, plot_height = width - 2 * margin, height - 2 * margin
    points: list[str] = []
    for index, (row, value) in enumerate(zip(rows, values)):
        x = margin + (plot_width * index / max(1, len(rows) - 1))
        y = height - margin - (plot_height * value / max_value)
        request_id = html.escape(str(row.get("request_id") or ""))
        points.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#2563eb"><title>{request_id}: {value:.6f}s</title></circle>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(title)}</text>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>
<text x="{width / 2}" y="{height-12}" text-anchor="middle" font-family="sans-serif" font-size="12">request order / arrival time</text>
<text x="14" y="{height / 2}" transform="rotate(-90 14 {height / 2})" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(y_label)}</text>
{''.join(points)}
</svg>\n'''
    path.write_text(svg, encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_markdown(path: Path, manifest: JsonDict, rows: list[JsonDict], summary: JsonDict) -> None:
    lines = [
        "# Clean upstream Qwen80B baseline report",
        "",
        "## Validity and boundary",
        "",
        "This run passed the recorded artifact gates. TTFT and token-normalized request TPOT are client-observed timings; they are not GPU kernel or internal Prefill/Decode stage durations.",
        "Primary TPOT is `(last nonempty output event - first nonempty output event) / (usage.completion_tokens - 1)`. The retained CSV contains SSE stream-event gaps, which are diagnostic transport events and are not one row per token.",
        "",
        f"- Run: `{manifest['run_id']}`",
        f"- Image ID: `{manifest['image_id']}`",
        f"- Trace SHA256: `{manifest['trace_sha256']}`",
        f"- Model: `{manifest['model_id']}`",
        f"- Router SHA256: `{manifest['router_sha256']}`",
        f"- Topology: `{manifest['topology']}`",
        "",
        "This is one measured run and does not establish run-to-run statistical significance.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Requests | {summary['requests']} |",
        f"| TTFT mean (s) | {_fmt(summary['ttft']['all']['mean_s'])} |",
        f"| TTFT P95 (s) | {_fmt(summary['ttft']['all']['p95_s'])} |",
        f"| TTFT SLO attainment | {_fmt(summary['ttft']['all']['attainment'])} |",
        f"| Request token-normalized TPOT mean (s) | {_fmt(summary['tpot_requests']['mean_s'])} |",
        f"| Request token-normalized TPOT P95 (s) | {_fmt(summary['tpot_requests']['p95_s'])} |",
        f"| Request token-normalized TPOT attainment | {_fmt(summary['tpot_requests']['attainment'])} |",
        f"| SSE stream-event gap rows | {summary['stream_event_gap_rows']} |",
        f"| SSE stream-event gap P95 (s) | {_fmt(summary['stream_event_gaps']['p95_s'])} |",
        "",
        "## Requests",
        "",
        "| request_id | kind | arrival_s | TTFT_s | TTFT met | avg TPOT_s | P95 TPOT_s | latency_s | status |",
        "|---|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('request_id')} | {row.get('prompt_kind')} | {_fmt(row.get('arrival_offset_s'))} | "
            f"{_fmt(row.get('ttft_s'))} | {row.get('ttft_met')} | {_fmt(row.get('avg_tpot_s'))} | "
            f"{_fmt(row.get('p95_tpot_s'))} | {_fmt(row.get('latency_s'))} | {row.get('status')} |"
        )
    lines.extend(
        [
            "",
            "## Raw evidence",
            "",
            "- `../raw/slo_ledger.jsonl`",
            "- `../raw/upstream_baseline/request_metrics.jsonl`",
            "- `../raw/upstream_baseline/tpot_tokens.csv`",
            "- `../raw/upstream_baseline/responses.jsonl`",
            "- `../raw/upstream_baseline/errors.jsonl`",
            "- `../manifest.json`",
            "- `../logs/`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_report(
    run_dir: Path,
    *,
    expected_requests: int = 40,
    expected_tokens: int = 10_000,
) -> JsonDict:
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    _validate_manifest(manifest)
    raw = run_dir / "raw" / "upstream_baseline"
    rows = read_jsonl(raw / "request_metrics.jsonl")
    _validate_requests(rows, expected_requests, expected_tokens)
    errors = read_jsonl(raw / "errors.jsonl")
    if errors:
        raise ValueError(f"errors.jsonl must be empty, got {len(errors)} rows")
    ledger_rows = _count_nonempty_lines(run_dir / "raw" / "slo_ledger.jsonl")
    if ledger_rows < expected_requests * 2:
        raise ValueError(f"ledger row count is too small: expected at least {expected_requests * 2}, got {ledger_rows}")
    tpot_rows = _read_tpot(raw / "tpot_tokens.csv")
    if not tpot_rows:
        raise ValueError("SSE stream-event gap evidence is empty")
    gap_request_ids = {str(row.get("request_id") or "") for row in tpot_rows}
    missing_gap_ids = sorted({str(row["request_id"]) for row in rows} - gap_request_ids)
    if missing_gap_ids:
        raise ValueError("missing SSE stream-event gap evidence for: " + ", ".join(missing_gap_ids))

    rows = sorted(rows, key=lambda row: (float(row.get("arrival_offset_s") or 0), str(row.get("request_id") or "")))
    request_tpot = stats(float(row["avg_tpot_s"]) for row in rows)
    request_tpot["attainment"] = sum(float(row["avg_tpot_s"]) <= float(row.get("tpot_slo_s") or 0.05) for row in rows) / len(rows)
    summary = {
        "valid": True,
        "requests": len(rows),
        "ledger_rows": ledger_rows,
        "stream_event_gap_rows": len(tpot_rows),
        "ttft": _ttft_summary(rows),
        "tpot_requests": request_tpot,
        "stream_event_gaps": _stream_event_gap_summary(tpot_rows),
        "tpot_metric_source": EXPECTED_TPOT_SOURCE,
        "instrumentation": "client-observed streaming event times from time.monotonic(); usage.completion_tokens supplies token count",
        "limitation": "one measured run; no run-to-run statistical significance",
    }
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(report_dir / "request_metrics.csv", rows)
    _write_scatter(report_dir / "ttft_scatter.svg", rows, "ttft_s", "TTFT by request", "TTFT (s)")
    _write_scatter(report_dir / "tpot_scatter.svg", rows, "avg_tpot_s", "Average TPOT by request", "TPOT (s)")
    _write_markdown(report_dir / "report.md", manifest, rows, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = generate_report(args.run_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
