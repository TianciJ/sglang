#!/usr/bin/env python3
"""Normalize SGLang ReqTimeStats log lines into request stage events."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence


_REQ_TIME_STATS_RE = re.compile(r"ReqTimeStats\((?P<meta>.*?)\):\s*(?P<stats>.*)$")
_DURATION_RE = re.compile(r"^(-?\d+(?:\.\d+)?)ms$")
_FLOAT_PREFIX_RE = re.compile(r"^(-?\d+(?:\.\d+)?)")
_DOCKER_TIMESTAMP_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\s+"
)


def _fields(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in value.split(","):
        if "=" not in item:
            continue
        key, field_value = item.strip().split("=", 1)
        result[key] = field_value.strip()
    return result


def _duration_seconds(stats: dict[str, str], key: str) -> float | None:
    raw = stats.get(key)
    if raw is None:
        return None
    match = _DURATION_RE.match(raw)
    if match is None:
        return None
    return float(match.group(1)) / 1000.0


def _float_prefix(stats: dict[str, str], key: str) -> float | None:
    raw = stats.get(key)
    if raw is None:
        return None
    match = _FLOAT_PREFIX_RE.match(raw)
    return float(match.group(1)) if match is not None else None


def _int_field(fields: dict[str, str], key: str) -> int | None:
    raw = fields.get(key)
    return int(raw) if raw is not None else None


def parse_req_time_stats_line(
    line: str, *, worker: str, log_timestamp: float | None = None
) -> dict[str, Any] | None:
    match = _REQ_TIME_STATS_RE.search(line)
    if match is None:
        return None

    meta = _fields(match.group("meta"))
    stats = _fields(match.group("stats"))
    role = meta.get("type", "unknown").lower()
    entry_time = _float_prefix(stats, "entry_time")

    if role == "prefill":
        detailed = "prefill_compute_duration" in stats
        stage_fields = (
            [
                ("prefill.bootstrap", "bootstrap_duration"),
                ("prefill.queue", "queue_duration"),
                ("prefill.compute", "prefill_compute_duration"),
                ("prefill.transfer_prepare", "transfer_prepare_duration"),
                ("prefill.transfer", "transfer_duration"),
                ("prefill.completion", "completion_duration"),
            ]
            if detailed
            else [
                ("prefill.bootstrap_queue", "bootstrap_queue_duration"),
                ("prefill.queue", "queue_duration"),
                ("prefill.forward_and_transfer", "forward_duration"),
            ]
        )
    elif role == "decode":
        detailed = True
        stage_fields = [
            ("decode.bootstrap", "bootstrap_duration"),
            ("decode.alloc_wait", "alloc_wait_duration"),
            ("decode.transfer", "transfer_duration"),
            ("decode.queue", "queue_duration"),
            ("decode.forward", "forward_duration"),
        ]
    else:
        detailed = False
        stage_fields = []

    events: list[dict[str, Any]] = []
    cursor = entry_time
    for stage, field in stage_fields:
        duration = _duration_seconds(stats, field)
        if duration is None:
            continue
        event: dict[str, Any] = {
            "stage": stage,
            "duration_s": duration,
            "measurement_kind": "exact_process_log",
        }
        if cursor is not None:
            event["started_at"] = cursor
            cursor += duration
            event["finished_at"] = cursor
        if not detailed and stage == "prefill.forward_and_transfer":
            event["inferred_substage"] = None
        events.append(event)

    transfer_total_mb = _float_prefix(stats, "transfer_total")
    transfer_speed = _float_prefix(stats, "transfer_speed")
    return {
        "request_id": meta.get("rid"),
        "bootstrap_room": _int_field(meta, "bootstrap_room"),
        "worker": worker,
        "role": role,
        "prompt_tokens": _int_field(meta, "input_len"),
        "cached_tokens": _int_field(meta, "cached_input_len"),
        "output_tokens": _int_field(meta, "output_len"),
        "log_timestamp": log_timestamp,
        "entry_time": entry_time,
        "timing_detail": "detailed" if detailed else "coarse",
        "transfer_speed_gb_s": transfer_speed,
        "transfer_total_bytes": (
            int(round(transfer_total_mb * 1024 * 1024))
            if transfer_total_mb is not None
            else None
        ),
        "events": events,
        "source_line": line,
    }


def _parse_docker_timestamp(line: str) -> float | None:
    match = _DOCKER_TIMESTAMP_RE.match(line)
    if match is None:
        return None
    value = match.group("timestamp")
    date_part, time_part = value[:-1].split("T", 1)
    if "." in time_part:
        clock, fraction = time_part.split(".", 1)
        time_part = f"{clock}.{(fraction + '000000')[:6]}"
    normalized = f"{date_part}T{time_part}+00:00"
    return dt.datetime.fromisoformat(normalized).timestamp()


def normalize_log_file(path: Path, *, worker: str) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    events: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.rstrip("\r\n")
            row = parse_req_time_stats_line(
                line, worker=worker, log_timestamp=_parse_docker_timestamp(line)
            )
            if row is None:
                continue
            row["source_file"] = str(path)
            row["source_line_number"] = line_number
            rows.append(row)
            shared = {
                key: value for key, value in row.items() if key not in {"events"}
            }
            for event in row["events"]:
                events.append({**shared, **event})
    return rows, events


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _parse_log_spec(value: str) -> tuple[str, Path]:
    worker, separator, path = value.partition("=")
    if not separator or not worker or not path:
        raise argparse.ArgumentTypeError("--log must be WORKER=PATH")
    return worker, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", type=_parse_log_spec, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows: list[dict] = []
    events: list[dict] = []
    for worker, path in args.log:
        file_rows, file_events = normalize_log_file(path, worker=worker)
        rows.extend(file_rows)
        events.extend(file_events)
    _write_jsonl(args.output, rows)
    _write_jsonl(args.events_output, events)
    return 0


if __name__ == "__main__":
    sys.exit(main())
