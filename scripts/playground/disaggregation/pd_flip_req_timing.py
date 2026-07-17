#!/usr/bin/env python3
"""Normalize SGLang ReqTimeStats log lines into request stage events."""

from __future__ import annotations

import re
from typing import Any


_REQ_TIME_STATS_RE = re.compile(r"ReqTimeStats\((?P<meta>.*?)\):\s*(?P<stats>.*)$")
_DURATION_RE = re.compile(r"^(-?\d+(?:\.\d+)?)ms$")
_FLOAT_PREFIX_RE = re.compile(r"^(-?\d+(?:\.\d+)?)")


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
