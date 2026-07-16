#!/usr/bin/env python3
"""Derive auditable PD Flip donor timing tables from one trace40 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


JsonDict = Dict[str, Any]
SHANGHAI = timezone(timedelta(hours=8))


def load_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_controller_log(path: Path) -> JsonDict:
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = "\n{\n"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"controller JSON payload not found in {path}")
    return json.loads(text[start + 1 :])


def merge_non_null(target: JsonDict, source: JsonDict) -> None:
    for key, value in source.items():
        if value is not None:
            target[key] = value


def session_belongs_to_run(session_id: Any, run_id: str) -> bool:
    if not session_id:
        return False
    return str(session_id).startswith(f"{run_id}-")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iso_shanghai(epoch: Optional[float]) -> Optional[str]:
    if epoch is None:
        return None
    return datetime.fromtimestamp(float(epoch), timezone.utc).astimezone(SHANGHAI).isoformat(
        timespec="microseconds"
    )


def ms(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value) * 1000.0, 6)


def diff_ms(end: Optional[float], start: Optional[float]) -> Optional[float]:
    if end is None or start is None:
        return None
    return round((float(end) - float(start)) * 1000.0, 6)


def build_full_timeline_rows(
    *,
    crossing_epoch: float,
    trigger_epoch: float,
    first: Optional[JsonDict],
    observation_epoch: Optional[float],
    final: Optional[JsonDict],
    controller_actions: list[JsonDict],
    controller_finished_epoch: Optional[float],
) -> list[JsonDict]:
    rows: list[JsonDict] = []

    def add(
        stage: str,
        *,
        phase: str,
        start: Optional[float] = None,
        end: Optional[float] = None,
        duration: Optional[float] = None,
        timing_source: str,
        lane: str = "critical_path",
        overlaps: str = "",
    ) -> None:
        rows.append(
            {
                "order": len(rows) + 1,
                "phase": phase,
                "stage": stage,
                "lane": lane,
                "duration_ms": duration if duration is not None else diff_ms(end, start),
                "start_epoch": start,
                "end_epoch": end,
                "start_shanghai": iso_shanghai(start),
                "end_shanghai": iso_shanghai(end),
                "timing_source": timing_source,
                "overlaps": overlaps,
            }
        )

    add(
        "slo_sample_gate_to_controller_decision",
        phase="slo_trigger",
        start=crossing_epoch,
        end=trigger_epoch,
        timing_source="request_ledger_and_controller_snapshot",
    )

    def add_batch(batch: str, points: JsonDict) -> None:
        for name, field in (
            ("prefill_restore_process", "prefill_restore_critical_ms"),
            ("prefill_transfer_process", "prefill_transfer_critical_ms"),
            ("source_base_transfer_process", "source_base_transfer_critical_ms"),
        ):
            add(
                f"{batch}_{name}",
                phase=f"{batch}_migration",
                duration=points.get(field),
                timing_source="worker_exact_process",
                lane="parallel_detail",
                overlaps=f"{batch}_base_receive_window",
            )
        add(
            f"{batch}_base_receive_window",
            phase=f"{batch}_migration",
            start=points.get("base_start"),
            end=points.get("base_ready"),
            timing_source="worker_epoch",
        )
        add(
            f"{batch}_base_ready_to_delta_start",
            phase=f"{batch}_migration",
            start=points.get("base_ready"),
            end=points.get("delta_start"),
            timing_source="worker_epoch",
        )
        add(
            f"{batch}_delta_transfer_window",
            phase=f"{batch}_migration",
            start=points.get("delta_start"),
            end=points.get("delta_complete"),
            timing_source="worker_epoch",
        )
        add(
            f"{batch}_delta_complete_to_commit_ready",
            phase=f"{batch}_migration",
            start=points.get("delta_complete"),
            end=points.get("commit_ready"),
            timing_source="worker_epoch",
        )
        add(
            f"{batch}_commit_ready_to_source_finish",
            phase=f"{batch}_migration",
            start=points.get("commit_ready"),
            end=points.get("source_finish"),
            timing_source="worker_epoch",
        )
        add(
            f"{batch}_source_finish_to_target_activate",
            phase=f"{batch}_migration",
            start=points.get("source_finish"),
            end=points.get("activation"),
            timing_source="worker_epoch",
        )

    if first is not None:
        add(
            "controller_decision_to_first_base_start",
            phase="first_migration",
            start=trigger_epoch,
            end=first.get("base_start"),
            timing_source="controller_snapshot_and_worker_epoch",
        )
        add_batch("first", first)
        if observation_epoch is not None:
            add(
                "observation_window",
                phase="observation",
                start=first.get("activation"),
                end=observation_epoch,
                timing_source="worker_epoch_and_controller_snapshot",
            )

    if final is not None:
        if observation_epoch is not None:
            add(
                "observation_decision_to_final_base_start",
                phase="final_migration",
                start=observation_epoch,
                end=final.get("base_start"),
                timing_source="controller_snapshot_and_worker_epoch",
            )
        add_batch("final", final)

        tail_started = False
        for action in controller_actions:
            step = str(action.get("step") or "")
            if step == "post_migration_idle_assertion":
                tail_started = True
            if not tail_started:
                continue
            add(
                step,
                phase="role_flip_tail",
                duration=ms(action.get("elapsed_seconds")),
                timing_source="controller_action_elapsed",
                lane="controller_action",
                overlaps="final_activate_to_controller_exit",
            )
        if controller_finished_epoch is not None:
            add(
                "final_activate_to_controller_exit",
                phase="role_flip_tail",
                start=final.get("activation"),
                end=controller_finished_epoch,
                timing_source="worker_epoch_and_runner_timeline",
            )

    return rows


def write_csv(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--slo-threshold", type=float, default=0.99)
    parser.add_argument("--min-prefill-samples", type=int, default=10)
    parser.add_argument("--min-decode-samples", type=int, default=10)
    parser.add_argument("--page-size", type=int, default=64)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    controller = load_controller_log(run_dir / "controller/controller.log")
    request_metrics = list(load_jsonl(run_dir / "workload/state_machine/request_metrics.jsonl"))
    by_trace = {str(row["request_id"]): row for row in request_metrics}
    by_upstream = {str(row["upstream_request_id"]): row for row in request_metrics}
    wall_offsets = [
        float(row["start_wall"]) - float(row["start_monotonic"])
        for row in request_metrics
    ]
    wall_offset = statistics.median(wall_offsets)

    snapshots = list(controller.get("snapshots") or [])
    trigger_index = next(
        index
        for index, snapshot in enumerate(snapshots)
        if int((snapshot.get("prefill_counts") or {}).get("total") or 0)
        >= args.min_prefill_samples
        and int((snapshot.get("decode_counts") or {}).get("total") or 0)
        >= args.min_decode_samples
        and float(snapshot.get("prefill_slo_attainment") or 0.0)
        < args.slo_threshold
    )
    trigger_snapshot = snapshots[trigger_index]
    previous_snapshot = snapshots[trigger_index - 1] if trigger_index else None
    trigger_mono = float(trigger_snapshot["timestamp"])
    previous_mono = (
        float(previous_snapshot["timestamp"]) if previous_snapshot is not None else None
    )

    earliest: dict[str, JsonDict] = {}
    latest: dict[str, JsonDict] = {}
    for record in load_jsonl(run_dir / "workload/trace_slo_ledger.jsonl"):
        event_time = float(record["event_time"])
        if event_time > trigger_mono:
            continue
        request_id = str(record["request_id"])
        earliest.setdefault(request_id, record)
        latest[request_id] = record

    ordered = sorted(
        earliest.values(), key=lambda row: (float(row["event_time"]), str(row["request_id"]))
    )
    crossing = ordered[args.min_prefill_samples - 1]
    crossing_id = str(crossing["request_id"])
    crossing_metric = by_trace[crossing_id]
    contributors: list[JsonDict] = []
    for rank, first_record in enumerate(ordered[: args.min_prefill_samples], start=1):
        request_id = str(first_record["request_id"])
        current = latest[request_id]
        metric = by_trace.get(request_id, {})
        first_mono = float(current["first_token_time"])
        contributors.append(
            {
                "sample_rank": rank,
                "request_id": request_id,
                "upstream_request_id": metric.get("upstream_request_id"),
                "prompt_kind": metric.get("prompt_kind"),
                "first_token_mono": first_mono,
                "first_token_shanghai": iso_shanghai(first_mono + wall_offset),
                "ttft_ms": ms(current.get("ttft_seconds")),
                "ttft_slo_ms": ms(current.get("ttft_slo_seconds")),
                "ttft_met": bool(current.get("ttft_met")),
                "new_since_previous_poll": previous_mono is not None
                and float(first_record["event_time"]) > previous_mono,
            }
        )
    write_csv(output_dir / "slo_trigger_contributors.csv", contributors)

    trigger_epoch = trigger_mono + wall_offset
    crossing_first_token_mono = float(crossing["first_token_time"])
    trigger_payload = {
        "decision": "prefill_risky_decode_healthy",
        "slo_threshold": args.slo_threshold,
        "min_prefill_samples": args.min_prefill_samples,
        "min_decode_samples": args.min_decode_samples,
        "previous_snapshot": {
            "timestamp_mono": previous_mono,
            "timestamp_shanghai": iso_shanghai(previous_mono + wall_offset)
            if previous_mono is not None
            else None,
            "prefill_good": int((previous_snapshot.get("prefill_counts") or {}).get("good") or 0)
            if previous_snapshot
            else None,
            "prefill_total": int((previous_snapshot.get("prefill_counts") or {}).get("total") or 0)
            if previous_snapshot
            else None,
        },
        "trigger_snapshot": {
            "timestamp_mono": trigger_mono,
            "timestamp_epoch": trigger_epoch,
            "timestamp_shanghai": iso_shanghai(trigger_epoch),
            "prefill_good": int((trigger_snapshot.get("prefill_counts") or {}).get("good") or 0),
            "prefill_total": int((trigger_snapshot.get("prefill_counts") or {}).get("total") or 0),
            "prefill_attainment": trigger_snapshot.get("prefill_slo_attainment"),
            "decode_good": int((trigger_snapshot.get("decode_counts") or {}).get("good") or 0),
            "decode_total": int((trigger_snapshot.get("decode_counts") or {}).get("total") or 0),
            "decode_attainment": trigger_snapshot.get("decode_slo_attainment"),
        },
        "sample_gate_crossing_request": {
            "request_id": crossing_id,
            "upstream_request_id": crossing_metric.get("upstream_request_id"),
            "prompt_kind": crossing_metric.get("prompt_kind"),
            "first_token_mono": crossing_first_token_mono,
            "first_token_shanghai": iso_shanghai(crossing_first_token_mono + wall_offset),
            "ttft_ms": ms(crossing.get("ttft_seconds")),
            "ttft_slo_ms": ms(crossing.get("ttft_slo_seconds")),
            "ttft_met": bool(crossing.get("ttft_met")),
            "controller_detection_lag_ms": diff_ms(trigger_mono, crossing_first_token_mono),
        },
        "interpretation": (
            "Prefill attainment was already below threshold before the transition. "
            "This request supplied the tenth eligible TTFT sample and opened the minimum-sample gate; "
            "the controller acted at the next polling snapshot."
        ),
        "monotonic_to_wall_offset_seconds": wall_offset,
        "offset_span_microseconds": (max(wall_offsets) - min(wall_offsets)) * 1_000_000,
    }
    (output_dir / "slo_trigger.json").write_text(
        json.dumps(trigger_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    timing: dict[tuple[str, str, str], JsonDict] = defaultdict(dict)
    measurements: dict[tuple[str, str, str], JsonDict] = defaultdict(dict)
    donor_entries: dict[tuple[str, str], JsonDict] = defaultdict(dict)
    sessions: set[str] = set()
    for event in load_jsonl(run_dir / "metrics/events.jsonl"):
        event_type = event.get("event_type")
        status = event.get("status") or {}
        session_id = status.get("session_id")
        node = str(event.get("node") or "")
        if session_belongs_to_run(session_id, run_dir.name):
            sessions.add(str(session_id))
        if event_type == "migration_status" and session_belongs_to_run(
            session_id, run_dir.name
        ):
            debug = status.get("timing_debug") or {}
            for entry in debug.get("entries") or []:
                rid = str(entry.get("rid") or "")
                merge_non_null(timing[(str(session_id), node, rid)], entry.get("timing") or {})
            for row in status.get("request_measurements") or []:
                rid = str(row.get("rid") or "")
                merge_non_null(measurements[(str(session_id), node, rid)], row)
        elif event_type == "prefill_donor_status" and session_belongs_to_run(
            session_id, run_dir.name
        ):
            for rid, row in (status.get("entries") or {}).items():
                merge_non_null(donor_entries[(str(session_id), str(rid))], row)

    request_rows: list[JsonDict] = []
    batch_rows: list[JsonDict] = []
    batch_points: dict[str, JsonDict] = {}
    for session_id in sorted(sessions):
        batch = "first" if session_id.endswith("-first") else "final"
        rids = sorted(
            {
                rid
                for sess, _node, rid in measurements
                if sess == session_id and rid
            }
        )
        for rid in rids:
            target = measurements.get((session_id, "node3", rid), {})
            source = measurements.get((session_id, "node2", rid), {})
            donor = donor_entries.get((session_id, rid), {})
            target_timing = timing.get((session_id, "node3", rid), {})
            source_timing = timing.get((session_id, "node2", rid), {})
            trace = by_upstream.get(rid, {})
            p_tokens = int(target.get("prompt_len") or source.get("prompt_len") or donor.get("prompt_len") or 0)
            b_tokens = int(target.get("prefill_donor_end") or source.get("prefill_donor_end") or donor.get("prefill_donor_end") or 0)
            c0_tokens = int(target.get("c0_tokens") or 0)
            c1_tokens = int(target.get("c1_tokens") or source.get("c1_tokens") or c0_tokens)
            row = {
                "batch": batch,
                "session_id": session_id,
                "trace_request_id": trace.get("request_id"),
                "upstream_request_id": rid,
                "prompt_kind": trace.get("prompt_kind"),
                "page_size": args.page_size,
                "p_tokens": p_tokens,
                "b_tokens": b_tokens,
                "c0_tokens": c0_tokens,
                "c1_tokens": c1_tokens,
                "source_logical_base_tokens": max(0, c0_tokens - b_tokens),
                "delta_logical_tokens": max(0, c1_tokens - c0_tokens),
                "prefill_restore_hit_tokens": donor.get("prefill_donor_restore_hit_len"),
                "prefill_pages": donor.get("prefill_donor_pages"),
                "prefill_bytes": donor.get("prefill_donor_transfer_bytes"),
                "prefill_restore_ms": ms(donor.get("prefill_donor_restore_seconds")),
                "prefill_transfer_ms": ms(donor.get("prefill_donor_transfer_seconds")),
                "source_base_pages": source.get("source_base_pages") or target.get("source_base_pages"),
                "source_base_bytes": source.get("source_base_transfer_bytes"),
                "source_base_transfer_ms": ms(source.get("source_duration_seconds")),
                "target_base_receive_ms": diff_ms(
                    target_timing.get("target_held_epoch"),
                    target_timing.get("target_donor_transferring_epoch"),
                ),
                "delta_bytes": source.get("delta_bytes"),
                "delta_transfer_ms": ms(source.get("delta_duration_seconds")),
                "base_ready_to_delta_start_ms": diff_ms(
                    source_timing.get("delta_transfer_started_epoch"),
                    target_timing.get("target_held_epoch"),
                ),
                "delta_complete_to_commit_ready_ms": diff_ms(
                    target_timing.get("target_commit_ready_epoch"),
                    source_timing.get("delta_transfer_completed_epoch"),
                ),
                "commit_ready_to_activate_ms": diff_ms(
                    target_timing.get("target_adopted_epoch"),
                    target_timing.get("target_commit_ready_epoch"),
                ),
                "target_base_start_to_activate_ms": diff_ms(
                    target_timing.get("target_adopted_epoch"),
                    target_timing.get("target_donor_transferring_epoch"),
                ),
                "base_start_shanghai": iso_shanghai(target_timing.get("target_donor_transferring_epoch")),
                "base_ready_shanghai": iso_shanghai(target_timing.get("target_held_epoch")),
                "delta_start_shanghai": iso_shanghai(source_timing.get("delta_transfer_started_epoch")),
                "delta_complete_shanghai": iso_shanghai(source_timing.get("delta_transfer_completed_epoch")),
                "commit_ready_shanghai": iso_shanghai(target_timing.get("target_commit_ready_epoch")),
                "source_finish_shanghai": iso_shanghai(source_timing.get("source_finish_migrated_epoch")),
                "activate_shanghai": iso_shanghai(target_timing.get("target_adopted_epoch")),
                "target_prefix_match_skipped": target.get("target_prefix_match_skipped"),
                "provenance_mode": "prefill_donor_and_source_decode",
            }
            request_rows.append(row)

        batch_request_rows = [row for row in request_rows if row["session_id"] == session_id]
        batch_target_timings = [timing[(session_id, "node3", row["upstream_request_id"])] for row in batch_request_rows]
        batch_source_timings = [timing[(session_id, "node2", row["upstream_request_id"])] for row in batch_request_rows]

        def values(items: list[JsonDict], key: str) -> list[float]:
            return [float(item[key]) for item in items if item.get(key) is not None]

        base_starts = values(batch_target_timings, "target_donor_transferring_epoch")
        base_readies = values(batch_target_timings, "target_held_epoch")
        delta_starts = values(batch_source_timings, "delta_transfer_started_epoch")
        delta_completes = values(batch_source_timings, "delta_transfer_completed_epoch")
        commit_readies = values(batch_target_timings, "target_commit_ready_epoch")
        source_finishes = values(batch_source_timings, "source_finish_migrated_epoch")
        activations = values(batch_target_timings, "target_adopted_epoch")
        base_start = min(base_starts) if base_starts else None
        base_ready = max(base_readies) if base_readies else None
        delta_start = min(delta_starts) if delta_starts else None
        delta_complete = max(delta_completes) if delta_completes else None
        commit_ready = max(commit_readies) if commit_readies else None
        source_finish = max(source_finishes) if source_finishes else None
        activation = max(activations) if activations else None
        batch_point = {
            "base_start": base_start,
            "base_ready": base_ready,
            "delta_start": delta_start,
            "delta_complete": delta_complete,
            "commit_ready": commit_ready,
            "source_finish": source_finish,
            "activation": activation,
            "prefill_restore_critical_ms": max(
                (row["prefill_restore_ms"] or 0) for row in batch_request_rows
            ),
            "prefill_transfer_critical_ms": max(
                (row["prefill_transfer_ms"] or 0) for row in batch_request_rows
            ),
            "source_base_transfer_critical_ms": max(
                (row["source_base_transfer_ms"] or 0) for row in batch_request_rows
            ),
        }
        batch_points[batch] = batch_point
        batch_rows.append(
            {
                "batch": batch,
                "session_id": session_id,
                "request_count": len(batch_request_rows),
                "trace_request_ids": ";".join(str(row["trace_request_id"]) for row in batch_request_rows),
                "prefill_restore_critical_ms": batch_point["prefill_restore_critical_ms"],
                "prefill_transfer_critical_ms": batch_point["prefill_transfer_critical_ms"],
                "source_base_transfer_critical_ms": batch_point["source_base_transfer_critical_ms"],
                "target_base_receive_critical_ms": diff_ms(base_ready, base_start),
                "base_ready_to_delta_start_ms": diff_ms(delta_start, base_ready),
                "delta_transfer_critical_ms": max((row["delta_transfer_ms"] or 0) for row in batch_request_rows),
                "delta_complete_to_commit_ready_ms": diff_ms(commit_ready, delta_complete),
                "commit_ready_to_source_finish_ms": diff_ms(source_finish, commit_ready),
                "source_finish_to_activate_ms": diff_ms(activation, source_finish),
                "batch_base_start_to_activate_ms": diff_ms(activation, base_start),
                "trigger_to_first_base_start_ms": diff_ms(base_start, trigger_epoch) if batch == "first" else None,
                "trigger_to_first_activate_ms": diff_ms(activation, trigger_epoch) if batch == "first" else None,
                "base_start_shanghai": iso_shanghai(base_start),
                "base_ready_shanghai": iso_shanghai(base_ready),
                "delta_start_shanghai": iso_shanghai(delta_start),
                "delta_complete_shanghai": iso_shanghai(delta_complete),
                "commit_ready_shanghai": iso_shanghai(commit_ready),
                "source_finish_shanghai": iso_shanghai(source_finish),
                "activate_shanghai": iso_shanghai(activation),
            }
        )

    first_activation = next(
        (row["activate_shanghai"] for row in batch_rows if row["batch"] == "first"), None
    )
    write_csv(output_dir / "request_stage_timings.csv", request_rows)
    write_csv(output_dir / "batch_stage_timings.csv", batch_rows)

    observation_epoch = None
    if trigger_index + 1 < len(snapshots):
        observation_epoch = float(snapshots[trigger_index + 1]["timestamp"]) + wall_offset
    runner_events = list(load_jsonl(run_dir / "runner_timeline.jsonl"))
    controller_finished_epoch = next(
        (
            float(row["epoch_ns"]) / 1_000_000_000.0
            for row in runner_events
            if row.get("event") == "controller_finished"
        ),
        None,
    )
    full_timeline_rows = build_full_timeline_rows(
        crossing_epoch=crossing_first_token_mono + wall_offset,
        trigger_epoch=trigger_epoch,
        first=batch_points.get("first"),
        observation_epoch=observation_epoch,
        final=batch_points.get("final"),
        controller_actions=list(controller.get("actions") or []),
        controller_finished_epoch=controller_finished_epoch,
    )
    write_csv(output_dir / "full_timeline.csv", full_timeline_rows)

    source_files = [
        "controller/controller.log",
        "metrics/events.jsonl",
        "workload/trace_slo_ledger.jsonl",
        "workload/state_machine/request_metrics.jsonl",
        "report/controller_actions.csv",
        "runner_timeline.jsonl",
    ]
    manifest = {
        "run_id": run_dir.name,
        "generated_at_shanghai": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
        "controller_success": bool(controller.get("success")),
        "controller_message": controller.get("message"),
        "request_count": len(request_metrics),
        "request_error_count": sum(1 for row in request_metrics if row.get("error")),
        "migrated_request_count": len(request_rows),
        "migration_batch_count": len(batch_rows),
        "all_target_prefix_matches_skipped": all(
            row.get("target_prefix_match_skipped") is True for row in request_rows
        ),
        "fallback_action_count": sum(
            1 for action in controller.get("actions") or [] if "fallback" in str(action.get("step", ""))
        ),
        "failed_controller_action_count": sum(
            1 for action in controller.get("actions") or [] if not action.get("success")
        ),
        "first_activation_shanghai": first_activation,
        "source_files": {
            relative: {
                "bytes": (run_dir / relative).stat().st_size,
                "sha256": sha256(run_dir / relative),
            }
            for relative in source_files
        },
        "derived_files": [
            "slo_trigger.json",
            "slo_trigger_contributors.csv",
            "request_stage_timings.csv",
            "batch_stage_timings.csv",
            "full_timeline.csv",
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
