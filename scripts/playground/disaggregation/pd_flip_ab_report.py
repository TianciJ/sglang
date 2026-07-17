#!/usr/bin/env python3
"""Regenerate the Qwen3-Next PD Flip A/B report from preserved raw files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]


def _read_json(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    return [
        value
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for value in [json.loads(line)]
        if isinstance(value, dict)
    ]


def _rate(good: int, total: int) -> float | None:
    return good / total if total else None


def _aggregate(rows: Iterable[JsonDict]) -> JsonDict:
    rows = list(rows)
    completed = [row for row in rows if row.get("status") == "completed"]
    tpot_good = sum(int(row.get("good_tpot_intervals") or 0) for row in rows)
    tpot_total = sum(int(row.get("total_tpot_intervals") or 0) for row in rows)
    return {
        "requests": len(rows),
        "completed_requests": len(completed),
        "ttft_attainment": _rate(
            sum(bool(row.get("ttft_met")) for row in rows), len(rows)
        ),
        "tpot_interval_attainment": _rate(tpot_good, tpot_total),
        "average_tpot_request_attainment": _rate(
            sum(bool(row.get("tpot_avg_met")) for row in rows), len(rows)
        ),
        "joint_attainment": _rate(
            sum(bool(row.get("all_met")) for row in rows), len(rows)
        ),
        "short": _aggregate_kind(rows, "short"),
        "long": _aggregate_kind(rows, "long"),
    }


def _aggregate_kind(rows: list[JsonDict], kind: str) -> JsonDict:
    selected = [row for row in rows if row.get("prompt_kind") == kind]
    return {
        "requests": len(selected),
        "ttft_attainment": _rate(
            sum(bool(row.get("ttft_met")) for row in selected), len(selected)
        ),
        "joint_attainment": _rate(
            sum(bool(row.get("all_met")) for row in selected), len(selected)
        ),
    }


def _write_csv(path: Path, rows: list[JsonDict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _stage_rows(run_dir: Path) -> list[JsonDict]:
    result: list[JsonDict] = []
    for mode in ("baseline", "state_machine"):
        for row in _read_jsonl(
            run_dir / mode / "metrics" / "request_stage_events.jsonl"
        ):
            result.append({"mode": mode, **row})
    return result


def _slo_rows(run_dir: Path, controller: JsonDict) -> list[JsonDict]:
    result: list[JsonDict] = []
    for mode in ("baseline", "state_machine"):
        for row in _read_jsonl(
            run_dir / mode / "observer" / "snapshots.jsonl"
        ):
            result.append(
                {
                    "mode": mode,
                    "source": "observer",
                    "timestamp": row.get("observed_at"),
                    "ttft_attainment": row.get("ttft_attainment"),
                    "tpot_attainment": row.get("tpot_attainment"),
                    "decision": row.get("decision"),
                    "trigger_request_id": row.get("trigger_request_id"),
                    "threshold_crossing_time": row.get("threshold_crossing_time"),
                    "poll_detection_time": row.get("poll_detection_time"),
                    "poll_lag_seconds": row.get("poll_lag_seconds"),
                    "ttft_good": row.get("ttft_good"),
                    "ttft_total": row.get("ttft_total"),
                    "tpot_good": row.get("tpot_good"),
                    "tpot_total": row.get("tpot_total"),
                }
            )
    for row in controller.get("snapshots") or []:
        if not isinstance(row, dict):
            continue
        trigger = row.get("trigger") if isinstance(row.get("trigger"), dict) else {}
        result.append(
            {
                "mode": "state_machine",
                "source": "controller",
                "timestamp": row.get("timestamp"),
                "ttft_attainment": row.get("prefill_slo_attainment"),
                "tpot_attainment": row.get("decode_slo_attainment"),
                "decision": None,
                "trigger_request_id": trigger.get("trigger_request_id"),
                "threshold_crossing_time": trigger.get("threshold_crossing_time"),
                "poll_detection_time": trigger.get("poll_detection_time"),
                "poll_lag_seconds": trigger.get("poll_lag_seconds"),
                "ttft_good": None,
                "ttft_total": None,
                "tpot_good": None,
                "tpot_total": None,
            }
        )
    return sorted(result, key=lambda row: (row.get("timestamp") or 0, row["mode"]))


def _trigger_summary(
    trigger: JsonDict, rows: list[JsonDict], *, source: str
) -> JsonDict:
    request_id = trigger.get("trigger_request_id")
    request = next((row for row in rows if row.get("request_id") == request_id), {})
    return {
        "source": source,
        "request_id": request_id,
        "prompt_kind": request.get("prompt_kind"),
        "arrival_offset_s": request.get("arrival_offset_s"),
        "ttft_s": request.get("ttft_s"),
        "ttft_slo_s": request.get("ttft_slo_s"),
        "threshold_crossing_time": trigger.get("threshold_crossing_time"),
        "poll_detection_time": trigger.get("poll_detection_time"),
        "poll_lag_seconds": trigger.get("poll_lag_seconds"),
    }


def _migration_rows(run_dir: Path) -> list[JsonDict]:
    rows = _read_jsonl(
        run_dir
        / "state_machine"
        / "metrics"
        / "migration"
        / "migration_phase_events.jsonl"
    )
    previous_by_request: dict[str, int] = {}
    result: list[JsonDict] = []
    for row in sorted(
        rows,
        key=lambda item: (
            str(item.get("request_id") or ""),
            int(item.get("mono_ns") or 0),
        ),
    ):
        request_id = str(row.get("request_id") or "")
        mono_ns = row.get("mono_ns")
        previous = previous_by_request.get(request_id)
        duration = None
        if isinstance(mono_ns, int) and previous is not None:
            duration = (mono_ns - previous) / 1_000_000_000
        if isinstance(mono_ns, int):
            previous_by_request[request_id] = mono_ns
        result.append(
            {
                "request_id": request_id,
                "phase": row.get("phase"),
                "worker": row.get("worker"),
                "epoch_ns": row.get("epoch_ns"),
                "mono_ns": mono_ns,
                "duration_from_previous_s": duration,
                "bytes": row.get("bytes"),
                "page_count": row.get("page_count"),
                "logical_start": row.get("logical_start"),
                "logical_end": row.get("logical_end"),
            }
        )
    return result


def _validate(
    baseline_manifest: JsonDict,
    state_manifest: JsonDict,
    baseline_rows: list[JsonDict],
    state_rows: list[JsonDict],
    controller: JsonDict,
) -> list[str]:
    errors: list[str] = []
    for key in (
        "trace_sha256",
        "model_fingerprint",
        "code_hash",
        "image_id",
        "gpu_ids",
        "tp_size",
        "dp_size",
        "slo_window_seconds",
        "slo_enter_threshold",
        "slo_recover_threshold",
        "first_migration_ratio",
        "observation_seconds",
    ):
        if baseline_manifest.get(key) != state_manifest.get(key):
            errors.append(f"{key} mismatch")
    if baseline_manifest.get("initial_topology") != "1P3D" or state_manifest.get(
        "initial_topology"
    ) != "1P3D":
        errors.append("both modes must start at 1P3D")
    if baseline_manifest.get("state_machine_enabled") is not False or baseline_manifest.get(
        "runtime_role_switch_enabled"
    ) is not False:
        errors.append("baseline is not a static SGLang topology")
    if state_manifest.get("state_machine_enabled") is not True or state_manifest.get(
        "runtime_role_switch_enabled"
    ) is not True:
        errors.append("state machine runtime flags are not enabled")
    if state_manifest.get("hicache_stitch_enabled") is not False or state_manifest.get(
        "prefill_donor_enabled"
    ) is not False:
        errors.append("state machine must disable HiCache stitch and Prefill Donor")
    for mode, rows in (("baseline", baseline_rows), ("state_machine", state_rows)):
        if len(rows) != 40:
            errors.append(f"{mode} request count is not 40")
        if any(row.get("status") != "completed" for row in rows):
            errors.append(f"{mode} has non-terminal requests")
        if any(
            row.get("completion_tokens") != 10000
            or row.get("completion_token_match") is not True
            for row in rows
        ):
            errors.append(f"{mode} output contract failed")
    if controller.get("success") is not True:
        errors.append("state machine controller did not succeed")
    if controller.get("first_migration_ratio") != state_manifest.get(
        "first_migration_ratio"
    ):
        errors.append("controller first migration ratio does not match manifest")
    if controller.get("observation_seconds") != state_manifest.get(
        "observation_seconds"
    ):
        errors.append("controller observation duration does not match manifest")
    if controller.get("final_topology") != "2P2D":
        errors.append("final topology is not 2P2D")
    states = {
        row.get("state")
        for row in controller.get("state_trace") or []
        if isinstance(row, dict)
    }
    if not {"first_migrating", "observing", "second_migrating"}.issubset(states):
        errors.append("state machine did not observe both migration batches")
    return errors


def generate_report(run_dir: Path) -> JsonDict:
    baseline_manifest = _read_json(run_dir / "baseline" / "manifest.json")
    state_manifest = _read_json(run_dir / "state_machine" / "manifest.json")
    baseline_rows = _read_jsonl(
        run_dir / "baseline" / "raw" / "request_metrics.jsonl"
    )
    state_rows = _read_jsonl(
        run_dir / "state_machine" / "raw" / "request_metrics.jsonl"
    )
    baseline_observer = _read_json(
        run_dir / "baseline" / "observer" / "summary.json"
    )
    state_observer = _read_json(
        run_dir / "state_machine" / "observer" / "summary.json"
    )
    controller = _read_json(
        run_dir / "state_machine" / "controller" / "result.json"
    )
    errors = _validate(
        baseline_manifest, state_manifest, baseline_rows, state_rows, controller
    )
    baseline = _aggregate(baseline_rows)
    state_machine = _aggregate(state_rows)
    baseline_trigger = _trigger_summary(
        baseline_observer.get("first_trigger") or {},
        baseline_rows,
        source="observer",
    )
    controller_trigger = next(
        (
            row.get("trigger")
            for row in controller.get("snapshots") or []
            if isinstance(row, dict) and isinstance(row.get("trigger"), dict)
        ),
        None,
    )
    state_trigger = _trigger_summary(
        controller_trigger or state_observer.get("first_trigger") or {},
        state_rows,
        source="controller" if controller_trigger else "observer",
    )
    winner = None
    if not errors:
        baseline_joint = baseline.get("joint_attainment") or 0.0
        state_joint = state_machine.get("joint_attainment") or 0.0
        winner = (
            "state_machine"
            if state_joint > baseline_joint
            else "baseline" if baseline_joint > state_joint else "tie"
        )
    summary = {
        "experiment": "Qwen3-Next-80B-A3B-Instruct quick validation",
        "valid": not errors,
        "validity_errors": errors,
        "winner": winner,
        "trigger": state_trigger or baseline_trigger,
        "baseline_trigger": baseline_trigger,
        "state_machine_trigger": state_trigger,
        "baseline": baseline,
        "state_machine": state_machine,
        "controller": controller,
    }

    output = run_dir / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    state_by_id = {row.get("request_id"): row for row in state_rows}
    comparison_rows = []
    for base in baseline_rows:
        state = state_by_id.get(base.get("request_id"), {})
        comparison_rows.append(
            {
                "request_id": base.get("request_id"),
                "prompt_kind": base.get("prompt_kind"),
                "baseline_ttft_s": base.get("ttft_s"),
                "state_machine_ttft_s": state.get("ttft_s"),
                "baseline_avg_tpot_s": base.get("avg_tpot_s"),
                "state_machine_avg_tpot_s": state.get("avg_tpot_s"),
                "baseline_joint_met": base.get("all_met"),
                "state_machine_joint_met": state.get("all_met"),
            }
        )
    _write_csv(
        output / "request_comparison.csv",
        comparison_rows,
        list(comparison_rows[0]) if comparison_rows else ["request_id"],
    )
    _write_csv(
        output / "stage_timings.csv",
        _stage_rows(run_dir),
        [
            "mode",
            "request_id",
            "worker",
            "role",
            "stage",
            "duration_s",
            "started_at",
            "finished_at",
            "measurement_kind",
            "source_file",
            "source_line_number",
        ],
    )
    _write_csv(
        output / "slo_timeseries.csv",
        _slo_rows(run_dir, controller),
        [
            "mode",
            "source",
            "timestamp",
            "ttft_attainment",
            "tpot_attainment",
            "decision",
            "trigger_request_id",
            "threshold_crossing_time",
            "poll_detection_time",
            "poll_lag_seconds",
            "ttft_good",
            "ttft_total",
            "tpot_good",
            "tpot_total",
        ],
    )
    _write_csv(
        output / "migration_timings.csv",
        _migration_rows(run_dir),
        [
            "request_id",
            "phase",
            "worker",
            "epoch_ns",
            "mono_ns",
            "duration_from_previous_s",
            "bytes",
            "page_count",
            "logical_start",
            "logical_end",
        ],
    )

    verdict = (
        f"Provisional higher joint attainment: {winner}."
        if not errors
        else "Run is invalid; no performance winner is reported."
    )
    def percent(value: Any) -> str:
        return f"{float(value) * 100:.2f}%" if value is not None else "n/a"

    trigger_lines = (
        f"- Trigger request: `{state_trigger.get('request_id')}` "
        f"({state_trigger.get('prompt_kind') or 'unknown'} prompt).\n"
        f"- Threshold crossing: `{state_trigger.get('threshold_crossing_time')}`; "
        f"controller poll: `{state_trigger.get('poll_detection_time')}`; "
        f"poll lag: `{state_trigger.get('poll_lag_seconds')}` seconds.\n"
    )
    validity_text = (
        "- none\n" if not errors else "".join(f"- {error}\n" for error in errors)
    )
    (output / "report.md").write_text(
        "# Qwen3-Next 80B PD Flip A/B quick validation\n\n"
        + verdict
        + "\n\n## SLO result\n\n"
        + "| Mode | TTFT attainment | TPOT interval attainment | Joint attainment |\n"
        + "| --- | ---: | ---: | ---: |\n"
        + f"| Baseline 1P3D | {percent(baseline.get('ttft_attainment'))} | "
        + f"{percent(baseline.get('tpot_interval_attainment'))} | "
        + f"{percent(baseline.get('joint_attainment'))} |\n"
        + f"| PD Flip 1P3D to 2P2D | {percent(state_machine.get('ttft_attainment'))} | "
        + f"{percent(state_machine.get('tpot_interval_attainment'))} | "
        + f"{percent(state_machine.get('joint_attainment'))} |\n\n"
        + "## State-machine trigger\n\n"
        + trigger_lines
        + "\n## Validity errors\n\n"
        + validity_text
        + "\n## Raw-backed detail\n\n"
        + "- `request_comparison.csv`: paired client TTFT/TPOT and attainment.\n"
        + "- `stage_timings.csv`: SGLang Prefill/Decode process stages with worker and source log line.\n"
        + "- `slo_timeseries.csv`: observer and controller SLO windows, including trigger evidence.\n"
        + "- `migration_timings.csv`: request migration phase timestamps, deltas, and bytes.\n"
        + "- `timeline.svg`: compact baseline/state-machine topology timeline.\n\n"
        + "This is one baseline run and one state-machine run; it is not statistically significant. "
        + "The state-machine run measures full source-Decode hybrid-state migration, not Prefill Donor or HiCache stitching.\n",
        encoding="utf-8",
    )
    (output / "timeline.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="140">'
        '<rect width="900" height="140" fill="white"/>'
        '<text x="20" y="35" font-family="sans-serif" font-size="20">Qwen3-Next 80B A/B timeline</text>'
        '<text x="20" y="75" font-family="sans-serif" font-size="14">Baseline: static 1P3D</text>'
        '<text x="20" y="105" font-family="sans-serif" font-size="14">State machine: 1P3D → 50% → 2s observe → remaining → 2P2D</text>'
        "</svg>\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    generate_report(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
