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


def _validate(
    baseline_manifest: JsonDict,
    state_manifest: JsonDict,
    baseline_rows: list[JsonDict],
    state_rows: list[JsonDict],
    controller: JsonDict,
) -> list[str]:
    errors: list[str] = []
    for key in ("trace_sha256", "model_fingerprint", "code_hash", "gpu_ids"):
        if baseline_manifest.get(key) != state_manifest.get(key):
            errors.append(f"{key} mismatch")
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
    if controller.get("first_migration_ratio") != 0.5:
        errors.append("first migration ratio is not 0.5")
    if controller.get("observation_seconds") != 3.0:
        errors.append("observation duration is not 3 seconds")
    if controller.get("final_topology") != "2P2D":
        errors.append("final topology is not 2P2D")
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
    observer = _read_json(
        run_dir / "baseline" / "observer" / "summary.json"
    )
    controller = _read_json(
        run_dir / "state_machine" / "controller" / "result.json"
    )
    errors = _validate(
        baseline_manifest, state_manifest, baseline_rows, state_rows, controller
    )
    baseline = _aggregate(baseline_rows)
    state_machine = _aggregate(state_rows)
    trigger_raw = observer.get("first_trigger") or {}
    trigger = {
        "request_id": trigger_raw.get("trigger_request_id"),
        "threshold_crossing_time": trigger_raw.get("threshold_crossing_time"),
    }
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
        "trigger": trigger,
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
    _write_csv(output / "stage_timings.csv", [], ["mode", "request_id", "stage", "duration_s"])
    _write_csv(output / "slo_timeseries.csv", [], ["mode", "timestamp", "ttft_attainment", "tpot_attainment"])
    _write_csv(output / "migration_timings.csv", [], ["request_id", "phase", "duration_s", "bytes"])

    verdict = (
        f"Provisional higher joint attainment: {winner}."
        if not errors
        else "Run is invalid; no performance winner is reported."
    )
    (output / "report.md").write_text(
        "# Qwen3-Next 80B PD Flip A/B quick validation\n\n"
        + verdict
        + "\n\nThis is one baseline run and one state-machine run; it is not statistically significant.\n",
        encoding="utf-8",
    )
    (output / "timeline.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="140">'
        '<rect width="900" height="140" fill="white"/>'
        '<text x="20" y="35" font-family="sans-serif" font-size="20">Qwen3-Next 80B A/B timeline</text>'
        '<text x="20" y="75" font-family="sans-serif" font-size="14">Baseline: static 1P3D</text>'
        '<text x="20" y="105" font-family="sans-serif" font-size="14">State machine: 1P3D → 50% → 3s observe → remaining → 2P2D</text>'
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
