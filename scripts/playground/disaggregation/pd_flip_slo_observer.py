#!/usr/bin/env python3
"""Read-only request-ledger SLO observer for baseline experiment runs."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class SLOSnapshot:
    observed_at: float
    window_seconds: float
    ttft_good: int
    ttft_total: int
    tpot_good: int
    tpot_total: int
    ttft_attainment: float | None
    tpot_attainment: float | None
    decision: str
    trigger_request_id: str | None = None
    threshold_crossing_time: float | None = None
    poll_detection_time: float | None = None
    poll_lag_seconds: float | None = None
    terminal_requests: int = 0


def _load_records(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _latest_in_window(
    records: Iterable[JsonDict], *, now: float, window_seconds: float
) -> list[JsonDict]:
    cutoff = now - max(0.0, window_seconds)
    latest: dict[str, JsonDict] = {}
    for row in records:
        request_id = row.get("request_id")
        event_time = row.get("event_time")
        if request_id is None or not isinstance(event_time, (int, float)):
            continue
        if window_seconds > 0 and event_time < cutoff:
            continue
        previous = latest.get(str(request_id))
        if previous is None or event_time >= previous["event_time"]:
            latest[str(request_id)] = row
    return sorted(latest.values(), key=lambda row: (row["event_time"], row["request_id"]))


def evaluate_slo_window(
    records: Iterable[JsonDict],
    *,
    now: float,
    window_seconds: float,
    enter_threshold: float,
    recover_threshold: float,
    min_ttft_samples: int,
    min_tpot_intervals: int,
) -> SLOSnapshot:
    if not 0 <= enter_threshold <= recover_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= enter <= recover <= 1")
    rows = _latest_in_window(records, now=now, window_seconds=window_seconds)

    ttft_good = ttft_total = tpot_good = tpot_total = terminal_requests = 0
    for row in rows:
        ttft = row.get("ttft_seconds")
        ttft_slo = row.get("ttft_slo_seconds")
        if isinstance(ttft, (int, float)) and isinstance(ttft_slo, (int, float)):
            ttft_total += 1
            ttft_good += int(ttft <= ttft_slo)
        interval_total = row.get("total_tpot_intervals")
        interval_good = row.get("good_tpot_intervals")
        if isinstance(interval_total, int) and interval_total > 0:
            tpot_total += interval_total
            if isinstance(interval_good, int):
                tpot_good += max(0, min(interval_good, interval_total))
        terminal_requests += int(row.get("status") not in (None, "running"))

    ttft_attainment = ttft_good / ttft_total if ttft_total else None
    tpot_attainment = tpot_good / tpot_total if tpot_total else None
    eligible = ttft_total >= min_ttft_samples and tpot_total >= min_tpot_intervals
    decision = "insufficient_samples"
    trigger_request_id = None
    threshold_crossing_time = None
    if eligible:
        if (
            ttft_attainment is not None
            and tpot_attainment is not None
            and ttft_attainment < enter_threshold
            and tpot_attainment >= enter_threshold
        ):
            decision = "enter"
            if rows:
                trigger_request_id = str(rows[-1]["request_id"])
                threshold_crossing_time = float(rows[-1]["event_time"])
        elif (
            ttft_attainment is not None
            and tpot_attainment is not None
            and ttft_attainment >= recover_threshold
            and tpot_attainment >= recover_threshold
        ):
            decision = "recover"
        else:
            decision = "hold"

    return SLOSnapshot(
        observed_at=now,
        window_seconds=window_seconds,
        ttft_good=ttft_good,
        ttft_total=ttft_total,
        tpot_good=tpot_good,
        tpot_total=tpot_total,
        ttft_attainment=ttft_attainment,
        tpot_attainment=tpot_attainment,
        decision=decision,
        trigger_request_id=trigger_request_id,
        threshold_crossing_time=threshold_crossing_time,
        poll_detection_time=now if decision == "enter" else None,
        poll_lag_seconds=(
            max(0.0, now - threshold_crossing_time)
            if threshold_crossing_time is not None
            else None
        ),
        terminal_requests=terminal_requests,
    )


def observe_once(
    *,
    ledger_path: Path,
    journal_path: Path,
    now: float,
    window_seconds: float,
    enter_threshold: float,
    recover_threshold: float,
    min_ttft_samples: int,
    min_tpot_intervals: int,
) -> SLOSnapshot:
    snapshot = evaluate_slo_window(
        _load_records(ledger_path),
        now=now,
        window_seconds=window_seconds,
        enter_threshold=enter_threshold,
        recover_threshold=recover_threshold,
        min_ttft_samples=min_ttft_samples,
        min_tpot_intervals=min_tpot_intervals,
    )
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(snapshot), sort_keys=True) + "\n")
    return snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--enter-threshold", type=float, default=0.90)
    parser.add_argument("--recover-threshold", type=float, default=0.95)
    parser.add_argument("--min-ttft-samples", type=int, default=10)
    parser.add_argument("--min-tpot-intervals", type=int, default=100)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--expected-requests", type=int, default=40)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    first_trigger: SLOSnapshot | None = None
    while True:
        now = time.monotonic()
        snapshot = observe_once(
            ledger_path=args.ledger,
            journal_path=args.journal,
            now=now,
            window_seconds=args.window_seconds,
            enter_threshold=args.enter_threshold,
            recover_threshold=args.recover_threshold,
            min_ttft_samples=args.min_ttft_samples,
            min_tpot_intervals=args.min_tpot_intervals,
        )
        if snapshot.decision == "enter" and first_trigger is None:
            first_trigger = snapshot
        if snapshot.terminal_requests >= args.expected_requests:
            break
        time.sleep(max(0.01, args.poll_interval))

    summary = {
        "final": asdict(snapshot),
        "first_trigger": asdict(first_trigger) if first_trigger is not None else None,
        "observer_only": True,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
