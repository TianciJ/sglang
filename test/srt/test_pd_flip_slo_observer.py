import tempfile
import unittest
from pathlib import Path


def _record(index, *, event_time, ttft_good, tpot_good, terminal=False):
    return {
        "request_id": f"req-{index:02d}",
        "event_time": event_time,
        "ttft_seconds": 1.0 if ttft_good else 3.0,
        "ttft_slo_seconds": 2.0,
        "good_tpot_intervals": tpot_good,
        "total_tpot_intervals": 10,
        "status": "completed" if terminal else "running",
    }


class SLOObserverTest(unittest.TestCase):
    def test_requires_minimum_samples_and_uses_hysteresis(self):
        from scripts.playground.disaggregation.pd_flip_slo_observer import (
            evaluate_slo_window,
        )

        insufficient = evaluate_slo_window(
            [_record(i, event_time=100 + i, ttft_good=i < 8, tpot_good=10) for i in range(9)],
            now=109,
            window_seconds=10,
            enter_threshold=0.90,
            recover_threshold=0.95,
            min_ttft_samples=10,
            min_tpot_intervals=100,
        )
        self.assertEqual(insufficient.decision, "insufficient_samples")

        entering = evaluate_slo_window(
            [_record(i, event_time=100 + i, ttft_good=i < 8, tpot_good=10) for i in range(10)],
            now=109,
            window_seconds=10,
            enter_threshold=0.90,
            recover_threshold=0.95,
            min_ttft_samples=10,
            min_tpot_intervals=100,
        )
        self.assertEqual(entering.ttft_attainment, 0.8)
        self.assertEqual(entering.tpot_attainment, 1.0)
        self.assertEqual(entering.decision, "enter")
        self.assertEqual(entering.trigger_request_id, "req-09")
        self.assertEqual(entering.threshold_crossing_time, 109)

        recovered = evaluate_slo_window(
            [_record(i, event_time=200 + i, ttft_good=True, tpot_good=10) for i in range(20)],
            now=219,
            window_seconds=30,
            enter_threshold=0.90,
            recover_threshold=0.95,
            min_ttft_samples=10,
            min_tpot_intervals=100,
        )
        self.assertEqual(recovered.decision, "recover")

    def test_window_uses_latest_record_per_request(self):
        from scripts.playground.disaggregation.pd_flip_slo_observer import (
            evaluate_slo_window,
        )

        rows = [
            _record(i, event_time=100 + i, ttft_good=True, tpot_good=10)
            for i in range(10)
        ]
        rows.append(_record(0, event_time=110, ttft_good=False, tpot_good=5))
        snapshot = evaluate_slo_window(
            rows,
            now=110,
            window_seconds=20,
            enter_threshold=0.95,
            recover_threshold=0.99,
            min_ttft_samples=10,
            min_tpot_intervals=100,
        )

        self.assertEqual(snapshot.ttft_total, 10)
        self.assertEqual(snapshot.tpot_total, 100)
        self.assertEqual(snapshot.ttft_good, 9)
        self.assertEqual(snapshot.tpot_good, 95)

    def test_observer_writes_snapshots_without_mutation_interfaces(self):
        from scripts.playground.disaggregation.pd_flip_slo_observer import (
            observe_once,
        )

        rows = [
            _record(i, event_time=100 + i, ttft_good=i < 8, tpot_good=10, terminal=True)
            for i in range(10)
        ]
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.jsonl"
            journal = Path(directory) / "observer.jsonl"
            ledger.write_text(
                "".join(__import__("json").dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            snapshot = observe_once(
                ledger_path=ledger,
                journal_path=journal,
                now=109,
                window_seconds=10,
                enter_threshold=0.90,
                recover_threshold=0.95,
                min_ttft_samples=10,
                min_tpot_intervals=100,
            )

            source = Path(
                "scripts/playground/disaggregation/pd_flip_slo_observer.py"
            ).read_text(encoding="utf-8")
            journal_text = journal.read_text(encoding="utf-8")

        self.assertEqual(snapshot.decision, "enter")
        self.assertIn('"decision": "enter"', journal_text)
        for forbidden in ("post_json", "/drain", "/migrate", "runtime_role/switch"):
            self.assertNotIn(forbidden, source)

    def test_controller_exposes_and_uses_separate_recovery_threshold(self):
        source = Path(
            "scripts/playground/disaggregation/pd_flip_controller.py"
        ).read_text(encoding="utf-8")

        self.assertIn("slo_recovery_threshold: float = 0.95", source)
        self.assertIn("--slo-recovery-threshold", source)
        self.assertIn(
            "recover_threshold=self.config.slo_recovery_threshold", source
        )
        self.assertIn('snapshots[-1]["trigger"]', source)


if __name__ == "__main__":
    unittest.main()
