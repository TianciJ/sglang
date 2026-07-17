import json
import csv
import tempfile
import unittest
from pathlib import Path


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class ABReportTest(unittest.TestCase):
    def _make_run(self, root, *, mismatch=False):
        common = {
            "trace_sha256": "trace-sha",
            "model_fingerprint": "model-sha",
            "code_hash": "code-sha",
            "image_id": "image-sha",
            "gpu_ids": "0,1,2,3",
            "tp_size": 4,
            "dp_size": 1,
            "initial_topology": "1P3D",
            "slo_window_seconds": 10,
            "slo_enter_threshold": 0.9,
            "slo_recover_threshold": 0.95,
            "first_migration_ratio": 0.5,
            "observation_seconds": 2.0,
            "hicache_stitch_enabled": False,
            "prefill_donor_enabled": False,
        }
        for mode in ("baseline", "state_machine"):
            manifest = dict(common)
            manifest["state_machine_enabled"] = mode == "state_machine"
            manifest["runtime_role_switch_enabled"] = mode == "state_machine"
            if mismatch and mode == "state_machine":
                manifest["trace_sha256"] = "different"
            path = root / mode / "manifest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest), encoding="utf-8")
            rows = []
            for index in range(40):
                rows.append(
                    {
                        "request_id": f"req-{index:02d}",
                        "prompt_kind": "long" if index % 2 == 0 else "short",
                        "status": "completed",
                        "completion_tokens": 10000,
                        "completion_token_match": True,
                        "ttft_met": index < (36 if mode == "baseline" else 38),
                        "tpot_avg_met": index < (35 if mode == "baseline" else 37),
                        "all_met": index < (34 if mode == "baseline" else 36),
                        "good_tpot_intervals": 9000 + index,
                        "total_tpot_intervals": 9999,
                        "ttft_s": 4.0 if index % 2 == 0 else 1.5,
                        "avg_tpot_s": 0.04,
                    }
                )
            _write_jsonl(root / mode / "raw" / "request_metrics.jsonl", rows)
            _write_jsonl(
                root / mode / "metrics" / "request_stage_events.jsonl",
                [
                    {
                        "request_id": "req-00",
                        "worker": "node0",
                        "role": "prefill",
                        "stage": "prefill.compute",
                        "duration_s": 0.2 if mode == "baseline" else 0.18,
                        "started_at": 100.0,
                        "finished_at": 100.2,
                    }
                ],
            )
        (root / "baseline" / "observer").mkdir(parents=True)
        (root / "baseline" / "observer" / "summary.json").write_text(
            json.dumps(
                {
                    "first_trigger": {
                        "trigger_request_id": "req-09",
                        "threshold_crossing_time": 109.0,
                    }
                }
            ),
            encoding="utf-8",
        )
        _write_jsonl(
            root / "baseline" / "observer" / "snapshots.jsonl",
            [
                {
                    "observed_at": 109.25,
                    "ttft_attainment": 0.8,
                    "tpot_attainment": 0.99,
                    "decision": "enter",
                    "ttft_good": 8,
                    "ttft_total": 10,
                    "tpot_good": 990,
                    "tpot_total": 1000,
                }
            ],
        )
        (root / "state_machine" / "observer").mkdir(parents=True)
        (root / "state_machine" / "observer" / "summary.json").write_text(
            json.dumps(
                {
                    "first_trigger": {
                        "trigger_request_id": "req-11",
                        "threshold_crossing_time": 111.0,
                        "poll_detection_time": 111.25,
                        "poll_lag_seconds": 0.25,
                    }
                }
            ),
            encoding="utf-8",
        )
        _write_jsonl(
            root / "state_machine" / "observer" / "snapshots.jsonl",
            [
                {
                    "observed_at": 111.25,
                    "ttft_attainment": 0.8,
                    "tpot_attainment": 0.99,
                    "decision": "enter",
                    "trigger_request_id": "req-11",
                    "threshold_crossing_time": 111.0,
                    "poll_detection_time": 111.25,
                    "poll_lag_seconds": 0.25,
                    "ttft_good": 8,
                    "ttft_total": 10,
                    "tpot_good": 990,
                    "tpot_total": 1000,
                }
            ],
        )
        (root / "state_machine" / "controller").mkdir(parents=True)
        (root / "state_machine" / "controller" / "result.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "first_migration_ratio": 0.5,
                    "observation_seconds": 2.0,
                    "final_topology": "2P2D",
                    "state_trace": [
                        {"state": "first_migrating"},
                        {"state": "observing"},
                        {"state": "second_migrating"},
                        {"state": "flipping_role"},
                    ],
                    "snapshots": [
                        {
                            "timestamp": 110.0,
                            "prefill_slo_attainment": 0.85,
                            "decode_slo_attainment": 0.98,
                            "trigger": {
                                "trigger_request_id": "req-11",
                                "threshold_crossing_time": 110.75,
                                "poll_detection_time": 111.0,
                                "poll_lag_seconds": 0.25,
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        _write_jsonl(
            root
            / "state_machine"
            / "metrics"
            / "migration"
            / "migration_phase_events.jsonl",
            [
                {
                    "request_id": "req-00",
                    "phase": "base_transfer_started",
                    "mono_ns": 1_000_000_000,
                    "epoch_ns": 10_000_000_000,
                    "bytes": 4096,
                    "worker": "node2",
                },
                {
                    "request_id": "req-00",
                    "phase": "base_transfer_ready",
                    "mono_ns": 1_250_000_000,
                    "epoch_ns": 10_250_000_000,
                    "bytes": 4096,
                    "worker": "node3",
                },
            ],
        )

    def test_generates_raw_backed_comparison_artifacts(self):
        from scripts.playground.disaggregation.pd_flip_ab_report import (
            generate_report,
        )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._make_run(run_dir)
            summary = generate_report(run_dir)

            comparison = run_dir / "comparison"
            for name in (
                "summary.json",
                "request_comparison.csv",
                "stage_timings.csv",
                "slo_timeseries.csv",
                "migration_timings.csv",
                "report.md",
                "timeline.svg",
            ):
                self.assertTrue((comparison / name).exists(), name)
            report_text = (comparison / "report.md").read_text(encoding="utf-8")
            with (comparison / "stage_timings.csv").open(encoding="utf-8") as f:
                stage_rows = list(csv.DictReader(f))
            with (comparison / "slo_timeseries.csv").open(encoding="utf-8") as f:
                slo_rows = list(csv.DictReader(f))
            with (comparison / "migration_timings.csv").open(
                encoding="utf-8"
            ) as f:
                migration_rows = list(csv.DictReader(f))

        self.assertTrue(summary["valid"])
        self.assertEqual(summary["baseline_trigger"]["request_id"], "req-09")
        self.assertEqual(summary["state_machine_trigger"]["request_id"], "req-11")
        self.assertEqual(summary["state_machine_trigger"]["prompt_kind"], "short")
        self.assertEqual(summary["state_machine_trigger"]["poll_lag_seconds"], 0.25)
        self.assertEqual(summary["state_machine_trigger"]["source"], "controller")
        self.assertEqual(summary["baseline"]["ttft_attainment"], 0.9)
        self.assertEqual(summary["state_machine"]["joint_attainment"], 0.9)
        self.assertIn("quick validation", report_text.lower())
        self.assertIn("Qwen3-Next", report_text)
        self.assertIn("Trigger request", report_text)
        self.assertIn("stage_timings.csv", report_text)
        self.assertEqual({row["mode"] for row in stage_rows}, {"baseline", "state_machine"})
        self.assertEqual({row["mode"] for row in slo_rows}, {"baseline", "state_machine"})
        controller_slo = next(row for row in slo_rows if row["source"] == "controller")
        self.assertEqual(controller_slo["trigger_request_id"], "req-11")
        self.assertEqual(migration_rows[-1]["phase"], "base_transfer_ready")
        self.assertEqual(float(migration_rows[-1]["duration_from_previous_s"]), 0.25)

    def test_invalid_pair_does_not_claim_a_winner(self):
        from scripts.playground.disaggregation.pd_flip_ab_report import (
            generate_report,
        )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._make_run(run_dir, mismatch=True)
            summary = generate_report(run_dir)

        self.assertFalse(summary["valid"])
        self.assertIsNone(summary["winner"])
        self.assertIn("trace_sha256 mismatch", summary["validity_errors"])

    def test_invalid_pair_rejects_runtime_contract_and_missing_second_batch(self):
        from scripts.playground.disaggregation.pd_flip_ab_report import (
            generate_report,
        )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            self._make_run(run_dir)
            manifest_path = run_dir / "state_machine" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["image_id"] = "different-image"
            manifest["hicache_stitch_enabled"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result_path = run_dir / "state_machine" / "controller" / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["state_trace"] = [{"state": "first_migrating"}]
            result_path.write_text(json.dumps(result), encoding="utf-8")

            summary = generate_report(run_dir)

        self.assertFalse(summary["valid"])
        self.assertIn("image_id mismatch", summary["validity_errors"])
        self.assertIn(
            "state machine must disable HiCache stitch and Prefill Donor",
            summary["validity_errors"],
        )
        self.assertIn(
            "state machine did not observe both migration batches",
            summary["validity_errors"],
        )


if __name__ == "__main__":
    unittest.main()
