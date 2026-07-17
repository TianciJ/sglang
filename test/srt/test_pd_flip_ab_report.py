import json
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
            "gpu_ids": "0,1,2,3",
        }
        for mode in ("baseline", "state_machine"):
            manifest = dict(common)
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
        (root / "state_machine" / "controller").mkdir(parents=True)
        (root / "state_machine" / "controller" / "result.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "first_migration_ratio": 0.5,
                    "observation_seconds": 3.0,
                    "final_topology": "2P2D",
                }
            ),
            encoding="utf-8",
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

        self.assertTrue(summary["valid"])
        self.assertEqual(summary["trigger"]["request_id"], "req-09")
        self.assertEqual(summary["baseline"]["ttft_attainment"], 0.9)
        self.assertEqual(summary["state_machine"]["joint_attainment"], 0.9)
        self.assertIn("quick validation", report_text.lower())
        self.assertIn("Qwen3-Next", report_text)

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


if __name__ == "__main__":
    unittest.main()
