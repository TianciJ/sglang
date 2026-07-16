import ast
import sys
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List


SCHEDULER_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "sglang"
    / "srt"
    / "managers"
    / "scheduler.py"
)
MEASURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "playground"
    / "disaggregation"
    / "pd_flip_migration_measure.py"
)


def load_scheduler_method(name):
    source = SCHEDULER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    scheduler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    function = next(
        node
        for node in scheduler.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Any": Any, "Dict": Dict, "List": List}
    exec(compile(module, str(SCHEDULER_PATH), "exec"), namespace)
    return namespace[name]


def load_measure_module():
    spec = __import__("importlib.util").util.spec_from_file_location(
        "pd_flip_timeline_measure", MEASURE_PATH
    )
    module = __import__("importlib.util").util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TimelineMeasurementTest(unittest.TestCase):
    def test_ranked_migration_phase_event_has_reconstructable_identity(self):
        measure = load_scheduler_method("_pd_flip_migration_request_measurements")
        session = {
            "session_id": "s-rank",
            "role": "target",
            "dp_rank": 3,
            "target_entries": {
                "r0": {
                    "manifest": {
                        "rid": "r0",
                        "origin_input_ids": [1, 2, 3, 4],
                        "kv_committed_len": 6,
                        "source_decode_start": 2,
                        "source_decode_dp_rank": 1,
                        "target_decode_dp_rank": 3,
                        "page_size": 2,
                        "model_fingerprint": "model-layout-sha",
                    },
                    "target_committed_len": 8,
                    "source_index_size": 2,
                    "source_transfer_bytes": 1024,
                    "delta_transfer_bytes": 512,
                    "timing_debug": {
                        "target_receive_completed_mono": 12.5,
                        "target_receive_completed_epoch": 1002.5,
                    },
                }
            },
        }

        row = measure(session)[0]
        event = row["phase_events"][0]

        for field in (
            "request_id",
            "session_id",
            "worker",
            "dp_rank",
            "phase",
            "epoch_ns",
            "mono_ns",
            "logical_start",
            "logical_end",
            "actual_slot_count",
            "bytes",
            "model_fingerprint",
        ):
            self.assertIn(field, event)
        self.assertEqual(event["request_id"], "r0")
        self.assertEqual(event["session_id"], "s-rank")
        self.assertEqual(event["dp_rank"], 3)
        self.assertEqual(event["target_decode_dp_rank"], 3)
        self.assertEqual(event["model_fingerprint"], "model-layout-sha")

    def test_phase_event_flattener_uses_external_worker_node(self):
        module = load_measure_module()
        events = [
            {
                "event_type": "migration_status",
                "node": "decode-b",
                "status": {
                    "session_id": "s0",
                    "request_measurements": [
                        {
                            "rid": "r0",
                            "phase_events": [
                                {
                                    "request_id": "r0",
                                    "session_id": "s0",
                                    "worker": "target",
                                    "dp_rank": 5,
                                    "phase": "target_receive_completed",
                                    "epoch_ns": 10,
                                    "mono_ns": 20,
                                }
                            ],
                        }
                    ],
                },
            }
        ]

        rows = module.flatten_migration_phase_events(events)

        self.assertEqual(rows[0]["worker"], "decode-b")
        self.assertEqual(rows[0]["worker_role"], "target")
        self.assertEqual(rows[0]["dp_rank"], 5)

    def test_worker_collection_preserves_every_dp_rank_status(self):
        module = load_measure_module()

        class Client:
            def get_json(self, url):
                if url.endswith("/pd_flip/migration/status"):
                    return [
                        {"success": True, "status": {"dp_rank": 0, "state": "ready"}},
                        {"success": True, "status": {"dp_rank": 1, "state": "ready"}},
                    ]
                if url.endswith("/pd_flip/migration/prefill-donor/status"):
                    return [
                        {"success": True, "status": {"dp_rank": 0, "state": "idle"}},
                        {"success": True, "status": {"dp_rank": 1, "state": "idle"}},
                    ]
                return {}

        events = module.collect_worker_events(
            Client(), {"name": "decode-b", "worker_url": "http://decode-b"}
        )

        migration = [
            event for event in events if event["event_type"] == "migration_status"
        ]
        donor = [
            event for event in events if event["event_type"] == "prefill_donor_status"
        ]
        self.assertEqual([event["dp_rank"] for event in migration], [0, 1])
        self.assertEqual([event["dp_rank"] for event in donor], [0, 1])

    def test_fallback_summary_preserves_stitch_and_fallback_phases(self):
        module = load_measure_module()
        row = {
            "session_id": "s0",
            "rid": "r0",
            "ts_mono": 2.0,
            "fallback_attempted": True,
            "fallback_reason": "restore failed",
            "stitch_attempt_seconds": 0.03,
            "stitch_failure_detection_seconds": 0.02,
            "fallback_transfer_seconds": 0.05,
            "stitch_failure_to_fallback_complete_seconds": 0.06,
            "failed_stitch_added_cost_seconds": 0.03,
            "timing_measurement_kind": "exact_process",
        }

        summary = module.summarize_fallback([row], [])

        self.assertEqual(summary["stitch_attempt_seconds"], 0.03)
        self.assertEqual(summary["fallback_transfer_seconds"], 0.05)
        self.assertEqual(summary["failed_stitch_added_cost_seconds"], 0.03)
        self.assertIn("stitch_attempt_seconds", module.migration_request_fields())
        self.assertIn("timing_measurement_kind", module.migration_request_fields())

    def test_failed_stitch_and_full_fallback_are_separate_phases(self):
        measure = load_scheduler_method("_pd_flip_migration_request_measurements")
        session = {
            "target_entries": {
                "r0": {
                    "manifest": {
                        "origin_input_ids": [1, 2, 3, 4, 5, 6],
                        "kv_committed_len": 20,
                    },
                    "target_committed_len": 20,
                    "mooncake_hit_len": 6,
                    "stitch_mode": "source_decode_full_fallback",
                    "fallback_attempted": True,
                    "fallback_reason": "migration target HiCache restore failed",
                    "fallback_duration_seconds": 0.05,
                    "timing_debug": {
                        "target_prefix_query_started_mono": 100.0,
                        "target_prefix_query_completed_mono": 100.01,
                        "target_hicache_restore_started_mono": 100.01,
                        "target_hicache_restore_failed_mono": 100.03,
                        "target_fallback_required_mono": 100.03,
                        "target_fallback_prepare_received_mono": 100.04,
                        "target_fallback_receive_completed_mono": 100.09,
                    },
                }
            }
        }

        row = measure(session)[0]

        self.assertAlmostEqual(row["stitch_attempt_seconds"], 0.03)
        self.assertAlmostEqual(row["stitch_failure_detection_seconds"], 0.02)
        self.assertAlmostEqual(row["fallback_transfer_seconds"], 0.05)
        self.assertAlmostEqual(
            row["stitch_failure_to_fallback_complete_seconds"], 0.06
        )
        self.assertAlmostEqual(row["failed_stitch_added_cost_seconds"], 0.03)
        self.assertEqual(row["timing_measurement_kind"], "exact_process")


if __name__ == "__main__":
    unittest.main()
