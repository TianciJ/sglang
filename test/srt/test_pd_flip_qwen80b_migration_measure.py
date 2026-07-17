import unittest
from pathlib import Path

from test.srt.test_pd_flip_timeline_measurements import (
    load_measure_module,
    load_scheduler_method,
)


class Qwen80BMigrationMeasurementTest(unittest.TestCase):
    def test_layout_manifest_includes_runtime_auxiliary_state_types(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "python"
            / "sglang"
            / "srt"
            / "managers"
            / "scheduler.py"
        ).read_text(encoding="utf-8")

        self.assertIn("def _pd_flip_state_type_names(self)", source)
        self.assertIn('"state_types": self._pd_flip_state_type_names()', source)

    def test_scheduler_reports_combined_kv_and_mamba_transfer_contract(self):
        measure = load_scheduler_method("_pd_flip_migration_request_measurements")
        session = {
            "session_id": "qwen-session",
            "role": "source",
            "source_entries": {
                "req-1": {
                    "manifest": {
                        "origin_input_ids": [1, 2, 3],
                        "kv_committed_len": 64,
                        "source_decode_start": 0,
                        "page_size": 64,
                        "state_types": ["mamba"],
                    },
                    "committed_len": 80,
                    "source_transfer_bytes": 4096,
                    "delta_transfer_bytes": 1024,
                }
            },
        }

        row = measure(session)[0]

        self.assertEqual(row["state_types"], ["mamba"])
        self.assertIs(row["includes_mamba_state"], True)
        self.assertEqual(row["combined_transfer_bytes"], 5120)
        self.assertIsNone(row["kv_component_bytes"])
        self.assertIsNone(row["mamba_component_bytes"])
        self.assertIs(row["byte_breakdown_available"], False)
        self.assertEqual(row["provenance_mode"], "source_decode_full_state")

    def test_flattener_preserves_hybrid_state_fields(self):
        module = load_measure_module()
        events = [
            {
                "event_type": "migration_status",
                "node": "node2",
                "status": {
                    "session_id": "qwen-session",
                    "request_measurements": [
                        {
                            "rid": "req-1",
                            "state_types": ["mamba"],
                            "includes_mamba_state": True,
                            "combined_transfer_bytes": 5120,
                            "kv_component_bytes": None,
                            "mamba_component_bytes": None,
                            "byte_breakdown_available": False,
                        }
                    ],
                },
            }
        ]

        row = module.flatten_migration_request_samples(events)[0]

        self.assertEqual(row["state_types"], ["mamba"])
        self.assertIs(row["includes_mamba_state"], True)
        self.assertEqual(row["combined_transfer_bytes"], 5120)
        for field in (
            "state_types",
            "includes_mamba_state",
            "combined_transfer_bytes",
            "kv_component_bytes",
            "mamba_component_bytes",
            "byte_breakdown_available",
        ):
            self.assertIn(field, module.migration_request_fields())


if __name__ == "__main__":
    unittest.main()
