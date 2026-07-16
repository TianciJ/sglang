import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "superpowers"
    / "reports"
    / "scripts"
    / "analyze_pd_flip_prefill_donor_timing.py"
)
SPEC = importlib.util.spec_from_file_location("pd_flip_timing_analysis", SCRIPT_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)


class TestSessionFiltering(unittest.TestCase):
    def test_only_sessions_owned_by_current_run_are_accepted(self):
        run_id = "20260715T114406Z-full-timeline"

        self.assertTrue(
            ANALYSIS.session_belongs_to_run(
                "20260715T114406Z-full-timeline-first", run_id
            )
        )
        self.assertTrue(
            ANALYSIS.session_belongs_to_run(
                "20260715T114406Z-full-timeline-final", run_id
            )
        )
        self.assertFalse(
            ANALYSIS.session_belongs_to_run(
                "20260715T102321Z-previous-final", run_id
            )
        )
        self.assertFalse(ANALYSIS.session_belongs_to_run(None, run_id))


class TestFullTimeline(unittest.TestCase):
    def test_includes_observation_and_both_migration_batches(self):
        points = {
            "base_start": 10.0,
            "base_ready": 10.1,
            "delta_start": 11.1,
            "delta_complete": 11.2,
            "commit_ready": 11.4,
            "source_finish": 11.41,
            "activation": 11.42,
            "prefill_restore_critical_ms": 1.0,
            "prefill_transfer_critical_ms": 2.0,
            "source_base_transfer_critical_ms": 3.0,
        }
        final = dict(points)
        final.update(
            {
                "base_start": 21.5,
                "base_ready": 21.6,
                "delta_start": 22.6,
                "delta_complete": 22.7,
                "commit_ready": 22.9,
                "source_finish": 22.91,
                "activation": 22.92,
            }
        )

        rows = ANALYSIS.build_full_timeline_rows(
            crossing_epoch=9.5,
            trigger_epoch=9.8,
            first=points,
            observation_epoch=21.42,
            final=final,
            controller_actions=[
                {"step": "post_migration_idle_assertion", "elapsed_seconds": 0.5}
            ],
            controller_finished_epoch=23.5,
        )
        by_stage = {row["stage"]: row for row in rows}

        self.assertEqual(by_stage["observation_window"]["duration_ms"], 10000.0)
        self.assertEqual(by_stage["first_base_receive_window"]["duration_ms"], 100.0)
        self.assertEqual(by_stage["final_base_receive_window"]["duration_ms"], 100.0)
        self.assertEqual(
            by_stage["post_migration_idle_assertion"]["duration_ms"], 500.0
        )


if __name__ == "__main__":
    unittest.main()
