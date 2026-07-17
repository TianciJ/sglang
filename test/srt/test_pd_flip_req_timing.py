import json
import tempfile
import unittest
from pathlib import Path


class ReqTimingParserTest(unittest.TestCase):
    def test_normalizes_timestamped_docker_log_file_to_rows_and_events(self):
        from scripts.playground.disaggregation.pd_flip_req_timing import (
            normalize_log_file,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "node0.docker.log"
            path.write_text(
                "2026-07-17T02:00:00.123456789Z ReqTimeStats("
                "rid=req-7, bootstrap_room=701, input_len=2000, "
                "cached_input_len=32, output_len=1, type=Prefill): "
                "bootstrap_duration=10.00ms, queue_duration=20.00ms, "
                "prefill_compute_duration=30.00ms, "
                "transfer_prepare_duration=4.00ms, transfer_duration=6.00ms, "
                "completion_duration=1.00ms, entry_time=1000.000\n"
                "2026-07-17T02:00:01.000000000Z unrelated\n",
                encoding="utf-8",
            )

            rows, events = normalize_log_file(path, worker="node0")

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(events), 6)
        self.assertEqual(rows[0]["source_file"], str(path))
        self.assertEqual(rows[0]["source_line_number"], 1)
        self.assertAlmostEqual(rows[0]["log_timestamp"], 1784253600.123456)
        self.assertEqual(events[0]["request_id"], "req-7")
        self.assertEqual(events[0]["worker"], "node0")
        self.assertEqual(events[0]["source_line_number"], 1)

    def test_cli_writes_request_and_flat_stage_jsonl(self):
        from scripts.playground.disaggregation.pd_flip_req_timing import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "node2.log"
            rows_path = root / "rows.jsonl"
            events_path = root / "events.jsonl"
            log.write_text(
                "ReqTimeStats(rid=req-9, input_len=10, cached_input_len=0, "
                "output_len=2, type=Decode): bootstrap_duration=1.00ms, "
                "alloc_wait_duration=2.00ms, transfer_duration=3.00ms, "
                "queue_duration=4.00ms, forward_duration=5.00ms, "
                "entry_time=10.0\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "--log",
                    f"node2={log}",
                    "--output",
                    str(rows_path),
                    "--events-output",
                    str(events_path),
                ]
            )

            rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
            events = [
                json.loads(line) for line in events_path.read_text().splitlines()
            ]

        self.assertEqual(exit_code, 0)
        self.assertEqual(rows[0]["worker"], "node2")
        self.assertEqual(len(events), 5)
        self.assertEqual(events[-1]["stage"], "decode.forward")

    def test_parses_detailed_prefill_stages_into_absolute_events(self):
        from scripts.playground.disaggregation.pd_flip_req_timing import (
            parse_req_time_stats_line,
        )

        line = (
            "2026-07-17T02:00:00.000000000Z INFO ReqTimeStats("
            "rid=req-7, bootstrap_room=701, input_len=2000, cached_input_len=32, "
            "output_len=1, type=Prefill): bootstrap_duration=10.00ms, "
            "queue_duration=20.00ms, prefill_compute_duration=30.00ms, "
            "transfer_prepare_duration=4.00ms, transfer_duration=6.00ms, "
            "completion_duration=1.00ms, entry_time=1000.000, "
            "transfer_speed=12.50 GB/s, transfer_total=75.00 MB, #retries=0"
        )

        row = parse_req_time_stats_line(line, worker="node0")

        self.assertEqual(row["request_id"], "req-7")
        self.assertEqual(row["bootstrap_room"], 701)
        self.assertEqual(row["role"], "prefill")
        self.assertEqual(row["prompt_tokens"], 2000)
        self.assertEqual(row["cached_tokens"], 32)
        self.assertEqual(row["transfer_total_bytes"], 75 * 1024 * 1024)
        self.assertEqual(row["transfer_speed_gb_s"], 12.5)
        self.assertEqual(
            [event["stage"] for event in row["events"]],
            [
                "prefill.bootstrap",
                "prefill.queue",
                "prefill.compute",
                "prefill.transfer_prepare",
                "prefill.transfer",
                "prefill.completion",
            ],
        )
        self.assertAlmostEqual(row["events"][0]["started_at"], 1000.0)
        self.assertAlmostEqual(row["events"][-1]["finished_at"], 1000.071)
        self.assertEqual(row["source_line"], line)

    def test_parses_detailed_decode_stages(self):
        from scripts.playground.disaggregation.pd_flip_req_timing import (
            parse_req_time_stats_line,
        )

        line = (
            "ReqTimeStats(rid=req-7, bootstrap_room=701, input_len=2000, "
            "cached_input_len=32, output_len=10000, type=Decode): "
            "bootstrap_duration=5.00ms, alloc_wait_duration=7.00ms, "
            "transfer_duration=11.00ms, queue_duration=13.00ms, "
            "forward_duration=17.00ms, entry_time=1001.000"
        )

        row = parse_req_time_stats_line(line, worker="node2")

        self.assertEqual(row["role"], "decode")
        self.assertEqual(
            [event["stage"] for event in row["events"]],
            [
                "decode.bootstrap",
                "decode.alloc_wait",
                "decode.transfer",
                "decode.queue",
                "decode.forward",
            ],
        )
        self.assertAlmostEqual(row["events"][-1]["finished_at"], 1001.053)

    def test_old_prefill_log_is_preserved_as_a_coarse_measured_interval(self):
        from scripts.playground.disaggregation.pd_flip_req_timing import (
            parse_req_time_stats_line,
        )

        line = (
            "ReqTimeStats(rid=old, input_len=100, cached_input_len=0, "
            "output_len=1, type=Prefill): bootstrap_queue_duration=2.00ms, "
            "queue_duration=3.00ms, forward_duration=5.00ms, "
            "entry_time=50.000, transfer_speed=0.00 GB/s, "
            "transfer_total=0.00 MB, #retries=0"
        )

        row = parse_req_time_stats_line(line, worker="node0")

        self.assertEqual(
            [event["stage"] for event in row["events"]],
            [
                "prefill.bootstrap_queue",
                "prefill.queue",
                "prefill.forward_and_transfer",
            ],
        )
        self.assertEqual(row["timing_detail"], "coarse")
        self.assertIsNone(row["events"][-1].get("inferred_substage"))

    def test_ignores_unrelated_lines(self):
        from scripts.playground.disaggregation.pd_flip_req_timing import (
            parse_req_time_stats_line,
        )

        self.assertIsNone(
            parse_req_time_stats_line("INFO server started", worker="node0")
        )

    def test_server_prefill_log_exposes_compute_and_transfer_subphases(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "python"
            / "sglang"
            / "srt"
            / "observability"
            / "req_time_stats.py"
        ).read_text(encoding="utf-8")

        for field in (
            "prefill_compute_duration=",
            "transfer_prepare_duration=",
            "transfer_duration=",
            "completion_duration=",
        ):
            self.assertIn(field, source)

        duration_method = source.split("def convert_to_duration(self) -> str:", 1)[1]
        prefill_branch = duration_method.split(
            "elif self.disagg_mode == DisaggregationMode.PREFILL:", 1
        )[1].split("elif self.disagg_mode == DisaggregationMode.DECODE:", 1)[0]
        for assignment in (
            "prefill_compute_duration = self.duration_between(",
            "transfer_prepare_duration = self.duration_between(",
            "transfer_duration = self.duration_between(",
            "completion_duration = self.duration_between(",
        ):
            self.assertIn(assignment, prefill_branch)


if __name__ == "__main__":
    unittest.main()
