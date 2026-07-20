import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch


class PDFlipTraceReplayTest(unittest.TestCase):
    def test_natural_output_tpot_uses_usage_token_count_not_sse_event_count(self):
        from scripts.playground.disaggregation.pd_flip_trace_replay import (
            _apply_natural_output_tpot,
        )

        metrics = {
            "avg_tpot_s": 0.45,
            "p50_tpot_s": 0.4,
            "p95_tpot_s": 0.5,
            "max_tpot_s": 0.5,
            "tpot_slo_s": 0.05,
            "ttft_met": True,
        }
        _apply_natural_output_tpot(
            metrics,
            first_output_monotonic=10.0,
            last_output_monotonic=10.9,
            completion_tokens=10,
        )

        self.assertAlmostEqual(metrics["avg_tpot_s"], 0.1)
        self.assertAlmostEqual(metrics["token_normalized_tpot_s"], 0.1)
        self.assertEqual(metrics["avg_stream_event_gap_s"], 0.45)
        self.assertEqual(
            metrics["tpot_metric_source"],
            "client_first_last_output_over_usage_completion_tokens",
        )
        self.assertFalse(metrics["tpot_avg_met"])
        self.assertFalse(metrics["all_met"])

    def test_send_one_request_persists_cache_and_token_timestamps(self):
        from scripts.playground.disaggregation.pd_flip_trace_replay import (
            _send_one_request,
        )

        chunks = [
            {
                "id": "upstream-1",
                "choices": [{"delta": {"content": "测"}, "finish_reason": None}],
            },
            {
                "id": "upstream-1",
                "choices": [
                    {"delta": {"content": "测"}, "finish_reason": "length"}
                ],
            },
            {
                "id": "upstream-1",
                "choices": [],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 4},
                    "cached_tokens_details": {
                        "device": 1,
                        "host": 2,
                        "storage": 1,
                    },
                },
            },
        ]

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                for chunk in chunks:
                    yield f"data: {json.dumps(chunk)}\n".encode()
                    yield b"\n"
                yield b"data: [DONE]\n"

        record = {
            "request_id": "req-1",
            "arrival_offset_s": 0.0,
            "ttft_slo_s": 5.0,
            "tpot_slo_s": 5.0,
            "body": {
                "stream": True,
                "max_tokens": 2,
                "custom_params": {"forced_text": "测"},
            },
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "urllib.request.urlopen", return_value=FakeResponse()
        ):
            result = _send_one_request(
                record,
                "http://router",
                time.monotonic(),
                10.0,
                Path(directory) / "ledger.jsonl",
                threading.Lock(),
                None,
                "baseline",
            )

        response = result["response"]
        self.assertEqual(response["prompt_tokens"], 100)
        self.assertEqual(response["cached_tokens"], 4)
        self.assertEqual(response["cached_tokens_device"], 1)
        self.assertEqual(response["cached_tokens_host"], 2)
        self.assertEqual(response["cached_tokens_storage"], 1)
        self.assertEqual(response["prefix_hit_ratio"], 0.04)
        self.assertIsInstance(response["client_send_wall"], float)
        self.assertIsInstance(response["first_token_wall"], float)
        self.assertIsInstance(response["last_token_wall"], float)
        self.assertEqual(len(result["tpot_intervals"]), 1)
        self.assertIsInstance(
            result["tpot_intervals"][0]["token_received_at_wall"], float
        )

    def test_build_trace_creates_mixed_200_request_workload(self):
        from scripts.playground.disaggregation.pd_flip_trace_replay import build_trace

        trace = build_trace(
            num_requests=200,
            interval_seconds=1.0,
            model="deepseek_v3.1_terminus",
            seed=7,
        )

        self.assertEqual(len(trace), 200)
        self.assertEqual(
            [r["arrival_offset_s"] for r in trace[:4]], [0.0, 1.0, 2.0, 3.0]
        )
        self.assertEqual(trace[-1]["arrival_offset_s"], 199.0)

        kinds = {record["prompt_kind"] for record in trace}
        self.assertEqual(kinds, {"short", "medium", "long"})

        first = trace[0]
        self.assertEqual(first["request_id"], "trace-0000")
        self.assertGreater(first["ttft_slo_s"], 0)
        self.assertGreater(first["tpot_slo_s"], 0)
        body = first["body"]
        self.assertEqual(body["model"], "deepseek_v3.1_terminus")
        self.assertTrue(body["stream"])
        self.assertIn("messages", body)
        self.assertEqual(
            body["custom_params"]["pd_flip_slo"]["ttft_seconds"],
            first["ttft_slo_s"],
        )
        self.assertEqual(
            body["custom_params"]["pd_flip_slo"]["tpot_seconds"],
            first["tpot_slo_s"],
        )

    def test_build_trace_can_generate_non_streaming_requests(self):
        from scripts.playground.disaggregation.pd_flip_trace_replay import build_trace

        trace = build_trace(
            num_requests=3,
            interval_seconds=0.5,
            model="deepseek_v3.1_terminus",
            seed=7,
            stream=False,
        )

        self.assertFalse(trace[0]["stream"])
        self.assertFalse(trace[0]["body"]["stream"])

    def test_build_trace_can_generate_40_request_char_count_mix(self):
        from scripts.playground.disaggregation.pd_flip_trace_replay import build_trace

        trace = build_trace(
            num_requests=40,
            interval_seconds=0.5,
            model="deepseek_v3.1_terminus",
            seed=7,
            short_chars=1000,
            long_chars=10000,
            short_count=20,
            long_count=20,
        )

        self.assertEqual(len(trace), 40)
        self.assertEqual(trace[-1]["arrival_offset_s"], 19.5)
        kinds = [record["prompt_kind"] for record in trace]
        self.assertEqual(kinds.count("short"), 20)
        self.assertEqual(kinds.count("long"), 20)
        self.assertNotIn("medium", kinds)
        for record in trace:
            target = 1000 if record["prompt_kind"] == "short" else 10000
            self.assertLessEqual(abs(record["prompt_chars"] - target), target * 0.02)
            content = record["body"]["messages"][0]["content"]
            self.assertEqual(len(content), record["prompt_chars"])

    def test_trace40_has_one_output_budget_and_distinct_prompts(self):
        from scripts.playground.disaggregation.pd_flip_trace_replay import build_trace

        trace = build_trace(
            num_requests=40,
            interval_seconds=0.5,
            model="deepseek_v3.1_terminus",
            seed=7,
            short_chars=1000,
            long_chars=10000,
            short_count=20,
            long_count=20,
            max_tokens=10000,
            forced_text="字",
            forced_token_id=1234,
        )

        prompts = [row["body"]["messages"][0]["content"] for row in trace]
        first_lines = [prompt.splitlines()[0] for prompt in prompts]
        self.assertEqual(len(set(prompts)), 40)
        self.assertEqual(len(set(first_lines)), 40)
        self.assertTrue(all(row["max_tokens"] == 10000 for row in trace))
        self.assertTrue(all(row["body"]["max_tokens"] == 10000 for row in trace))
        self.assertTrue(all(row["body"]["ignore_eos"] is True for row in trace))
        self.assertTrue(all(row["body"]["stop"] is None for row in trace))
        self.assertTrue(
            all(
                row["body"]["stream_options"] == {"include_usage": True}
                for row in trace
            )
        )
        self.assertTrue(
            all(
                row["body"]["custom_params"]["forced_token_id"] == 1234 for row in trace
            )
        )
        self.assertTrue(
            all(row["body"]["custom_params"]["forced_text"] == "字" for row in trace)
        )

    def test_forced_trace_requires_complete_valid_output_contract(self):
        from scripts.playground.disaggregation.pd_flip_trace_replay import build_trace

        common = {
            "num_requests": 1,
            "interval_seconds": 0.0,
            "model": "deepseek_v3.1_terminus",
            "seed": 7,
        }
        with self.assertRaisesRegex(ValueError, "max_tokens must be positive"):
            build_trace(max_tokens=0, **common)
        with self.assertRaisesRegex(ValueError, "forced_text and forced_token_id"):
            build_trace(forced_text="字", **common)

    def test_generate_cli_accepts_forced_output_contract(self):
        from scripts.playground.disaggregation.pd_flip_trace_replay import build_parser

        args = build_parser().parse_args(
            [
                "generate",
                "--output-dir",
                "/tmp/trace",
                "--model",
                "deepseek_v3.1_terminus",
                "--max-tokens",
                "10000",
                "--forced-text",
                "字",
                "--forced-token-id",
                "1234",
            ]
        )
        self.assertEqual(args.max_tokens, 10000)
        self.assertEqual(args.forced_text, "字")
        self.assertEqual(args.forced_token_id, 1234)

    def test_compact_output_evidence_uses_usage_and_omits_full_text(self):
        from scripts.playground.disaggregation.pd_flip_trace_replay import (
            build_output_evidence,
        )

        evidence = build_output_evidence(
            ["字"] * 10000,
            expected_tokens=10000,
            forced_text="字",
            usage_completion_tokens=10000,
        )

        self.assertEqual(evidence["completion_tokens"], 10000)
        self.assertEqual(evidence["completion_tokens_source"], "usage")
        self.assertTrue(evidence["completion_token_match"])
        self.assertEqual(evidence["forced_text_mismatch_count"], 0)
        self.assertLessEqual(len(evidence["output_first"]), 32)
        self.assertLessEqual(len(evidence["output_last"]), 32)
        self.assertEqual(len(evidence["output_sha256"]), 64)
        self.assertNotIn("content", evidence)
        self.assertNotIn("output_text", evidence)

    def test_compact_output_evidence_detects_wrong_output(self):
        from scripts.playground.disaggregation.pd_flip_trace_replay import (
            build_output_evidence,
        )

        evidence = build_output_evidence(
            ["字", "错", "字"],
            expected_tokens=3,
            forced_text="字",
            usage_completion_tokens=2,
        )

        self.assertFalse(evidence["completion_token_match"])
        self.assertEqual(evidence["forced_text_mismatch_count"], 1)

    def test_extract_non_stream_text_handles_chat_message(self):
        from scripts.playground.disaggregation.pd_flip_trace_replay import (
            _extract_non_stream_text,
        )

        choice = {
            "message": {
                "reasoning_content": "think ",
                "content": "answer",
            }
        }

        self.assertEqual(_extract_non_stream_text(choice), "think answer")

    def test_extract_usage_cache_evidence_records_prefix_hit_breakdown(self):
        from scripts.playground.disaggregation.pd_flip_trace_replay import (
            extract_usage_cache_evidence,
        )

        evidence = extract_usage_cache_evidence(
            {
                "usage": {
                    "prompt_tokens": 2000,
                    "completion_tokens": 10000,
                    "prompt_tokens_details": {"cached_tokens": 32},
                    "cached_tokens_details": {
                        "device": 8,
                        "host": 16,
                        "storage": 8,
                    },
                }
            }
        )

        self.assertEqual(evidence["prompt_tokens"], 2000)
        self.assertEqual(evidence["completion_tokens"], 10000)
        self.assertEqual(evidence["cached_tokens"], 32)
        self.assertEqual(evidence["cached_tokens_device"], 8)
        self.assertEqual(evidence["cached_tokens_host"], 16)
        self.assertEqual(evidence["cached_tokens_storage"], 8)
        self.assertAlmostEqual(evidence["prefix_hit_ratio"], 0.016)

    def test_extract_usage_cache_evidence_accepts_sglang_extension_details(self):
        from scripts.playground.disaggregation.pd_flip_trace_replay import (
            extract_usage_cache_evidence,
        )

        evidence = extract_usage_cache_evidence(
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "prompt_tokens_details": {"cached_tokens": 7},
                },
                "sglang": {
                    "cached_tokens_details": {
                        "device": 2,
                        "host": 3,
                        "storage": 2,
                    }
                },
            }
        )

        self.assertEqual(evidence["cached_tokens"], 7)
        self.assertEqual(evidence["cached_tokens_device"], 2)
        self.assertEqual(evidence["cached_tokens_host"], 3)
        self.assertEqual(evidence["cached_tokens_storage"], 2)

    def test_compute_metrics_reports_ttft_tpot_and_slo_attainment(self):
        from scripts.playground.disaggregation.pd_flip_trace_replay import (
            compute_metrics,
        )

        record = {
            "request_id": "trace-0001",
            "arrival_offset_s": 1.0,
            "ttft_slo_s": 0.50,
            "tpot_slo_s": 0.20,
        }

        metrics = compute_metrics(
            record,
            scheduled_monotonic=10.0,
            start_monotonic=10.0,
            first_token_monotonic=10.4,
            token_monotonic_times=[10.4, 10.55, 10.90],
            end_monotonic=11.0,
            status="completed",
            error=None,
        )

        self.assertAlmostEqual(metrics["ttft_s"], 0.4)
        self.assertAlmostEqual(metrics["avg_tpot_s"], 0.25)
        self.assertAlmostEqual(metrics["p95_tpot_s"], 0.35)
        self.assertEqual(metrics["good_tpot_intervals"], 1)
        self.assertEqual(metrics["total_tpot_intervals"], 2)
        self.assertTrue(metrics["ttft_met"])
        self.assertFalse(metrics["tpot_avg_met"])
        self.assertFalse(metrics["all_met"])


if __name__ == "__main__":
    unittest.main()
