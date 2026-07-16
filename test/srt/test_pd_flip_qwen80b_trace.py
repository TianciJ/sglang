import hashlib
import json
import tempfile
import unittest
from pathlib import Path


class Qwen80BTraceTest(unittest.TestCase):
    def _build(self):
        from scripts.playground.disaggregation.pd_flip_qwen80b_trace import (
            build_qwen80b_trace,
        )

        return build_qwen80b_trace(
            run_nonce="run-20260717",
            model="Qwen3-Next-80B-A3B-Instruct",
            forced_token_id=12345,
            forced_text="测",
            custom_logit_processor="serialized-forced-token-processor",
        )

    def test_builds_the_agreed_40_request_population(self):
        rows = self._build()

        self.assertEqual(len(rows), 40)
        self.assertEqual(len({row["request_id"] for row in rows}), 40)
        self.assertEqual(
            [row["prompt_kind"] for row in rows], ["long", "short"] * 20
        )
        self.assertEqual(
            sum(row["prompt_kind"] == "short" for row in rows), 20
        )
        self.assertEqual(sum(row["prompt_kind"] == "long" for row in rows), 20)
        for row in rows:
            expected_chars = 1_000 if row["prompt_kind"] == "short" else 10_000
            self.assertEqual(row["prompt_chars"], expected_chars)
            self.assertEqual(len(row["user_content"]), expected_chars)

    def test_nonce_is_the_first_user_content_and_prevents_shared_user_prefixes(self):
        rows = self._build()
        contents = [row["body"]["messages"][0]["content"] for row in rows]

        for index, content in enumerate(contents):
            nonce = f"{chr(0x4E00 + index)}|run-20260717:req-{index:02d}:"
            self.assertTrue(content.startswith(nonce))
            self.assertEqual(rows[index]["user_content"], content)

        for left_index, left in enumerate(contents):
            for right in contents[left_index + 1 :]:
                self.assertNotEqual(left[0], right[0])

    def test_applies_output_and_slo_contract(self):
        rows = self._build()

        for row in rows:
            body = row["body"]
            expected_ttft = 2.0 if row["prompt_kind"] == "short" else 5.0
            self.assertEqual(row["ttft_slo_s"], expected_ttft)
            self.assertEqual(row["tpot_slo_s"], 0.05)
            self.assertEqual(body["model"], "Qwen3-Next-80B-A3B-Instruct")
            self.assertEqual(body["max_tokens"], 10_000)
            self.assertIs(body["stream"], True)
            self.assertIs(body["ignore_eos"], True)
            self.assertIsNone(body["stop"])
            self.assertEqual(body["stream_options"], {"include_usage": True})
            self.assertEqual(
                body["custom_logit_processor"],
                "serialized-forced-token-processor",
            )
            self.assertEqual(
                body["custom_params"]["forced_token_id"], 12345
            )
            self.assertEqual(body["custom_params"]["forced_text"], "测")
            self.assertEqual(
                body["custom_params"]["pd_flip_slo"],
                {"ttft_seconds": expected_ttft, "tpot_seconds": 0.05},
            )

    def test_uses_four_ten_request_waves(self):
        rows = self._build()
        offsets = [row["arrival_offset_s"] for row in rows]

        expected = [
            wave_start + within_wave * 0.5
            for wave_start in (0.0, 7.5, 15.0, 22.5)
            for within_wave in range(10)
        ]
        self.assertEqual(offsets, expected)
        self.assertEqual(offsets[-1], 27.0)

    def test_write_trace_is_canonical_and_hashes_the_effective_jsonl(self):
        from scripts.playground.disaggregation.pd_flip_qwen80b_trace import (
            write_trace,
        )

        rows = self._build()
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.jsonl"
            manifest_path = Path(directory) / "manifest.json"
            manifest = write_trace(rows, trace_path, manifest_path)

            expected_hash = hashlib.sha256(trace_path.read_bytes()).hexdigest()
            reloaded = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]
            persisted_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )

        self.assertEqual(reloaded, rows)
        self.assertEqual(manifest["trace_sha256"], expected_hash)
        self.assertEqual(persisted_manifest, manifest)
        self.assertEqual(manifest["request_count"], 40)
        self.assertEqual(manifest["last_arrival_offset_s"], 27.0)


if __name__ == "__main__":
    unittest.main()
