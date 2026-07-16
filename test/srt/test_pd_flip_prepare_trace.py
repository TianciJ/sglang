import json
import tempfile
import unittest
from pathlib import Path


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("special tokens must be disabled")
        return [2024] if text == "字" else [1, 2]

    def decode(self, token_ids):
        return "字" if token_ids == [2024] else "wrong"


class PDFlipPrepareTraceTest(unittest.TestCase):
    @staticmethod
    def _source_rows():
        rows = []
        for index in range(40):
            prompt_kind = "long" if index % 2 == 0 else "short"
            prompt_chars = 10000 if prompt_kind == "long" else 1000
            prefix = f"request-{index:04d}-"
            content = prefix + ("x" * (prompt_chars - len(prefix)))
            rows.append(
                {
                    "request_id": f"trace-{index:04d}",
                    "prompt_kind": prompt_kind,
                    "prompt_chars": prompt_chars,
                    "ttft_slo_s": 1.0,
                    "tpot_slo_s": 0.1,
                    "max_tokens": 1,
                    "body": {
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens": 1,
                        "custom_params": {},
                    },
                }
            )
        return rows

    def test_resolve_forced_token_requires_one_round_trip_token(self):
        from scripts.playground.disaggregation.pd_flip_prepare_trace import (
            resolve_forced_token,
        )

        self.assertEqual(resolve_forced_token(FakeTokenizer(), "字"), 2024)
        with self.assertRaisesRegex(ValueError, "exactly one token"):
            resolve_forced_token(FakeTokenizer(), "两个")

    def test_apply_output_contract_updates_budget_and_processor(self):
        from scripts.playground.disaggregation.pd_flip_prepare_trace import (
            apply_output_contract,
        )

        row = {
            "max_tokens": 1,
            "body": {"max_tokens": 1, "custom_params": {}},
        }

        apply_output_contract(
            row,
            max_tokens=10000,
            forced_text="字",
            forced_token_id=2024,
            custom_logit_processor="serialized-processor",
        )

        self.assertEqual(row["max_tokens"], 10000)
        self.assertEqual(row["body"]["max_tokens"], 10000)
        self.assertEqual(row["body"]["temperature"], 0.0)
        self.assertTrue(row["body"]["ignore_eos"])
        self.assertIsNone(row["body"]["stop"])
        self.assertEqual(row["body"]["custom_logit_processor"], "serialized-processor")
        self.assertEqual(row["body"]["custom_params"]["forced_token_id"], 2024)
        self.assertEqual(row["body"]["custom_params"]["forced_text"], "字")

    def test_apply_output_contract_rejects_incomplete_values(self):
        from scripts.playground.disaggregation.pd_flip_prepare_trace import (
            apply_output_contract,
        )

        row = {"body": {}}
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            apply_output_contract(
                row,
                max_tokens=0,
                forced_text="字",
                forced_token_id=2024,
                custom_logit_processor="serialized-processor",
            )
        with self.assertRaisesRegex(ValueError, "forced_text"):
            apply_output_contract(
                row,
                max_tokens=10000,
                forced_text="",
                forced_token_id=2024,
                custom_logit_processor="serialized-processor",
            )
        with self.assertRaisesRegex(ValueError, "custom_logit_processor"):
            apply_output_contract(
                row,
                max_tokens=10000,
                forced_text="字",
                forced_token_id=2024,
                custom_logit_processor="",
            )

    def test_validate_trace_rejects_duplicate_request_identity_or_prompt(self):
        from scripts.playground.disaggregation.pd_flip_prepare_trace import (
            _validate_trace,
        )

        rows = self._source_rows()
        rows[1]["request_id"] = rows[0]["request_id"]
        with self.assertRaisesRegex(ValueError, "request_id"):
            _validate_trace(rows)

        rows = self._source_rows()
        rows[1]["body"]["messages"][0]["content"] = rows[0]["body"]["messages"][0][
            "content"
        ]
        with self.assertRaisesRegex(ValueError, "Prompt"):
            _validate_trace(rows)

    def test_prepare_trace_applies_forced_output_contract_to_all_rows(self):
        from scripts.playground.disaggregation.pd_flip_prepare_trace import (
            prepare_trace,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.jsonl"
            output = root / "effective.jsonl"
            manifest = root / "manifest.json"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in self._source_rows()),
                encoding="utf-8",
            )

            prepare_trace(
                source=source,
                output=output,
                manifest=manifest,
                wave_size=10,
                wave_gap_seconds=6.0,
                intra_wave_interval_seconds=0.1,
                max_tokens=10000,
                forced_text="字",
                forced_token_id=2024,
                custom_logit_processor="serialized-processor",
            )

            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            schedule = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual(len(rows), 40)
        self.assertTrue(
            all(row["max_tokens"] == row["body"]["max_tokens"] == 10000 for row in rows)
        )
        self.assertTrue(
            all(row["body"]["custom_params"]["forced_token_id"] == 2024 for row in rows)
        )
        self.assertTrue(
            all(
                row["body"]["custom_logit_processor"] == "serialized-processor"
                for row in rows
            )
        )
        self.assertEqual(schedule["max_tokens"], 10000)
        self.assertEqual(schedule["forced_text"], "字")
        self.assertEqual(schedule["forced_token_id"], 2024)

    def test_cli_requires_deepseek_output_contract_inputs(self):
        from scripts.playground.disaggregation.pd_flip_prepare_trace import (
            build_parser,
        )

        args = build_parser().parse_args(
            [
                "--source",
                "source.jsonl",
                "--output",
                "effective.jsonl",
                "--manifest",
                "manifest.json",
                "--wave-size",
                "10",
                "--wave-gap-seconds",
                "6",
                "--intra-wave-interval-seconds",
                "0.1",
                "--max-tokens",
                "10000",
                "--forced-text",
                "字",
                "--tokenizer-path",
                "/models/deepseek_v3.1_terminus",
            ]
        )

        self.assertEqual(args.max_tokens, 10000)
        self.assertEqual(args.forced_text, "字")
        self.assertEqual(str(args.tokenizer_path), "/models/deepseek_v3.1_terminus")


if __name__ == "__main__":
    unittest.main()
