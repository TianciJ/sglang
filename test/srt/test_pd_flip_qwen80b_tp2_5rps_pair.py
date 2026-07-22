import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "pd_flip_qwen80b_tp2_5rps_pair.sh"
UPSTREAM_ENV = ROOT / "experiments" / "pd_upstream_qwen80b_baseline.env.example"
STATE_ENV = ROOT / "experiments" / "pd_flip_qwen80b_ab.env.example"
TRACE = ROOT / "pd-flip-artifacts" / "qwen80b-trace40-5rps-slo025-045" / "trace.jsonl"
TRACE_MANIFEST = TRACE.with_name("manifest.json")


def bash_path(path: Path) -> str:
    path = path.resolve()
    if os.name != "nt":
        return str(path)
    return f"/mnt/{path.drive[0].lower()}{path.as_posix()[2:]}"


class TP2FiveRPSPairRunnerTest(unittest.TestCase):
    def test_runner_freezes_order_trace_slo_and_topology_contract(self):
        source = RUNNER.read_text(encoding="utf-8")
        pair_env_source = (
            ROOT / "experiments" / "pd_flip_qwen80b_tp2_5rps_pair.env.example"
        ).read_text(encoding="utf-8")
        self.assertLess(
            source.index('"${UPSTREAM_RUNNER}" run'),
            source.index('"${STATE_RUNNER}" state-machine'),
        )
        for expected in (
            "TRACE_SHA256",
            "TRACE_LONG_TTFT_SLO_SECONDS",
            "TRACE_SHORT_TTFT_SLO_SECONDS",
            'controller.get("final_topology") == "2P2D"',
            'controller.get("first_migration_ratio") == 0.5',
            'controller.get("observation_seconds") == 2.0',
            'observer.get("first_trigger")',
            "write_pair_design",
            "validate_pair_provenance",
            "BASELINE_RUN_ID_OVERRIDE",
            "run-state-only",
            '"model_fingerprint"',
            "model_fingerprint_reconciliation.json",
            "legacy_reconciliation_used",
            '"mooncake_hosts"',
        ):
            self.assertIn(expected, source)
        self.assertIn("TRACE_INTERVAL_SECONDS=0.2", pair_env_source)
        self.assertNotIn("docker restart", source)
        self.assertNotIn("docker rm -f", source)
        self.assertNotIn("pkill", source)
        self.assertNotIn("kill -9", source)

    def test_validate_accepts_the_checked_in_trace_and_matching_mode_envs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pair_env = Path(temp_dir) / "pair.env"
            pair_env.write_text(
                "\n".join(
                    (
                        "ADMIN_API_KEY_SOURCE_ENV=/not-read-by-validate",
                        f'UPSTREAM_ENV_FILE="{bash_path(UPSTREAM_ENV)}"',
                        f'STATE_ENV_FILE="{bash_path(STATE_ENV)}"',
                        f'ARTIFACT_ROOT="{bash_path(Path(temp_dir))}"',
                        f'TRACE_SOURCE="{bash_path(TRACE)}"',
                        f'TRACE_MANIFEST_SOURCE="{bash_path(TRACE_MANIFEST)}"',
                        "TRACE_SHA256=d82d0f7fc5b745f43a48d6d91451794887b4a3f2e5f049d6e7a30a38652c9508",
                        "SOURCE_TRACE_SHA256=c5dbbf75c997dfc5d67a18251082f2f246d6c055eb4af5040fbe147f49f4ce5d",
                        "TRACE_INTERVAL_SECONDS=0.2",
                        "TRACE_LONG_TTFT_SLO_SECONDS=0.45",
                        "TRACE_SHORT_TTFT_SLO_SECONDS=0.25",
                        "TRACE_TPOT_SLO_SECONDS=0.05",
                        "SLO_WINDOW_SECONDS=10",
                        "SLO_ENTER_THRESHOLD=0.90",
                        "SLO_RECOVER_THRESHOLD=0.95",
                        "MIN_TTFT_SAMPLES=10",
                        "MIN_TPOT_INTERVALS=100",
                        "PD_FLIP_FIRST_MIGRATION_RATIO=0.5",
                        "PD_FLIP_OBSERVATION_SECONDS=2",
                        "ENABLE_CANDIDATE_PREFILL_WARMUP=1",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                ["bash", bash_path(RUNNER), "validate"],
                cwd=ROOT,
                env={**os.environ, "PAIR_ENV_FILE": bash_path(pair_env), "PAIR_ID": "unit-pair"},
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
