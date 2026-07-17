import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "pd_flip_qwen80b_ab.sh"
ENV_EXAMPLE = ROOT / "experiments" / "pd_flip_qwen80b_ab.env.example"
WORKER = ROOT / "scripts/playground/disaggregation/pd_flip_docker/run_worker.sh"


def _bash_path(path):
    path = Path(path).resolve()
    if os.name != "nt":
        return str(path)
    return f"/mnt/{path.drive[0].lower()}{path.as_posix()[2:]}"


class Qwen80BABRunnerTest(unittest.TestCase):
    def test_example_env_is_sourceable_by_bash(self):
        if shutil.which("bash") is None:
            self.skipTest("bash is unavailable")
        result = subprocess.run(
            ["bash", "-n", _bash_path(ENV_EXAMPLE)], text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        source = ENV_EXAMPLE.read_text(encoding="utf-8")
        self.assertIn(
            'EXTRA_SGLANG_ARGS="--trust-remote-code --mamba-scheduler-strategy '
            'extra_buffer --enable-metrics"',
            source,
        )

    def test_env_freezes_the_agreed_quick_validation(self):
        source = ENV_EXAMPLE.read_text(encoding="utf-8")
        for value in (
            "MODEL_ID=Qwen3-Next-80B-A3B-Instruct",
            "MODEL_PATH=/models/Qwen3-Next-80B-A3B-Instruct",
            "TP_SIZE=4",
            "DP_SIZE=1",
            "TRACE_REQUESTS=40",
            "TRACE_MAX_TOKENS=10000",
            "TRACE_INTRA_WAVE_INTERVAL_SECONDS=0.5",
            "TRACE_WAVE_START_INTERVAL_SECONDS=7.5",
            "PD_FLIP_FIRST_MIGRATION_RATIO=0.5",
            "PD_FLIP_OBSERVATION_SECONDS=2",
            "SLO_WINDOW_SECONDS=10",
            "SLO_ENTER_THRESHOLD=0.90",
            "SLO_RECOVER_THRESHOLD=0.95",
            "MIN_TTFT_SAMPLES=10",
            "MIN_TPOT_INTERVALS=100",
            "CONTROLLER_POLL_SECONDS=0.25",
        ):
            self.assertIn(value, source)

    def test_runner_has_separate_clean_baseline_and_state_machine_launches(self):
        source = RUNNER.read_text(encoding="utf-8")
        for action in ("preflight", "prepare", "baseline", "state-machine", "compare", "run"):
            self.assertIn(action, source)
        self.assertIn("ENABLE_PD_FLIP_STATE_MACHINE=0", source)
        self.assertIn("ENABLE_PD_RUNTIME_ROLE_SWITCH=0", source)
        self.assertIn("ENABLE_PD_FLIP_STATE_MACHINE=1", source)
        self.assertIn("ENABLE_PD_RUNTIME_ROLE_SWITCH=1", source)
        self.assertIn("ENABLE_PD_FLIP_HICACHE_STITCH=0", source)
        self.assertIn("ENABLE_PD_FLIP_PREFILL_DONOR=0", source)
        self.assertIn("ENABLE_REQUEST_TIME_STATS_LOGGING=1", source)
        self.assertIn("pd_flip_slo_observer.py", source)
        self.assertIn("pd_flip_req_timing.py", source)
        self.assertIn("pd_flip_migration_measure.py", source)
        self.assertIn("monitor-progressive", source)
        self.assertIn("--first-migration-ratio", source)
        self.assertIn("--observation-seconds", source)
        self.assertIn("--slo-recovery-threshold", source)
        self.assertIn("--force-second-migration-after-observation", source)
        self.assertIn("--window-seconds", source)
        self.assertIn("--max-tokens '${TRACE_MAX_TOKENS}'", source)
        self.assertIn("model.safetensors.index.json", source)
        self.assertIn("weight_map", source)
        self.assertIn(".git/refs/heads/main", source)
        self.assertIn("-printf '%f:%s", source)
        self.assertIn("ACTIVE_MODE", source)
        self.assertIn("trap 'on_failure", source)
        self.assertNotIn("docker restart", source)
        self.assertNotIn("huggingface-cli download", source)
        self.assertNotIn("snapshot_download", source)
        self.assertNotIn("pkill", source)
        self.assertNotIn("kill -9", source)

    def test_worker_supports_explicit_gpu_and_feature_gates(self):
        source = WORKER.read_text(encoding="utf-8")
        self.assertIn('ENABLE_DP_ATTENTION:-1', source)
        self.assertIn('GPU_IDS:-all', source)
        self.assertIn('ENABLE_REQUEST_TIME_STATS_LOGGING:-0', source)
        self.assertIn('--served-model-name "${MODEL_ID}"', source)

    def test_dry_run_prints_safe_commands_without_external_actions(self):
        if shutil.which("bash") is None:
            self.skipTest("bash is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / "qwen.env"
            artifact_root = Path(directory) / "artifacts"
            env_path.write_text(
                "\n".join(
                    [
                        "ADMIN_API_KEY=local-test-secret",
                        "IMAGE=test-image",
                        "SGLANG_REPO=/srv/sglang",
                        "MODEL_PATH=/models/Qwen3-Next-80B-A3B-Instruct",
                        "MODEL_ID=Qwen3-Next-80B-A3B-Instruct",
                        f"ARTIFACT_ROOT={shlex.quote(_bash_path(artifact_root))}",
                    ]
                ),
                encoding="utf-8",
            )
            command = " ".join(
                [
                    f"ENV_FILE={shlex.quote(_bash_path(env_path))}",
                    "DRY_RUN=1",
                    "RUN_ID=test-run",
                    shlex.quote(_bash_path(RUNNER)),
                    "run",
                ]
            )
            result = subprocess.run(
                ["bash", "-lc", command], text=True, capture_output=True
            )

        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertNotIn("local-test-secret", output)
        self.assertIn("DRY-RUN preflight", output)
        self.assertIn("DRY-RUN baseline", output)
        self.assertIn("DRY-RUN state-machine", output)
        self.assertIn("DRY-RUN compare", output)


if __name__ == "__main__":
    unittest.main()
