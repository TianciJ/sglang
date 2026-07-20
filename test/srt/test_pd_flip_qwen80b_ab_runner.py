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
WARMUP = (
    ROOT
    / "scripts"
    / "playground"
    / "disaggregation"
    / "pd_flip_candidate_prefill_warmup.py"
)


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
            "MC_GID_INDEX=3",
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
            "ENABLE_CANDIDATE_PREFILL_WARMUP=0",
            "COMPILE_CACHE_ROOT=/home/tiancij/sglang-compile-cache",
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
        self.assertIn("PYTHONPATH=python:.", source)
        self.assertIn("printf -v extra_sglang_args_quoted '%q'", source)
        self.assertIn(
            'wait_worker "${host}" "${ROLES[$index]}" "${mode}" "${index}"',
            source,
        )
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

    def test_worker_health_gate_only_requires_runtime_role_for_state_machine(self):
        source = RUNNER.read_text(encoding="utf-8")
        start = source.index("wait_worker()")
        end = source.index("\n}\n\nwrite_mode_manifest", start)
        body = source[start:end]

        self.assertIn('worker_ip="${NODE_IPS[$index]}"', body)
        self.assertIn('if [[ "${mode}" == "state_machine" ]]', body)
        self.assertIn("http://${worker_ip}:${PORT}/health", body)
        self.assertNotIn("http://127.0.0.1:${PORT}/health", body)

    def test_worker_launch_detaches_before_waiting_for_health(self):
        source = RUNNER.read_text(encoding="utf-8")
        start = source.index("start_mode()")
        end = source.index("\n}\n\nstart_sampler", start)
        body = source[start:end]

        self.assertIn("cd '${SGLANG_REPO}'; nohup env", body)
        self.assertNotIn("cd '${SGLANG_REPO}' && nohup env", body)
        self.assertEqual(body.count("for index in 0 1 2 3; do"), 2)
        self.assertLess(body.index("run_worker.sh"), body.index("wait_worker"))
        self.assertIn(
            "done\n  for index in 0 1 2 3; do\n    host=\"${SSH_HOSTS[$index]}\"",
            body,
        )

    def test_measurement_helpers_use_experiment_image_before_trace_replay(self):
        source = RUNNER.read_text(encoding="utf-8")
        start = source.index("start_sampler()")
        end = source.index("\n}\n\nvalidate_workload", start)
        sampler_body = source[start:end]
        start = source.index("run_workload()")
        end = source.index("\n}\n\ncollect_and_stop", start)
        workload_body = source[start:end]

        self.assertIn("cd '${SGLANG_REPO}'; nohup env", sampler_body)
        self.assertNotIn("cd '${SGLANG_REPO}' && nohup env", sampler_body)
        self.assertIn("start_observer_container", workload_body)
        self.assertIn("start_controller_container", workload_body)
        self.assertNotIn("nohup python3", workload_body)

        observer_start = source.index("start_observer_container()")
        observer_end = source.index("\n}\n\nstart_controller_container", observer_start)
        observer_body = source[observer_start:observer_end]
        controller_start = source.index("start_controller_container()")
        controller_end = source.index("\n}\n\nwait_helper_container", controller_start)
        controller_body = source[controller_start:controller_end]

        for body in (observer_body, controller_body):
            self.assertIn("docker run -d --name", body)
            self.assertIn("--network host", body)
            self.assertIn("'${IMAGE}'", body)
            self.assertIn("/sgl-workspace/sglang", body)
        self.assertIn("${SGLANG_REPO}:/sgl-workspace/sglang:ro", observer_body)
        self.assertIn("${SGLANG_REPO}:/sgl-workspace/sglang:ro", controller_body)
        self.assertIn("--session-journal-path", controller_body)
        self.assertIn("${RUN_DIR}/${mode}/controller/session.json", controller_body)

    def test_helper_containers_have_exact_owned_names_and_cleanup(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("helper_name()", source)
        self.assertIn("docker wait", source)
        self.assertIn("docker rm -f", source)
        self.assertNotIn("pd_flip_slo_observer.py'; do", source)

    def test_worker_supports_explicit_gpu_and_feature_gates(self):
        source = WORKER.read_text(encoding="utf-8")
        self.assertIn('ENABLE_DP_ATTENTION:-1', source)
        self.assertIn('GPU_IDS:-all', source)
        self.assertIn('gpu_request="\\"device=${gpu_request}\\""', source)
        self.assertIn('CUDA_VISIBLE_DEVICES=${GPU_IDS}', source)
        self.assertIn('MC_GID_INDEX', source)
        self.assertIn(
            'extra_docker_args+=(-e "ADMIN_API_KEY=${ADMIN_API_KEY}")', source
        )
        self.assertNotIn('extra_docker_args+=(-e ADMIN_API_KEY)', source)
        self.assertIn('ENABLE_REQUEST_TIME_STATS_LOGGING:-0', source)
        self.assertIn('--served-model-name "${MODEL_ID}"', source)

    def test_worker_mounts_the_provenance_keyed_persistent_compile_cache(self):
        source = WORKER.read_text(encoding="utf-8")

        self.assertIn("SGLANG_COMPILE_CACHE_HOST_DIR", source)
        self.assertIn("SGLANG_COMPILE_CACHE_CONTAINER_DIR", source)
        self.assertIn('SGLANG_CACHE_DIR=', source)
        self.assertIn('TORCHINDUCTOR_CACHE_DIR=', source)
        self.assertIn('TRITON_CACHE_DIR=', source)
        self.assertIn('CUDA_CACHE_PATH=', source)
        self.assertIn('TORCH_EXTENSIONS_DIR=', source)
        self.assertIn('TVM_FFI_CACHE_DIR=', source)

    def test_state_machine_warms_every_candidate_p_before_measurement(self):
        source = RUNNER.read_text(encoding="utf-8")
        run_start = source.index("run_one_mode()")
        run_end = source.index("\n}\n\nbaseline()", run_start)
        run_body = source[run_start:run_end]

        self.assertIn('if [[ "${mode}" == "state_machine" ]]', run_body)
        self.assertIn('warm_candidate_prefill_nodes "${mode}"', run_body)
        self.assertLess(
            run_body.index('warm_candidate_prefill_nodes "${mode}"'),
            run_body.index('run_workload "${mode}"'),
        )

        self.assertIn("pd_flip_candidate_prefill_warmup.py", source)
        self.assertIn("--candidate-prefill-name node0", source)
        self.assertIn("--candidate-prefill-name node1", source)
        self.assertIn("--candidate-prefill-name node2", source)
        self.assertIn("--candidate-prefill-name node3", source)
        self.assertIn("--trace-jsonl '${RUN_DIR}/trace/trace.jsonl'", source)
        self.assertIn("--output-dir '${RUN_DIR}/${mode}/warmup'", source)
        manifest_start = source.index("write_mode_manifest()")
        manifest_end = source.index("\n}\n\nstart_mode", manifest_start)
        manifest_body = source[manifest_start:manifest_end]
        self.assertIn("compile_cache_namespace", manifest_body)
        self.assertIn("compile_cache_container_dir", manifest_body)
        self.assertIn("gpu_model", manifest_body)
        self.assertIn("driver_version", manifest_body)
        self.assertIn("warmup_profile_version", manifest_body)
        self.assertIn("compile_cache_provenance_hash", manifest_body)
        self.assertIn("compile_cache_snapshot_before", manifest_body)
        self.assertIn("compile_cache_snapshot_after_warmup", manifest_body)

        cache_start = source.index("ensure_cache_namespace()")
        cache_end = source.index("\n}\n\ncapture_cache_snapshot", cache_start)
        cache_body = source[cache_start:cache_end]
        self.assertIn("for index in 0 1 2 3; do", cache_body)
        self.assertIn("cache provenance mismatch", cache_body)
        self.assertIn("CACHE_PROVENANCE_HASH", cache_body)
        snapshot_start = source.index("capture_cache_snapshot()")
        snapshot_end = source.index("\n}\n\nwrite_remote_env", snapshot_start)
        snapshot_body = source[snapshot_start:snapshot_end]
        self.assertIn("provenance_material", snapshot_body)
        self.assertIn("onerror", snapshot_body)
        self.assertIn("ls-files --others --exclude-standard", source)

    def test_persistent_compile_cache_is_only_mounted_for_the_diagnostic(self):
        source = RUNNER.read_text(encoding="utf-8")
        start = source.index("start_mode()")
        end = source.index("\n}\n\nstart_sampler", start)
        body = source[start:end]

        self.assertIn('cache_enabled="0"', body)
        self.assertIn(
            'if [[ "${mode}" == "state_machine" && '
            '"${ENABLE_CANDIDATE_PREFILL_WARMUP}" == "1" ]]',
            body,
        )
        self.assertIn('cache_enabled="1"', body)
        self.assertIn(
            'write_remote_env "${host}" "${mode}" "${index}" "${flags}" '
            '"${cache_enabled}"',
            body,
        )

        env_start = source.index("write_remote_env()")
        env_end = source.index("\n}\n\nwait_worker", env_start)
        env_body = source[env_start:env_end]
        self.assertIn('if [[ "${cache_enabled}" == "1" ]]', env_body)

    def test_candidate_prefill_diagnostic_cannot_be_reported_as_an_ab_run(self):
        source = RUNNER.read_text(encoding="utf-8")
        run_all_start = source.index("run_all()")
        run_all_end = source.index("\n}\n\ncase", run_all_start)
        run_all_body = source[run_all_start:run_all_end]

        self.assertIn('ENABLE_CANDIDATE_PREFILL_WARMUP', run_all_body)
        self.assertIn("state-machine diagnostic", run_all_body)
        self.assertLess(run_all_body.index("exit 2"), run_all_body.index("preflight"))

        compare_start = source.index("compare()")
        compare_end = source.index("\n}\n\nrun_all", compare_start)
        compare_body = source[compare_start:compare_end]
        self.assertIn("candidate_prefill_warmup_enabled", compare_body)
        self.assertIn("state-machine diagnostic", compare_body)

    def test_candidate_prefill_warmup_helper_is_checked_in(self):
        self.assertTrue(WARMUP.is_file(), WARMUP)

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
