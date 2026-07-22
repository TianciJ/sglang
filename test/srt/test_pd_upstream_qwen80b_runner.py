import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[2]
RUNNER = ROOT / "experiments" / "pd_upstream_qwen80b_baseline.sh"
ENV_EXAMPLE = ROOT / "experiments" / "pd_upstream_qwen80b_baseline.env.example"
RUNBOOK = ROOT / "docs" / "runbooks" / "pd_upstream_qwen80b_baseline.md"


def source() -> str:
    return RUNNER.read_text(encoding="utf-8")


def wsl_path(path: Path) -> str:
    if not path.drive:
        return path.as_posix()
    drive = path.drive.rstrip(":").lower()
    tail = path.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{tail}"


def test_pins_clean_image_natural_trace_and_fixed_workload_contract():
    text = source()
    assert "tiancij/sglang-upstream:v0.5.15-clean" in text
    assert "sha256:7dd92779d739364d79af34af65815ddc14e567728e5256f65ac922367161213e" in text
    assert "c5dbbf75c997dfc5d67a18251082f2f246d6c055eb4af5040fbe147f49f4ce5d" in text
    assert "qwen80b-trace40-natural" in text
    assert "EXPECTED_REQUESTS=40" in text
    assert "EXPECTED_TOKENS=10000" in text
    assert "--max-workers 40" in text
    assert "--portable-forced-token-processor" not in text
    assert "client_runtime_processor" not in text
    assert "--enable-custom-logit-processor" not in text
    assert "'output_contract':'natural'" in text
    assert "client_first_last_output_over_usage_completion_tokens" in text


def test_smoke_includes_unmeasured_10000_token_natural_output_probe_and_cold_gate():
    text = source()
    assert "natural-10k-probe.json" in text
    assert '"max_tokens": 10000' in text
    assert '"ignore_eos": True' in text
    assert "completion_tokens == 10000" in text
    assert 'finish_reason == "length"' in text
    assert "flush_cache" in text


def test_smoke_warms_long_and_short_trace_prompts_before_one_flush():
    text = source()
    assert 'warmup_kinds = ("long", "short")' in text
    assert "long-prefill-warmup.json" in text
    assert "short-prefill-warmup.json" in text
    assert 'trace_path = os.path.join(run_dir, "trace", "trace.jsonl")' in text
    assert 'next(row for row in trace_rows if row["prompt_kind"] == prompt_kind)' in text
    assert 'warmup_body = dict(trace_row["body"])' in text
    assert 'warmup_body.pop("custom_params", None)' in text
    assert 'warmup_body["max_tokens"] = 1' in text
    assert 'assert prompt_tokens > 6000' in text
    assert 'assert 500 <= prompt_tokens <= 1000' in text
    assert '"measured": False' in text
    assert '"kv_cache_flushed_after": True' in text
    assert '"started_utc"' in text
    assert '"first_output_utc"' in text
    assert '"finished_utc"' in text
    assert 'warmup-node${index}.docker.log' in text
    assert "warmup-router.docker.log" in text

    long_record = text.index("long-prefill-warmup.json")
    short_record = text.index("short-prefill-warmup.json")
    flush = text.index("flush_cache", short_record)
    measure = text.index("measure\n", text.index("run_all()"))
    assert long_record < short_record < flush < measure


def test_dual_warmup_log_window_is_compatible_with_python36_host():
    text = source()
    assert "datetime.fromisoformat" not in text
    assert "warmup_window_start = datetime.now(timezone.utc) - timedelta(seconds=2)" in text
    assert "warmup_window_end = datetime.now(timezone.utc) + timedelta(seconds=2)" in text


def test_dual_warmup_flush_failure_is_forensic_instead_of_relaunching():
    text = source()
    assert "post-warmup cache flush failed" in text
    flush_failure = text.index("post-warmup cache flush failed")
    measure = text.index("measure\n", text.index("run_all()"))
    assert flush_failure < measure


def test_manifest_keeps_gid_index_and_mooncake_hosts_as_separate_fields():
    text = source()
    assert "'mc_gid_index':${MC_GID_INDEX},'mooncake_hosts':" in text


def test_never_mounts_host_code_into_worker_or_router_and_has_no_custom_flags():
    text = source()
    assert ":/sgl-workspace/sglang" not in text
    for forbidden in (
        "--enable-pd-flip-state-machine",
        "--enable-pd-runtime-role-switch",
        "--enable-pd-flip-hicache-stitch",
        "--enable-pd-flip-prefill-donor",
        "--enable-hierarchical-cache",
        "--disaggregation-decode-enable-radix-cache",
        "--enable-custom-logit-processor",
    ):
        assert forbidden not in text
    assert "cd /sgl-workspace/sglang" in text
    assert "-v \"${REMOTE_HELPER_REPO}:/work/sglang:ro\"" in text


def test_uses_only_exact_owned_names_and_safe_stop_primitives():
    text = source()
    assert "set -Eeuo pipefail" in text
    assert "tiancij-upstream-%s-node%s" in text
    assert "tiancij-upstream-%s-router" in text
    assert "tiancij-upstream-router-build-%s" in text
    assert "docker stop --time 1800" in text
    assert "docker rm \"${name}\"" in text
    assert "trap 'on_failure" in text
    for unsafe in ("docker restart", "pkill", "killall", "kill -9", "docker rm -f"):
        assert unsafe not in text
    assert "docker ps -aq --filter name=" not in text


def test_gpu_device_request_keeps_all_four_ids_in_one_docker_argument():
    text = source()
    assert '--gpus "\\\"device=$gpus\\\""' in text


def test_safe_stop_removes_created_containers_without_stopping_them():
    text = source()
    assert 'status="$(docker inspect "$name" --format \'{{.State.Status}}\')"' in text
    assert 'running|paused|restarting)' in text
    assert 'docker rm "${name}"' in text


def test_has_bounded_gates_concurrent_start_and_purity_inspection():
    text = source()
    assert "seq 1" in text and "WORKER_HEALTH_ATTEMPTS" in text
    assert "seq 1" in text and "ROUTER_HEALTH_ATTEMPTS" in text
    assert "start_worker \"${index}\" &" in text
    assert "wait \"${pid}\"" in text
    assert "docker inspect" in text
    assert "Mounts" in text
    assert "Config.Cmd" in text
    assert "Config.Env" in text
    assert "nvidia-smi -L" in text
    assert "chronyc tracking" in text
    assert "ibv_devinfo" in text


def test_prepare_creates_real_directories_and_purity_gate_checks_mounts_only():
    text = source()
    assert "'{trace,raw,logs,inspect,status,smoke,report}'" not in text
    assert "mkdir -p '${RUN_DIR}/trace' '${RUN_DIR}/raw' '${RUN_DIR}/logs'" in text
    mount_checks = [
        line
        for line in text.splitlines()
        if "grep -F '/sgl-workspace/sglang'" in line
    ]
    assert len(mount_checks) == 2
    assert all("{{json .Mounts}}" in line for line in mount_checks)


def test_exposes_complete_lifecycle_and_evidence_inventory():
    text = source()
    for command in (
        "preflight)",
        "build-router)",
        "prepare)",
        "start)",
        "smoke)",
        "measure)",
        "collect-stop)",
        "report)",
        "dry-run)",
        "run)",
    ):
        assert command in text
    for artifact in (
        "slo_ledger.jsonl",
        "request_metrics.jsonl",
        "responses.jsonl",
        "errors.jsonl",
        "tpot_tokens.csv",
        "manifest.json",
        "INVENTORY.txt",
        "source_manifest.json",
    ):
        assert artifact in text


def test_env_example_contains_no_real_secret_and_fixed_topology():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "ADMIN_API_KEY=replace-with-a-private-secret" in text
    assert "ADMIN_API_KEY_FILE=" in text
    assert "GPU_IDS=0,1" in text
    assert "TP_SIZE=2" in text
    assert "DP_SIZE=1" in text
    assert "NODE0_ROLE=prefill" in text
    assert text.count("_ROLE=decode") == 3
    assert "IB_DEVICE=mlx5_bond_1" in text
    assert "MC_USE_IPV6=1" in text
    assert "MC_GID_INDEX=3" in text
    for suffix in ("6241", "7b81", "6601", "5f01"):
        assert f"fd03:4514:80:{suffix}::1" in text


def test_preflight_does_not_dump_full_process_arguments_or_secrets():
    text = source()
    assert "ps -eo pid,user,args" not in text
    assert "ps -eo pid,user,comm" in text


def test_can_load_admin_key_from_private_file_without_printing_it():
    text = source()
    assert 'if [[ -n "${ADMIN_API_KEY_FILE:-}" ]]' in text
    assert '[[ -r "${ADMIN_API_KEY_FILE}" ]]' in text
    assert 'ADMIN_API_KEY="${ADMIN_API_KEY#ADMIN_API_KEY=}"' in text


def test_collected_logs_redact_server_args_key_and_teardown_removes_secret_envs():
    text = source()
    assert "admin_api_key=" in text
    assert "<redacted>" in text
    assert "rm -f -- '${RUN_DIR}/helper.env'" in text
    assert "for index in 0 1 2 3" in text
    assert "'${RUN_DIR}/node${index}/worker.env'" in text
    cleanup = text.index("rm -f -- '${RUN_DIR}/helper.env'")
    inventory = text.rindex("INVENTORY.txt")
    assert cleanup < inventory


def test_passes_validated_ipv6_mooncake_identity_separately_from_http_ip():
    text = source()
    assert 'MOONCAKE_HOSTS=(' in text
    assert 'moon_host="${MOONCAKE_HOSTS[$index]}"' in text
    assert '-e "MOONCAKE_LOCAL_HOSTNAME=$moon_host"' in text
    assert '-e "SGLANG_HOST_IP=$moon_host"' in text
    assert '-e "MC_USE_IPV6=$use_ipv6"' in text
    assert "show_gids" in text
    assert "mooncake_hosts" in text
    assert "mc_use_ipv6" in text


def test_smoke_reads_secret_from_remote_env_without_putting_it_in_ssh_command():
    text = source()
    assert "key='${ADMIN_API_KEY}'" not in text
    assert 'ssh "${SSH_HOSTS[$index]}" "curl' not in text
    assert 'key = os.environ["ADMIN_API_KEY"]' in text
    assert 'source "$env_file"' in text


def test_dry_run_is_redacted_and_does_not_contact_nodes():
    env = os.environ.copy()
    result = subprocess.run(
        [
            "bash",
            "-lc",
            f"RUN_ID=unit-dry-run ENV_FILE='{wsl_path(ENV_EXAMPLE)}' '{wsl_path(RUNNER)}' dry-run",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "unit-dry-run" in result.stdout
    assert "replace-with-a-private-secret" not in result.stdout
    assert "ssh " not in result.stdout
    assert re.search(r"node0.*prefill", result.stdout)
    assert re.search(r"node[123].*decode", result.stdout)


def test_runbook_covers_operator_sequence_and_artifact_boundary():
    text = RUNBOOK.read_text(encoding="utf-8")
    for command in ("preflight", "build-router", "dry-run", "run", "collect-stop", "report"):
        assert f" {command}" in text
    for artifact in (
        "slo_ledger.jsonl",
        "request_metrics.jsonl",
        "responses.jsonl",
        "errors.jsonl",
        "tpot_tokens.csv",
        "manifest.json",
        "INVENTORY.txt",
    ):
        assert artifact in text
    assert "client-observed" in text
    assert "one measured run" in text
    assert "P2PHANDSHAKE" in text
    assert "forensic" in text
