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
    drive = path.drive.rstrip(":").lower()
    tail = path.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{tail}"


def test_pins_clean_image_trace_and_fixed_workload_contract():
    text = source()
    assert "tiancij/sglang-upstream:v0.5.15-clean" in text
    assert "sha256:7dd92779d739364d79af34af65815ddc14e567728e5256f65ac922367161213e" in text
    assert "82da848d68c9662a7aaaf76deb547b1d8cc6c4f562586f0d60dd212bc114e964" in text
    assert "EXPECTED_REQUESTS=40" in text
    assert "EXPECTED_TOKENS=10000" in text
    assert "EXPECTED_LEDGER_ROWS=400040" in text
    assert "EXPECTED_TPOT_ROWS=399960" in text
    assert "--max-workers 40" in text


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
    ):
        assert forbidden not in text
    assert "cd /sgl-workspace/sglang" in text
    assert "-v \"${REMOTE_HELPER_REPO}:/work/sglang:ro\"" in text


def test_uses_only_exact_owned_names_and_safe_stop_primitives():
    text = source()
    assert "tiancij-upstream-%s-node%s" in text
    assert "tiancij-upstream-%s-router" in text
    assert "tiancij-upstream-router-build-%s" in text
    assert "docker stop --time 1800" in text
    assert "docker rm \"${name}\"" in text
    assert "trap 'on_failure" in text
    for unsafe in ("docker restart", "pkill", "killall", "kill -9", "docker rm -f"):
        assert unsafe not in text
    assert "docker ps -aq --filter name=" not in text


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
    assert "GPU_IDS=0,1,2,3" in text
    assert "TP_SIZE=4" in text
    assert "DP_SIZE=1" in text
    assert "NODE0_ROLE=prefill" in text
    assert text.count("_ROLE=decode") == 3
    assert "IB_DEVICE=mlx5_bond_0" in text
    assert "MC_USE_IPV6=1" in text
    assert "MC_GID_INDEX=3" in text
    for suffix in ("6240", "7b80", "6600", "5f00"):
        assert f"fd03:4514:80:{suffix}::1" in text


def test_can_load_admin_key_from_private_file_without_printing_it():
    text = source()
    assert 'if [[ -n "${ADMIN_API_KEY_FILE:-}" ]]' in text
    assert '[[ -r "${ADMIN_API_KEY_FILE}" ]]' in text
    assert 'ADMIN_API_KEY="${ADMIN_API_KEY#ADMIN_API_KEY=}"' in text


def test_passes_validated_ipv6_mooncake_identity_separately_from_http_ip():
    text = source()
    assert 'MOONCAKE_HOSTS=(' in text
    assert 'moon_host="${MOONCAKE_HOSTS[$index]}"' in text
    assert '-e "MOONCAKE_LOCAL_HOSTNAME=$moon_host"' in text
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
