#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAIR_ENV_FILE="${PAIR_ENV_FILE:-${SCRIPT_DIR}/pd_flip_qwen80b_tp2_5rps_pair.env.example}"
source "${PAIR_ENV_FILE}"

PAIR_ID="${PAIR_ID:-$(date -u +%Y%m%dT%H%M%SZ)-qwen80b-tp2-5rps-pair}"
BASELINE_RUN_ID="${PAIR_ID}-upstream"
STATE_RUN_ID="${PAIR_ID}-state"
PAIR_DIR="${ARTIFACT_ROOT}/${PAIR_ID}-pair"
UPSTREAM_RUNNER="${SCRIPT_DIR}/pd_upstream_qwen80b_baseline.sh"
STATE_RUNNER="${SCRIPT_DIR}/pd_flip_qwen80b_ab.sh"

die() { echo "pair runner: $*" >&2; exit 2; }

load_admin_key() {
  [[ -r "${ADMIN_API_KEY_SOURCE_ENV}" ]] || die "private admin-key source is unreadable"
  ADMIN_API_KEY="$(sed -n 's/^ADMIN_API_KEY=//p' "${ADMIN_API_KEY_SOURCE_ENV}" | tail -n 1)"
  case "${ADMIN_API_KEY}" in
    ""|replace-with-*|changeme|CHANGE_ME) die "private admin key is missing" ;;
  esac
  export ADMIN_API_KEY
}

env_value() {
  local file="$1" key="$2"
  bash -c 'set -a; source "$1"; eval "printf %s \"\${$2-}\""' _ "${file}" "${key}"
}

validate_trace() {
  [[ -f "${TRACE_SOURCE}" && -f "${TRACE_MANIFEST_SOURCE}" ]] || die "trace or manifest is missing"
  [[ "$(sha256sum "${TRACE_SOURCE}" | awk '{print $1}')" == "${TRACE_SHA256}" ]] || die "trace hash mismatch"
  python3 - "${TRACE_SOURCE}" "${TRACE_MANIFEST_SOURCE}" <<'PY'
import hashlib, json, sys
trace, manifest = sys.argv[1:]
rows = [json.loads(line) for line in open(trace, encoding="utf-8") if line.strip()]
meta = json.load(open(manifest, encoding="utf-8"))
actual = hashlib.sha256(open(trace, "rb").read()).hexdigest()
assert len(rows) == 40 and meta["request_count"] == 40
assert actual == meta["effective_sha256"]
assert [r["arrival_offset_s"] for r in rows] == [round(i * 0.2, 9) for i in range(40)]
assert all(r["ttft_slo_s"] == 0.45 for r in rows[::2])
assert all(r["ttft_slo_s"] == 0.25 for r in rows[1::2])
assert all(r["tpot_slo_s"] == 0.05 for r in rows)
assert all(r["body"]["max_tokens"] == 10000 and r["body"]["ignore_eos"] is True for r in rows)
assert all(r["body"]["custom_params"]["pd_flip_slo"]["ttft_seconds"] == r["ttft_slo_s"] for r in rows)
PY
}

validate_pair_config() {
  [[ "${PAIR_ID}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || die "invalid PAIR_ID"
  [[ -r "${UPSTREAM_ENV_FILE}" && -r "${STATE_ENV_FILE}" ]] || die "private mode env file is missing"
  validate_trace
  local key left right
  for key in MODEL_PATH MODEL_ID GPU_IDS TP_SIZE DP_SIZE MEM_FRACTION_STATIC IB_DEVICE MC_GID_INDEX MC_USE_IPV6 ROUTER_PORT BOOTSTRAP_PORT NODE0_IP NODE1_IP NODE2_IP NODE3_IP NODE0_MOONCAKE_HOST NODE1_MOONCAKE_HOST NODE2_MOONCAKE_HOST NODE3_MOONCAKE_HOST; do
    left="$(env_value "${UPSTREAM_ENV_FILE}" "${key}")"
    right="$(env_value "${STATE_ENV_FILE}" "${key}")"
    [[ -n "${left}" && "${left}" == "${right}" ]] || die "mode config mismatch for ${key}: ${left} != ${right}"
  done
  [[ "$(env_value "${UPSTREAM_ENV_FILE}" WORKER_PORT)" == "$(env_value "${STATE_ENV_FILE}" PORT)" ]] || die "mode config mismatch for worker port"
  [[ "$(env_value "${UPSTREAM_ENV_FILE}" GPU_IDS)" == "0,1" ]] || die "this run requires GPUs 0,1"
  [[ "$(env_value "${UPSTREAM_ENV_FILE}" TP_SIZE)" == "2" ]] || die "this run requires TP=2"
  [[ "${TRACE_INTERVAL_SECONDS}" == "0.2" ]] || die "arrival interval must be 0.2 seconds"
  [[ "${TRACE_LONG_TTFT_SLO_SECONDS}" == "0.45" ]] || die "long TTFT SLO must be 0.45 seconds"
  [[ "${TRACE_SHORT_TTFT_SLO_SECONDS}" == "0.25" ]] || die "short TTFT SLO must be 0.25 seconds"
  [[ "${TRACE_TPOT_SLO_SECONDS}" == "0.05" ]] || die "TPOT SLO must be 0.05 seconds"
  [[ "${SLO_WINDOW_SECONDS}" == "10" ]] || die "SLO window must be 10 seconds"
  [[ "${SLO_ENTER_THRESHOLD}" == "0.90" ]] || die "SLO enter threshold must be 0.90"
  [[ "${SLO_RECOVER_THRESHOLD}" == "0.95" ]] || die "SLO recovery threshold must be 0.95"
  [[ "${MIN_TTFT_SAMPLES}" == "10" ]] || die "minimum TTFT samples must be 10"
  [[ "${MIN_TPOT_INTERVALS}" == "100" ]] || die "minimum TPOT intervals must be 100"
  [[ "${PD_FLIP_OBSERVATION_SECONDS}" == "2" ]] || die "observation period must be 2 seconds"
  [[ "${PD_FLIP_FIRST_MIGRATION_RATIO}" == "0.5" ]] || die "first migration ratio must be 0.5"
  [[ "${ENABLE_CANDIDATE_PREFILL_WARMUP}" == "1" ]] || die "candidate Prefill warmup must be enabled"
}

common_env() {
  export TRACE_SOURCE TRACE_MANIFEST_SOURCE TRACE_SHA256 SOURCE_TRACE_SHA256
  export TRACE_INTERVAL_SECONDS TRACE_LONG_TTFT_SLO_SECONDS TRACE_SHORT_TTFT_SLO_SECONDS TRACE_TPOT_SLO_SECONDS
}

state_env() {
  common_env
  export SLO_WINDOW_SECONDS SLO_ENTER_THRESHOLD SLO_RECOVER_THRESHOLD
  export MIN_TTFT_SAMPLES MIN_TPOT_INTERVALS PD_FLIP_FIRST_MIGRATION_RATIO
  export PD_FLIP_OBSERVATION_SECONDS ENABLE_CANDIDATE_PREFILL_WARMUP
  export TRACE_OUTPUT_CONTRACT=natural ENABLE_CUSTOM_LOGIT_PROCESSOR=0
}

preflight() {
  load_admin_key
  validate_pair_config
  common_env
  RUN_ID="${BASELINE_RUN_ID}" ENV_FILE="${UPSTREAM_ENV_FILE}" "${UPSTREAM_RUNNER}" preflight
  state_env
  RUN_ID="${STATE_RUN_ID}" ENV_FILE="${STATE_ENV_FILE}" "${STATE_RUNNER}" preflight
}

validate_baseline() {
  python3 - "${ARTIFACT_ROOT}/${BASELINE_RUN_ID}" "${TRACE_SHA256}" <<'PY'
import json, pathlib, sys
root, expected_trace = pathlib.Path(sys.argv[1]), sys.argv[2]
manifest = json.load(open(root / "manifest.json"))
summary = json.load(open(root / "report" / "summary.json"))
assert manifest["validity"] == "valid" and manifest["trace_sha256"] == expected_trace
assert summary["valid"] is True and summary["requests"] == 40
PY
}

validate_state() {
  python3 - "${ARTIFACT_ROOT}/${STATE_RUN_ID}" "${TRACE_SHA256}" <<'PY'
import json, pathlib, sys
root, expected_trace = pathlib.Path(sys.argv[1]), sys.argv[2]
manifest = json.load(open(root / "state_machine" / "manifest.json"))
rows = [json.loads(x) for x in open(root / "state_machine" / "raw" / "request_metrics.jsonl") if x.strip()]
errors = [x for x in open(root / "state_machine" / "raw" / "state_machine" / "errors.jsonl") if x.strip()]
controller = json.load(open(root / "state_machine" / "controller" / "result.json"))
observer = json.load(open(root / "state_machine" / "observer" / "summary.json"))
assert manifest["trace_sha256"] == expected_trace
assert len(rows) == 40 and not errors
assert all(r["status"] == "completed" and r["completion_tokens"] == 10000 and r["finish_reason"] == "length" for r in rows)
assert controller.get("success") is True and controller.get("final_topology") == "2P2D"
assert controller.get("first_migration_ratio") == 0.5 and controller.get("observation_seconds") == 2.0
assert observer.get("first_trigger")
PY
}

write_pair_design() {
  mkdir -p "${PAIR_DIR}"
  local revision dirty upstream_image state_image
  revision="$(git -C "${SCRIPT_DIR}/.." rev-parse HEAD 2>/dev/null || printf unknown)"
  dirty=false
  git -C "${SCRIPT_DIR}/.." diff --quiet --ignore-submodules HEAD -- 2>/dev/null || dirty=true
  upstream_image="$(env_value "${UPSTREAM_ENV_FILE}" IMAGE)"
  state_image="$(env_value "${STATE_ENV_FILE}" IMAGE)"
  python3 - "${PAIR_DIR}/design.json" "${PAIR_ID}" "${BASELINE_RUN_ID}" "${STATE_RUN_ID}" \
    "${TRACE_SHA256}" "${SOURCE_TRACE_SHA256}" "${revision}" "${dirty}" \
    "${upstream_image}" "${state_image}" \
    "$(env_value "${UPSTREAM_ENV_FILE}" MODEL_ID)" "$(env_value "${UPSTREAM_ENV_FILE}" MODEL_PATH)" \
    "$(env_value "${UPSTREAM_ENV_FILE}" GPU_IDS)" "$(env_value "${UPSTREAM_ENV_FILE}" TP_SIZE)" \
    "$(env_value "${UPSTREAM_ENV_FILE}" DP_SIZE)" "$(env_value "${UPSTREAM_ENV_FILE}" MEM_FRACTION_STATIC)" \
    "$(env_value "${UPSTREAM_ENV_FILE}" IB_DEVICE)" "$(env_value "${UPSTREAM_ENV_FILE}" MC_GID_INDEX)" \
    "$(env_value "${UPSTREAM_ENV_FILE}" MC_USE_IPV6)" \
    "$(env_value "${UPSTREAM_ENV_FILE}" NODE0_MOONCAKE_HOST)" \
    "$(env_value "${UPSTREAM_ENV_FILE}" NODE1_MOONCAKE_HOST)" \
    "$(env_value "${UPSTREAM_ENV_FILE}" NODE2_MOONCAKE_HOST)" \
    "$(env_value "${UPSTREAM_ENV_FILE}" NODE3_MOONCAKE_HOST)" \
    "$(env_value "${UPSTREAM_ENV_FILE}" WORKER_PORT)" \
    "$(env_value "${UPSTREAM_ENV_FILE}" ROUTER_PORT)" \
    "$(env_value "${UPSTREAM_ENV_FILE}" BOOTSTRAP_PORT)" <<'PY'
import json, pathlib, sys
(
    output, pair_id, baseline_id, state_id, trace_sha, source_sha, revision,
    dirty, upstream_image, state_image, model_id, model_path, gpu_ids,
    tp_size, dp_size, mem_fraction, ib_device, gid_index, use_ipv6,
    host0, host1, host2, host3, worker_port, router_port, bootstrap_port,
) = sys.argv[1:]
design = {
    "pair_id": pair_id,
    "question": "clean upstream static 1P3D versus custom state machine 1P3D to 2P2D",
    "claim_boundary": "end-to-end system comparison; images/code differ, so this does not isolate state-machine overhead",
    "baseline_run_id": baseline_id,
    "state_machine_run_id": state_id,
    "trace_sha256": trace_sha,
    "source_trace_sha256": source_sha,
    "runner_revision": revision,
    "runner_worktree_dirty": dirty == "true",
    "upstream_image": upstream_image,
    "state_machine_image": state_image,
    "model_id": model_id,
    "model_path": model_path,
    "gpu_ids": gpu_ids,
    "tp_size": int(tp_size),
    "dp_size": int(dp_size),
    "mem_fraction_static": float(mem_fraction),
    "ib_device": ib_device,
    "mc_gid_index": int(gid_index),
    "mc_use_ipv6": int(use_ipv6),
    "mooncake_hosts": [host0, host1, host2, host3],
    "worker_port": int(worker_port),
    "router_port": int(router_port),
    "bootstrap_port": int(bootstrap_port),
    "workload": {
        "requests": 40, "arrival_interval_seconds": 0.2,
        "long_ttft_slo_seconds": 0.45, "short_ttft_slo_seconds": 0.25,
        "tpot_slo_seconds": 0.05, "max_tokens": 10000, "ignore_eos": True,
    },
    "policy": {
        "window_seconds": 10, "enter_threshold": 0.90,
        "recover_threshold": 0.95, "first_migration_ratio": 0.5,
        "observation_seconds": 2,
    },
}
pathlib.Path(output).write_text(json.dumps(design, indent=2, sort_keys=True) + "\n")
PY
}

validate_pair_provenance() {
  python3 - "${ARTIFACT_ROOT}/${BASELINE_RUN_ID}" "${ARTIFACT_ROOT}/${STATE_RUN_ID}" <<'PY'
import json, pathlib, sys
baseline, state = map(pathlib.Path, sys.argv[1:])
b = json.load(open(baseline / "manifest.json"))
s = json.load(open(state / "state_machine" / "manifest.json"))
keys = (
    "trace_sha256", "model_id", "model_fingerprint", "gpu_ids", "tp_size",
    "dp_size", "mem_fraction_static", "ib_device", "mc_gid_index",
    "mc_use_ipv6", "mooncake_hosts", "worker_port", "router_port",
    "bootstrap_port", "output_contract",
)
mismatches = {key: [b.get(key), s.get(key)] for key in keys if b.get(key) != s.get(key)}
assert not mismatches, f"baseline/state provenance mismatch: {mismatches}"
assert b.get("topology") == s.get("initial_topology") == "1P3D"
PY
}

write_pair_summary() {
  mkdir -p "${PAIR_DIR}"
  python3 - "${ARTIFACT_ROOT}/${BASELINE_RUN_ID}" "${ARTIFACT_ROOT}/${STATE_RUN_ID}" "${PAIR_DIR}" "${TRACE_SHA256}" <<'PY'
import json, pathlib, statistics, sys
baseline, state, out = map(pathlib.Path, sys.argv[1:4])
expected_trace = sys.argv[4]
b = json.load(open(baseline / "report" / "summary.json"))
rows = [json.loads(x) for x in open(state / "state_machine" / "raw" / "request_metrics.jsonl") if x.strip()]
controller = json.load(open(state / "state_machine" / "controller" / "result.json"))
def percentile(xs, q):
    xs = sorted(xs); return xs[max(0, int((len(xs) * q + 0.999999)) - 1)]
ttft = [float(r["ttft_s"]) for r in rows]
tpot = [float(r["avg_tpot_s"]) for r in rows]
summary = {
  "valid": True,
  "claim_boundary": "end-to-end clean upstream versus custom state-machine system; not isolated state-machine overhead",
  "trace_sha256": expected_trace,
  "baseline": b,
  "state_machine": {
    "requests": len(rows), "ttft_mean_s": statistics.fmean(ttft), "ttft_p95_s": percentile(ttft, .95),
    "ttft_attainment": sum(bool(r.get("ttft_met")) for r in rows) / len(rows),
    "tpot_mean_s": statistics.fmean(tpot), "tpot_p95_s": percentile(tpot, .95),
    "final_topology": controller["final_topology"], "controller": controller,
  },
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY
}

run_pair() {
  preflight
  write_pair_design
  common_env
  RUN_ID="${BASELINE_RUN_ID}" ENV_FILE="${UPSTREAM_ENV_FILE}" "${UPSTREAM_RUNNER}" run
  validate_baseline
  state_env
  RUN_ID="${STATE_RUN_ID}" ENV_FILE="${STATE_ENV_FILE}" "${STATE_RUNNER}" preflight
  RUN_ID="${STATE_RUN_ID}" ENV_FILE="${STATE_ENV_FILE}" "${STATE_RUNNER}" prepare
  RUN_ID="${STATE_RUN_ID}" ENV_FILE="${STATE_ENV_FILE}" "${STATE_RUNNER}" state-machine
  validate_state
  validate_pair_provenance
  write_pair_summary
}

case "${1:-}" in
  validate) validate_pair_config ;;
  preflight) preflight ;;
  run) run_pair ;;
  report) validate_baseline; validate_state; validate_pair_provenance; write_pair_summary ;;
  *) echo "usage: $0 validate|preflight|run|report" >&2; exit 2 ;;
esac
