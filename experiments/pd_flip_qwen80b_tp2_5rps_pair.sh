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
  for key in MODEL_PATH MODEL_ID GPU_IDS TP_SIZE DP_SIZE MEM_FRACTION_STATIC IB_DEVICE MC_GID_INDEX MC_USE_IPV6 NODE0_IP NODE1_IP NODE2_IP NODE3_IP NODE0_MOONCAKE_HOST NODE1_MOONCAKE_HOST NODE2_MOONCAKE_HOST NODE3_MOONCAKE_HOST; do
    left="$(env_value "${UPSTREAM_ENV_FILE}" "${key}")"
    right="$(env_value "${STATE_ENV_FILE}" "${key}")"
    [[ -n "${left}" && "${left}" == "${right}" ]] || die "mode config mismatch for ${key}: ${left} != ${right}"
  done
  [[ "$(env_value "${UPSTREAM_ENV_FILE}" GPU_IDS)" == "0,1" ]] || die "this run requires GPUs 0,1"
  [[ "$(env_value "${UPSTREAM_ENV_FILE}" TP_SIZE)" == "2" ]] || die "this run requires TP=2"
  [[ "${PD_FLIP_OBSERVATION_SECONDS}" == "2" ]] || die "observation period must be 2 seconds"
  [[ "${PD_FLIP_FIRST_MIGRATION_RATIO}" == "0.5" ]] || die "first migration ratio must be 0.5"
}

common_env() {
  export TRACE_SOURCE TRACE_MANIFEST_SOURCE TRACE_SHA256 SOURCE_TRACE_SHA256
  export TRACE_INTERVAL_SECONDS TRACE_LONG_TTFT_SLO_SECONDS TRACE_SHORT_TTFT_SLO_SECONDS
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
  python3 - "${ARTIFACT_ROOT}/${BASELINE_RUN_ID}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.load(open(root / "manifest.json"))
summary = json.load(open(root / "report" / "summary.json"))
assert manifest["validity"] == "valid" and manifest["trace_sha256"] == "d82d0f7fc5b745f43a48d6d91451794887b4a3f2e5f049d6e7a30a38652c9508"
assert summary["valid"] is True and summary["requests"] == 40
PY
}

validate_state() {
  python3 - "${ARTIFACT_ROOT}/${STATE_RUN_ID}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.load(open(root / "state_machine" / "manifest.json"))
rows = [json.loads(x) for x in open(root / "state_machine" / "raw" / "request_metrics.jsonl") if x.strip()]
errors = [x for x in open(root / "state_machine" / "raw" / "state_machine" / "errors.jsonl") if x.strip()]
controller = json.load(open(root / "state_machine" / "controller" / "result.json"))
observer = json.load(open(root / "state_machine" / "observer" / "summary.json"))
assert manifest["trace_sha256"] == "d82d0f7fc5b745f43a48d6d91451794887b4a3f2e5f049d6e7a30a38652c9508"
assert len(rows) == 40 and not errors
assert all(r["status"] == "completed" and r["completion_tokens"] == 10000 and r["finish_reason"] == "length" for r in rows)
assert controller.get("success") is True and controller.get("final_topology") == "2P2D"
assert controller.get("first_migration_ratio") == 0.5 and controller.get("observation_seconds") == 2.0
assert observer.get("first_trigger")
PY
}

write_pair_summary() {
  mkdir -p "${PAIR_DIR}"
  python3 - "${ARTIFACT_ROOT}/${BASELINE_RUN_ID}" "${ARTIFACT_ROOT}/${STATE_RUN_ID}" "${PAIR_DIR}" <<'PY'
import json, pathlib, statistics, sys
baseline, state, out = map(pathlib.Path, sys.argv[1:])
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
  "trace_sha256": "d82d0f7fc5b745f43a48d6d91451794887b4a3f2e5f049d6e7a30a38652c9508",
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
  mkdir -p "${PAIR_DIR}"
  common_env
  RUN_ID="${BASELINE_RUN_ID}" ENV_FILE="${UPSTREAM_ENV_FILE}" "${UPSTREAM_RUNNER}" run
  validate_baseline
  state_env
  RUN_ID="${STATE_RUN_ID}" ENV_FILE="${STATE_ENV_FILE}" "${STATE_RUNNER}" preflight
  RUN_ID="${STATE_RUN_ID}" ENV_FILE="${STATE_ENV_FILE}" "${STATE_RUNNER}" prepare
  RUN_ID="${STATE_RUN_ID}" ENV_FILE="${STATE_ENV_FILE}" "${STATE_RUNNER}" state-machine
  validate_state
  write_pair_summary
}

case "${1:-}" in
  validate) validate_pair_config ;;
  preflight) preflight ;;
  run) run_pair ;;
  report) validate_baseline; validate_state; write_pair_summary ;;
  *) echo "usage: $0 validate|preflight|run|report" >&2; exit 2 ;;
esac
