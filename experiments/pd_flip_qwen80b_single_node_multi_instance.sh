#!/usr/bin/env bash
set -euo pipefail

# Single-node feasibility runner: four TP=2 Qwen80B instances, 1P3D -> 2P2D.
# Run this script on the selected coordinator/worker host.

if [[ -n "${ENV_FILE:-}" ]]; then
  COMMAND_ARG="${1:-run}"
else
  ENV_FILE="${1:-}"
  COMMAND_ARG="${2:-run}"
fi
if [[ -z "${ENV_FILE}" || ! -f "${ENV_FILE}" ]]; then
  echo "ENV_FILE must point to a chmod-600 private environment file" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"

case "${ADMIN_API_KEY:-}" in
  ""|replace-with-*|changeme|CHANGE_ME) echo "invalid ADMIN_API_KEY in private env" >&2; exit 2 ;;
esac

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-qwen80b-single-node-mi}"
RUN_DIR="${ARTIFACT_ROOT:-/home/tiancij/pd-artifacts}/${RUN_ID}"
IMAGE="${IMAGE:-sglang-pd-switch:tianciJ}"
SGLANG_REPO="${SGLANG_REPO:-/home/tiancij/sglang}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3-Next-80B-A3B-Instruct}"
MODEL_ID="${MODEL_ID:-Qwen3-Next-80B-A3B-Instruct}"
NODE_IP="${NODE_IP:-192.168.0.42}"
MOONCAKE_HOST="${MOONCAKE_HOST:-fd03:4514:80:6241::1}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
TP_SIZE="${TP_SIZE:-2}"
DP_SIZE="${DP_SIZE:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
IB_DEVICE="${IB_DEVICE:-mlx5_bond_1}"
MC_GID_INDEX="${MC_GID_INDEX:-3}"
MC_USE_IPV6="${MC_USE_IPV6:-1}"
COMPILE_CACHE_ROOT="${COMPILE_CACHE_ROOT:-/home/tiancij/sglang-compile-cache}"
COMPILE_CACHE_NAMESPACE="${COMPILE_CACHE_NAMESPACE:-qwen80b-ad0d00526372dcbfeca64743}"
COMPILE_CACHE_CONTAINER_DIR="${COMPILE_CACHE_CONTAINER_DIR:-/var/cache/sglang-compile}"
PD_FLIP_FIRST_MIGRATION_RATIO="${PD_FLIP_FIRST_MIGRATION_RATIO:-0.5}"
PD_FLIP_OBSERVATION_SECONDS="${PD_FLIP_OBSERVATION_SECONDS:-2}"
SLO_WINDOW_SECONDS="${SLO_WINDOW_SECONDS:-10}"
SLO_ENTER_THRESHOLD="${SLO_ENTER_THRESHOLD:-0.90}"
SLO_RECOVER_THRESHOLD="${SLO_RECOVER_THRESHOLD:-0.95}"
MIN_TTFT_SAMPLES="${MIN_TTFT_SAMPLES:-10}"
MIN_TPOT_INTERVALS="${MIN_TPOT_INTERVALS:-100}"
CONTROLLER_POLL_SECONDS="${CONTROLLER_POLL_SECONDS:-0.25}"
TRACE_SOURCE="${TRACE_SOURCE:?TRACE_SOURCE is required}"
TRACE_MANIFEST_SOURCE="${TRACE_MANIFEST_SOURCE:?TRACE_MANIFEST_SOURCE is required}"
TRACE_SHA256="${TRACE_SHA256:?TRACE_SHA256 is required}"

read -r -a GPU_PAIRS_A <<< "${GPU_PAIRS:-0,1 2,3 4,5 6,7}"
read -r -a WORKER_PORTS_A <<< "${WORKER_PORTS:-30000 30001 30002 30003}"
read -r -a BOOTSTRAP_PORTS_A <<< "${BOOTSTRAP_PORTS:-18998 18999 19000 19001}"
ROLES=(prefill decode decode decode)
if ((${#GPU_PAIRS_A[@]} != 4 || ${#WORKER_PORTS_A[@]} != 4 || ${#BOOTSTRAP_PORTS_A[@]} != 4)); then
  echo "exactly four GPU pairs, worker ports, and bootstrap ports are required" >&2
  exit 2
fi

worker_name() { printf 'tiancij-qwen80b-%s-mi%s' "${RUN_ID}" "$1"; }
helper_name() { printf 'tiancij-qwen80b-%s-%s' "${RUN_ID}" "$1"; }
router_name() { printf 'tiancij-qwen80b-%s-router' "${RUN_ID}"; }

capture_container_log() {
  local name="$1" output="$2"
  docker inspect "${name}" >/dev/null 2>&1 || return 0
  docker logs --timestamps "${name}" > "${output}" 2>&1 || true
}

redact_log_file() {
  local path="$1"
  [[ -f "${path}" ]] || return 0
  ADMIN_API_KEY="${ADMIN_API_KEY}" python3 - "${path}" <<'PY'
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
secret = os.environ["ADMIN_API_KEY"].encode()
data = path.read_bytes()
redacted = data.replace(secret, b"<redacted>")
if redacted != data:
    temporary = path.with_name(path.name + ".redacting")
    temporary.write_bytes(redacted)
    temporary.replace(path)
PY
}

stop_exact_if_present() {
  local name="$1" timeout="$2"
  docker inspect "${name}" >/dev/null 2>&1 || return 0
  case "$(docker inspect "${name}" --format '{{.State.Status}}')" in
    running|paused|restarting) docker stop --time "${timeout}" "${name}" >/dev/null ;;
  esac
}

interrupt_sampler_if_present() {
  local name="$1" state attempt
  docker inspect "${name}" >/dev/null 2>&1 || return 0
  state="$(docker inspect "${name}" --format '{{.State.Status}}')"
  if [[ "${state}" == "running" ]]; then
    docker kill --signal=INT "${name}" >/dev/null
    for attempt in $(seq 1 60); do
      state="$(docker inspect "${name}" --format '{{.State.Status}}' 2>/dev/null || true)"
      [[ "${state}" != "running" ]] && return 0
      sleep 1
    done
    echo "sampler did not exit after targeted SIGINT: ${name}" >&2
    return 1
  fi
}

cleanup_owned() {
  local reason="${1:-normal}" name index
  set +e
  mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/status"
  for component in warmup observer controller sampler; do
    name="$(helper_name "${component}")"
    capture_container_log "${name}" "${RUN_DIR}/logs/${component}.docker.log"
    if [[ "${component}" == sampler ]]; then
      interrupt_sampler_if_present "${name}"
    else
      stop_exact_if_present "${name}" 300
    fi
    docker rm "${name}" >/dev/null 2>&1 || true
  done
  name="$(router_name)"
  capture_container_log "${name}" "${RUN_DIR}/logs/router.docker.log"
  stop_exact_if_present "${name}" 1800
  for index in 0 1 2 3; do
    name="$(worker_name "${index}")"
    capture_container_log "${name}" "${RUN_DIR}/logs/mi${index}.docker.log"
    stop_exact_if_present "${name}" 1800
  done
  for name in "${RUN_DIR}"/logs/*; do
    redact_log_file "${name}"
  done
  printf '%s\n' "${reason}" > "${RUN_DIR}/status/teardown_reason.txt"
  nvidia-smi -L > "${RUN_DIR}/status/nvidia-smi-L-after.txt" 2>&1 || true
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader > "${RUN_DIR}/status/gpu-processes-after.csv" 2>&1 || true
  ss -ltnp > "${RUN_DIR}/status/listeners-after.txt" 2>&1 || true
}

on_failure() {
  local status="$?"
  trap - ERR INT TERM
  echo "run failed; preserving ${RUN_DIR} and stopping exact run-owned resources" >&2
  cleanup_owned failure
  exit "${status}"
}
on_signal() {
  trap - ERR INT TERM
  echo "run interrupted; preserving ${RUN_DIR} and stopping exact run-owned resources" >&2
  cleanup_owned interrupted
  exit 130
}
trap on_failure ERR
trap on_signal INT TERM

preflight() {
  [[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "invalid RUN_ID" >&2; return 2; }
  [[ "${TP_SIZE}" == 2 && "${DP_SIZE}" == 1 ]] || { echo "this feasibility runner requires TP=2, DP=1" >&2; return 2; }
  [[ "$(stat -c '%a' "${ENV_FILE}")" =~ ^(600|400)$ ]] || { echo "private env must be chmod 600 or 400" >&2; return 2; }
  test -d "${SGLANG_REPO}" && test -f "${MODEL_PATH}/config.json" && test -f "${MODEL_PATH}/tokenizer.json"
  test -x "${SGLANG_REPO}/experimental/sgl-router/target/release/sgl-router"
  docker image inspect "${IMAGE}" >/dev/null
  test -f "${TRACE_SOURCE}" && test -f "${TRACE_MANIFEST_SOURCE}"
  [[ "$(sha256sum "${TRACE_SOURCE}" | awk '{print $1}')" == "${TRACE_SHA256}" ]]
  test -d "/sys/class/infiniband/${IB_DEVICE}"
  local gid_file="/sys/class/infiniband/${IB_DEVICE}/ports/1/gids/${MC_GID_INDEX}"
  test -r "${gid_file}" && [[ "$(cat "${gid_file}")" != 0000:0000:0000:0000:0000:0000:0000:0000 ]]
  show_gids | python3 -c "import ipaddress,sys; rows=[x.split() for x in sys.stdin if x.startswith('${IB_DEVICE}')]; assert any(x[2]=='${MC_GID_INDEX}' and ipaddress.ip_address(x[3])==ipaddress.ip_address('${MOONCAKE_HOST}') for x in rows)"
  local all_ports="${ROUTER_PORT} ${WORKER_PORTS_A[*]} ${BOOTSTRAP_PORTS_A[*]}" port gpu pair name index
  for port in ${all_ports}; do
    ! ss -ltn | awk '{print $4}' | grep -Eq "(:${port})$" || { echo "port ${port} is occupied" >&2; return 1; }
  done
  for index in 0 1 2 3; do
    name="$(worker_name "${index}")"
    ! docker inspect "${name}" >/dev/null 2>&1 || { echo "owned name already exists: ${name}" >&2; return 1; }
    pair="${GPU_PAIRS_A[$index]//,/ }"
    for gpu in ${pair}; do
      [[ -z "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -E '^[[:space:]]*[0-9]+' || true)" ]] || { echo "GPU ${gpu} is busy" >&2; return 1; }
    done
  done
  for name in "$(router_name)" "$(helper_name warmup)" "$(helper_name observer)" "$(helper_name controller)" "$(helper_name sampler)"; do
    ! docker inspect "${name}" >/dev/null 2>&1 || { echo "owned name already exists: ${name}" >&2; return 1; }
  done
  nvidia-smi -L >/dev/null
}

prepare() {
  mkdir -p "${RUN_DIR}"/{trace,raw,state_machine,logs,status,controller,observer,metrics,warmup,env}
  cp "${TRACE_SOURCE}" "${RUN_DIR}/trace/trace.jsonl"
  cp "${TRACE_MANIFEST_SOURCE}" "${RUN_DIR}/trace/manifest.json"
  mkdir -p "${COMPILE_CACHE_ROOT}/${COMPILE_CACHE_NAMESPACE}"
  date --iso-8601=ns > "${RUN_DIR}/status/start-time.txt"
  hostname > "${RUN_DIR}/status/hostname.txt"
  git -C "${SGLANG_REPO}" rev-parse HEAD > "${RUN_DIR}/status/code-revision.txt" 2>/dev/null || true
  git -C "${SGLANG_REPO}" status --short > "${RUN_DIR}/status/code-status.txt" 2>/dev/null || true
  docker image inspect "${IMAGE}" --format '{{.Id}}' > "${RUN_DIR}/status/image-id.txt"
  nvidia-smi -q > "${RUN_DIR}/status/nvidia-smi-before.txt"
  ss -ltnp > "${RUN_DIR}/status/listeners-before.txt"
  python3 - "${RUN_DIR}/manifest.json" <<PY
import hashlib, json, sys
json.dump({
  'run_id': '${RUN_ID}', 'experiment_class': 'single_node_multi_instance_feasibility',
  'performance_comparison_valid': False, 'model_id': '${MODEL_ID}',
  'model_path': '${MODEL_PATH}', 'trace_sha256': '${TRACE_SHA256}',
  'trace_requests': 40, 'arrival_interval_seconds': 0.2,
  'instances': [
    {'name':'mi%d'%i,'gpu_ids':g,'worker_port':int(p),'bootstrap_port':int(b),'initial_role':r}
    for i,(g,p,b,r) in enumerate(zip('${GPU_PAIRS_A[*]}'.split(),'${WORKER_PORTS_A[*]}'.split(),'${BOOTSTRAP_PORTS_A[*]}'.split(),'${ROLES[*]}'.split()))
  ],
  'initial_topology':'1P3D', 'expected_final_topology':'2P2D',
  'migration_source':'mi2', 'migration_target':'mi3',
  'first_migration_ratio':float('${PD_FLIP_FIRST_MIGRATION_RATIO}'),
  'observation_seconds':float('${PD_FLIP_OBSERVATION_SECONDS}'),
  'slo_window_seconds':float('${SLO_WINDOW_SECONDS}'),
  'slo_enter_threshold':float('${SLO_ENTER_THRESHOLD}'),
  'slo_recover_threshold':float('${SLO_RECOVER_THRESHOLD}'),
  'tp_size':2, 'dp_size':1, 'mem_fraction_static':float('${MEM_FRACTION_STATIC}'),
  'rdma_device':'${IB_DEVICE}', 'gid_index':int('${MC_GID_INDEX}'),
  'measurement_boundary':'client-observed streaming events; not GPU kernel time'
}, open(sys.argv[1],'w'), indent=2, sort_keys=True)
PY
}

write_envs() {
  local index worker_urls="" worker_urls_quoted env_file extra_sglang_args_quoted
  printf -v extra_sglang_args_quoted '%q' "--trust-remote-code --mamba-scheduler-strategy extra_buffer --enable-metrics"
  for index in 0 1 2 3; do
    worker_urls+=" http://${NODE_IP}:${WORKER_PORTS_A[$index]}"
  done
  worker_urls="${worker_urls# }"
  printf -v worker_urls_quoted '%q' "${worker_urls}"
  for index in 0 1 2 3; do
    env_file="${RUN_DIR}/env/mi${index}.env"
    umask 077
    cat > "${env_file}" <<EOF
ADMIN_API_KEY=${ADMIN_API_KEY}
IMAGE=${IMAGE}
SGLANG_REPO=${SGLANG_REPO}
MODEL_PATH=${MODEL_PATH}
MODEL_ID=${MODEL_ID}
PORT=${WORKER_PORTS_A[$index]}
ROUTER_PORT=${ROUTER_PORT}
BOOTSTRAP_PORT=${BOOTSTRAP_PORTS_A[$index]}
TRANSFER_BACKEND=mooncake
IB_DEVICE=${IB_DEVICE}
MC_GID_INDEX=${MC_GID_INDEX}
MC_USE_IPV6=${MC_USE_IPV6}
MOONCAKE_LOCAL_HOSTNAME=${MOONCAKE_HOST}
SGLANG_HOST_IP=${MOONCAKE_HOST}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC}
GPU_IDS=${GPU_PAIRS_A[$index]}
TP_SIZE=${TP_SIZE}
DP_SIZE=${DP_SIZE}
SGLANG_COMPILE_CACHE_HOST_DIR=${COMPILE_CACHE_ROOT}/${COMPILE_CACHE_NAMESPACE}
SGLANG_COMPILE_CACHE_CONTAINER_DIR=${COMPILE_CACHE_CONTAINER_DIR}
ENABLE_DP_ATTENTION=0
ENABLE_CUSTOM_LOGIT_PROCESSOR=0
ENABLE_REQUEST_TIME_STATS_LOGGING=1
ENABLE_PD_FLIP_STATE_MACHINE=1
ENABLE_PD_RUNTIME_ROLE_SWITCH=1
ENABLE_PD_FLIP_HICACHE_STITCH=0
ENABLE_PD_FLIP_PREFILL_DONOR=0
EXTRA_SGLANG_ARGS=${extra_sglang_args_quoted}
WORKER_URLS=${worker_urls_quoted}
PD_FLIP_WORKER_CONTAINER_NAME=$(worker_name "${index}")
PD_FLIP_ROUTER_CONTAINER_NAME=$(router_name)
ROUTER_DYNAMO_TARBALL_FALLBACK=0
CARGO_NET_OFFLINE=true
EOF
    chmod 600 "${env_file}"
  done
}

validate_env_files() {
  local index expected_worker_urls="" expected_extra
  for index in 0 1 2 3; do
    expected_worker_urls+=" http://${NODE_IP}:${WORKER_PORTS_A[$index]}"
  done
  expected_worker_urls="${expected_worker_urls# }"
  expected_extra="--trust-remote-code --mamba-scheduler-strategy extra_buffer --enable-metrics"
  for index in 0 1 2 3; do
    (
      unset WORKER_URLS EXTRA_SGLANG_ARGS
      # shellcheck disable=SC1090
      source "${RUN_DIR}/env/mi${index}.env"
      [[ "${WORKER_URLS}" == "${expected_worker_urls}" ]]
      [[ "${EXTRA_SGLANG_ARGS}" == "${expected_extra}" ]]
    )
  done
}

start_workers() {
  local index
  for index in 0 1 2 3; do
    nohup env ENV_FILE="${RUN_DIR}/env/mi${index}.env" \
      "${SGLANG_REPO}/scripts/playground/disaggregation/pd_flip_docker/run_worker.sh" \
      "${ROLES[$index]}" "${NODE_IP}" > "${RUN_DIR}/logs/mi${index}.launcher.log" 2>&1 < /dev/null &
    echo "$!" > "${RUN_DIR}/status/mi${index}.launcher.pid"
  done
  for index in 0 1 2 3; do
    local launcher_pid
    launcher_pid="$(cat "${RUN_DIR}/status/mi${index}.launcher.pid")"
    for attempt in $(seq 1 1800); do
      if curl -fsS "http://${NODE_IP}:${WORKER_PORTS_A[$index]}/health" >/dev/null 2>&1 && \
         curl -fsS -H "Authorization: Bearer ${ADMIN_API_KEY}" "http://${NODE_IP}:${WORKER_PORTS_A[$index]}/pd_flip/runtime_role/status" | \
           python3 -c "import json,sys; v=json.load(sys.stdin); xs=v if isinstance(v,list) else [v]; assert xs and all(x.get('success') is True and x.get('status',{}).get('role')=='${ROLES[$index]}' and x.get('status',{}).get('active_event_loop_role')=='${ROLES[$index]}' for x in xs)"; then
        break
      fi
      kill -0 "${launcher_pid}" 2>/dev/null || {
        echo "mi${index} launcher exited before health gate" >&2
        return 1
      }
      [[ "${attempt}" != 1800 ]] || return 1
      sleep 2
    done
  done
}

start_router() {
  nohup env ENV_FILE="${RUN_DIR}/env/mi0.env" \
    "${SGLANG_REPO}/scripts/playground/disaggregation/pd_flip_docker/run_router.sh" \
    > "${RUN_DIR}/logs/router.launcher.log" 2>&1 < /dev/null &
  echo "$!" > "${RUN_DIR}/status/router.launcher.pid"
  for attempt in $(seq 1 300); do
    curl -fsS "http://127.0.0.1:${ROUTER_PORT}/v1/models" >/dev/null 2>&1 && break
    [[ "${attempt}" != 300 ]] || return 1
    sleep 1
  done
  curl -fsS -H "Authorization: Bearer ${ADMIN_API_KEY}" "http://127.0.0.1:${ROUTER_PORT}/pd_flip/router/workers" > "${RUN_DIR}/status/router-initial.json"
  python3 - "${RUN_DIR}/status/router-initial.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); roles=[str(x.get('role','')).lower() for x in d.get('workers',[])]
assert len(roles)==4 and roles.count('prefill')==1 and roles.count('decode')==3, roles
assert not any(bool(x.get('draining')) for x in d['workers'])
PY
}

warmup() {
  local name="$(helper_name warmup)" nodes
  nodes="$(node_args_string)"
  docker run --name "${name}" --network host --env-file "${RUN_DIR}/env/mi0.env" \
    -v "${SGLANG_REPO}:/sgl-workspace/sglang:ro" -v "${RUN_DIR}:${RUN_DIR}" "${IMAGE}" \
    bash -lc "cd /sgl-workspace/sglang && exec env PYTHONPATH=python:. python3 scripts/playground/disaggregation/pd_flip_candidate_prefill_warmup.py --router-url http://127.0.0.1:${ROUTER_PORT} ${nodes} --initial-prefill-name mi0 --candidate-prefill-name mi0 --candidate-prefill-name mi1 --candidate-prefill-name mi2 --candidate-prefill-name mi3 --trace-jsonl '${RUN_DIR}/trace/trace.jsonl' --output-dir '${RUN_DIR}/warmup' --api-key-env ADMIN_API_KEY --request-timeout-seconds 900 --role-timeout-seconds 180 --role-poll-seconds 0.25"
  capture_container_log "${name}" "${RUN_DIR}/logs/warmup.docker.log"
  docker rm "${name}" >/dev/null
  python3 - "${RUN_DIR}/warmup/summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert d.get('success') is True, d
assert d.get('warmup_request_count')==8 and d.get('final_topology')=='1P3D', d
assert d.get('kv_cache_flushed_after') is True, d
PY
}

node_args_string() {
  local index out=""
  for index in 0 1 2 3; do
    out+=" --node name=mi${index},worker_url=http://${NODE_IP}:${WORKER_PORTS_A[$index]},router_worker_id=http://${NODE_IP}:${WORKER_PORTS_A[$index]},bootstrap_port=${BOOTSTRAP_PORTS_A[$index]}"
  done
  printf '%s' "${out}"
}

run_measured_workload() {
  local ledger="${RUN_DIR}/raw/slo_ledger.jsonl" nodes
  nodes="$(node_args_string)"
  docker run -d --name "$(helper_name sampler)" --network host --env-file "${RUN_DIR}/env/mi0.env" \
    -v "${SGLANG_REPO}:/sgl-workspace/sglang:ro" -v "${RUN_DIR}:${RUN_DIR}" "${IMAGE}" \
    bash -lc "cd /sgl-workspace/sglang && exec env PYTHONPATH=python:. python3 scripts/playground/disaggregation/pd_flip_migration_measure.py sample --router-url http://127.0.0.1:${ROUTER_PORT} ${nodes} --output-events '${RUN_DIR}/raw/migration_events.jsonl' --interval-seconds 0.05 --duration-seconds 7200 --api-key-env ADMIN_API_KEY" >/dev/null
  docker run -d --name "$(helper_name observer)" --network host \
    -v "${SGLANG_REPO}:/sgl-workspace/sglang:ro" -v "${RUN_DIR}:${RUN_DIR}" "${IMAGE}" \
    bash -lc "cd /sgl-workspace/sglang && exec env PYTHONPATH=python:. python3 scripts/playground/disaggregation/pd_flip_slo_observer.py --ledger '${ledger}' --journal '${RUN_DIR}/observer/snapshots.jsonl' --summary '${RUN_DIR}/observer/summary.json' --window-seconds '${SLO_WINDOW_SECONDS}' --enter-threshold '${SLO_ENTER_THRESHOLD}' --recover-threshold '${SLO_RECOVER_THRESHOLD}' --min-ttft-samples '${MIN_TTFT_SAMPLES}' --min-tpot-intervals '${MIN_TPOT_INTERVALS}' --poll-interval '${CONTROLLER_POLL_SECONDS}' --expected-requests 40" >/dev/null
  docker run -d --name "$(helper_name controller)" --network host --env-file "${RUN_DIR}/env/mi0.env" \
    -v "${SGLANG_REPO}:/sgl-workspace/sglang:ro" -v "${RUN_DIR}:${RUN_DIR}" "${IMAGE}" \
    bash -lc "cd /sgl-workspace/sglang && exec env PYTHONPATH=python:. python3 scripts/playground/disaggregation/pd_flip_controller.py --router-url http://127.0.0.1:${ROUTER_PORT} ${nodes} --api-key-env ADMIN_API_KEY --first-migration-ratio '${PD_FLIP_FIRST_MIGRATION_RATIO}' --observation-seconds '${PD_FLIP_OBSERVATION_SECONDS}' --slo-threshold '${SLO_ENTER_THRESHOLD}' --slo-recovery-threshold '${SLO_RECOVER_THRESHOLD}' --force-second-migration-after-observation --min-prefill-slo-samples '${MIN_TTFT_SAMPLES}' --min-decode-slo-samples '${MIN_TPOT_INTERVALS}' --session-journal-path '${RUN_DIR}/controller/session.json' monitor-progressive --trace-slo-ledger '${ledger}' --window-seconds '${SLO_WINDOW_SECONDS}' --source-name mi2 --migration-target-name mi3 --iterations 2400 --poll-interval '${CONTROLLER_POLL_SECONDS}' > '${RUN_DIR}/controller/result.json' 2> '${RUN_DIR}/logs/controller.stderr.log'" >/dev/null
  cd "${SGLANG_REPO}"
  python3 scripts/playground/disaggregation/pd_flip_trace_replay.py replay \
    --trace-jsonl "${RUN_DIR}/trace/trace.jsonl" --router-url "http://127.0.0.1:${ROUTER_PORT}" \
    --mode state_machine --output-dir "${RUN_DIR}/raw" --ledger-path "${ledger}" \
    --timeout-seconds 7200 --max-workers 40
  cp "${RUN_DIR}/raw/state_machine/request_metrics.jsonl" "${RUN_DIR}/raw/request_metrics.jsonl"
  [[ "$(docker wait "$(helper_name observer)")" == 0 ]]
  [[ "$(docker wait "$(helper_name controller)")" == 0 ]]
  capture_container_log "$(helper_name observer)" "${RUN_DIR}/logs/observer.docker.log"
  capture_container_log "$(helper_name controller)" "${RUN_DIR}/logs/controller.docker.log"
  docker rm "$(helper_name observer)" "$(helper_name controller)" >/dev/null
  interrupt_sampler_if_present "$(helper_name sampler)"
  capture_container_log "$(helper_name sampler)" "${RUN_DIR}/logs/sampler.docker.log"
  docker rm "$(helper_name sampler)" >/dev/null
}

validate() {
  curl -fsS -H "Authorization: Bearer ${ADMIN_API_KEY}" "http://127.0.0.1:${ROUTER_PORT}/pd_flip/router/workers" > "${RUN_DIR}/controller/final_router.json"
  python3 - "${RUN_DIR}" <<'PY'
import json, os, sys
root=sys.argv[1]
rows=[json.loads(x) for x in open(root+'/raw/request_metrics.jsonl') if x.strip()]
assert len(rows)==40, len(rows)
assert all(x.get('status')=='completed' for x in rows)
assert all(x.get('completion_tokens')==10000 for x in rows)
errors=[json.loads(x) for x in open(root+'/raw/state_machine/errors.jsonl') if x.strip()]
assert errors==[], errors
c=json.load(open(root+'/controller/result.json')); assert c.get('success') is True, c
r=json.load(open(root+'/controller/final_router.json'))
roles=[str(x.get('role','')).lower() for x in r.get('workers',[])]
assert len(roles)==4 and roles.count('prefill')==2 and roles.count('decode')==2, roles
events=c.get('state_trace') or c.get('events') or []
assert any((x.get('reason')=='role_flip_complete' or x.get('event')=='role_flip_complete') for x in events), events[-5:]
json.dump({'valid':True,'request_count':40,'error_count':0,'final_topology':'2P2D'}, open(root+'/status/validity.json','w'), indent=2)
PY
  cd "${SGLANG_REPO}"
  python3 scripts/playground/disaggregation/pd_flip_migration_measure.py summarize \
    --events-jsonl "${RUN_DIR}/raw/migration_events.jsonl" --output-dir "${RUN_DIR}/metrics/migration" \
    --controller-log "${RUN_DIR}/controller/result.json" --request-metrics-jsonl "${RUN_DIR}/raw/request_metrics.jsonl" \
    --errors-jsonl "${RUN_DIR}/raw/state_machine/errors.jsonl" > "${RUN_DIR}/metrics/migration-summary.stdout.json"
}

post_teardown_gate() {
  local port gpu pair
  for port in "${ROUTER_PORT}" "${WORKER_PORTS_A[@]}" "${BOOTSTRAP_PORTS_A[@]}"; do
    ! ss -ltn | awk '{print $4}' | grep -Eq "(:${port})$" || { echo "port remains occupied: ${port}" >&2; return 1; }
  done
  for pair in "${GPU_PAIRS_A[@]}"; do
    for gpu in ${pair//,/ }; do
      [[ -z "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -E '^[[:space:]]*[0-9]+' || true)" ]] || return 1
    done
  done
  nvidia-smi -L >/dev/null
  printf '%s\n' valid > "${RUN_DIR}/status/teardown-valid.txt"
}

case "${COMMAND:-${COMMAND_ARG}}" in
  preflight) preflight ;;
  prepare)
    preflight
    prepare
    write_envs
    validate_env_files
    echo "multi-instance environment validation passed: ${RUN_DIR}"
    ;;
  run)
    preflight
    prepare
    write_envs
    validate_env_files
    start_workers
    start_router
    warmup
    run_measured_workload
    validate
    cleanup_owned success
    post_teardown_gate
    trap - ERR INT TERM
    echo "single-node multi-instance feasibility run valid: ${RUN_DIR}"
    ;;
  *) echo "usage: ENV_FILE=/private/file $0 [env-file] preflight|prepare|run" >&2; exit 2 ;;
esac
