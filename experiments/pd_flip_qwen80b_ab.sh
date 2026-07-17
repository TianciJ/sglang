#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/pd_flip_qwen80b_ab.env.example}"
source "${ENV_FILE}"

DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-qwen80b-ab}"
RUN_DIR="${ARTIFACT_ROOT:-/home/tiancij/pd-artifacts}/${RUN_ID}"
MODEL_ID="${MODEL_ID:-Qwen3-Next-80B-A3B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3-Next-80B-A3B-Instruct}"
TP_SIZE="${TP_SIZE:-4}"
DP_SIZE="${DP_SIZE:-1}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PORT="${PORT:-30000}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
TRACE_REQUESTS="${TRACE_REQUESTS:-40}"
TRACE_MAX_TOKENS="${TRACE_MAX_TOKENS:-10000}"
TRACE_FORCED_TEXT="${TRACE_FORCED_TEXT:-的}"
PD_FLIP_FIRST_MIGRATION_RATIO="${PD_FLIP_FIRST_MIGRATION_RATIO:-0.5}"
PD_FLIP_OBSERVATION_SECONDS="${PD_FLIP_OBSERVATION_SECONDS:-3}"
SLO_WINDOW_SECONDS="${SLO_WINDOW_SECONDS:-10}"
SLO_ENTER_THRESHOLD="${SLO_ENTER_THRESHOLD:-0.90}"
SLO_RECOVER_THRESHOLD="${SLO_RECOVER_THRESHOLD:-0.95}"
MIN_TTFT_SAMPLES="${MIN_TTFT_SAMPLES:-10}"
MIN_TPOT_INTERVALS="${MIN_TPOT_INTERVALS:-100}"
CONTROLLER_POLL_SECONDS="${CONTROLLER_POLL_SECONDS:-0.25}"

SSH_HOSTS=("${NODE0_SSH:-cloud-099}" "${NODE1_SSH:-cloud-100}" "${NODE2_SSH:-cloud-101}" "${NODE3_SSH:-cloud-102}")
NODE_IPS=("${NODE0_IP:-192.168.0.42}" "${NODE1_IP:-192.168.0.40}" "${NODE2_IP:-192.168.0.39}" "${NODE3_IP:-192.168.0.41}")
ROLES=(prefill decode decode decode)
ACTIVE_MODE=""

redacted() {
  local value="$*"
  if [[ -n "${ADMIN_API_KEY:-}" ]]; then
    value="${value//${ADMIN_API_KEY}/<redacted>}"
  fi
  printf '%s\n' "${value}"
}

remote_code_hash() {
  local host="$1"
  ssh "${host}" "git -C '${SGLANG_REPO}' rev-parse HEAD 2>/dev/null || cat '${SGLANG_REPO}/.git/refs/heads/main'"
}

dry_note() {
  redacted "DRY-RUN $*"
}

require_secret() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  case "${ADMIN_API_KEY:-}" in
    ""|replace-with-*|changeme|CHANGE_ME)
      echo "ADMIN_API_KEY must be supplied through a private ENV_FILE" >&2
      exit 2
      ;;
  esac
}

container_name() {
  local mode="$1" index="$2"
  printf 'tiancij-qwen80b-%s-%s-node%s' "${RUN_ID}" "${mode}" "${index}"
}

router_name() {
  printf 'tiancij-qwen80b-%s-%s-router' "${RUN_ID}" "$1"
}

helper_name() {
  local mode="$1" component="$2"
  printf 'tiancij-qwen80b-%s-%s-%s' "${RUN_ID}" "${mode}" "${component}"
}

on_failure() {
  local status="${1:-1}" mode="${ACTIVE_MODE}"
  trap - ERR INT TERM
  if [[ "${DRY_RUN}" == "1" || -z "${mode}" ]]; then
    return "${status}"
  fi
  set +e
  echo "abnormal exit; gracefully cleaning exact owned resources for ${RUN_ID}/${mode}" >&2
  ssh "${SSH_HOSTS[0]}" "pid_file='${RUN_DIR}/${mode}/pids/migration_sampler.pid'; if test -f \"\$pid_file\"; then pid=\$(cat \"\$pid_file\"); if kill -0 \"\$pid\" 2>/dev/null && tr '\\0' ' ' < /proc/\$pid/cmdline | grep -F 'pd_flip_migration_measure.py sample' >/dev/null; then kill -TERM \"\$pid\"; fi; fi"
  local component helper
  for component in observer controller; do
    helper="$(helper_name "${mode}" "${component}")"
    ssh "${SSH_HOSTS[0]}" "if docker inspect '${helper}' >/dev/null 2>&1; then docker logs --timestamps '${helper}' >> '${RUN_DIR}/${mode}/logs/${component}.log' 2>&1 || true; docker rm -f '${helper}' >/dev/null; fi"
  done
  local router
  router="$(router_name "${mode}")"
  ssh "${SSH_HOSTS[0]}" "if docker inspect '${router}' >/dev/null 2>&1; then docker logs --timestamps '${router}' > '${RUN_DIR}/${mode}/logs/router.failure.docker.log' 2>&1 || true; docker stop --time 1800 '${router}' >/dev/null; fi"
  for index in 0 1 2 3; do
    local host="${SSH_HOSTS[$index]}" name
    name="$(container_name "${mode}" "${index}")"
    ssh "${host}" "if docker inspect '${name}' >/dev/null 2>&1; then docker logs --timestamps '${name}' > '${RUN_DIR}/${mode}/logs/node${index}.failure.docker.log' 2>&1 || true; docker stop --time 1800 '${name}' >/dev/null; fi"
  done
  return "${status}"
}

trap 'on_failure $?; exit $?' ERR
trap 'on_failure 130; exit 130' INT TERM

preflight() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    dry_note "preflight four nodes; check SSH, exact model files, image, ports, GPUs, clocks, and owned names"
    return
  fi
  require_secret
  local missing=() expected_code="" expected_model="" expected_image=""
  local gpu_list="${GPU_IDS//,/ }"
  for index in 0 1 2 3; do
    local host="${SSH_HOSTS[$index]}" code_hash model_hash image_id
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${host}" true
    if ! ssh "${host}" "python3 -c \"import json; from pathlib import Path; root=Path('${MODEL_PATH}'); index=root/'model.safetensors.index.json'; weights=list(root.glob('*.safetensors'))+list(root.glob('*.bin')); data=json.loads(index.read_text()) if index.exists() else {}; refs=set(data.get('weight_map',{}).values()); missing=[name for name in refs if not (root/name).is_file()]; assert (root/'config.json').is_file(); assert ((bool(refs) and not missing) if index.exists() else bool(weights))\""; then
      missing+=("${host}")
    fi
    ssh "${host}" "test -d '${SGLANG_REPO}' && docker image inspect '${IMAGE}' >/dev/null && test -z \"\$(docker ps -aq --filter name='^/tiancij-qwen80b-${RUN_ID}-')\""
    ssh "${host}" "! ss -ltn | awk '{print \$4}' | grep -Eq '(:${PORT}|:${BOOTSTRAP_PORT})$'"
    if [[ "${index}" == "0" ]]; then
      ssh "${host}" "! ss -ltn | awk '{print \$4}' | grep -Eq '(:${ROUTER_PORT})$'"
    fi
    ssh "${host}" "for gpu in ${gpu_list}; do test -z \"\$(nvidia-smi -i \"\$gpu\" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -E '^[[:space:]]*[0-9]+' || true)\" || { echo selected GPU \$gpu is busy >&2; exit 1; }; done"
    code_hash="$(remote_code_hash "${host}")"
    model_hash="$(ssh "${host}" "{ sha256sum '${MODEL_PATH}/config.json'; find '${MODEL_PATH}' -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) -printf '%f:%s\\n' | LC_ALL=C sort; } | sha256sum | awk '{print \$1}'")"
    image_id="$(ssh "${host}" "docker image inspect '${IMAGE}' --format '{{.Id}}'")"
    if [[ -z "${expected_code}" ]]; then
      expected_code="${code_hash}"; expected_model="${model_hash}"; expected_image="${image_id}"
    elif [[ "${code_hash}" != "${expected_code}" || "${model_hash}" != "${expected_model}" || "${image_id}" != "${expected_image}" ]]; then
      echo "code, model config, or image fingerprint mismatch at ${host}" >&2
      exit 2
    fi
    ssh "${host}" "date --iso-8601=ns; chronyc tracking 2>/dev/null || true"
  done
  if ((${#missing[@]})); then
    printf 'model is missing or incomplete on nodes: %s\n' "${missing[*]}" >&2
    exit 2
  fi
  ssh "${SSH_HOSTS[0]}" "test -x '${SGLANG_REPO}/experimental/sgl-router/target/release/sgl-router'"
}

prepare() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    dry_note "prepare immutable 40-request trace once at ${RUN_DIR}/trace"
    return
  fi
  require_secret
  local host="${SSH_HOSTS[0]}"
  ssh "${host}" "mkdir -p '${RUN_DIR}/trace' '${RUN_DIR}/comparison'; for mode in baseline state_machine; do for part in raw logs metrics observer controller pids status; do mkdir -p '${RUN_DIR}/'\$mode/\$part; done; done"
  ssh "${host}" "docker run --rm --network none -v '${SGLANG_REPO}:/sgl-workspace/sglang:ro' -v '${MODEL_PATH}:${MODEL_PATH}:ro' -v '${RUN_DIR}:${RUN_DIR}' '${IMAGE}' bash -lc \"cd /sgl-workspace/sglang && PYTHONPATH=python:. python3 scripts/playground/disaggregation/pd_flip_qwen80b_trace.py --run-nonce '${RUN_ID}' --model '${MODEL_ID}' --forced-text '${TRACE_FORCED_TEXT}' --tokenizer-path '${MODEL_PATH}' --max-tokens '${TRACE_MAX_TOKENS}' --output '${RUN_DIR}/trace/trace.jsonl' --manifest '${RUN_DIR}/trace/manifest.json'\""
  ssh "${host}" "python3 -c \"import json; p='${RUN_DIR}/trace/trace.jsonl'; rows=[json.loads(x) for x in open(p) if x.strip()]; assert len(rows)==${TRACE_REQUESTS}; assert all(r['body']['max_tokens']==${TRACE_MAX_TOKENS} for r in rows)\""
}

write_remote_env() {
  local host="$1" mode="$2" index="$3" flags="$4"
  local node0="http://${NODE_IPS[0]}:${PORT}"
  local node1="http://${NODE_IPS[1]}:${PORT}"
  local node2="http://${NODE_IPS[2]}:${PORT}"
  local node3="http://${NODE_IPS[3]}:${PORT}"
  local content extra_sglang_args_quoted
  printf -v extra_sglang_args_quoted '%q' "${EXTRA_SGLANG_ARGS:---trust-remote-code --mamba-scheduler-strategy extra_buffer --enable-metrics}"
  content="ADMIN_API_KEY=${ADMIN_API_KEY}
IMAGE=${IMAGE}
SGLANG_REPO=${SGLANG_REPO}
MODEL_PATH=${MODEL_PATH}
MODEL_ID=${MODEL_ID}
PORT=${PORT}
ROUTER_PORT=${ROUTER_PORT}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT}
TRANSFER_BACKEND=${TRANSFER_BACKEND:-mooncake}
IB_DEVICE=${IB_DEVICE:-mlx5_0}
MC_GID_INDEX=${MC_GID_INDEX:-}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.88}
GPU_IDS=${GPU_IDS}
TP_SIZE=${TP_SIZE}
DP_SIZE=${DP_SIZE}
ENABLE_DP_ATTENTION=${ENABLE_DP_ATTENTION:-0}
ENABLE_CUSTOM_LOGIT_PROCESSOR=1
ENABLE_REQUEST_TIME_STATS_LOGGING=1
ENABLE_PD_FLIP_HICACHE_STITCH=0
ENABLE_PD_FLIP_PREFILL_DONOR=0
${flags}
EXTRA_SGLANG_ARGS=${extra_sglang_args_quoted}
NODE0=${node0}
NODE1=${node1}
NODE2=${node2}
NODE3=${node3}
PD_FLIP_WORKER_CONTAINER_NAME=$(container_name "${mode}" "${index}")
PD_FLIP_ROUTER_CONTAINER_NAME=$(router_name "${mode}")
ROUTER_DYNAMO_TARBALL_FALLBACK=0
CARGO_NET_OFFLINE=true"
  local encoded
  encoded="$(printf '%s\n' "${content}" | base64 | tr -d '\n')"
  ssh "${host}" "umask 077; mkdir -p '${RUN_DIR}/${mode}/logs'; printf '%s' '${encoded}' | base64 -d > '${RUN_DIR}/${mode}/node${index}.env'"
}

wait_worker() {
  local host="$1" role="$2" mode="$3" index="$4" worker_ip="${NODE_IPS[$index]}"
  if [[ "${mode}" == "state_machine" ]]; then
    ssh "${host}" "source '${RUN_DIR}/${mode}/node${index}.env'; for attempt in \$(seq 1 1800); do if curl -fsS 'http://${worker_ip}:${PORT}/health' >/dev/null && curl -fsS -H \"Authorization: Bearer \${ADMIN_API_KEY}\" 'http://${worker_ip}:${PORT}/pd_flip/runtime_role/status' | python3 -c \"import json,sys; v=json.load(sys.stdin); xs=v if isinstance(v,list) else [v]; role='${role}'; assert xs and all(x.get('success') is True and x.get('status',{}).get('role')==role and x.get('status',{}).get('active_event_loop_role')==role for x in xs)\"; then exit 0; fi; sleep 2; done; exit 1"
  else
    ssh "${host}" "for attempt in \$(seq 1 1800); do if curl -fsS 'http://${worker_ip}:${PORT}/health' >/dev/null; then exit 0; fi; sleep 2; done; exit 1"
  fi
}

write_mode_manifest() {
  local mode="$1" host="${SSH_HOSTS[0]}" topology trace_sha code_hash model_hash image_id content encoded state_enabled role_switch_enabled
  topology="1P3D"
  state_enabled=false
  role_switch_enabled=false
  if [[ "${mode}" == "state_machine" ]]; then
    state_enabled=true
    role_switch_enabled=true
  fi
  trace_sha="$(ssh "${host}" "python3 -c \"import json; print(json.load(open('${RUN_DIR}/trace/manifest.json'))['trace_sha256'])\"")"
  code_hash="$(remote_code_hash "${host}")"
  model_hash="$(ssh "${host}" "{ sha256sum '${MODEL_PATH}/config.json'; find '${MODEL_PATH}' -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' \) -printf '%f:%s\\n' | LC_ALL=C sort; } | sha256sum | awk '{print \$1}'")"
  image_id="$(ssh "${host}" "docker image inspect '${IMAGE}' --format '{{.Id}}'")"
  content="{\"run_id\":\"${RUN_ID}\",\"mode\":\"${mode}\",\"trace_sha256\":\"${trace_sha}\",\"model_id\":\"${MODEL_ID}\",\"model_fingerprint\":\"${model_hash}\",\"code_hash\":\"${code_hash}\",\"image_id\":\"${image_id}\",\"gpu_ids\":\"${GPU_IDS}\",\"tp_size\":${TP_SIZE},\"dp_size\":${DP_SIZE},\"initial_topology\":\"${topology}\",\"state_machine_enabled\":${state_enabled},\"runtime_role_switch_enabled\":${role_switch_enabled},\"hicache_stitch_enabled\":false,\"prefill_donor_enabled\":false,\"slo_window_seconds\":${SLO_WINDOW_SECONDS},\"slo_enter_threshold\":${SLO_ENTER_THRESHOLD},\"slo_recover_threshold\":${SLO_RECOVER_THRESHOLD},\"first_migration_ratio\":${PD_FLIP_FIRST_MIGRATION_RATIO},\"observation_seconds\":${PD_FLIP_OBSERVATION_SECONDS}}"
  encoded="$(printf '%s\n' "${content}" | base64 | tr -d '\n')"
  ssh "${host}" "printf '%s' '${encoded}' | base64 -d > '${RUN_DIR}/${mode}/manifest.json'"
}

start_mode() {
  local mode="$1" flags
  if [[ "${mode}" == "baseline" ]]; then
    flags="ENABLE_PD_FLIP_STATE_MACHINE=0
ENABLE_PD_RUNTIME_ROLE_SWITCH=0"
  else
    flags="ENABLE_PD_FLIP_STATE_MACHINE=1
ENABLE_PD_RUNTIME_ROLE_SWITCH=1"
  fi
  for index in 0 1 2 3; do
    local host="${SSH_HOSTS[$index]}"
    write_remote_env "${host}" "${mode}" "${index}" "${flags}"
    ssh "${host}" "cd '${SGLANG_REPO}'; nohup env ENV_FILE='${RUN_DIR}/${mode}/node${index}.env' scripts/playground/disaggregation/pd_flip_docker/run_worker.sh '${ROLES[$index]}' '${NODE_IPS[$index]}' > '${RUN_DIR}/${mode}/logs/node${index}.log' 2>&1 < /dev/null &"
  done
  for index in 0 1 2 3; do
    host="${SSH_HOSTS[$index]}"
    wait_worker "${host}" "${ROLES[$index]}" "${mode}" "${index}"
  done
  ssh "${SSH_HOSTS[0]}" "cd '${SGLANG_REPO}'; nohup env ENV_FILE='${RUN_DIR}/${mode}/node0.env' scripts/playground/disaggregation/pd_flip_docker/run_router.sh > '${RUN_DIR}/${mode}/logs/router.log' 2>&1 < /dev/null &"
  ssh "${SSH_HOSTS[0]}" "for attempt in \$(seq 1 300); do curl -fsS 'http://127.0.0.1:${ROUTER_PORT}/v1/models' >/dev/null && exit 0; sleep 1; done; exit 1"
  write_mode_manifest "${mode}"
}

start_sampler() {
  local mode="$1" host="${SSH_HOSTS[0]}" node_args=""
  for index in 0 1 2 3; do
    node_args+=" --node 'name=node${index},worker_url=http://${NODE_IPS[$index]}:${PORT}'"
  done
  ssh "${host}" "cd '${SGLANG_REPO}'; nohup env ADMIN_API_KEY='${ADMIN_API_KEY}' python3 scripts/playground/disaggregation/pd_flip_migration_measure.py sample --router-url 'http://127.0.0.1:${ROUTER_PORT}' ${node_args} --output-events '${RUN_DIR}/${mode}/raw/migration_events.jsonl' --interval-seconds '${MIGRATION_SAMPLE_INTERVAL_SECONDS:-0.05}' --duration-seconds '${MEASUREMENT_DURATION_SECONDS:-7200}' --api-key-env ADMIN_API_KEY > '${RUN_DIR}/${mode}/logs/migration_sampler.log' 2>&1 < /dev/null & echo \$! > '${RUN_DIR}/${mode}/pids/migration_sampler.pid'"
}

start_observer_container() {
  local mode="$1" ledger="$2" host="${SSH_HOSTS[0]}" name
  name="$(helper_name "${mode}" observer)"
  ssh "${host}" "docker run -d --name '${name}' --network host -v '${SGLANG_REPO}:/sgl-workspace/sglang:ro' -v '${RUN_DIR}:${RUN_DIR}' '${IMAGE}' bash -lc \"cd /sgl-workspace/sglang && exec env PYTHONPATH=python:. python3 scripts/playground/disaggregation/pd_flip_slo_observer.py --ledger '${ledger}' --journal '${RUN_DIR}/${mode}/observer/snapshots.jsonl' --summary '${RUN_DIR}/${mode}/observer/summary.json' --window-seconds '${SLO_WINDOW_SECONDS}' --enter-threshold '${SLO_ENTER_THRESHOLD}' --recover-threshold '${SLO_RECOVER_THRESHOLD}' --min-ttft-samples '${MIN_TTFT_SAMPLES}' --min-tpot-intervals '${MIN_TPOT_INTERVALS}' --poll-interval '${CONTROLLER_POLL_SECONDS}' --expected-requests '${TRACE_REQUESTS}'\" >/dev/null"
}

start_controller_container() {
  local mode="$1" ledger="$2" host="${SSH_HOSTS[0]}" name
  name="$(helper_name "${mode}" controller)"
  ssh "${host}" "docker run -d --name '${name}' --network host --env ADMIN_API_KEY='${ADMIN_API_KEY}' -v '${SGLANG_REPO}:/sgl-workspace/sglang:ro' -v '${RUN_DIR}:${RUN_DIR}' '${IMAGE}' bash -lc \"cd /sgl-workspace/sglang && exec env PYTHONPATH=python:. python3 scripts/playground/disaggregation/pd_flip_controller.py --router-url 'http://127.0.0.1:${ROUTER_PORT}' --node 'name=node0,worker_url=http://${NODE_IPS[0]}:${PORT},router_worker_id=http://${NODE_IPS[0]}:${PORT},bootstrap_port=${BOOTSTRAP_PORT}' --node 'name=node1,worker_url=http://${NODE_IPS[1]}:${PORT},router_worker_id=http://${NODE_IPS[1]}:${PORT},bootstrap_port=${BOOTSTRAP_PORT}' --node 'name=node2,worker_url=http://${NODE_IPS[2]}:${PORT},router_worker_id=http://${NODE_IPS[2]}:${PORT},bootstrap_port=${BOOTSTRAP_PORT}' --node 'name=node3,worker_url=http://${NODE_IPS[3]}:${PORT},router_worker_id=http://${NODE_IPS[3]}:${PORT},bootstrap_port=${BOOTSTRAP_PORT}' --api-key-env ADMIN_API_KEY --first-migration-ratio '${PD_FLIP_FIRST_MIGRATION_RATIO}' --observation-seconds '${PD_FLIP_OBSERVATION_SECONDS}' --slo-threshold '${SLO_ENTER_THRESHOLD}' --slo-recovery-threshold '${SLO_RECOVER_THRESHOLD}' --force-second-migration-after-observation --min-prefill-slo-samples '${MIN_TTFT_SAMPLES}' --min-decode-slo-samples '${MIN_TPOT_INTERVALS}' --session-journal-path '${RUN_DIR}/${mode}/controller/session.json' monitor-progressive --trace-slo-ledger '${ledger}' --window-seconds '${SLO_WINDOW_SECONDS}' --source-name node2 --migration-target-name node3 --iterations '${PD_FLIP_MONITOR_ITERATIONS:-2400}' --poll-interval '${CONTROLLER_POLL_SECONDS}' > '${RUN_DIR}/${mode}/controller/result.json' 2> '${RUN_DIR}/${mode}/logs/controller.log'\" >/dev/null"
}

wait_helper_container() {
  local mode="$1" component="$2" output="$3" host="${SSH_HOSTS[0]}" name
  name="$(helper_name "${mode}" "${component}")"
  ssh "${host}" "status=\$(docker wait '${name}'); docker logs --timestamps '${name}' >> '${RUN_DIR}/${mode}/logs/${component}.log' 2>&1 || true; result=1; if test \"\$status\" = 0 && test -s '${output}'; then result=0; fi; docker rm '${name}' >/dev/null; exit \"\$result\""
}

stop_sampler() {
  local mode="$1" host="${SSH_HOSTS[0]}"
  ssh "${host}" "pid=\$(cat '${RUN_DIR}/${mode}/pids/migration_sampler.pid'); if kill -0 \"\$pid\" 2>/dev/null; then tr '\\0' ' ' < /proc/\$pid/cmdline | grep -F 'pd_flip_migration_measure.py sample' >/dev/null; kill -TERM \"\$pid\"; for attempt in \$(seq 1 60); do kill -0 \"\$pid\" 2>/dev/null || exit 0; sleep 1; done; exit 1; fi"
}

validate_workload() {
  local mode="$1" host="${SSH_HOSTS[0]}"
  ssh "${host}" "python3 -c \"import json; p='${RUN_DIR}/${mode}/raw/request_metrics.jsonl'; xs=[json.loads(x) for x in open(p) if x.strip()]; assert len(xs)==${TRACE_REQUESTS}, len(xs); assert all(x.get('status')=='completed' for x in xs); assert all(x.get('completion_tokens')==${TRACE_MAX_TOKENS} and x.get('completion_token_match') is True for x in xs)\""
}

summarize_measurements() {
  local mode="$1" host="${SSH_HOSTS[0]}" controller_arg=""
  if [[ "${mode}" == "state_machine" ]]; then
    controller_arg="--controller-log '${RUN_DIR}/${mode}/controller/result.json'"
  fi
  ssh "${host}" "cd '${SGLANG_REPO}' && python3 scripts/playground/disaggregation/pd_flip_migration_measure.py summarize --events-jsonl '${RUN_DIR}/${mode}/raw/migration_events.jsonl' --output-dir '${RUN_DIR}/${mode}/metrics/migration' ${controller_arg} --request-metrics-jsonl '${RUN_DIR}/${mode}/raw/request_metrics.jsonl' --errors-jsonl '${RUN_DIR}/${mode}/raw/${mode}/errors.jsonl' > '${RUN_DIR}/${mode}/metrics/migration_summary.stdout.json'"
}

finalize_controller_contract() {
  local mode="state_machine" host="${SSH_HOSTS[0]}"
  ssh "${host}" "curl -fsS -H 'Authorization: Bearer ${ADMIN_API_KEY}' 'http://127.0.0.1:${ROUTER_PORT}/pd_flip/router/workers' > '${RUN_DIR}/${mode}/controller/final_router.json'"
  ssh "${host}" "python3 -c \"import json; p='${RUN_DIR}/${mode}/controller/result.json'; q='${RUN_DIR}/${mode}/controller/final_router.json'; d=json.load(open(p)); r=json.load(open(q)); roles=[str(x.get('role','')).lower() for x in r.get('workers',[])]; d.update(first_migration_ratio=${PD_FLIP_FIRST_MIGRATION_RATIO}, observation_seconds=float(${PD_FLIP_OBSERVATION_SECONDS}), final_topology=f'{roles.count(\"prefill\")}P{roles.count(\"decode\")}D'); open(p,'w').write(json.dumps(d,indent=2,sort_keys=True)+'\\n')\""
}

run_workload() {
  local mode="$1" host="${SSH_HOSTS[0]}" ledger="${RUN_DIR}/${mode}/raw/slo_ledger.jsonl"
  start_sampler "${mode}"
  start_observer_container "${mode}" "${ledger}"
  if [[ "${mode}" == "state_machine" ]]; then
    start_controller_container "${mode}" "${ledger}"
  fi
  ssh "${host}" "cd '${SGLANG_REPO}' && python3 scripts/playground/disaggregation/pd_flip_trace_replay.py replay --trace-jsonl '${RUN_DIR}/trace/trace.jsonl' --router-url 'http://127.0.0.1:${ROUTER_PORT}' --mode '${mode}' --output-dir '${RUN_DIR}/${mode}/raw' --ledger-path '${ledger}' --timeout-seconds '${WORKLOAD_TIMEOUT_SECONDS:-7200}' --max-workers 40"
  ssh "${host}" "cp '${RUN_DIR}/${mode}/raw/${mode}/request_metrics.jsonl' '${RUN_DIR}/${mode}/raw/request_metrics.jsonl'"
  validate_workload "${mode}"
  wait_helper_container "${mode}" observer "${RUN_DIR}/${mode}/observer/summary.json"
  if [[ "${mode}" == "state_machine" ]]; then
    wait_helper_container "${mode}" controller "${RUN_DIR}/${mode}/controller/result.json"
    finalize_controller_contract
  fi
  stop_sampler "${mode}"
  summarize_measurements "${mode}"
}

collect_and_stop() {
  local mode="$1"
  ssh "${SSH_HOSTS[0]}" "docker logs --timestamps '$(router_name "${mode}")' > '${RUN_DIR}/${mode}/logs/router.docker.log' 2>&1 || true; docker stop --time 1800 '$(router_name "${mode}")' >/dev/null"
  for index in 0 1 2 3; do
    local host="${SSH_HOSTS[$index]}" name
    name="$(container_name "${mode}" "${index}")"
    ssh "${host}" "docker logs --timestamps '${name}' > '${RUN_DIR}/${mode}/logs/node${index}.docker.log' 2>&1 || true; docker stop --time 1800 '${name}' >/dev/null; ! ss -ltn | awk '{print \$4}' | grep -Eq '(:${PORT}|:${BOOTSTRAP_PORT})$'; nvidia-smi -L >/dev/null"
    if [[ "${index}" != "0" ]]; then
      ssh "${host}" "cat '${RUN_DIR}/${mode}/logs/node${index}.docker.log'" | ssh "${SSH_HOSTS[0]}" "cat > '${RUN_DIR}/${mode}/logs/node${index}.docker.log'"
    fi
  done
  ssh "${SSH_HOSTS[0]}" "cd '${SGLANG_REPO}' && python3 scripts/playground/disaggregation/pd_flip_req_timing.py --log 'node0=${RUN_DIR}/${mode}/logs/node0.docker.log' --log 'node1=${RUN_DIR}/${mode}/logs/node1.docker.log' --log 'node2=${RUN_DIR}/${mode}/logs/node2.docker.log' --log 'node3=${RUN_DIR}/${mode}/logs/node3.docker.log' --output '${RUN_DIR}/${mode}/metrics/req_time_stats.jsonl' --events-output '${RUN_DIR}/${mode}/metrics/request_stage_events.jsonl'"
}

run_one_mode() {
  local mode="$1"
  if [[ "${DRY_RUN}" == "1" ]]; then
    dry_note "${mode//_/-} parallel workers, health gates, router, observer/controller, replay, graceful stop"
    return
  fi
  ACTIVE_MODE="${mode}"
  start_mode "${mode}"
  run_workload "${mode}"
  collect_and_stop "${mode}"
  ACTIVE_MODE=""
}

baseline() { run_one_mode baseline; }
state_machine() { run_one_mode state_machine; }

compare() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    dry_note "compare raw artifacts and regenerate report"
    return
  fi
  ssh "${SSH_HOSTS[0]}" "cd '${SGLANG_REPO}' && python3 scripts/playground/disaggregation/pd_flip_ab_report.py --run-dir '${RUN_DIR}'"
}

run_all() {
  preflight
  prepare
  baseline
  state_machine
  compare
}

case "${1:-}" in
  preflight) preflight ;;
  prepare) prepare ;;
  baseline) baseline ;;
  state-machine) state_machine ;;
  compare) compare ;;
  run) run_all ;;
  *) echo "usage: $0 preflight|prepare|baseline|state-machine|compare|run" >&2; exit 2 ;;
esac
