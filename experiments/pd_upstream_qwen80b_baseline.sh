#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${ENV_FILE:-${SCRIPT_DIR}/pd_upstream_qwen80b_baseline.env.example}"

if [[ -n "${ADMIN_API_KEY_FILE:-}" ]]; then
  [[ -r "${ADMIN_API_KEY_FILE}" ]] || { echo "ADMIN_API_KEY_FILE is not readable" >&2; exit 2; }
  ADMIN_API_KEY="$(tr -d '\r\n' < "${ADMIN_API_KEY_FILE}")"
  ADMIN_API_KEY="${ADMIN_API_KEY#ADMIN_API_KEY=}"
fi

IMAGE="tiancij/sglang-upstream:v0.5.15-clean"
EXPECTED_IMAGE_ID="sha256:7dd92779d739364d79af34af65815ddc14e567728e5256f65ac922367161213e"
TRACE_SHA256="${TRACE_SHA256:-c5dbbf75c997dfc5d67a18251082f2f246d6c055eb4af5040fbe147f49f4ce5d}"
SOURCE_TRACE_SHA256="${SOURCE_TRACE_SHA256:-82da848d68c9662a7aaaf76deb547b1d8cc6c4f562586f0d60dd212bc114e964}"
TRACE_SOURCE="${TRACE_SOURCE:-${REPO_ROOT}/pd-flip-artifacts/qwen80b-trace40-natural/trace.jsonl}"
TRACE_MANIFEST_SOURCE="${TRACE_MANIFEST_SOURCE:-$(dirname "${TRACE_SOURCE}")/manifest.json}"
EXPECTED_REQUESTS=40
EXPECTED_TOKENS=10000
TRACE_INTERVAL_SECONDS="${TRACE_INTERVAL_SECONDS:-0.2}"
TRACE_LONG_TTFT_SLO_SECONDS="${TRACE_LONG_TTFT_SLO_SECONDS:-0.45}"
TRACE_SHORT_TTFT_SLO_SECONDS="${TRACE_SHORT_TTFT_SLO_SECONDS:-0.25}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-upstream-qwen80b-$(openssl rand -hex 3)}"
RUN_DIR="${REMOTE_ARTIFACT_ROOT}/${RUN_ID}"
HELPER_IMAGE="${HELPER_IMAGE:-${IMAGE}}"
WORKER_HEALTH_ATTEMPTS="${WORKER_HEALTH_ATTEMPTS:-1800}"
ROUTER_HEALTH_ATTEMPTS="${ROUTER_HEALTH_ATTEMPTS:-300}"
HEALTH_POLL_SECONDS="${HEALTH_POLL_SECONDS:-2}"
WORKLOAD_TIMEOUT_SECONDS="${WORKLOAD_TIMEOUT_SECONDS:-7200}"

SSH_HOSTS=("${NODE0_HOST}" "${NODE1_HOST}" "${NODE2_HOST}" "${NODE3_HOST}")
NODE_IPS=("${NODE0_IP}" "${NODE1_IP}" "${NODE2_IP}" "${NODE3_IP}")
MOONCAKE_HOSTS=("${NODE0_MOONCAKE_HOST}" "${NODE1_MOONCAKE_HOST}" "${NODE2_MOONCAKE_HOST}" "${NODE3_MOONCAKE_HOST}")
ROLES=("${NODE0_ROLE}" "${NODE1_ROLE}" "${NODE2_ROLE}" "${NODE3_ROLE}")
ACTIVE=0

worker_name() { local index="$1"; printf 'tiancij-upstream-%s-node%s' "${RUN_ID}" "${index}"; }
router_name() { printf 'tiancij-upstream-%s-router' "${RUN_ID}"; }
helper_name() { printf 'tiancij-upstream-%s-helper' "${RUN_ID}"; }
builder_name() { printf 'tiancij-upstream-router-build-%s' "${RUN_ID}"; }

require_secret() {
  case "${ADMIN_API_KEY:-}" in
    ""|replace-with-*|changeme|CHANGE_ME)
      echo "ADMIN_API_KEY must come from a private ENV_FILE" >&2
      return 2
      ;;
  esac
}

assert_fixed_config() {
  [[ "${GPU_IDS}" == "0,1" ]] || { echo "GPU_IDS must be 0,1" >&2; return 2; }
  [[ "${TP_SIZE}" == "2" && "${DP_SIZE}" == "1" ]] || { echo "TP_SIZE=2 and DP_SIZE=1 are required" >&2; return 2; }
  [[ "${ROLES[*]}" == "prefill decode decode decode" ]] || { echo "roles must be 1P3D" >&2; return 2; }
  [[ "${IB_DEVICE}" == "mlx5_bond_1" ]] || { echo "IB_DEVICE must be mlx5_bond_1" >&2; return 2; }
  [[ "${MC_USE_IPV6}" == "1" && "${MC_GID_INDEX}" == "3" ]] || { echo "Mooncake bond1 requires MC_USE_IPV6=1 and MC_GID_INDEX=3" >&2; return 2; }
  [[ "${MEM_FRACTION_STATIC}" == "0.80" ]] || { echo "MEM_FRACTION_STATIC must be 0.80" >&2; return 2; }
  [[ "${RUN_ID}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo "invalid RUN_ID" >&2; return 2; }
}

dry_run() {
  assert_fixed_config
  echo "run_id=${RUN_ID} image=${IMAGE} image_id=${EXPECTED_IMAGE_ID}"
  echo "trace=${TRACE_SOURCE} trace_sha256=${TRACE_SHA256} requests=${EXPECTED_REQUESTS} max_tokens=${EXPECTED_TOKENS}"
  for index in 0 1 2 3; do
    echo "node${index} ${ROLES[$index]} ${SSH_HOSTS[$index]} http=${NODE_IPS[$index]} mooncake=${MOONCAKE_HOSTS[$index]} ib=${IB_DEVICE} gid=${MC_GID_INDEX} GPUs=${GPU_IDS} TP=${TP_SIZE} DP=${DP_SIZE}"
  done
  echo "router $(router_name) workers=${NODE_IPS[*]} port=${ROUTER_PORT}"
  echo "helper $(helper_name) mounts host helper code at /work/sglang read-only; no GPU"
  echo "worker/router source=/sgl-workspace/sglang from clean image; host source mount=none"
  echo "raw=slo_ledger.jsonl,request_metrics.jsonl,responses.jsonl,errors.jsonl,tpot_tokens.csv"
  echo "validation=${EXPECTED_REQUESTS} requests x ${EXPECTED_TOKENS} usage tokens; dynamic SSE event counts; report=TTFT+token-normalized-TPOT"
}

preflight() {
  require_secret
  assert_fixed_config
  [[ -f "${TRACE_SOURCE}" ]] || { echo "missing trace: ${TRACE_SOURCE}" >&2; return 2; }
  [[ -f "${TRACE_MANIFEST_SOURCE}" ]] || { echo "missing trace manifest: ${TRACE_MANIFEST_SOURCE}" >&2; return 2; }
  [[ "$(sha256sum "${TRACE_SOURCE}" | awk '{print $1}')" == "${TRACE_SHA256}" ]] || { echo "local trace hash mismatch" >&2; return 2; }
  python3 - "${TRACE_SOURCE}" "${TRACE_INTERVAL_SECONDS}" "${TRACE_LONG_TTFT_SLO_SECONDS}" "${TRACE_SHORT_TTFT_SLO_SECONDS}" <<'PY'
import json, sys
path, interval, long_slo, short_slo = sys.argv[1], *map(float, sys.argv[2:])
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
assert len(rows) == 40
assert [row["arrival_offset_s"] for row in rows] == [round(i * interval, 9) for i in range(40)]
assert all(row["ttft_slo_s"] == long_slo for row in rows[::2])
assert all(row["ttft_slo_s"] == short_slo for row in rows[1::2])
assert all(row["body"]["custom_params"]["pd_flip_slo"]["ttft_seconds"] == row["ttft_slo_s"] for row in rows)
PY
  local expected_model="" index host image_id model_hash name gpu_list
  gpu_list="${GPU_IDS//,/ }"
  for index in 0 1 2 3; do
    host="${SSH_HOSTS[$index]}"
    name="$(worker_name "${index}")"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${host}" true
    ssh "${host}" "docker inspect '${name}' >/dev/null 2>&1 && { echo owned-name collision: '${name}' >&2; exit 1; } || true"
    if [[ "${index}" == "0" ]]; then
      ssh "${host}" "docker inspect '$(router_name)' >/dev/null 2>&1 && { echo owned-name collision: '$(router_name)' >&2; exit 1; } || true"
    fi
    image_id="$(ssh "${host}" "docker image inspect '${IMAGE}' --format '{{.Id}}'")"
    [[ "${image_id}" == "${EXPECTED_IMAGE_ID}" ]] || { echo "${host}: clean image ID mismatch" >&2; return 2; }
    ssh "${host}" "python3 -c \"import json; from pathlib import Path; p=Path('${MODEL_PATH}'); d=json.loads((p/'model.safetensors.index.json').read_text()); missing=[x for x in set(d['weight_map'].values()) if not (p/x).is_file()]; assert (p/'config.json').is_file() and (p/'tokenizer.json').is_file() and not missing, missing\""
    model_hash="$(ssh "${host}" "{ sha256sum '${MODEL_PATH}/config.json' '${MODEL_PATH}/tokenizer.json'; find '${MODEL_PATH}' -maxdepth 1 -type f -name '*.safetensors' -printf '%f:%s\\n' | LC_ALL=C sort; } | sha256sum | awk '{print \$1}'")"
    if [[ -z "${expected_model}" ]]; then expected_model="${model_hash}"; fi
    [[ "${model_hash}" == "${expected_model}" ]] || { echo "${host}: model fingerprint mismatch" >&2; return 2; }
    ssh "${host}" "for gpu in ${gpu_list}; do test -z \"\$(nvidia-smi -i \"\$gpu\" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -E '^[[:space:]]*[0-9]+' || true)\" || { echo GPU \$gpu busy >&2; exit 1; }; done"
    ssh "${host}" "! ss -ltn | awk '{print \$4}' | grep -Eq '(:${WORKER_PORT}|:${BOOTSTRAP_PORT})$'"
    ssh "${host}" "ibv_devinfo -d '${IB_DEVICE}' >/dev/null; show_gids | python3 -c \"import ipaddress,sys; rows=[x.split() for x in sys.stdin if x.startswith('${IB_DEVICE}')]; assert any(x[2]=='${MC_GID_INDEX}' and ipaddress.ip_address(x[3])==ipaddress.ip_address('${MOONCAKE_HOSTS[$index]}') for x in rows)\"; ip -6 addr show dev bond1 | grep -F '${MOONCAKE_HOSTS[$index]}/' >/dev/null"
    ssh "${host}" "docker_root=\$(docker info --format '{{.DockerRootDir}}'); nvidia-smi -L; nvidia-smi --query-gpu=index,uuid,memory.total,memory.used --format=csv,noheader; df -h '${MODEL_PATH}' \"\${docker_root}\"; docker ps --format '{{.Names}} {{.Status}} {{.Image}}'; ps -eo pid,user,comm --sort=pid | grep -E 'sglang|mooncake|sgl-router' || true; ip -brief address; show_gids | grep -E '^${IB_DEVICE}[[:space:]]+1[[:space:]]+${MC_GID_INDEX}[[:space:]]+'; ibv_devinfo -d '${IB_DEVICE}' | sed -n '1,80p'; date --iso-8601=ns; chronyc tracking 2>/dev/null || true"
  done
  ssh "${SSH_HOSTS[0]}" "! ss -ltn | awk '{print \$4}' | grep -Eq '(:${ROUTER_PORT})$'"
  echo "preflight passed; model_fingerprint=${expected_model}"
}

build_router() {
  require_secret
  assert_fixed_config
  local host="${SSH_HOSTS[0]}" name
  name="$(builder_name)"
  ssh "${host}" bash -s -- "${name}" "${IMAGE}" "${EXPECTED_IMAGE_ID}" "${ROUTER_ARTIFACT_DIR}" <<'REMOTE'
set -euo pipefail
name="$1"; image="$2"; expected_image_id="$3"; output="$4"
test "$(docker image inspect "$image" --format '{{.Id}}')" = "$expected_image_id"
if test -x "$output/sgl-router" && test -s "$output/provenance.json"; then
  python3 -c "import json; d=json.load(open('$output/provenance.json')); assert d['image_id']=='$expected_image_id'"
  sha256sum "$output/sgl-router"
  exit 0
fi
if docker inspect "$name" >/dev/null 2>&1; then
  echo "router builder name collision: $name" >&2
  exit 2
fi
mkdir -p "$output"
docker run --name "$name" --network host -v "$output:/out" "$image" bash -lc '
  set -euo pipefail
  if ! command -v cargo >/dev/null 2>&1; then
    curl --fail --location --retry 5 https://sh.rustup.rs | sh -s -- -y --profile minimal
    source "$HOME/.cargo/env"
  fi
  cd /sgl-workspace/sglang/experimental/sgl-router
  if test -f Cargo.lock; then cargo build --release --locked; else cargo build --release; fi
  install -m 0755 target/release/sgl-router /out/sgl-router
  sha256sum /out/sgl-router
' >"$output/build-${name}.log" 2>&1
docker inspect "$name" >"$output/build-${name}.inspect.json"
docker rm "$name" >/dev/null
router_sha="$(sha256sum "$output/sgl-router" | awk '{print $1}')"
printf '{"image_id":"%s","router_sha256":"%s","source":"/sgl-workspace/sglang/experimental/sgl-router"}\n' "$expected_image_id" "$router_sha" >"$output/provenance.json"
cat "$output/provenance.json"
REMOTE
}

prepare() {
  require_secret
  assert_fixed_config
  local host="${SSH_HOSTS[0]}" index model_hash router_sha
  for index in 0 1 2 3; do
    ssh "${SSH_HOSTS[$index]}" "umask 077; mkdir -p '${RUN_DIR}/node${index}'"
  done
  ssh "${host}" "mkdir -p '${RUN_DIR}/trace' '${RUN_DIR}/raw' '${RUN_DIR}/logs' '${RUN_DIR}/inspect' '${RUN_DIR}/status' '${RUN_DIR}/smoke' '${RUN_DIR}/report'"
  scp "${TRACE_SOURCE}" "${host}:${RUN_DIR}/trace/trace.jsonl" >/dev/null
  scp "${TRACE_MANIFEST_SOURCE}" "${host}:${RUN_DIR}/trace/source_manifest.json" >/dev/null
  ssh "${host}" "test \"\$(sha256sum '${RUN_DIR}/trace/trace.jsonl' | awk '{print \$1}')\" = '${TRACE_SHA256}'"
  router_sha="$(ssh "${host}" "sha256sum '${ROUTER_ARTIFACT_DIR}/sgl-router' | awk '{print \$1}'")"
  model_hash="$(ssh "${host}" "{ sha256sum '${MODEL_PATH}/config.json' '${MODEL_PATH}/tokenizer.json'; find '${MODEL_PATH}' -maxdepth 1 -type f -name '*.safetensors' -printf '%f:%s\\n' | LC_ALL=C sort; } | sha256sum | awk '{print \$1}'")"
  ssh "${host}" "python3 -c \"import json; d={'run_id':'${RUN_ID}','mode':'upstream_baseline','validity':'pending','image':'${IMAGE}','image_id':'${EXPECTED_IMAGE_ID}','trace_sha256':'${TRACE_SHA256}','source_trace_sha256':'${SOURCE_TRACE_SHA256}','model_id':'${MODEL_ID}','model_fingerprint':'${model_hash}','router_sha256':'${router_sha}','topology':'1P3D','gpu_ids':'${GPU_IDS}','tp_size':${TP_SIZE},'dp_size':${DP_SIZE},'worker_port':${WORKER_PORT},'router_port':${ROUTER_PORT},'bootstrap_port':${BOOTSTRAP_PORT},'ib_device':'${IB_DEVICE}','mc_use_ipv6':${MC_USE_IPV6},'mc_gid_index':${MC_GID_INDEX},'mooncake_hosts':['${MOONCAKE_HOSTS[0]}','${MOONCAKE_HOSTS[1]}','${MOONCAKE_HOSTS[2]}','${MOONCAKE_HOSTS[3]}'],'mooncake_metadata':'P2PHANDSHAKE','mem_fraction_static':${MEM_FRACTION_STATIC},'output_contract':'natural','tpot_metric_source':'client_first_last_output_over_usage_completion_tokens','client_instrumentation':'time.monotonic streaming receive events plus usage.completion_tokens'}; open('${RUN_DIR}/manifest.json','w').write(json.dumps(d,indent=2,sort_keys=True)+'\\n')\""
}

write_secret_env() {
  local index="$1" host="${SSH_HOSTS[$index]}" encoded
  encoded="$(printf 'ADMIN_API_KEY=%s\n' "${ADMIN_API_KEY}" | base64 | tr -d '\n')"
  ssh "${host}" "umask 077; printf '%s' '${encoded}' | base64 -d > '${RUN_DIR}/node${index}/worker.env'"
}

start_worker() {
  local index="$1" host="${SSH_HOSTS[$index]}" ip="${NODE_IPS[$index]}" role="${ROLES[$index]}" moon_host="${MOONCAKE_HOSTS[$index]}" name
  name="$(worker_name "${index}")"
  write_secret_env "${index}"
  ssh "${host}" bash -s -- "${name}" "${RUN_ID}" "${IMAGE}" "${EXPECTED_IMAGE_ID}" "${MODEL_PATH}" "${MODEL_ID}" "${ip}" "${role}" "${WORKER_PORT}" "${TP_SIZE}" "${DP_SIZE}" "${BOOTSTRAP_PORT}" "${IB_DEVICE}" "${MEM_FRACTION_STATIC}" "${GPU_IDS}" "${MC_GID_INDEX}" "${MOONCAKE_PROTOCOL}" "${moon_host}" "${MC_USE_IPV6}" "${RUN_DIR}/node${index}/worker.env" <<'REMOTE'
set -euo pipefail
name="$1"; run_id="$2"; image="$3"; expected_image_id="$4"; model_path="$5"; model_id="$6"; ip="$7"; role="$8"; port="$9"; shift 9
tp="$1"; dp="$2"; bootstrap="$3"; ib="$4"; mem="$5"; gpus="$6"; gid="$7"; protocol="$8"; moon_host="$9"; use_ipv6="${10}"; env_file="${11}"
test "$(docker image inspect "$image" --format '{{.Id}}')" = "$expected_image_id"
test -d /dev/infiniband
docker run -d --name "$name" \
  --label tiancij.experiment=pd-upstream-qwen80b --label "tiancij.run_id=$run_id" \
  --gpus "\"device=$gpus\"" --network host --ipc host --privileged \
  --env-file "$env_file" -e "CUDA_VISIBLE_DEVICES=$gpus" -e "MOONCAKE_LOCAL_HOSTNAME=$moon_host" \
  -e "SGLANG_HOST_IP=$moon_host" \
  -e "MOONCAKE_PROTOCOL=$protocol" -e "MC_USE_IPV6=$use_ipv6" -e "MC_GID_INDEX=$gid" \
  -v "$model_path:$model_path:ro" -v /dev/infiniband:/dev/infiniband \
  "$image" bash -lc 'cd /sgl-workspace/sglang && exec python3 -m sglang.launch_server \
    --model-path '"$model_path"' --served-model-name '"$model_id"' --host '"$ip"' --port '"$port"' \
    --tp-size '"$tp"' --dp-size '"$dp"' --disaggregation-mode '"$role"' \
    --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port '"$bootstrap"' \
    --disaggregation-ib-device '"$ib"' --mem-fraction-static '"$mem"' \
    --enable-request-time-stats-logging \
    --trust-remote-code --mamba-scheduler-strategy extra_buffer --enable-metrics \
    --admin-api-key "${ADMIN_API_KEY}"'
REMOTE
}

wait_worker() {
  local index="$1" host="${SSH_HOSTS[$index]}" ip="${NODE_IPS[$index]}"
  ssh "${host}" "for attempt in \$(seq 1 \"${WORKER_HEALTH_ATTEMPTS}\"); do curl -fsS 'http://${ip}:${WORKER_PORT}/health' >/dev/null && exit 0; sleep '${HEALTH_POLL_SECONDS}'; done; docker logs --tail 200 '$(worker_name "${index}")' >&2; exit 1"
}

save_worker_inspect() {
  local index="$1" host="${SSH_HOSTS[$index]}" name
  name="$(worker_name "${index}")"
  ssh "${host}" "docker inspect '${name}' | sed -E 's/(ADMIN_API_KEY=)[^\"]+/\1<redacted>/g'" | \
    ssh "${SSH_HOSTS[0]}" "cat > '${RUN_DIR}/inspect/node${index}.json'"
  ssh "${host}" "docker inspect '${name}' --format '{{json .Config.Cmd}} {{json .Config.Env}}' | sed -E 's/(ADMIN_API_KEY=)[^\"]+/\1<redacted>/g'" | \
    ssh "${SSH_HOSTS[0]}" "cat > '${RUN_DIR}/inspect/node${index}.effective.txt'"
  ssh "${host}" "docker inspect '${name}' --format '{{json .Mounts}}' | grep -F '/sgl-workspace/sglang' && exit 1 || true"
}

start_router() {
  local host="${SSH_HOSTS[0]}" name
  name="$(router_name)"
  ssh "${host}" bash -s -- "${name}" "${RUN_ID}" "${IMAGE}" "${EXPECTED_IMAGE_ID}" "${ROUTER_ARTIFACT_DIR}/sgl-router" "${MODEL_PATH}" "${TOKENIZER_PATH}" "${MODEL_ID}" "${ROUTER_PORT}" "${WORKER_PORT}" "${NODE_IPS[0]}" "${NODE_IPS[1]}" "${NODE_IPS[2]}" "${NODE_IPS[3]}" <<'REMOTE'
set -euo pipefail
name="$1"; run_id="$2"; image="$3"; expected_image_id="$4"; binary="$5"; model_path="$6"; tokenizer="$7"; model_id="$8"; router_port="$9"; shift 9
worker_port="$1"; ip0="$2"; ip1="$3"; ip2="$4"; ip3="$5"
test "$(docker image inspect "$image" --format '{{.Id}}')" = "$expected_image_id"
docker run -d --name "$name" --network host \
  --label tiancij.experiment=pd-upstream-qwen80b --label "tiancij.run_id=$run_id" \
  -v "$binary:/opt/tiancij/bin/sgl-router:ro" -v "$model_path:$model_path:ro" \
  "$image" /opt/tiancij/bin/sgl-router --host 0.0.0.0 --port "$router_port" \
  --model-id "$model_id" --tokenizer-path "$tokenizer" --request-timeout-secs 7200 \
  --worker-urls "http://$ip0:$worker_port" "http://$ip1:$worker_port" "http://$ip2:$worker_port" "http://$ip3:$worker_port"
REMOTE
  ssh "${host}" "for attempt in \$(seq 1 \"${ROUTER_HEALTH_ATTEMPTS}\"); do curl -fsS 'http://127.0.0.1:${ROUTER_PORT}/v1/models' >/dev/null && exit 0; sleep '${HEALTH_POLL_SECONDS}'; done; docker logs --tail 200 '${name}' >&2; exit 1"
  ssh "${host}" "docker inspect '${name}' > '${RUN_DIR}/inspect/router.json'; docker inspect '${name}' --format '{{json .Config.Cmd}} {{json .Config.Env}}' > '${RUN_DIR}/inspect/router.effective.txt'; docker inspect '${name}' --format '{{json .Mounts}}' | grep -F '/sgl-workspace/sglang' && exit 1 || true"
}

start_all() {
  require_secret
  ACTIVE=1
  local pids=() index pid
  for index in 0 1 2 3; do
    start_worker "${index}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}"; done
  for index in 0 1 2 3; do wait_worker "${index}"; save_worker_inspect "${index}"; done
  start_router
}

smoke() {
  require_secret
  local host="${SSH_HOSTS[0]}" index
  ssh "${host}" bash -s -- "${RUN_DIR}/node0/worker.env" "${ROUTER_PORT}" "${MODEL_ID}" "${RUN_ID}" "${RUN_DIR}" <<'REMOTE'
set -euo pipefail
env_file="$1"; router_port="$2"; model_id="$3"; run_id="$4"; run_dir="$5"
set -a
source "$env_file"
set +a
python3 - "$router_port" "$model_id" "$run_id" "$run_dir" <<'PY'
import json
import hashlib
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

router_port, model_id, run_id, run_dir = sys.argv[1:]
key = os.environ["ADMIN_API_KEY"]
url = f"http://127.0.0.1:{router_port}/v1/chat/completions"
base = {"model": model_id, "max_tokens": 32, "stream": False, "ignore_eos": True}
for index in range(2):
    body = dict(
        base,
        messages=[
            {
                "role": "user",
                "content": f"upstream-smoke-{run_id}-{index}-never-in-formal-trace",
            }
        ],
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        output = json.load(response)
    assert output.get("choices") and output["choices"][0].get("message", {}).get("content")
    with open(f"{run_dir}/smoke/response-{index}.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle)
        handle.write("\n")

probe_body = {
    "model": model_id,
    "messages": [
        {
            "role": "user",
            "content": f"upstream-natural-10k-probe-{run_id}-never-in-formal-trace. Continue writing plain text without stopping.",
        }
    ],
    "max_tokens": 10000,
    "stream": True,
    "stream_options": {"include_usage": True},
    "ignore_eos": True,
    "stop": None,
    "temperature": 0,
}
probe_request = urllib.request.Request(
    url,
    data=json.dumps(probe_body).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
)
started = time.monotonic()
first_output = None
last_output = None
output_events = 0
output_bytes = 0
digest = hashlib.sha256()
completion_tokens = None
finish_reason = None
with urllib.request.urlopen(probe_request, timeout=7200) as response:
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        event = json.loads(line[6:])
        usage = event.get("usage") or {}
        if usage.get("completion_tokens") is not None:
            completion_tokens = int(usage["completion_tokens"])
        choices = event.get("choices") or []
        if choices and choices[0].get("finish_reason") is not None:
            finish_reason = choices[0]["finish_reason"]
        content = ""
        if choices:
            delta = choices[0].get("delta") or {}
            content = delta.get("content") or ""
        if content:
            now = time.monotonic()
            if first_output is None:
                first_output = now
            last_output = now
            output_events += 1
            encoded = content.encode("utf-8")
            output_bytes += len(encoded)
            digest.update(encoded)
completion_tokens = int(completion_tokens or -1)
assert completion_tokens == 10000, completion_tokens
assert finish_reason == "length", finish_reason
assert first_output is not None and last_output is not None and output_events > 0
probe_summary = {
    "completion_tokens": completion_tokens,
    "finish_reason": finish_reason,
    "output_events": output_events,
    "output_bytes": output_bytes,
    "output_sha256": digest.hexdigest(),
    "ttft_s": first_output - started,
    "output_span_s": last_output - first_output,
    "token_normalized_tpot_s": (last_output - first_output) / (completion_tokens - 1),
    "metric_source": "client_first_last_output_over_usage_completion_tokens",
}
with open(f"{run_dir}/smoke/natural-10k-probe.json", "w", encoding="utf-8") as handle:
    json.dump(probe_summary, handle, indent=2, sort_keys=True)
    handle.write("\n")

trace_path = os.path.join(run_dir, "trace", "trace.jsonl")
with open(trace_path, encoding="utf-8") as handle:
    trace_rows = [json.loads(line) for line in handle if line.strip()]
warmup_kinds = ("long", "short")
selected_rows = {
    prompt_kind: next(row for row in trace_rows if row["prompt_kind"] == prompt_kind)
    for prompt_kind in warmup_kinds
}


def run_prefill_warmup(prompt_kind, trace_row):
    warmup_body = dict(trace_row["body"])
    warmup_body.pop("custom_params", None)
    warmup_body["max_tokens"] = 1
    warmup_body["stream"] = True
    warmup_body["stream_options"] = {"include_usage": True}
    warmup_request = urllib.request.Request(
        url,
        data=json.dumps(warmup_body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
        },
    )
    started_utc = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    first_output_utc = None
    first_output_monotonic = None
    completion_tokens = None
    prompt_tokens = None
    finish_reason = None
    response_status = None
    with urllib.request.urlopen(warmup_request, timeout=600) as response:
        response_status = response.status
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or {}
            if usage.get("prompt_tokens") is not None:
                prompt_tokens = int(usage["prompt_tokens"])
            if usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])
            choices = event.get("choices") or []
            if choices and choices[0].get("finish_reason") is not None:
                finish_reason = choices[0]["finish_reason"]
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or delta.get("reasoning_content") or ""
                if content and first_output_monotonic is None:
                    first_output_monotonic = time.monotonic()
                    first_output_utc = datetime.now(timezone.utc)
    finished_monotonic = time.monotonic()
    finished_utc = datetime.now(timezone.utc)
    assert response_status == 200, response_status
    assert first_output_monotonic is not None and first_output_utc is not None
    assert completion_tokens == 1, completion_tokens
    assert prompt_tokens is not None, prompt_tokens
    assert finish_reason == "length", finish_reason
    return {
        "trace_request_id": trace_row["request_id"],
        "trace_prompt_kind": prompt_kind,
        "trace_prompt_chars": trace_row.get("prompt_chars"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "response_status": response_status,
        "started_utc": started_utc.isoformat(),
        "first_output_utc": first_output_utc.isoformat(),
        "finished_utc": finished_utc.isoformat(),
        "ttft_s": first_output_monotonic - started_monotonic,
        "total_duration_s": finished_monotonic - started_monotonic,
        "measured": False,
        "kv_cache_flushed_after": True,
    }


warmup_window_start = datetime.now(timezone.utc) - timedelta(seconds=2)
warmup_results = {}
for prompt_kind in warmup_kinds:
    result = run_prefill_warmup(prompt_kind, selected_rows[prompt_kind])
    prompt_tokens = result["prompt_tokens"]
    if prompt_kind == "long":
        assert prompt_tokens > 6000, prompt_tokens
    else:
        assert 500 <= prompt_tokens <= 1000, prompt_tokens
    warmup_results[prompt_kind] = result

warmup_window_end = datetime.now(timezone.utc) + timedelta(seconds=2)
log_window_start = warmup_window_start.isoformat()
log_window_end = warmup_window_end.isoformat()
for result in warmup_results.values():
    result["log_window_start_utc"] = log_window_start
    result["log_window_end_utc"] = log_window_end

warmup_files = {
    "long": "long-prefill-warmup.json",
    "short": "short-prefill-warmup.json",
}
for prompt_kind in warmup_kinds:
    with open(
        f"{run_dir}/smoke/{warmup_files[prompt_kind]}", "w", encoding="utf-8"
    ) as handle:
        json.dump(warmup_results[prompt_kind], handle, indent=2, sort_keys=True)
        handle.write("\n")
PY
REMOTE
  local warmup_since warmup_until
  read -r warmup_since warmup_until < <(
    ssh "${host}" "python3 -c \"import json; a=json.load(open('${RUN_DIR}/smoke/long-prefill-warmup.json')); b=json.load(open('${RUN_DIR}/smoke/short-prefill-warmup.json')); print(a['log_window_start_utc'], b['log_window_end_utc'])\""
  )
  sleep 2
  for index in 0 1 2 3; do
    ssh "${SSH_HOSTS[$index]}" "test \"\$(docker inspect '$(worker_name "${index}")' --format '{{index .Config.Labels \"tiancij.run_id\"}}')\" = '${RUN_ID}'; docker logs --timestamps --since '${warmup_since}' --until '${warmup_until}' '$(worker_name "${index}")' 2>&1" | \
      ssh "${host}" "cat > '${RUN_DIR}/logs/warmup-node${index}.docker.log'"
  done
  ssh "${host}" "test \"\$(docker inspect '$(router_name)' --format '{{index .Config.Labels \"tiancij.run_id\"}}')\" = '${RUN_ID}'; docker logs --timestamps --since '${warmup_since}' --until '${warmup_until}' '$(router_name)' 2>&1" | \
    ssh "${host}" "cat > '${RUN_DIR}/logs/warmup-router.docker.log'"
  local flush_ok=1 flush_response
  for index in 0 1 2 3; do
    if ! flush_response="$(ssh "${SSH_HOSTS[$index]}" bash -s -- "${RUN_DIR}/node${index}/worker.env" "http://${NODE_IPS[$index]}:${WORKER_PORT}/flush_cache" <<'REMOTE'
set -euo pipefail
env_file="$1"; url="$2"
set -a
source "$env_file"
set +a
curl -fsS -X POST -H "Authorization: Bearer ${ADMIN_API_KEY}" "$url"
REMOTE
)"; then
      flush_ok=0
    else
      printf '%s\n' "${flush_response}" | ssh "${host}" "cat > '${RUN_DIR}/smoke/flush-node${index}.json'"
    fi
  done
  if [[ "${flush_ok}" != "1" ]]; then
    echo "post-warmup cache flush failed; preserving forensic run without relaunch" >&2
    return 1
  fi
  ssh "${host}" "printf '%s\n' 'process warm; KV cold after successful dual-prefill warmup flush' > '${RUN_DIR}/smoke/cold-state.txt'"
}

measure() {
  require_secret
  local host="${SSH_HOSTS[0]}" name encoded
  name="$(helper_name)"
  encoded="$(printf 'ADMIN_API_KEY=%s\n' "${ADMIN_API_KEY}" | base64 | tr -d '\n')"
  ssh "${host}" "umask 077; printf '%s' '${encoded}' | base64 -d > '${RUN_DIR}/helper.env'"
  ssh "${host}" bash -s -- "${name}" "${RUN_ID}" "${HELPER_IMAGE}" "${REMOTE_HELPER_REPO}" "${RUN_DIR}" "${MODEL_ID}" "${ROUTER_PORT}" "${WORKLOAD_TIMEOUT_SECONDS}" <<'REMOTE'
set -euo pipefail
name="$1"; run_id="$2"; image="$3"; repo="$4"; run_dir="$5"; model="$6"; router_port="$7"; timeout="$8"
REMOTE_HELPER_REPO="$repo"
docker run --name "$name" --network host --env-file "$run_dir/helper.env" \
  --label tiancij.experiment=pd-upstream-qwen80b --label "tiancij.run_id=$run_id" \
  -v "${REMOTE_HELPER_REPO}:/work/sglang:ro" -v "$run_dir:/run" \
  "$image" bash -lc 'cd /work/sglang && exec python3 scripts/playground/disaggregation/pd_flip_trace_replay.py replay \
    --trace-jsonl /run/trace/trace.jsonl --router-url http://127.0.0.1:'"$router_port"' \
    --mode upstream_baseline --output-dir /run/raw --ledger-path /run/raw/slo_ledger.jsonl \
    --timeout-seconds '"$timeout"' --max-workers 40 \
    --api-key "${ADMIN_API_KEY}"' \
  >"$run_dir/logs/client.log" 2>&1
docker inspect "$name" | sed -E 's/(ADMIN_API_KEY=)[^"]+/\1<redacted>/g' >"$run_dir/inspect/helper.json"
docker rm "$name" >/dev/null
REMOTE
  validate_raw
}

validate_raw() {
  local host="${SSH_HOSTS[0]}"
  ssh "${host}" "python3 -c \"import csv,json; from pathlib import Path; root=Path('${RUN_DIR}/raw'); mode=root/'upstream_baseline'; rows=[json.loads(x) for x in (mode/'request_metrics.jsonl').read_text().splitlines() if x.strip()]; ids=[x['request_id'] for x in rows]; errors=[x for x in (mode/'errors.jsonl').read_text().splitlines() if x.strip()]; ledger=sum(1 for x in (root/'slo_ledger.jsonl').open() if x.strip()); gaps=list(csv.DictReader((mode/'tpot_tokens.csv').open())); expected_source='client_first_last_output_over_usage_completion_tokens'; assert len(rows)==${EXPECTED_REQUESTS}; assert len(set(ids))==${EXPECTED_REQUESTS}; assert not errors; assert all(x.get('status')=='completed' and x.get('completion_tokens')==${EXPECTED_TOKENS} and x.get('completion_token_match') is True and x.get('finish_reason')=='length' and x.get('tpot_metric_source')==expected_source and x.get('avg_tpot_s') is not None and x.get('ttft_s') is not None for x in rows); assert ledger>=${EXPECTED_REQUESTS}*2, ledger; assert gaps; assert set(ids)<=set(x['request_id'] for x in gaps)\""
}

safe_stop_container() {
  local host="$1" name="$2"
  ssh "${host}" bash -s -- "${name}" "${RUN_ID}" <<'REMOTE'
set -euo pipefail
name="$1"; run_id="$2"
if ! docker inspect "$name" >/dev/null 2>&1; then exit 0; fi
test "$(docker inspect "$name" --format '{{index .Config.Labels "tiancij.run_id"}}')" = "$run_id"
status="$(docker inspect "$name" --format '{{.State.Status}}')"
case "$status" in
  running|paused|restarting) docker stop --time 1800 "$name" >/dev/null ;;
esac
docker rm "${name}" >/dev/null
REMOTE
}

stop_inference() {
  local index
  safe_stop_container "${SSH_HOSTS[0]}" "$(router_name)"
  for index in 0 1 2 3; do safe_stop_container "${SSH_HOSTS[$index]}" "$(worker_name "${index}")"; done
}

collect_logs() {
  local index host name
  for index in 0 1 2 3; do
    host="${SSH_HOSTS[$index]}"; name="$(worker_name "${index}")"
    ssh "${host}" "docker logs --timestamps '${name}' 2>&1 || true" | \
      sed -E "s/(admin_api_key=)'[^']*'/\1'<redacted>'/g; s/(ADMIN_API_KEY=)[^\" ,]+/\1<redacted>/g" | \
      ssh "${SSH_HOSTS[0]}" "cat > '${RUN_DIR}/logs/node${index}.docker.log'"
    ssh "${host}" "docker inspect '${name}' | sed -E 's/(ADMIN_API_KEY=)[^\"]+/\1<redacted>/g' || true" | ssh "${SSH_HOSTS[0]}" "cat > '${RUN_DIR}/inspect/node${index}.final.json'"
  done
  ssh "${SSH_HOSTS[0]}" "docker logs --timestamps '$(router_name)' 2>&1 || true" | \
    sed -E "s/(admin_api_key=)'[^']*'/\1'<redacted>'/g; s/(ADMIN_API_KEY=)[^\" ,]+/\1<redacted>/g" | \
    ssh "${SSH_HOSTS[0]}" "cat > '${RUN_DIR}/logs/router.docker.log'"
  ssh "${SSH_HOSTS[0]}" "docker inspect '$(router_name)' > '${RUN_DIR}/inspect/router.final.json' 2>/dev/null || true"
}

cleanup_secret_files() {
  local index
  ssh "${SSH_HOSTS[0]}" "rm -f -- '${RUN_DIR}/helper.env'"
  for index in 0 1 2 3; do
    ssh "${SSH_HOSTS[$index]}" "rm -f -- '${RUN_DIR}/node${index}/worker.env'"
  done
}

collect_stop() {
  collect_logs
  stop_inference
  cleanup_secret_files
  local index host
  for index in 0 1 2 3; do
    host="${SSH_HOSTS[$index]}"
    ssh "${host}" "! ss -ltn | awk '{print \$4}' | grep -Eq '(:${WORKER_PORT}|:${BOOTSTRAP_PORT})$'; nvidia-smi -L; date --iso-8601=ns" | \
      ssh "${SSH_HOSTS[0]}" "cat > '${RUN_DIR}/status/node${index}-after.txt'"
  done
  ssh "${SSH_HOSTS[0]}" "! ss -ltn | awk '{print \$4}' | grep -Eq '(:${ROUTER_PORT})$'"
  ACTIVE=0
}

report() {
  local host="${SSH_HOSTS[0]}" name="$(helper_name)-report"
  ssh "${host}" "status=0; docker run --name '${name}' --network none --label tiancij.experiment=pd-upstream-qwen80b --label 'tiancij.run_id=${RUN_ID}' -v '${REMOTE_HELPER_REPO}:/work/sglang:ro' -v '${RUN_DIR}:/run' '${HELPER_IMAGE}' bash -lc 'cd /work/sglang && python3 scripts/playground/disaggregation/pd_upstream_baseline_report.py --run-dir /run' > '${RUN_DIR}/logs/report.log' 2>&1 || status=\$?; docker rm '${name}' >/dev/null; exit \"\$status\""
  ssh "${host}" "python3 -c \"import json; p='${RUN_DIR}/manifest.json'; d=json.load(open(p)); d['validity']='valid'; open(p,'w').write(json.dumps(d,indent=2,sort_keys=True)+'\\n')\"; find '${RUN_DIR}' -type f ! -name INVENTORY.txt -print0 | sort -z | xargs -0 sha256sum > '${RUN_DIR}/INVENTORY.txt'"
}

on_failure() {
  local status="$1"
  trap - ERR INT TERM
  set +e
  if [[ "${ACTIVE}" == "1" ]]; then collect_logs; stop_inference; cleanup_secret_files; fi
  ssh "${SSH_HOSTS[0]}" "if test -f '${RUN_DIR}/manifest.json'; then python3 -c \"import json; p='${RUN_DIR}/manifest.json'; d=json.load(open(p)); d['validity']='forensic'; d['failure_exit_code']=${status}; open(p,'w').write(json.dumps(d,indent=2,sort_keys=True)+'\\n')\"; fi" >/dev/null 2>&1
  return "${status}"
}

run_all() {
  preflight
  build_router
  prepare
  start_all
  smoke
  measure
  collect_stop
  report
}

trap 'on_failure $?; exit $?' ERR
trap 'on_failure 130; exit 130' INT TERM

case "${1:-}" in
  preflight) preflight ;;
  build-router) build_router ;;
  prepare) prepare ;;
  start) start_all ;;
  smoke) smoke ;;
  measure) measure ;;
  collect-stop) collect_stop ;;
  report) report ;;
  dry-run) dry_run ;;
  run) run_all ;;
  *) echo "usage: $0 preflight|build-router|prepare|start|smoke|measure|collect-stop|report|dry-run|run" >&2; exit 2 ;;
esac
