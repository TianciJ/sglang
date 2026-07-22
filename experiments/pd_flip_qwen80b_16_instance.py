#!/usr/bin/env python3
"""Four-node, sixteen-worker Qwen80B PD Flip feasibility runner.

Run on the coordinator host. The runner uses exact RUN_ID-owned names and never
stops or reuses resources it does not own.
"""

from __future__ import print_function

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_env(path):
    values = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def split_words(value, default):
    return shlex.split(value if value is not None else default)


class Runner(object):
    def __init__(self, env_file, run_id):
        self.env_file = env_file
        file_values = parse_env(env_file)
        cfg = dict(file_values)
        cfg.update(os.environ)
        self.cfg = cfg
        self.secret = cfg.get("ADMIN_API_KEY", "")
        if not self.secret or self.secret.startswith("replace-with"):
            raise ValueError("private env has no valid ADMIN_API_KEY")
        self.run_id = run_id
        if not run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for ch in run_id):
            raise ValueError("invalid RUN_ID")
        self.image = cfg.get("IMAGE", "sglang-pd-switch:tianciJ")
        self.repo = cfg.get("SGLANG_REPO", "/home/tiancij/sglang-pd-qwen80b")
        self.model_path = cfg.get("MODEL_PATH", "/models/Qwen3-Next-80B-A3B-Instruct")
        self.model_id = cfg.get("MODEL_ID", "Qwen3-Next-80B-A3B-Instruct")
        self.artifact_root = cfg.get("ARTIFACT_ROOT", "/home/tiancij/pd-artifacts")
        self.run_dir = self.artifact_root + "/" + run_id
        self.ssh_hosts = split_words(cfg.get("SSH_HOSTS"), "root@192.168.0.42 root@192.168.0.40 root@192.168.0.39 root@192.168.0.41")
        self.node_ips = split_words(cfg.get("NODE_IPS"), "192.168.0.42 192.168.0.40 192.168.0.39 192.168.0.41")
        self.mooncake_hosts = split_words(cfg.get("MOONCAKE_HOSTS"), "fd03:4514:80:6241::1 fd03:4514:80:7b81::1 fd03:4514:80:6601::1 fd03:4514:80:5f01::1")
        if not (len(self.ssh_hosts) == len(self.node_ips) == len(self.mooncake_hosts) == 4):
            raise ValueError("exactly four SSH, node IP, and Mooncake hosts are required")
        self.gpu_pairs = split_words(cfg.get("GPU_PAIRS"), "0,1 2,3 4,5 6,7")
        self.worker_ports = [int(x) for x in split_words(cfg.get("WORKER_PORTS"), "30000 30001 30002 30003")]
        self.bootstrap_ports = [int(x) for x in split_words(cfg.get("BOOTSTRAP_PORTS"), "18998 18999 19000 19001")]
        if not (len(self.gpu_pairs) == len(self.worker_ports) == len(self.bootstrap_ports) == 4):
            raise ValueError("each host requires four GPU pairs and unique ports")
        self.router_port = int(cfg.get("ROUTER_PORT", "8000"))
        self.ib_device = cfg.get("IB_DEVICE", "mlx5_bond_1")
        self.gid_index = int(cfg.get("MC_GID_INDEX", "3"))
        self.mem_fraction = float(cfg.get("MEM_FRACTION_STATIC", "0.80"))
        self.trace_source = cfg.get("TRACE_SOURCE", "")
        self.trace_manifest_source = cfg.get("TRACE_MANIFEST_SOURCE", "")
        self.trace_sha = cfg.get("TRACE_SHA256", "")
        self.sample_interval = float(cfg.get("MIGRATION_SAMPLE_INTERVAL_SECONDS", "0.20"))
        self.cache_root = cfg.get("COMPILE_CACHE_ROOT", "/home/tiancij/sglang-compile-cache")
        self.cache_namespace = cfg.get("COMPILE_CACHE_NAMESPACE", "qwen80b-ad0d00526372dcbfeca64743")
        self.cache_container_dir = cfg.get("COMPILE_CACHE_CONTAINER_DIR", "/var/cache/sglang-compile")
        self.instances = []
        for host_index in range(4):
            for local_index in range(4):
                name = "h{}i{}".format(host_index, local_index)
                self.instances.append({
                    "name": name,
                    "host_index": host_index,
                    "local_index": local_index,
                    "ssh_host": self.ssh_hosts[host_index],
                    "ip": self.node_ips[host_index],
                    "mooncake_host": self.mooncake_hosts[host_index],
                    "gpu_ids": self.gpu_pairs[local_index],
                    "worker_port": self.worker_ports[local_index],
                    "bootstrap_port": self.bootstrap_ports[local_index],
                    "role": "prefill" if name == "h0i0" else "decode",
                })
        self.source_name = cfg.get("MIGRATION_SOURCE_NAME", "h2i2")
        self.target_name = cfg.get("MIGRATION_TARGET_NAME", "h3i3")
        self.cancelled = False

    def q(self, value):
        return shlex.quote(str(value))

    def remote(self, host, command, check=True, capture=True, input_data=None, timeout=None):
        args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command]
        completed = subprocess.run(
            args,
            input=input_data,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            universal_newlines=True,
            timeout=timeout,
        )
        if check and completed.returncode:
            raise RuntimeError("remote command failed on {}: {}".format(host, completed.stderr[-2000:]))
        return completed

    def http_json(self, url, method="GET", payload=None, timeout=10):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(url, data=data, method=method)
        request.add_header("Authorization", "Bearer " + self.secret)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def worker_name(self, instance):
        return "tiancij-qwen80b-{}-{}".format(self.run_id, instance["name"])

    def helper_name(self, component):
        return "tiancij-qwen80b-{}-{}".format(self.run_id, component)

    def router_name(self):
        return self.helper_name("router")

    def worker_url(self, instance):
        return "http://{}:{}".format(instance["ip"], instance["worker_port"])

    def env_path(self, instance):
        return "{}/env/{}.env".format(self.run_dir, instance["name"])

    def node_args(self, include_router=True):
        values = []
        for item in self.instances:
            spec = "name={0},worker_url={1}".format(item["name"], self.worker_url(item))
            if include_router:
                spec += ",router_worker_id={0},bootstrap_port={1}".format(
                    self.worker_url(item), item["bootstrap_port"]
                )
            values.extend(["--node", spec])
        return values

    def preflight_host(self, host_index):
        host = self.ssh_hosts[host_index]
        ports = list(self.worker_ports) + list(self.bootstrap_ports)
        if host_index == 0:
            ports.append(self.router_port)
        gpu_ids = " ".join(str(x) for pair in self.gpu_pairs for x in pair.split(","))
        script = """set -euo pipefail
test -d {repo}
test -f {model}/config.json
test -f {model}/tokenizer.json
test $(find {model} -maxdepth 1 -name '*.safetensors' | wc -l) -eq 41
docker image inspect {image} >/dev/null
test -z "$(docker ps -aq --filter name='^/tiancij-qwen80b-{run_id}-')"
for port in {ports}; do
  ! ss -ltn | awk '{{print $4}}' | grep -Eq "(:${{port}})$" || {{ echo occupied_port=${{port}} >&2; exit 1; }}
done
for gpu in {gpus}; do
  test -z "$(nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -E '^[[:space:]]*[0-9]+' || true)" || {{ echo busy_gpu=${{gpu}} >&2; exit 1; }}
done
test -r /sys/class/infiniband/{ib}/ports/1/gids/{gid}
test "$(cat /sys/class/infiniband/{ib}/ports/1/gids/{gid})" != 0000:0000:0000:0000:0000:0000:0000:0000
nvidia-smi -L >/dev/null
{{ docker image inspect {image} --format '{{{{.Id}}}}'; sha256sum {model}/config.json {model}/tokenizer.json; find {model} -maxdepth 1 -name '*.safetensors' -printf '%f:%s\n' | LC_ALL=C sort; sha256sum {repo}/experiments/pd_flip_qwen80b_16_instance.py {repo}/scripts/playground/disaggregation/pd_flip_controller.py {repo}/scripts/playground/disaggregation/pd_flip_candidate_prefill_warmup.py {repo}/scripts/playground/disaggregation/pd_flip_docker/run_worker.sh {repo}/scripts/playground/disaggregation/pd_flip_docker/run_router.sh; }} | sha256sum
date --iso-8601=ns
cat /sys/class/infiniband/{ib}/ports/1/gids/{gid}
df -Pk {artifact_root} {model} | tail -n +2
""".format(
            repo=self.q(self.repo), model=self.q(self.model_path), image=self.q(self.image),
            run_id=self.run_id, ports=" ".join(str(x) for x in ports), gpus=gpu_ids,
            ib=self.q(self.ib_device), gid=self.gid_index, artifact_root=self.q(self.artifact_root),
        )
        result = self.remote(host, script, check=False, timeout=60)
        return host_index, result

    def preflight(self):
        mode = os.stat(self.env_file).st_mode & 0o777
        if mode not in (0o400, 0o600):
            raise RuntimeError("private env must have mode 400 or 600")
        if not Path(self.trace_source).is_file() or not Path(self.trace_manifest_source).is_file():
            raise RuntimeError("frozen trace or manifest is missing on coordinator")
        actual = hashlib.sha256(Path(self.trace_source).read_bytes()).hexdigest()
        if actual != self.trace_sha:
            raise RuntimeError("trace SHA mismatch")
        router_binary = self.repo + "/experimental/sgl-router/target/release/sgl-router"
        router_result = self.remote(
            self.ssh_hosts[0],
            "test -x {0} && sha256sum {0}".format(self.q(router_binary)),
            check=False,
            timeout=30,
        )
        if router_result.returncode:
            raise RuntimeError("coordinator router binary is missing or not executable")
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(self.preflight_host, index) for index in range(4)]
            for future in futures:
                results.append(future.result())
        failures = []
        fingerprints = []
        for index, result in sorted(results):
            if result.returncode:
                failures.append({"host": self.ssh_hosts[index], "error": result.stderr.strip()[-1000:]})
            else:
                fingerprints.append(result.stdout)
        report = {"run_id": self.run_id, "success": not failures, "failures": failures,
                  "hosts": self.ssh_hosts, "checked_at": time.time()}
        print(json.dumps(report, indent=2, sort_keys=True))
        if failures:
            raise RuntimeError("preflight failed on {} host(s)".format(len(failures)))
        stable = [x.splitlines()[0] for x in fingerprints]
        if len(set(stable)) != 1:
            raise RuntimeError("image/model/code/router fingerprint mismatch")
        return report

    def prepare(self):
        Path(self.run_dir).mkdir(parents=True, exist_ok=False)
        for part in ("trace", "raw", "logs", "status", "controller", "observer", "metrics", "warmup", "env"):
            Path(self.run_dir, part).mkdir()
        Path(self.run_dir, "trace", "trace.jsonl").write_bytes(Path(self.trace_source).read_bytes())
        Path(self.run_dir, "trace", "manifest.json").write_bytes(Path(self.trace_manifest_source).read_bytes())
        for host in self.ssh_hosts:
            self.remote(host, "umask 077; mkdir -p {}/{{logs,status,env}} {}".format(
                self.q(self.run_dir), self.q(self.cache_root + "/" + self.cache_namespace)))
        manifest = {
            "run_id": self.run_id, "experiment_class": "four_node_16_instance_feasibility",
            "performance_comparison_valid": False, "model_id": self.model_id,
            "trace_sha256": self.trace_sha, "trace_requests": 40,
            "initial_topology": "1P15D", "expected_final_topology": "2P14D",
            "migration_source": self.source_name, "migration_target": self.target_name,
            "instances": self.instances, "sample_interval_seconds": self.sample_interval,
            "measurement_boundary": "client-observed streaming events; not GPU kernel time",
        }
        Path(self.run_dir, "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def write_envs(self):
        worker_urls = " ".join(self.worker_url(item) for item in self.instances)
        extra = "--trust-remote-code --mamba-scheduler-strategy extra_buffer --enable-metrics"
        for item in self.instances:
            fields = {
                "ADMIN_API_KEY": self.secret, "IMAGE": self.image, "SGLANG_REPO": self.repo,
                "MODEL_PATH": self.model_path, "MODEL_ID": self.model_id,
                "PORT": item["worker_port"], "ROUTER_PORT": self.router_port,
                "BOOTSTRAP_PORT": item["bootstrap_port"], "TRANSFER_BACKEND": "mooncake",
                "IB_DEVICE": self.ib_device, "MC_GID_INDEX": self.gid_index, "MC_USE_IPV6": 1,
                "MOONCAKE_LOCAL_HOSTNAME": item["mooncake_host"], "SGLANG_HOST_IP": item["mooncake_host"],
                "MEM_FRACTION_STATIC": self.mem_fraction, "GPU_IDS": item["gpu_ids"],
                "TP_SIZE": 2, "DP_SIZE": 1, "ENABLE_DP_ATTENTION": 0,
                "SGLANG_COMPILE_CACHE_HOST_DIR": self.cache_root + "/" + self.cache_namespace,
                "SGLANG_COMPILE_CACHE_CONTAINER_DIR": self.cache_container_dir,
                "ENABLE_CUSTOM_LOGIT_PROCESSOR": 0, "ENABLE_REQUEST_TIME_STATS_LOGGING": 1,
                "ENABLE_PD_FLIP_STATE_MACHINE": 1, "ENABLE_PD_RUNTIME_ROLE_SWITCH": 1,
                "ENABLE_PD_FLIP_HICACHE_STITCH": 0, "ENABLE_PD_FLIP_PREFILL_DONOR": 0,
                "EXTRA_SGLANG_ARGS": extra, "WORKER_URLS": worker_urls,
                "PD_FLIP_WORKER_CONTAINER_NAME": self.worker_name(item),
                "PD_FLIP_ROUTER_CONTAINER_NAME": self.router_name(),
                "ROUTER_DYNAMO_TARBALL_FALLBACK": 0, "CARGO_NET_OFFLINE": "true",
            }
            content = "".join("{}={}\n".format(key, shlex.quote(str(value))) for key, value in fields.items())
            self.remote(item["ssh_host"], "umask 077; cat > " + self.q(self.env_path(item)), input_data=content)

    def start_one_worker(self, item):
        log = "{}/logs/{}.launcher.log".format(self.run_dir, item["name"])
        pid = "{}/status/{}.launcher.pid".format(self.run_dir, item["name"])
        command = "cd {repo}; nohup env ENV_FILE={env} {runner} {role} {ip} > {log} 2>&1 < /dev/null & echo $! > {pid}".format(
            repo=self.q(self.repo), env=self.q(self.env_path(item)),
            runner=self.q(self.repo + "/scripts/playground/disaggregation/pd_flip_docker/run_worker.sh"),
            role=item["role"], ip=self.q(item["ip"]), log=self.q(log), pid=self.q(pid),
        )
        self.remote(item["ssh_host"], command)

    def wait_worker(self, item, deadline):
        url = self.worker_url(item)
        last = None
        while time.time() < deadline:
            try:
                with urlopen(url + "/health", timeout=2) as response:
                    if response.status != 200:
                        raise ValueError("health status {}".format(response.status))
                status = self.http_json(url + "/pd_flip/runtime_role/status", timeout=3)
                shards = status if isinstance(status, list) else [status]
                if shards and all(x.get("success") is True and x.get("status", {}).get("role") == item["role"] and x.get("status", {}).get("active_event_loop_role") == item["role"] for x in shards):
                    return
            except Exception as exc:
                last = repr(exc)
            time.sleep(2)
        raise RuntimeError("worker {} health/role timeout: {}".format(item["name"], last))

    def start_workers(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(self.start_one_worker, self.instances))
        deadline = time.time() + 3600
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(self.wait_worker, item, deadline) for item in self.instances]
            for future in futures:
                future.result()

    def coordinator(self, command, **kwargs):
        return self.remote(self.ssh_hosts[0], command, **kwargs)

    def start_router(self):
        env = self.env_path(self.instances[0])
        log = self.run_dir + "/logs/router.launcher.log"
        command = "cd {repo}; nohup env ENV_FILE={env} {runner} > {log} 2>&1 < /dev/null & echo $! > {pid}".format(
            repo=self.q(self.repo), env=self.q(env),
            runner=self.q(self.repo + "/scripts/playground/disaggregation/pd_flip_docker/run_router.sh"),
            log=self.q(log), pid=self.q(self.run_dir + "/status/router.launcher.pid"),
        )
        self.coordinator(command)
        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                with urlopen("http://127.0.0.1:{}/v1/models".format(self.router_port), timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("router health timeout")
        topology = self.http_json("http://127.0.0.1:{}/pd_flip/router/workers".format(self.router_port))
        self.validate_topology(topology, 1, 15)
        Path(self.run_dir, "status", "router-initial.json").write_text(json.dumps(topology, indent=2, sort_keys=True) + "\n")

    def validate_topology(self, topology, prefill, decode):
        workers = topology.get("workers", [])
        roles = [str(x.get("effective_role") or x.get("role") or "").lower() for x in workers]
        if len(workers) != prefill + decode or roles.count("prefill") != prefill or roles.count("decode") != decode or any(x.get("draining") for x in workers):
            raise RuntimeError("unexpected topology: {}".format(roles))

    def docker_helper_command(self, name, argv, detached=False, env_file=None, stdout_path=None, stderr_path=None):
        parts = ["docker", "run"]
        if detached:
            parts.append("-d")
        parts += ["--name", name, "--network", "host"]
        if env_file:
            parts += ["--env-file", env_file]
        inner = "cd /sgl-workspace/sglang && exec env PYTHONPATH=python:. " + " ".join(self.q(x) for x in argv)
        if stdout_path:
            inner += " > " + self.q(stdout_path)
        if stderr_path:
            inner += " 2> " + self.q(stderr_path)
        parts += ["-v", self.repo + ":/sgl-workspace/sglang:ro", "-v", self.run_dir + ":" + self.run_dir, self.image,
                  "bash", "-lc", inner]
        command = " ".join(self.q(x) for x in parts)
        return command

    def warmup(self):
        argv = ["python3", "scripts/playground/disaggregation/pd_flip_candidate_prefill_warmup.py",
                "--router-url", "http://127.0.0.1:{}".format(self.router_port)] + self.node_args()
        argv += ["--initial-prefill-name", "h0i0"]
        for item in self.instances:
            argv += ["--candidate-prefill-name", item["name"]]
        argv += ["--trace-jsonl", self.run_dir + "/trace/trace.jsonl", "--output-dir", self.run_dir + "/warmup",
                 "--api-key-env", "ADMIN_API_KEY", "--request-timeout-seconds", "900",
                 "--role-timeout-seconds", "180", "--role-poll-seconds", "0.25"]
        name = self.helper_name("warmup")
        result = self.coordinator(self.docker_helper_command(name, argv, env_file=self.env_path(self.instances[0])), check=False, timeout=7200)
        self.coordinator("docker logs --timestamps {name} > {log} 2>&1 || true; docker rm {name} >/dev/null".format(
            name=self.q(name), log=self.q(self.run_dir + "/logs/warmup.docker.log")))
        if result.returncode:
            raise RuntimeError("16-candidate warmup failed")
        summary = json.loads(Path(self.run_dir, "warmup", "summary.json").read_text())
        if not summary.get("success") or summary.get("warmup_request_count") != 32 or summary.get("final_topology") != "1P15D" or summary.get("kv_cache_flushed_after") is not True:
            raise RuntimeError("warmup contract failed")

    def start_helpers(self):
        ledger = self.run_dir + "/raw/slo_ledger.jsonl"
        node_args = self.node_args()
        sampler = ["python3", "scripts/playground/disaggregation/pd_flip_migration_measure.py", "sample",
                   "--router-url", "http://127.0.0.1:{}".format(self.router_port)] + node_args + [
                   "--output-events", self.run_dir + "/raw/migration_events.jsonl", "--interval-seconds", str(self.sample_interval),
                   "--duration-seconds", "7200", "--api-key-env", "ADMIN_API_KEY"]
        observer = ["python3", "scripts/playground/disaggregation/pd_flip_slo_observer.py",
                    "--ledger", ledger, "--journal", self.run_dir + "/observer/snapshots.jsonl",
                    "--summary", self.run_dir + "/observer/summary.json", "--window-seconds", self.cfg.get("SLO_WINDOW_SECONDS", "10"),
                    "--enter-threshold", self.cfg.get("SLO_ENTER_THRESHOLD", "0.90"), "--recover-threshold", self.cfg.get("SLO_RECOVER_THRESHOLD", "0.95"),
                    "--min-ttft-samples", self.cfg.get("MIN_TTFT_SAMPLES", "10"), "--min-tpot-intervals", self.cfg.get("MIN_TPOT_INTERVALS", "100"),
                    "--poll-interval", "0.25", "--expected-requests", "40"]
        controller = ["python3", "scripts/playground/disaggregation/pd_flip_controller.py",
                      "--router-url", "http://127.0.0.1:{}".format(self.router_port)] + node_args + [
                      "--api-key-env", "ADMIN_API_KEY", "--first-migration-ratio", self.cfg.get("PD_FLIP_FIRST_MIGRATION_RATIO", "0.5"),
                      "--observation-seconds", self.cfg.get("PD_FLIP_OBSERVATION_SECONDS", "2"),
                      "--slo-threshold", self.cfg.get("SLO_ENTER_THRESHOLD", "0.90"), "--slo-recovery-threshold", self.cfg.get("SLO_RECOVER_THRESHOLD", "0.95"),
                      "--force-second-migration-after-observation", "--min-prefill-slo-samples", self.cfg.get("MIN_TTFT_SAMPLES", "10"),
                      "--min-decode-slo-samples", self.cfg.get("MIN_TPOT_INTERVALS", "100"), "--session-journal-path", self.run_dir + "/controller/session.json",
                      "monitor-progressive", "--trace-slo-ledger", ledger, "--window-seconds", self.cfg.get("SLO_WINDOW_SECONDS", "10"),
                      "--source-name", self.source_name, "--migration-target-name", self.target_name,
                      "--iterations", "2400", "--poll-interval", "0.25"]
        specs = [
            ("sampler", sampler, True, True, None, None),
            ("observer", observer, True, False, None, None),
            ("controller", controller, True, True, self.run_dir + "/controller/result.json", self.run_dir + "/logs/controller.stderr.log"),
        ]
        for component, argv, detached, needs_env, out, err in specs:
            command = self.docker_helper_command(self.helper_name(component), argv, detached=detached,
                                                 env_file=self.env_path(self.instances[0]) if needs_env else None,
                                                 stdout_path=out, stderr_path=err)
            self.coordinator(command)

    def replay(self):
        command = "cd {repo} && python3 scripts/playground/disaggregation/pd_flip_trace_replay.py replay --trace-jsonl {trace} --router-url {router} --mode state_machine --output-dir {raw} --ledger-path {ledger} --timeout-seconds 7200 --max-workers 40".format(
            repo=self.q(self.repo), trace=self.q(self.run_dir + "/trace/trace.jsonl"),
            router=self.q("http://127.0.0.1:{}".format(self.router_port)), raw=self.q(self.run_dir + "/raw"),
            ledger=self.q(self.run_dir + "/raw/slo_ledger.jsonl"))
        self.coordinator(command, capture=False, timeout=7200)
        Path(self.run_dir, "raw", "request_metrics.jsonl").write_bytes(Path(self.run_dir, "raw", "state_machine", "request_metrics.jsonl").read_bytes())

    def wait_helper(self, component, required_path):
        name = self.helper_name(component)
        result = self.coordinator("status=$(docker wait {0}); docker logs --timestamps {0} > {1} 2>&1 || true; docker rm {0} >/dev/null; test \"$status\" = 0".format(
            self.q(name), self.q(self.run_dir + "/logs/{}.docker.log".format(component))), check=False, timeout=3600)
        if result.returncode or not Path(required_path).is_file():
            raise RuntimeError("{} helper failed".format(component))

    def stop_sampler(self):
        name = self.helper_name("sampler")
        self.coordinator("docker inspect {0} >/dev/null 2>&1 || exit 0; state=$(docker inspect {0} --format '{{{{.State.Status}}}}'); if test \"$state\" = running; then docker kill --signal=INT {0} >/dev/null; fi; for i in $(seq 1 60); do state=$(docker inspect {0} --format '{{{{.State.Status}}}}' 2>/dev/null || true); test \"$state\" != running && break; sleep 1; done; docker logs --timestamps {0} > {1} 2>&1 || true; docker rm {0} >/dev/null".format(
            self.q(name), self.q(self.run_dir + "/logs/sampler.docker.log")), timeout=120)

    def validate(self):
        rows = [json.loads(x) for x in Path(self.run_dir, "raw", "request_metrics.jsonl").read_text().splitlines() if x.strip()]
        errors = [x for x in Path(self.run_dir, "raw", "state_machine", "errors.jsonl").read_text().splitlines() if x.strip()]
        if len(rows) != 40 or errors or not all(x.get("status") == "completed" and x.get("completion_tokens") == 10000 and x.get("completion_token_match") is True and x.get("finish_reason") == "length" for x in rows):
            raise RuntimeError("request validity gate failed")
        controller = json.loads(Path(self.run_dir, "controller", "result.json").read_text())
        if controller.get("success") is not True or not any(x.get("reason") == "role_flip_complete" for x in controller.get("state_trace", [])):
            raise RuntimeError("controller contract failed")
        topology = self.http_json("http://127.0.0.1:{}/pd_flip/router/workers".format(self.router_port))
        self.validate_topology(topology, 2, 14)
        Path(self.run_dir, "controller", "final_router.json").write_text(json.dumps(topology, indent=2, sort_keys=True) + "\n")
        Path(self.run_dir, "status", "validity.json").write_text(json.dumps({"valid": True, "request_count": 40, "error_count": 0, "final_topology": "2P14D"}, indent=2) + "\n")

    def summarize(self):
        command = "cd {repo} && python3 scripts/playground/disaggregation/pd_flip_migration_measure.py summarize --events-jsonl {events} --output-dir {output} --controller-log {controller} --request-metrics-jsonl {requests} --errors-jsonl {errors} > {stdout}".format(
            repo=self.q(self.repo), events=self.q(self.run_dir + "/raw/migration_events.jsonl"),
            output=self.q(self.run_dir + "/metrics/migration"), controller=self.q(self.run_dir + "/controller/result.json"),
            requests=self.q(self.run_dir + "/raw/request_metrics.jsonl"), errors=self.q(self.run_dir + "/raw/state_machine/errors.jsonl"),
            stdout=self.q(self.run_dir + "/metrics/migration-summary.stdout.json"))
        self.coordinator(command, capture=False, timeout=3600)

    def capture_and_stop_worker(self, item):
        name = self.worker_name(item)
        log = "{}/logs/{}.docker.log".format(self.run_dir, item["name"])
        command = "if docker inspect {name} >/dev/null 2>&1; then docker logs --timestamps {name} > {log} 2>&1 || true; state=$(docker inspect {name} --format '{{{{.State.Status}}}}'); case \"$state\" in running|paused|restarting) docker stop --time 1800 {name} >/dev/null;; esac; fi; touch {log}".format(
            name=self.q(name), log=self.q(log))
        self.remote(item["ssh_host"], command, check=False, timeout=1900)
        if item["host_index"] != 0:
            for suffix in ("docker.log", "launcher.log"):
                remote_log = "{}/logs/{}.{}".format(self.run_dir, item["name"], suffix)
                local = Path(self.run_dir, "logs", item["name"] + "." + suffix)
                with local.open("wb") as output:
                    completed = subprocess.run(["ssh", item["ssh_host"], "cat " + self.q(remote_log)], stdout=output, stderr=subprocess.PIPE)
                if completed.returncode:
                    raise RuntimeError("failed to collect {} {}".format(item["name"], suffix))

    def cleanup(self):
        # Helpers and router exist only on the coordinator.
        for component in ("warmup", "observer", "controller", "sampler"):
            name = self.helper_name(component)
            if component == "sampler":
                command = "docker inspect {0} >/dev/null 2>&1 || exit 0; state=$(docker inspect {0} --format '{{{{.State.Status}}}}'); if test \"$state\" = running; then docker kill --signal=INT {0} >/dev/null; fi; docker rm {0} >/dev/null 2>&1 || true".format(self.q(name))
            else:
                command = "if docker inspect {0} >/dev/null 2>&1; then state=$(docker inspect {0} --format '{{{{.State.Status}}}}'); case \"$state\" in running|paused|restarting) docker stop --time 300 {0} >/dev/null;; esac; docker rm {0} >/dev/null 2>&1 || true; fi".format(self.q(name))
            self.coordinator(command, check=False, timeout=360)
        router = self.router_name()
        self.coordinator("if docker inspect {0} >/dev/null 2>&1; then docker logs --timestamps {0} > {1} 2>&1 || true; docker stop --time 1800 {0} >/dev/null; fi".format(
            self.q(router), self.q(self.run_dir + "/logs/router.docker.log")), check=False, timeout=1900)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(self.capture_and_stop_worker, self.instances))
        self.redact_artifacts()

    def redact_artifacts(self):
        secret = self.secret.encode("utf-8")
        for path in Path(self.run_dir).rglob("*"):
            if not path.is_file() or "env" in path.relative_to(self.run_dir).parts:
                continue
            found = False
            overlap = b""
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    data = overlap + chunk
                    if secret in data:
                        found = True
                        break
                    overlap = data[-max(0, len(secret) - 1):]
            if not found:
                continue
            data = path.read_bytes()
            temporary = path.with_name(path.name + ".redacting")
            temporary.write_bytes(data.replace(secret, b"<redacted>"))
            temporary.replace(path)

    def teardown_gate(self):
        failures = []
        for host_index, host in enumerate(self.ssh_hosts):
            ports = list(self.worker_ports) + list(self.bootstrap_ports)
            if host_index == 0:
                ports.append(self.router_port)
            command = "set -e; test -z \"$(docker ps -aq --filter name='^/tiancij-qwen80b-{run}-')\"; for port in {ports}; do ! ss -ltn | awk '{{print $4}}' | grep -Eq \"(:${{port}})$\"; done; for gpu in 0 1 2 3 4 5 6 7; do test -z \"$(nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader,nounits | grep -E '^[[:space:]]*[0-9]+' || true)\"; done; nvidia-smi -L >/dev/null".format(run=self.run_id, ports=" ".join(str(x) for x in ports))
            result = self.remote(host, command, check=False, timeout=60)
            if result.returncode:
                failures.append({"host": host, "error": result.stderr[-1000:]})
        Path(self.run_dir, "status", "teardown.json").write_text(json.dumps({"valid": not failures, "failures": failures}, indent=2, sort_keys=True) + "\n")
        if failures:
            raise RuntimeError("teardown gate failed")

    def postprocess(self):
        log_args = []
        for item in self.instances:
            log_args.extend(["--log", "{}={}/logs/{}.docker.log".format(item["name"], self.run_dir, item["name"])])
        command = "cd {repo} && python3 scripts/playground/disaggregation/pd_flip_req_timing.py {logs} --output {stats} --events-output {events}".format(
            repo=self.q(self.repo), logs=" ".join(self.q(x) for x in log_args),
            stats=self.q(self.run_dir + "/metrics/req_time_stats.jsonl"),
            events=self.q(self.run_dir + "/metrics/request_stage_events.jsonl"))
        self.coordinator(command, capture=False, timeout=1800)
        secret = self.secret.encode("utf-8")
        hits = []
        for path in Path(self.run_dir).rglob("*"):
            if not path.is_file() or "env" in path.relative_to(self.run_dir).parts or path.name == "INVENTORY.txt":
                continue
            overlap = b""
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    data = overlap + chunk
                    if secret in data:
                        hits.append(str(path.relative_to(self.run_dir)))
                        break
                    overlap = data[-max(0, len(secret) - 1):]
        if hits:
            raise RuntimeError("secret remained in non-env artifacts: {}".format(hits))
        security = {"env_directory_excluded_from_shared_artifacts": True, "secret_hits": 0,
                    "logs_and_status_redacted": True}
        Path(self.run_dir, "status", "security-redaction.json").write_text(json.dumps(security, indent=2, sort_keys=True) + "\n")
        inventory = Path(self.run_dir, "INVENTORY.txt")
        with inventory.open("wb") as output:
            command = "cd {root} && find . -path ./env -prune -o -name INVENTORY.txt -prune -o -type f -print0 | sort -z | xargs -0 sha256sum".format(root=self.q(self.run_dir))
            result = self.coordinator(command, capture=True, timeout=3600)
            output.write(result.stdout.encode("utf-8"))

    def run(self):
        self.preflight()
        self.prepare()
        self.write_envs()
        try:
            self.start_workers()
            self.start_router()
            self.warmup()
            self.start_helpers()
            self.replay()
            self.wait_helper("observer", self.run_dir + "/observer/summary.json")
            self.wait_helper("controller", self.run_dir + "/controller/result.json")
            self.stop_sampler()
            self.validate()
            self.summarize()
        finally:
            self.cleanup()
        self.teardown_gate()
        self.postprocess()
        print("16-instance feasibility run valid: " + self.run_dir)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "prepare", "run"))
    parser.add_argument("--env-file", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    runner = Runner(args.env_file, args.run_id)
    if args.command == "preflight":
        runner.preflight()
    elif args.command == "prepare":
        runner.preflight(); runner.prepare(); runner.write_envs()
        print("16-instance configuration prepared: " + runner.run_dir)
    else:
        runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
