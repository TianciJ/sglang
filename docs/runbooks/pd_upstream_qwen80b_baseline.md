# Clean upstream Qwen80B 1P3D baseline runbook

This runbook executes one 40-request TTFT/TPOT measurement on the owned clean
SGLang v0.5.15 image. It is deliberately separate from
`pd_flip_qwen80b_ab.sh`: no state-machine, runtime role switch, HiCache,
Prefill-donor, or modified router code enters the inference data plane.

## Fixed contract

- Image: `tiancij/sglang-upstream:v0.5.15-clean`.
- Image ID: `sha256:7dd92779d739364d79af34af65815ddc14e567728e5256f65ac922367161213e`.
- Model: `Qwen3-Next-80B-A3B-Instruct`.
- Placement: four nodes, GPUs `0,1`, TP 2, DP 1, `1P3D`.
- RDMA: active `mlx5_bond_1` on every node, per-node bond IPv6 address,
  `MC_USE_IPV6=1`, and GID index 3.
- Natural-output trace SHA256: `c5dbbf75c997dfc5d67a18251082f2f246d6c055eb4af5040fbe147f49f4ce5d`.
- Source trace SHA256: `82da848d68c9662a7aaaf76deb547b1d8cc6c4f562586f0d60dd212bc114e964`.
- Workload: 40 requests, native model output, `max_tokens=10000`, `ignore_eos=true`, concurrency 40.
- No custom logit processor is enabled or serialized in a request.
- SSE event counts are dynamic because one stream event can contain multiple
  tokens. Validity is based on 40 unique completed requests, exactly 10,000
  `usage.completion_tokens` each, `finish_reason=length`, and complete raw
  client evidence—not a fixed event-row count.

Upstream PD Mooncake uses transfer-engine `P2PHANDSHAKE` metadata for this
path. It does not need the HiCache/L3 Mooncake Store service. The runner
therefore neither stops nor resets any existing Mooncake Store; each clean
worker process creates fresh transfer-engine sessions, and the manifest records
`mooncake_metadata=P2PHANDSHAKE`.

## Prepare the private environment

Copy the example without committing the copy:

```bash
cp experiments/pd_upstream_qwen80b_baseline.env.example /root/pd-upstream-qwen80b.env
chmod 600 /root/pd-upstream-qwen80b.env
```

Set `ADMIN_API_KEY` in that private file. Confirm the four SSH aliases, node
IPs, model path, per-node `NODE*_MOONCAKE_HOST`, routable `MC_GID_INDEX`,
controller-side helper repository, and artifact paths. Never paste the private
file into a report. HTTP service IPs and Mooncake IPv6 identities are separate
on purpose.

From the repository that owns the fixed trace:

```bash
export ENV_FILE=/root/pd-upstream-qwen80b.env
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-upstream-qwen80b-clean-r1"
```

Keep the same `RUN_ID` for all subcommands belonging to one attempt. A repaired
rerun must use a new ID.

## Read-only preflight

```bash
bash experiments/pd_upstream_qwen80b_baseline.sh preflight
```

Preflight contacts all four nodes and checks exact image ID, complete and
matching model files, selected-GPU occupancy, ports, exact owned-name
collisions, driver health, disk, running containers/processes, network/RDMA,
GID visibility, and clocks. It makes no remote changes. Do not continue if a
node is unreachable, a selected GPU is busy, a port is occupied, or the model,
image, driver, RDMA, or clocks are unhealthy.

Pre-existing containers shown by preflight are evidence, not cleanup targets.
The runner never stops a container unless its exact name and
`tiancij.run_id=$RUN_ID` label both match.

## Extract and inspect the official router

```bash
bash experiments/pd_upstream_qwen80b_baseline.sh build-router
```

This creates a no-GPU container from the clean image and extracts the
image-provided `/usr/local/bin/sglang-router` launcher. It does not download a
Rust toolchain, rebuild router code, or mount the modified repository. The
launcher hash, image ID, source path, and inspect record are retained under
`ROUTER_ARTIFACT_DIR`; a matching artifact may be reused.

Review the complete redacted command plan before loading the model:

```bash
bash experiments/pd_upstream_qwen80b_baseline.sh dry-run
```

`dry-run` performs no SSH or Docker operation and must not print the secret.

## Formal run

The complete checked-in sequence is:

```bash
bash experiments/pd_upstream_qwen80b_baseline.sh run
```

It performs `preflight`, `build-router`, `prepare`, `start`, `smoke`,
`measure`, `collect-stop`, and `report` in that order:

1. Copy and re-hash the fixed serialized trace.
2. Start all four clean workers concurrently and wait for bounded health gates.
3. Start the official router only after every worker is healthy.
4. Send two unique, non-measured 32-token smoke requests through Prefill-to-Decode.
5. Send one unique, non-measured natural-output 10,000-token probe and require
   exact usage count plus `finish_reason=length`.
6. Copy the first long and first short request bodies from the frozen trace.
   Run them sequentially in `long` then `short` order, generate exactly one
   token from each, and retain separate client timing records plus one Docker
   log window spanning both requests on every worker and the router. These
   exercise the formal approximately 6.4k- and 650-token Prefill shapes without
   adding either request to measured rows.
7. Flush all four upstream caches after both warmups. This clears KV/Radix
   reuse while retaining process-level compiled kernels, allocator state, and
   workspaces. If any post-warmup flush cannot be proven, preserve the attempt
   as `forensic` and stop exact run-owned resources; do not relaunch under the
   same run ID because that would erase the warmup state.
8. Replay the trace once through an external no-GPU helper.
9. Reject the run unless all request integrity and raw-evidence gates pass.
10. Capture redacted inspect records and logs, gracefully stop exact owned
   containers, verify ports, GPUs, and driver health, and generate the report.

The worker and router obtain SGLang code only from the clean image. The modified
repository is mounted read-only at `/work/sglang` only in the non-GPU client and
report helpers, which serialize traffic and analyze received timestamps.

## Recovery and forensic handling

If startup or measurement fails, the error trap preserves the run directory as
`forensic` and tries to capture logs before gracefully stopping only exact
run-owned inference containers. Do not overwrite or relabel that directory.

Inspect ownership before any manual action:

```bash
docker inspect "tiancij-upstream-${RUN_ID}-node0" \
  --format '{{.Name}} {{index .Config.Labels "tiancij.run_id"}} {{.State.Status}}'
```

If the normal error path did not finish teardown, rerun only the checked-in
exact-name collection/stop step with the same ID:

```bash
bash experiments/pd_upstream_qwen80b_baseline.sh collect-stop
```

After evidence is complete, `report` can be regenerated without GPU use:

```bash
bash experiments/pd_upstream_qwen80b_baseline.sh report
```

Never use broad process/container matching, `docker restart`, or repeated model
reloads on an unhealthy or unreachable node.

## Artifact layout

The controller node stores `${REMOTE_ARTIFACT_ROOT}/${RUN_ID}`:

```text
manifest.json
INVENTORY.txt
trace/trace.jsonl
trace/source_manifest.json
raw/slo_ledger.jsonl
raw/upstream_baseline/request_metrics.jsonl
raw/upstream_baseline/responses.jsonl
raw/upstream_baseline/errors.jsonl
raw/upstream_baseline/ttft.csv
raw/upstream_baseline/tpot.csv
raw/upstream_baseline/tpot_tokens.csv
logs/client.log
logs/node0.docker.log ... logs/node3.docker.log
logs/router.docker.log
inspect/
status/
smoke/
  long-prefill-warmup.json
  short-prefill-warmup.json
logs/
  warmup-node0.docker.log ... warmup-node3.docker.log
  warmup-router.docker.log
report/summary.json
report/request_metrics.csv
report/ttft_scatter.svg
report/tpot_scatter.svg
report/report.md
```

`INVENTORY.txt` contains SHA256 checksums of the retained files. The report is
valid only when `manifest.json` says `valid`; a `pending` or `forensic` run is
not a result.

## Metric interpretation

The client uses `time.monotonic()` at HTTP request send and each non-empty
stream event receive, and obtains the exact generated-token count from
`usage.completion_tokens`:

```text
TTFT = first_nonempty_output_receive_time - request_start_time
request_TPOT = (last_nonempty_output_receive_time - first_nonempty_output_receive_time)
               / (usage.completion_tokens - 1)
```

These are client-observed timings. `tpot_tokens.csv` retains transport-level
SSE stream-event gaps for diagnosis; those rows are not individual tokens and
must not be presented as per-token TPOT. None of these metrics is GPU kernel
duration or a direct internal Prefill/Decode stage duration. Worker timestamped
logs are retained separately for later server-stage analysis.

This is one measured run. It is useful as a clean-upstream absolute baseline,
but it does not establish run-to-run statistical significance and cannot by
itself attribute a delta against a different image/revision solely to the state
machine.
