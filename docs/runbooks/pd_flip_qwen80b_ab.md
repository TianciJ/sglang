# Qwen3-Next 80B PD Flip A/B quick-validation runbook

This runbook compares the same deterministic 40-request workload on two four-node configurations:

- Baseline: stock static SGLang `1P3D`.
- State machine: PD Flip starts at `1P3D`, migrates 50% of the selected decode source's active requests, observes for 2 seconds, migrates the remainder, and publishes the final `2P2D` topology.

The harness does not run Prefill Donor or HiCache stitching. It measures full source-decode request migration, including the model's auxiliary Mamba/GDN state when the runtime exposes it.

## Safety boundary

The runner never downloads or copies model weights. It never uses `docker restart`, broad process matching, `pkill`, or `kill -9`. It only stops containers whose exact names include the run ID and only terminates the migration sampler through the PID file it created after verifying the PID command line.

Do not bypass preflight. A failure is intentionally non-mutating.

At the time this harness was prepared, `Qwen3-Next-80B-A3B-Instruct` was known to be present on node 102 but not yet confirmed on all other nodes. Until all four nodes have a complete, identical model directory, preflight should fail and print the missing nodes. The harness will not repair this condition.

## Frozen quick-validation configuration

- Four nodes: cloud-099, cloud-100, cloud-101, cloud-102.
- Four selected GPUs per node: `0,1,2,3`; TP=4 and DP=1.
- Initial roles: node0=P, node1=D, node2=D, node3=D.
- Fixed migration pair: node2 source D to node3 target D; node2 becomes P.
- 40 requests: 20 short prompts near 1,000 Chinese characters and 20 long prompts near 10,000 characters, interleaved.
- Four waves of ten requests. Requests within a wave are 0.5 seconds apart; wave starts are 7.5 seconds apart.
- Each request uses `max_tokens=10000`, `ignore_eos=true`, and a locally serialized single-token logit processor that repeatedly emits the token for `的`.
- Short/long TTFT SLO: 2/5 seconds. TPOT SLO: 50 ms.
- Rolling SLO window: 10 seconds. Entry below 90%; recovery at or above 95%.
- Minimum evidence: 10 TTFT samples and 100 TPOT intervals. Poll interval: 250 ms.

## One-time preparation

On every node, confirm that the same Git commit is checked out at `SGLANG_REPO`, the same Docker image ID is available, and the same model config and complete weight shards exist at `MODEL_PATH`. The runner checks these again.

Copy the example environment to a private file on the machine from which the runner will be invoked:

```bash
cp experiments/pd_flip_qwen80b_ab.env.example /path/to/private-qwen80b.env
chmod 600 /path/to/private-qwen80b.env
```

Set the real `ADMIN_API_KEY` only in that private file. Do not commit it. Verify the node aliases, IPs, image, repository path, model path, IB device, GPU IDs, and memory fraction.

## Dry run and preflight

The dry run performs no SSH, Docker, file creation, or process mutation:

```bash
DRY_RUN=1 \
RUN_ID=local-plan-check \
ENV_FILE=/path/to/private-qwen80b.env \
bash experiments/pd_flip_qwen80b_ab.sh run
```

Run the real read-only preflight next:

```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-qwen80b-ab
ENV_FILE=/path/to/private-qwen80b.env \
bash experiments/pd_flip_qwen80b_ab.sh preflight
```

Preflight rejects the run when any selected port is occupied, any selected GPU has a compute PID, an owned name already exists, SSH is unavailable, the router binary is missing, the model is incomplete, or code/model/image fingerprints differ. This prevents loading another 80B replica on top of an existing experiment.

## Execute the quick validation

Keep the same `RUN_ID` for every action. To execute the whole sequence:

```bash
RUN_ID=20260717T000000Z-qwen80b-ab \
ENV_FILE=/path/to/private-qwen80b.env \
bash experiments/pd_flip_qwen80b_ab.sh run
```

The sequence is:

1. Preflight the four hosts.
2. Build one immutable trace offline from the local model tokenizer and record its SHA-256.
3. Start baseline workers sequentially. Each worker must pass health and role checks before the next starts. Start the router only after all four workers pass.
4. Start the read-only SLO observer and the 50 ms telemetry sampler, replay all 40 requests, and verify every request completed with exactly 10,000 matching output tokens.
5. Capture logs, gracefully stop the exact baseline containers, verify ports and GPU driver health, and normalize request-stage evidence.
6. Load the state-machine configuration through different owned container names and repeat the sequential health gates.
7. Start telemetry and the progressive controller. On the first eligible TTFT violation, migrate 50%, observe for 2 seconds, and then migrate the remainder according to the configured experiment policy.
8. Verify the controller completed and the router reports `2P2D`, stop the sampler and owned containers, and normalize all migration evidence.
9. Validate the pair and generate the comparison report.

If baseline teardown or host-health verification fails, the shell exits before the second model load. Preserve the run directory and diagnose the host instead of retrying with `docker restart`.

## Artifact layout

Artifacts are rooted at `${ARTIFACT_ROOT}/${RUN_ID}` on node0:

```text
trace/
  trace.jsonl
  manifest.json
baseline/
  manifest.json
  raw/request_metrics.jsonl
  raw/slo_ledger.jsonl
  raw/migration_events.jsonl
  observer/snapshots.jsonl
  observer/summary.json
  logs/
  metrics/req_time_stats.jsonl
  metrics/request_stage_events.jsonl
  metrics/migration/
state_machine/
  manifest.json
  raw/request_metrics.jsonl
  raw/slo_ledger.jsonl
  raw/migration_events.jsonl
  controller/result.json
  controller/final_router.json
  logs/
  metrics/req_time_stats.jsonl
  metrics/request_stage_events.jsonl
  metrics/migration/
comparison/
  summary.json
  request_comparison.csv
  stage_timings.csv
  slo_timeseries.csv
  migration_timings.csv
  report.md
  timeline.svg
```

`request_metrics.jsonl` and token-interval raw files contain client scheduling, send, first-token, last-token, TTFT, TPOT, output-integrity, prompt-token, cached-token, and cache-tier evidence. `request_stage_events.jsonl` contains measured SGLang Prefill/Decode process stages with source worker, source log, and line number. Migration outputs contain router role/drain samples, worker queue/load samples, controller actions, request-level phase events, combined transfer bytes, and the declared state types.

Per-component KV-versus-Mamba byte counts remain `null` unless the backend reports a trustworthy split. The combined Mooncake transfer count is retained.

## Check validity before interpreting performance

Open `comparison/summary.json` first. Do not quote a winner unless `valid` is true. Pair validity requires matching trace, model, code, image/GPU configuration, 40 completed requests in both modes, the 10,000-token output contract, successful controller completion, the configured 50%/2-second policy, and final `2P2D` evidence.

The baseline observer is read-only; it records when the same threshold would have fired but never mutates routing or worker state. In the state-machine result, use the controller snapshots/state trace and `migration_phase_events.jsonl` to identify the request and timestamp that triggered the switch and to divide requests into pre-trigger, migration, observation, and post-switch intervals.

One baseline run and one state-machine run are a quick validation, not a statistically significant performance conclusion. Repeat paired runs after the chain is stable.

## Regenerate only the report

No servers are required to regenerate the report from preserved raw data:

```bash
cd "$SGLANG_REPO"
python3 scripts/playground/disaggregation/pd_flip_ab_report.py \
  --run-dir "${ARTIFACT_ROOT}/${RUN_ID}"
```

## Manual stop after an interrupted runner

Prefer letting the runner's graceful teardown finish. If the controlling terminal is interrupted, inspect `${RUN_DIR}/{baseline,state_machine}/pids` and exact container names in the manifests/logs. Stop only names containing the intended `RUN_ID` with a long Docker timeout. Do not use a wildcard process kill and do not start another model load until the four selected ports are free and `nvidia-smi -L` succeeds on every node.
