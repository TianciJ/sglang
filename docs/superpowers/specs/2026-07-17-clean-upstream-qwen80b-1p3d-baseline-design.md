# Clean upstream Qwen80B 1P3D baseline experiment design

## Objective

Measure one complete 40-request TTFT/TPOT run on unmodified upstream SGLang
while preserving the topology, model, serialized trace, arrival schedule,
generation contract, client instrumentation, and metric definitions used by the
2026-07-17 PD Flip experiment.

The experiment is an absolute clean-upstream baseline. It is intentionally one
measured run, so the report may describe this run and its 40 requests but must
not claim run-to-run statistical significance.

## Purity boundary

The inference data plane must run only code embedded in the owned clean image:

```text
tiancij/sglang-upstream:v0.5.15-clean
sha256:7dd92779d739364d79af34af65815ddc14e567728e5256f65ac922367161213e
```

This image is an owned immutable child of official SGLang v0.5.15, upstream
commit `f63458b5beaceabbd9d749b9fc956370e1b649e6`. Worker and router containers
must not bind-mount the modified host repository over `/sgl-workspace/sglang`
and must not set `PYTHONPATH` to host code.

The benchmark client and post-processing tools may run from the modified
repository in a separate, non-GPU helper process or helper container. They are
outside the inference data plane and only serialize requests, record client
receive times, validate output, and generate reports. Their revision and image
must still be recorded.

## Fixed experiment configuration

- Model: `Qwen3-Next-80B-A3B-Instruct`.
- Nodes: four.
- GPU allocation: GPUs `0,1,2,3` on every node.
- Tensor parallelism: 4.
- Data parallelism: 1.
- Topology: one Prefill worker and three Decode workers (`1P3D`).
- Transfer backend: upstream Mooncake.
- RDMA device: `mlx5_0`; use the preflight-validated routable GID configuration.
- Static memory fraction: 0.88.
- DP attention: disabled.
- Prefix state: cold for the measured trace after a successful smoke test.

The run must use a unique ID and exact run-owned names for worker, router,
Mooncake, helper, and artifact resources. It must not stop or modify unrelated
containers or services already present on the nodes.

## Worker configuration

Each worker is started from the clean image with only the read-only model path,
the required InfiniBand devices, and run-specific Mooncake configuration
mounted or injected. The worker command uses upstream arguments equivalent to:

```text
python3 -m sglang.launch_server
--model-path /models/Qwen3-Next-80B-A3B-Instruct
--served-model-name Qwen3-Next-80B-A3B-Instruct
--tp-size 4
--dp-size 1
--disaggregation-mode prefill|decode
--disaggregation-transfer-backend mooncake
--disaggregation-bootstrap-port 8998
--disaggregation-ib-device mlx5_0
--mem-fraction-static 0.88
--enable-custom-logit-processor
--enable-request-time-stats-logging
--trust-remote-code
--mamba-scheduler-strategy extra_buffer
--enable-metrics
```

Host, port, Mooncake namespace, and per-node role are supplied by the owned
runner. The following flags are forbidden:

```text
--enable-pd-flip-state-machine
--enable-pd-runtime-role-switch
--enable-pd-flip-hicache-stitch
--enable-pd-flip-prefill-donor
--enable-hierarchical-cache
--disaggregation-decode-enable-radix-cache
```

The manifest must contain the exact `docker inspect` command, mounts, image ID,
effective arguments, model fingerprint, node, GPU IDs, role, and network/RDMA
configuration for each worker.

## Upstream router

The clean image contains official router source under
`/sgl-workspace/sglang/experimental/sgl-router` but no prebuilt router binary.
Before the experiment, compile the router from that embedded source in an
isolated, non-GPU build container derived from the same clean image. Do not
mount or compile the modified host router source.

Store the router binary as a run-independent owned artifact, record its SHA256,
and run it from the clean image with the four upstream worker URLs. The router
must expose the standard OpenAI-compatible endpoint used by the existing trace
replayer. No PD Flip control endpoint, mutable role API, or state-machine
controller participates in this experiment.

## Mooncake isolation

Use a fresh run-owned Mooncake namespace and run-owned service/container names.
Do not reuse mutable request, metadata, or storage state from the previous PD
Flip experiment. If existing shared Mooncake infrastructure must be used for a
platform reason, create an isolated namespace and record that dependency in the
manifest; do not reset or modify another user's namespace.

HiCache, L2/L3 prefix restoration, Prefill donor, and PD Flip cache stitching
are outside this baseline. Mooncake is used only for upstream `1P3D` KV transfer.

## Workload contract

Reuse the existing serialized trace without regeneration:

```text
pd-flip-artifacts/qwen80b-trace40-source/trace.jsonl
SHA256: 82da848d68c9662a7aaaf76deb547b1d8cc6c4f562586f0d60dd212bc114e964
```

The trace contains:

- 40 requests with stable request IDs `qwen80b-00` through `qwen80b-39`;
- 20 1,000-character short prompts and 20 10,000-character long prompts,
  interleaved;
- arrivals every 0.5 seconds, with a 3-second pause after each group of ten;
- last arrival at 27 seconds;
- `max_tokens=10000`, `stream=true`, `ignore_eos=true`, and finish reason
  `length`;
- a serialized custom logit processor that emits token `的`;
- short-request TTFT SLO 2 seconds, long-request TTFT SLO 5 seconds, and TPOT
  SLO 0.05 seconds.

The trace replayer uses a maximum concurrency of 40 and follows the recorded
arrival offsets. No prompt, request ID, sampling parameter, SLO, or output
contract may be rewritten for the measured run.

## Experiment sequence

1. Perform read-only preflight on all nodes: ownership, active containers,
   ports, GPUs, driver, model files, image ID, disk, RDMA device/GID, clocks,
   and connectivity.
2. Create the run ID, artifact directory, redacted manifest, and exact owned
   resource names.
3. Verify the trace SHA256, four-node image ID, model fingerprint, and router
   binary SHA256.
4. Start the fresh Mooncake namespace/services required by upstream transfer.
5. Start all four workers concurrently. Wait for every worker's bounded health
   gate before starting the router.
6. Start the official upstream router and verify its model and health endpoints.
7. Send two non-measured smoke requests whose prefixes cannot match any formal
   trace prefix. Validate Prefill-to-Decode transfer and streaming output.
8. Flush upstream prefix/Radix caches through a verified upstream mechanism and
   confirm the measured trace starts from cold prefix state. If cache clearing
   cannot be proven, restart only the run-owned worker containers and repeat
   their health gates before measuring.
9. Start the external client collector and replay the 40-request trace exactly
   once.
10. Validate all request and event-level output before accepting the run.
11. Save worker/router/Mooncake/client logs and before/after inspect/status
    evidence.
12. Gracefully stop only the exact run-owned containers, verify their ports are
    free, and recheck node/GPU health.
13. Generate the TTFT/TPOT report from preserved raw client event times.

## Timing instrumentation

The client collector records `time.monotonic()` when it sends the HTTP request
and when each non-empty streaming output event is received:

```text
TTFT = first_token_time - request_start_time
TPOT(i) = token_time(i) - token_time(i - 1)
```

These are client-observed stream timings. The report must not describe them as
GPU kernel durations or internal Prefill/Decode stage times. Worker request-time
logs are retained separately for server-stage analysis.

The collector writes:

- `slo_ledger.jsonl`: one running row per output event plus one terminal row per
  request;
- `request_metrics.jsonl`: request-level anchors, TTFT, TPOT statistics, output
  integrity, and status;
- `responses.jsonl` and `errors.jsonl`;
- `ttft.csv`, `tpot.csv`, `tpot_tokens.csv`, `slo_attainment.csv`, and
  `slo_summary.csv`.

## Acceptance gates

The run is valid only if all checks pass:

- trace SHA256 equals
  `82da848d68c9662a7aaaf76deb547b1d8cc6c4f562586f0d60dd212bc114e964`;
- four worker image IDs equal
  `sha256:7dd92779d739364d79af34af65815ddc14e567728e5256f65ac922367161213e`;
- no worker/router mount hides `/sgl-workspace/sglang` and no command injects
  modified host Python code;
- no forbidden state-machine, role-switch, HiCache, or Prefill-donor flag appears
  in effective worker/router commands;
- exactly 40 unique request records complete successfully;
- every request reports 10,000 completion tokens, finish reason `length`, a
  matching forced output, and no error;
- `errors.jsonl` is empty;
- `slo_ledger.jsonl` contains 400,040 JSON records: 400,000 running output-event
  rows and 40 terminal rows;
- `tpot_tokens.csv` contains 399,960 data rows;
- all required manifests, raw event timing, worker/router logs, and teardown
  evidence exist;
- run-owned containers stop cleanly and pre-existing containers remain
  unaffected.

Any failed gate makes the directory a forensic run. A repaired attempt receives
a new run ID and does not overwrite the failed evidence.

## Final report

The report includes:

1. Validity and purity statement with image, model, trace, router, topology,
   hardware, and instrumentation provenance.
2. A 40-row request table containing request ID, prompt class, arrival offset,
   prompt tokens, TTFT, TTFT SLO result, average/P50/P95/max TPOT, TPOT SLO
   attainment, latency, and status.
3. TTFT summary for all requests and separately for short and long prompts:
   mean, median, P95, maximum, and SLO attainment.
4. Request-level TPOT summary and token-interval P50/P95/P99/maximum over all
   399,960 intervals, plus 50 ms interval attainment.
5. TTFT and TPOT scatter plots aligned by stable request ID and arrival time.
6. Links to raw event ledgers, per-request metrics, per-token intervals,
   manifests, and logs.
7. A clear limitation: this is one measured upstream run and does not establish
   run-to-run statistical significance.

Today's modified-code baseline and state-machine results may be shown only as
separately labeled historical references. Because their code/image provenance
differs, this clean-upstream run must not be used to attribute an observed delta
solely to the state machine.
