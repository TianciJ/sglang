# SGLang repository agent instructions

These instructions apply to the entire repository. More specific descendant
`AGENTS.md` files may add local rules, but they must not weaken the performance
experiment safety, evidence, or validity requirements below.

## Performance experiment policy

Treat every performance experiment as a reproducible systems experiment, not
as an ad hoc sequence of shell commands. This applies to single-node and
multi-node tests, baseline benchmarks, PD disaggregation, PD Flip, cache and
migration experiments, model comparisons, throughput tests, latency tests, and
SLO tests.

For four-node PD Flip work, start from the executable runner
`experiments/pd_flip_qwen80b_ab.sh` and the operator guide
`docs/runbooks/pd_flip_qwen80b_ab.md`. The approved repository-wide design is
`docs/superpowers/specs/2026-07-17-performance-experiment-agent-policy-design.md`.
The retained successful-run inventory, when present in this workspace, is
`pd-flip-artifacts/pd-switch-raw-20260717/INVENTORY.txt`.

Executable behavior and retained manifests take precedence over prose. If the
runner, runbook, and an old report disagree, inspect the current runner and the
actual run manifests instead of guessing or copying an old command.

## Mandatory safety and ownership rules

- Inspect every target node before making changes. Record reachable hosts,
  existing containers and processes, occupied GPUs, listening ports, mounts,
  model paths, disk capacity, driver health, and RDMA devices/GIDs.
- Determine ownership before stopping anything. Never stop, restart, kill,
  rename, move, or reuse another person's containers, processes, GPUs, ports,
  mounts, model files, experiment directories, or services.
- Give every run a unique `RUN_ID`. Use that ID in exact container names,
  helper names, PID files, logs, manifests, and artifact directories.
- Stop only exact run-owned container names. Stop a helper PID only when its PID
  file was created by the current run and its command line still matches the
  expected helper.
- Never use `docker restart`, wildcard or substring process killing, `pkill`,
  `killall`, or `kill -9` as experiment orchestration. Do not emulate these
  operations with a broad loop.
- Do not start a second model load on a node that is unreachable, unhealthy,
  partially torn down, or still holds an earlier experiment's GPU allocation.
  Preserve evidence and diagnose the node first.
- Secrets must come from a private environment file or environment variable.
  Never print an admin key, place it in a committed file, or include private
  environment files in a shared artifact bundle.

## Required experiment lifecycle

Follow this order for every performance experiment:

1. Define the question, primary metrics, SLOs, workload, comparison groups,
   warm-up policy, repeat count, and acceptance criteria before starting nodes.
2. Run a read-only preflight on every node. Verify SSH, model completeness and
   fingerprint, tokenizer, image ID, repository revision, GPU assignment,
   driver health, ports, disk, mounts, network/RDMA selection, and clock sync.
3. Generate a unique run-owned directory and manifest. Store the full effective
   configuration, not only defaults.
4. Prepare the trace once and hash it. Reuse the identical serialized trace for
   all groups in a comparison.
5. Start workers through a checked-in runner. Concurrent worker startup is
   allowed when the runner owns all resources, but every worker must pass a
   bounded health gate and its expected role/configuration check before the
   router starts.
6. Start the router only after all workers are ready. Start observers,
   controllers, and samplers only after their input paths and credentials have
   been verified.
7. Send no measured workload until every required readiness gate passes. Treat
   HTTP 503 during bounded startup polling as "not ready"; a 503 during the
   measured run is a failure, not a normal baseline result. Do not call a
   state-machine-only status endpoint in a baseline configuration that does not
   expose it.
8. Capture raw evidence continuously. A monitoring-helper failure must be
   visible and must not silently convert a run into a valid result.
9. Validate request completion and output integrity before teardown.
10. Save logs and status snapshots, gracefully stop exact run-owned containers,
    confirm their ports are free, and recheck GPU/driver and node health.
11. Finish and validate the entire first group before loading the next group.
12. Generate comparisons only after both groups independently pass their
    validity gates.

If a required behavior is missing from the runner, update and test the runner
first. Do not replace a missing orchestration step with an unrecorded one-off
command on the cluster.

## Comparison validity

A baseline and candidate form a valid performance pair only when they use the
same:

- serialized trace and trace hash;
- model ID, model fingerprint, tokenizer, and model files;
- code revision and uncommitted patch state;
- container image and relevant libraries;
- nodes, GPU IDs, TP/DP settings, clocks, ports, and network/RDMA configuration;
- prompt construction, generation parameters, forced-output contract, SLOs,
  observer thresholds, sample intervals, and timeout policy;
- warm-up, cache-state, startup, teardown, and measurement procedure.

Record these fields in each mode's manifest and make the comparison script
reject mismatches. Do not describe an unmatched pair as an A/B result. A run
may prove that a chain completes while still being invalid for performance
comparison.

Run the baseline first, collect it, tear it down gracefully, and verify all
nodes before loading the candidate. Do not overlap baseline and candidate model
loads or measured traffic on the same reserved resources.

## Raw evidence and artifact contract

Preserve enough event-level evidence to recompute metrics independently.
Request-level percentiles and summary CSV files alone are not raw data.

At minimum, retain:

- the exact trace, trace manifest, trace hash, and effective run configuration;
- per-output client receive timestamps or `slo_ledger.jsonl` with `start_time`,
  `first_token_time`, and each output event's `last_token_time`;
- `request_metrics.jsonl`, `responses.jsonl`, `errors.jsonl`, derived
  `ttft.csv`, `tpot.csv`, and per-output `tpot_tokens.csv`;
- worker, router, workload, observer, controller, and sampler logs with
  timestamps;
- before/after worker and router status, role/topology snapshots, controller
  journal/result, observer snapshots, migration events, and load samples;
- normalized server request-stage events with source node, source log, line
  number, request ID mapping, and wall-clock timestamp;
- model/code/image/GPU/network provenance and clock/timezone evidence;
- validation output, failures, teardown result, and a redacted artifact
  inventory with checksums.

For the client event ledger, recompute:

```text
TTFT = first_token_time - start_time
TPOT(i) = last_token_time(i) - last_token_time(i - 1)
```

These are client-observed stream-event timings. Do not present them as GPU
kernel time or an internal Prefill/Decode stage duration. Use normalized server
stage events for internal-chain timing, and state the clock and instrumentation
boundary in every report.

Keep failed, partial, and invalid attempts in separately named forensic
directories. Never merge their rows or logs into the final valid run.

## Successful Qwen80B reference run

Use this historical run as the operational reference for a chain that completed:

- Run ID: `20260717T042000Z-qwen80b-ab-obs2-gpu0123-gid3`.
- Model: `Qwen3-Next-80B-A3B-Instruct`.
- Allocation: four nodes, GPUs `0,1,2,3` on each node, TP 4, DP 1.
- Workload: 40 requests with interleaved short and long, deliberately distinct
  prefixes; each request generated exactly 10,000 forced `的` tokens with
  `ignore_eos=true`.
- Initial topology: `1P3D`.
- SLO policy: 10-second window, 0.90 enter threshold, 0.95 recover threshold.
- PD Flip policy: first migration 50%, observation period 2 seconds, then the
  remaining migration, ending at `2P2D`.
- The first recorded SLO trigger was request `qwen80b-02`.
- Both retained modes completed 40 requests with zero request errors. The final
  state-machine controller reached `role_flip_complete`.

This retained pair is an operational success record, not a controlled
performance A/B result. Its baseline manifest records code `420bb4ad9`, while
its final state-machine manifest records `f25c090c4`. Do not quote their metric
difference as the state machine's performance effect. Rerun both modes from one
identical revision and image before making that comparison.

At the time this policy was written, the current chain also contained later
experiment-safety fixes through `99dc46469`. When porting or rebasing the runner,
verify equivalent fixes for concurrent worker startup, routable RoCE GID
selection, helper execution inside the experiment image, admin-key propagation,
controller-session isolation, migrated-request chunk-cache adoption, and
observer termination on the global terminal-request count.

## PD Flip reference workflow

For a direct reproduction, use `experiments/pd_flip_qwen80b_ab.sh`; do not
retype its Docker and controller commands manually.

- Baseline: static `1P3D`, state machine disabled, runtime role switching
  disabled, and the SLO observer read-only.
- Candidate: start at `1P3D`, enable the state machine and runtime role
  switching, use the same trace, and require controller completion plus final
  `2P2D` evidence.
- Keep HiCache stitching and prefill-donor behavior disabled for this specific
  stock-baseline-versus-state-machine experiment unless the experiment question
  explicitly changes them for both appropriate comparison groups.
- Explicitly set first migration ratio `0.5` and observation period `2 seconds`.
  Do not rely on the runner's default observation value.
- Start all four workers concurrently through the runner, then wait for all
  health/role gates before starting the router.
- Replay the 40-request trace once per mode and require all 40 requests to
  finish with exactly 10,000 matching output tokens and finish reason `length`.
- Use the observer's first-trigger record and controller/migration events to
  locate the triggering request, first migration, observation period, second
  migration, role flip, and final topology.

Parameters may change for a different research question, but the change must be
declared in the experiment design and manifest. The safety, provenance,
event-level evidence, paired-comparison, and validation requirements do not
change.

## Failure handling and teardown

- On startup or workload failure, stop scheduling new requests, preserve the
  failing logs/status, and gracefully stop only current run-owned containers.
- A lost SSH connection, failed health gate, driver error, stuck model load,
  missing model shard, invalid GID, or host-level pressure is a blocker. Do not
  repeatedly reload the model or issue `docker restart` until the host is
  understood and healthy.
- Use long graceful Docker stop timeouts for large models. After stopping,
  verify exact ports are free and `nvidia-smi -L` works before the next load.
- If teardown cannot prove ownership or health, stop and request operator input.
- Preserve the invalid run with a reason in its manifest. A repaired rerun gets
  a new run ID; never overwrite the failed attempt.

## Performance claim gate

Do not publish or report a winner unless all of the following are true:

- every group independently passed its planned request-count, uniqueness,
  status, token-count, forced-output, finish-reason, error, and teardown checks;
- all comparison provenance fields match except the intended independent
  variable;
- state-machine runs show the intended policy, controller completion,
  request-to-migration linkage, and final topology;
- raw event timing and server-stage evidence are present and the metric boundary
  is stated;
- invalid attempts are excluded and disclosed;
- the result contains enough matched repetitions to describe variation. One
  baseline/state-machine pair is a quick chain validation, not a statistically
  reliable performance conclusion.

Lead reports with validity and limitations before TTFT, TPOT, throughput, SLO
attainment, or migration-delay comparisons.
