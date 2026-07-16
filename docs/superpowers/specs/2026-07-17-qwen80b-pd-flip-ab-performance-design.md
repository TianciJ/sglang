# Qwen3-Next 80B PD Flip A/B Performance Experiment Design

**Date:** 2026-07-17

**Status:** Approved for implementation

**Model:** `Qwen3-Next-80B-A3B-Instruct`

## Goal

Build a reproducible two-run performance experiment that compares:

1. stock SGLang disaggregated inference with a static `1P3D` topology; and
2. the same initial `1P3D` topology with the PD Flip state machine enabled,
   progressively changing it to `2P2D` after an SLO violation.

Both runs must use the same frozen 40-request trace and the same measurement
definitions. The report must preserve raw data, show SLO attainment, separate
stock SGLang request-path time from PD Flip overhead, and identify the request
and timestamp that opened the SLO trigger gate.

## Experiment Boundary

Qwen3-Next is a hybrid full-attention and GDN model. SGLang represents its GDN
recurrent state as auxiliary Mamba state. The current Prefill Donor path rejects
auxiliary state types, so this experiment will not enable Prefill Donor or
HiCache stitch.

The state-machine run therefore migrates the source Decode worker's complete
request state:

- ordinary full-attention KV pages;
- GDN/Mamba convolution state;
- GDN/Mamba temporal state; and
- request metadata and index mappings.

This experiment evaluates stock static PD inference versus dynamic role
reallocation and active-request migration. It does not evaluate the newer
"Prefill supplies prompt pages while Decode supplies the boundary page and
delta" optimization.

## Chosen Implementation Approach

Use two independent launch modes with shared trace and reporting code.

- The baseline workers start without PD Flip state-machine or runtime role
  switching flags and remain static `1P3D`.
- The state-machine workers start with the runtime role-switch and migration
  capabilities required for a `1P3D -> 2P2D` transition.
- The two modes use distinct container names and artifact directories.
- The model is gracefully stopped and its resources are conditionally verified
  between modes. `docker restart` and automatic kill-and-relaunch loops are
  forbidden.

This design requires a second model load, but it keeps the baseline semantically
clean. A baseline produced by merely leaving the controller stopped while PD
Flip components are initialized is explicitly rejected.

## Initial and Final Topology

Each node runs one SGLang worker using four GPUs (`TP=4`) by default. GPU IDs
remain configurable, but both A and B runs must use the same GPU set.

Initial topology for both runs:

- `node0`: Prefill;
- `node1`: Decode;
- `node2`: Decode source candidate; and
- `node3`: Decode migration target candidate.

The baseline remains in this topology for the full trace.

When the state-machine run triggers:

1. the router drains the selected source Decode worker;
2. 50% of the source worker's active requests are migrated to the target Decode
   worker;
3. the controller observes the system for exactly 3 seconds;
4. the remaining source requests are migrated;
5. old source state is released only after target commit; and
6. the source changes role from Decode to Prefill and the router publishes the
   final `2P2D` topology.

## Frozen Trace

The runner generates one immutable trace and reuses it for both runs.

### Request population

- request count: 40;
- 20 short prompts of approximately 1,000 Chinese characters;
- 20 long prompts of approximately 10,000 Chinese characters;
- short and long requests are deterministically interleaved;
- `max_tokens=10000` for every request;
- streaming is enabled;
- EOS is ignored;
- a custom logit processor forces one known token/character so every successful
  request keeps generating until the maximum token count; and
- the final raw output is represented by hashes, first/last samples, token
  count, and forced-token mismatch count rather than a huge duplicated text
  file.

### Prefix construction

The only shared request prefix is the model's unavoidable chat template. A
run-specific and request-specific nonce appears at the first user-content
position, followed by request-specific prompt text. Shared text later in a
prompt cannot form a radix prefix hit.

The experiment records actual prompt token count, matched prefix token count,
and hit ratio. Expected hit ratios are below about 5% for short prompts and
below about 1% for long prompts, but measured values, not expectations, are
authoritative.

### Arrival schedule

Requests arrive in four waves of ten:

- within a wave: one request every 0.5 seconds;
- each wave lasts 4.5 seconds from its first to its last request;
- after the last request in a wave, no new request arrives for 3 seconds;
- wave starts are therefore at 0.0, 7.5, 15.0, and 22.5 seconds; and
- all requests have been submitted by approximately 27 seconds.

The burst arrival rate is 2 requests/second and the average injection rate over
the 27-second schedule is approximately 1.48 requests/second. Existing requests
continue decoding during gaps.

## SLO Definitions

### Per-request latency objectives

- short-prompt TTFT: at most 2 seconds;
- long-prompt TTFT: at most 5 seconds; and
- TPOT: at most 50 milliseconds per token for both prompt classes.

For reference, 10,000 generated tokens at the TPOT objective imply an
approximate end-to-end budget of 502 seconds for short prompts and 505 seconds
for long prompts. End-to-end latency is reported but is not used as an online
trigger because it is only known after a long request completes.

### Cluster trigger objectives

- rolling SLO window: 10 seconds;
- enter threshold: attainment below 90%;
- recovery threshold: attainment at or above 95%;
- minimum eligible TTFT samples: 10 requests;
- minimum eligible TPOT samples: 100 token intervals; and
- controller poll interval: 250 milliseconds.

The online Decode signal is TPOT interval attainment, not completed-request
average TPOT, so the controller can react while 10,000-token streams are still
running.

### Reported attainment rates

The report includes:

1. TTFT request attainment;
2. TPOT interval attainment;
3. average-TPOT request attainment; and
4. joint request attainment, requiring both TTFT and average TPOT to pass.

The baseline also runs the same external, read-only SLO observer. It records the
time at which a state machine would have triggered but performs no mutation.
This permits aligned pre-trigger and post-trigger slices without adding state
machine logic to the baseline workers.

## Timing Model

### Stock SGLang request path

Both runs record the same stages for every request:

1. scheduled arrival;
2. actual client send;
3. router receive, selection, and dispatch;
4. tokenization completion;
5. radix-prefix match completion;
6. Prefill queue entry and exit;
7. Prefill compute;
8. Prefill-side standard PD metadata and transfer preparation;
9. standard P-to-D state transfer;
10. Decode-side receive, polling, validation, and index commit;
11. Decode queue entry and exit;
12. first Decode token;
13. subsequent per-token Decode intervals; and
14. the 10,000th generated token and request completion.

Derived request metrics include scheduled-send delay, router time, tokenizer
time, prefix-match time, Prefill queue time, Prefill compute time, P-to-D
transfer time, Decode receive/commit time, Decode queue time, TTFT, average/P50/
P95/maximum TPOT, end-to-end latency, completion-token integrity, and errors.

The same stage-event schema and instrumentation are used in both runs so the
state-machine report necessarily contains the full stock path rather than only
migration phases.

### PD Flip-only stages

The state-machine run additionally records:

1. first SLO threshold crossing;
2. controller detection and polling lag;
3. router source drain;
4. request snapshot and manifest construction;
5. target KV and Mamba slot allocation;
6. first-batch base-state transfer;
7. first-batch delta transfer, validation, and commit;
8. the exact 3-second observation period;
9. second-batch base-state transfer;
10. second-batch delta transfer, validation, and commit;
11. source-state release;
12. runtime role mutation;
13. router role publication and undrain; and
14. first token after target activation.

Migration accounting separates ordinary KV pages, GDN/Mamba convolution state,
GDN/Mamba temporal state, and metadata. It records logical token ranges, slot or
page counts, bytes, duration, effective bandwidth, target validation, and commit
time. Per-request reporting also identifies the largest migration-period token
gap and the time required for TPOT attainment to recover.

### Comparison windows

Both runs are summarized over:

- the full run;
- each arrival wave;
- the interval before threshold crossing;
- the first migration-equivalent window;
- the 3-second observation-equivalent window;
- the second migration-equivalent window; and
- the stable post-switch-equivalent window.

For the baseline, equivalent windows are aligned using its observer timestamp
and the state-machine run's measured phase durations. Raw, unaligned wall-clock
series remain available so this normalization cannot hide behavior.

## Artifacts

Each quick-validation pair creates one run directory:

```text
run/
|-- manifest.json
|-- trace/
|-- baseline/
|   |-- raw/
|   |-- logs/
|   `-- metrics/
|-- state_machine/
|   |-- raw/
|   |-- logs/
|   |-- controller/
|   `-- metrics/
`-- comparison/
    |-- summary.json
    |-- request_comparison.csv
    |-- stage_timings.csv
    |-- slo_timeseries.csv
    |-- migration_timings.csv
    |-- report.md
    `-- timeline.svg
```

Raw data includes the frozen request trace, schedule manifest, per-request
events, token interval timestamps, SLO ledger, router events, worker status and
load samples, migration state, controller journal, process logs, configuration
snapshots, code hashes, model fingerprint, clock information, and lightweight
host/GPU/network telemetry.

The comparison report splits short and long prompts, reports every request,
identifies the request that opened the trigger gate, separates stock-path time
from PD Flip overhead, and links every aggregate back to its raw source.

## Run Safety

### Preflight

Before either run, the runner verifies:

- SSH reachability to all four nodes;
- matching repository code hashes;
- the configured container image;
- the complete model directory on every node;
- synchronized clocks or recorded clock offsets;
- configured GPU IDs are free of unrelated processes;
- required ports are free;
- no conflicting experiment container exists; and
- the full launch command passes local argument validation.

The runner never downloads or copies model files automatically. Based on the
latest inventory, the model initially exists only on node102; preflight must
fail with an explicit missing-node list until an operator has safely distributed
it.

### Start and stop gates

- Workers start sequentially, and the next worker starts only after the current
  worker is healthy.
- The router starts only after all four workers are healthy and expose the
  expected roles.
- A run cannot finish successfully until all 40 requests reach a terminal
  state and raw artifacts are flushed.
- Before changing modes, router admission is paused and workers are drained.
- Containers receive a graceful stop with a long timeout.
- The next mode starts only after the prior mode's owned processes, GPU
  contexts, ports, and containers are gone and node health remains normal.
- Any stop timeout, stale owned GPU process, NVIDIA Xid, kernel hung task,
  RDMA/NCCL teardown error, DNS failure, or SSH degradation aborts the pair.
- An abnormal teardown is preserved for diagnosis and is never followed by an
  automatic SIGKILL-and-relaunch sequence.

`docker restart`, unbounded retries, and commands that target other users'
processes are forbidden.

## Quick Validation Scope

The first implementation defaults to one baseline run and one state-machine
run. The report labels this as a functional quick validation and does not claim
statistical significance. A later configuration may run three or more repeats
per mode without changing the trace, metric, or report schema.

## Error Handling and Validity

The paired experiment is invalid if any of the following occurs:

- the trace does not contain exactly 40 valid requests;
- a successful response produces other than 10,000 completion tokens;
- the baseline role topology changes;
- the state-machine run does not perform the 50% batch, 3-second observation,
  remaining batch, and final `2P2D` commit;
- required timing fields cannot be joined by request and run identifiers;
- monotonic timestamps move backward on a single process;
- model, code, GPU allocation, trace, or SLO definitions differ across modes;
- a worker or measurement process fails; or
- raw aggregates cannot be recomputed.

Invalid runs retain all artifacts and produce a validity report explaining the
failure. They must not produce a conclusion that either mode is faster or has
better SLO attainment.

## Verification Strategy

Implementation follows test-driven development. Automated tests cover:

- deterministic 40-request construction;
- 20/20 short-long composition;
- nonce placement and low-prefix construction;
- exact four-wave arrival offsets with 0.5-second spacing and 3-second gaps;
- forced 10,000-token request settings;
- TTFT, TPOT, joint attainment, and trigger-gate calculation;
- observer-only baseline behavior;
- two-batch 50%/remaining partitioning and exact observation duration;
- stock and PD Flip stage-event joins;
- KV versus Mamba byte accounting;
- A/B aggregation and validity checks;
- report regeneration from raw fixtures; and
- a runner dry-run that performs no SSH, Docker, process, or filesystem
  mutations outside its temporary test directory and never prints secrets.

The implementation is complete only when the focused tests pass, the existing
PD Flip test suite remains green, shell syntax checks pass, and a local dry-run
produces the expected commands and artifact manifest without external side
effects.

## Explicit Non-Goals

- Running the experiment during implementation without a separate user request.
- Automatically copying the 152 GB model between nodes.
- Reusing DeepSeek-specific TP8/DP8 or Prefill Donor runner assumptions.
- Claiming final performance conclusions from one run per mode.
- Adding Qwen3-Next Prefill Donor support in this work item.
