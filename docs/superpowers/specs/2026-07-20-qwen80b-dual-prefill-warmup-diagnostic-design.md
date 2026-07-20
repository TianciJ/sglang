# Qwen80B Dual-Prefill Warmup Diagnostic Design

## Question

Determine whether the repeatable first-wave TTFT spike in the clean upstream
Qwen80B `1P3D` trace disappears when both representative Prefill shape classes
execute before measurement. The experiment must distinguish a process-level
warmup effect from KV-prefix reuse and must retain enough evidence to locate any
remaining delay in Prefill, PD transfer waiting, or Decode.

## Controlled change

Keep the clean image, model, four nodes, GPUs, `1P3D` topology, router, frozen
40-row serialized trace and trace hash, request schedule, generation parameters,
SLOs, validation, reporting, and teardown identical to the retained clean
upstream run `20260720T023903Z-upstream-qwen80b-longwarm-r1`.

The only intentional change is replacing its single long-Prompt warmup with two
sequential non-measured warmups:

1. Copy the body of the first `long` trace row, remove experiment-only metadata,
   set `max_tokens=1`, and wait for successful streamed completion.
2. Copy the body of the first `short` trace row, apply the same transformations,
   and wait for successful streamed completion.

The expected representative token counts are approximately 6,403 and 647, but
the runner must select by `prompt_kind` and validate the tokenizer-reported
counts instead of hard-coding those values. The long request runs first and the
short request second. Warmups are not concurrent and do not enter measured
TTFT, TPOT, SLO, throughput, or completion statistics.

After both warmups complete, flush KV/Radix cache on all four workers without
restarting any run-owned process. This removes Prefix/KV reuse while retaining
compiled kernels, allocator state, communication setup, and other process-level
warm state. If every flush cannot be proven, preserve the attempt as invalid;
do not relaunch and continue under the same run ID because relaunching would
erase the independent variable.

## Warmup validation and evidence

Each warmup must independently satisfy HTTP 200, one non-empty first output,
exactly one completion token, and `finish_reason=length`. Validate that the
long warmup reports more than 6,000 prompt tokens and the short warmup reports
between 500 and 1,000 prompt tokens.

Retain separate client records:

- `smoke/long-prefill-warmup.json`
- `smoke/short-prefill-warmup.json`

Each record contains the selected trace request ID and kind, prompt characters,
tokenizer-reported prompt and completion counts, response status, finish reason,
UTC start/first-output/end, monotonic TTFT and duration, `measured=false`, and
`kv_cache_flushed_after=true`.

Retain timestamp-window worker and router logs spanning both warmups, with the
window start taken two seconds before the long warmup and the window end two
seconds after the short warmup. Preserve the complete unfiltered worker and
router logs separately. Search the retained windows for Prefill batches,
request-time statistics, transfer waiting, compile/JIT/Triton/autotune messages,
CUDA/NCCL faults, OOMs, and tracebacks. Absence of an explicit compilation line
is absence of direct proof, not evidence that compilation did not occur.

## Execution and safety

Use a new unique `RUN_ID` in all container names, PID files, logs, manifests,
and artifact directories. Run the complete four-node read-only preflight before
starting. Do not touch containers, processes, GPUs, ports, mounts, models, or
directories without proving they belong to the new run. Start workers only on
healthy, reachable, unoccupied nodes and pass bounded health and role gates
before starting the router or sending warmups.

On any warmup, flush, workload, validation, monitoring, SSH, GPU, or driver
failure, stop scheduling new requests, preserve the forensic directory, and
gracefully stop only exact run-owned resources. Never use `docker restart`,
wildcard process killing, `pkill`, `killall`, or `kill -9`.

## Validity and interpretation

The run is valid only if both warmups pass, all four post-warmup flushes are
proven, the serialized formal trace hash matches the retained run, all 40
formal requests complete uniquely with exactly 10,000 completion tokens and
`finish_reason=length`, no request errors occur, raw stream-event evidence is
complete, and teardown and postflight health checks pass.

Compare request-level TTFT and absolute first-token times for requests 00-11
against the retained long-only warmup run. The primary diagnostic success
criterion is that request 01 no longer introduces a multi-second barrier and
requests 02-09 no longer form the associated staircase. Also report the long
and short warmup TTFTs and the Prefill-side request-time records.

One run is a diagnostic chain validation, not a statistically reliable
performance comparison. If the spike disappears, the result supports missing
short-shape process warmup as the cause but does not identify a particular GPU
kernel without profiler evidence. If the spike remains, use the retained stage
logs to form the next single hypothesis; do not modify additional variables in
the same run.
