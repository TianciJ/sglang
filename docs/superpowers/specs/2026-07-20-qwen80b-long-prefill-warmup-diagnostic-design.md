# Qwen80B Long-Prefill Warmup Diagnostic Design

## Question

Determine whether the repeatable first-wave TTFT spike in the clean upstream
Qwen80B `1P3D` trace is removed when the process executes one representative
long-Prompt Prefill before measurement, and retain enough logs to distinguish
an observed effect from a kernel-level explanation.

## Controlled change

Keep the clean image, model, four nodes, GPUs, topology, router, frozen 40-row
trace and trace hash, request schedule, generation parameters, validation, and
reporting identical to the retained clean-upstream runs. The only intentional
change is one non-measured long-Prompt warmup before cache flush and formal
trace replay.

The warmup copies the request body from the first serialized trace row, removes
the experiment-only SLO metadata, changes `max_tokens` to `1`, and waits for a
successful response. It therefore exercises the exact 6.4k-token Prompt shape
without spending time on a 10,000-token Decode. After it completes, the runner
flushes all four workers' KV caches. Process-level compiled code, allocator
state, and workspaces remain warm while Prefix/KV reuse is removed.

## Evidence

Record a `smoke/long-prefill-warmup.json` client record containing the selected
trace request ID, declared prompt-token count, UTC start/first-output/end,
monotonic TTFT and total duration, response status, completion-token count, and
finish reason. Preserve the complete timestamped worker/router logs as before.

Also retain one timestamp-window log per worker and router covering the warmup.
Search those windows without filtering the retained originals for `compile`,
`JIT`, `Triton`, `autotune`, `workspace`, allocator, Prefill/extend, request-time
stats, CUDA, NCCL, OOM, and traceback evidence. Absence of such a line must be
reported as absence of direct proof, not proof that compilation did not occur.

## Validity and interpretation

The run is valid only if the warmup succeeds, all four cache flushes are proven,
all 40 formal requests complete uniquely with exactly 10,000 completion tokens
and `finish_reason=length`, no request errors occur, complete raw event evidence
is present, and run-owned resources are torn down safely.

Compare the first ten and remaining formal TTFTs against the three retained cold
runs. A removed first-wave spike supports the hypothesis that process-level
long-shape warmup was missing. It does not by itself identify a particular GPU
kernel; that stronger claim requires profiler evidence.
