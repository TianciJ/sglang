# Clean upstream natural-output baseline design

## Objective

Run one reproducible Qwen3-Next-80B-A3B-Instruct clean-upstream 1P3D baseline
with 40 requests and 10,000 naturally sampled completion tokens per request.
The inference data plane remains the immutable
`tiancij/sglang-upstream:v0.5.15-clean` image and contains no PD Flip state
machine, runtime role switching, HiCache, Prefill donor, or host source mount.

## Frozen workload

Create a new trace once from the retained 40-request source. Preserve request
IDs, prompt text, short/long alternation, arrival offsets, SLOs, model,
`temperature=0`, `stream=true`, `max_tokens=10000`, `ignore_eos=true`,
`stop=null`, and streaming usage. Remove `custom_logit_processor`,
`forced_text`, and `forced_token_id`. Record both the source hash and the new
trace hash. Never rewrite the new trace during replay. Any later candidate
comparison must reuse the identical new trace and hash.

## Timing boundary

TTFT is the elapsed time from client request send to the first non-empty output
event. Natural output can buffer or coalesce tokenizer text, so SSE event count
is not assumed to equal completion-token count. The primary request-level TPOT
is:

```text
(last non-empty output receive time - first non-empty output receive time)
/ (usage.completion_tokens - 1)
```

All SSE receive timestamps and event gaps remain raw evidence. Their
distribution is labeled `stream_event_gap`, not per-token TPOT.

## Readiness and measurement

Run the checked-in preflight first. Start four clean workers on GPUs 0-3 with
TP4/DP1 and roles 1P3D, then start the official upstream router. Run a unique
32-token smoke and one unmeasured 10,000-token natural-output probe. Flush all
four worker caches; if flushing cannot be proven, relaunch only exact run-owned
workers and repeat readiness gates. Replay the frozen 40-request trace once.

## Validity gates

The run is valid only if all 40 unique requests complete without error, usage
reports exactly 10,000 completion tokens for every request, every finish reason
is `length`, the trace and provenance hashes match, every request has first and
last output timestamps, all raw SSE events are retained, and exact run-owned
containers stop cleanly. Ledger and event-gap row counts are recorded rather
than fixed at the forced-output values 400,040 and 399,960.

This is one measured run. It is an absolute baseline, not a statistically
reliable winner and not a controlled comparison with the historical forced
output run.
