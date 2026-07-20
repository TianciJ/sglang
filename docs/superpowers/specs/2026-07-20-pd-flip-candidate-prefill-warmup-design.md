# PD Flip Candidate-P Prefill Warmup Design

## Objective

Determine whether the first-use TTFT staircase previously seen in the Qwen80B
PD Flip state-machine experiment is removed when every worker that may execute
Prefill has already exercised representative Prefill shapes in its own live
process before measured traffic starts.

This is a state-machine chain diagnostic. It is not a controlled comparison
against the historical baseline because the historical pair used different
revisions and did not apply the same candidate-role warmup policy.

## Controlled workload

Use the existing four-node Qwen3-Next-80B-A3B-Instruct runner, TP 4, DP 1,
GPUs 0-3, frozen 40-request trace, 10,000 forced output tokens, 1P3D initial
topology, 50% first migration, two-second observation period, and the existing
SLO/controller settings. HiCache stitching and Prefill donor behavior remain
disabled.

The measured state-machine trace must not start until the cache and warmup
gates below pass. Observer, controller, and migration sampling also start only
after those gates, so warmup traffic cannot enter measured metrics or trigger
the state machine.

## Persistent compilation cache

Each host receives a node-local persistent cache directory mounted into the
worker container. The cache namespace is derived from immutable execution
provenance: image ID, code revision, model fingerprint, GPU model, TP/DP,
dtype/backend-affecting launch arguments, and the warmup profile version.

The container redirects SGLang, TorchInductor, Triton, CUDA, Torch extension,
and TVM-FFI/JIT cache roots beneath that mount. The cache is reusable across
run-owned container lifetimes with matching provenance but is never treated as
proof that a process is warm. A cache provenance mismatch creates a distinct
namespace instead of deleting or overwriting an existing one. Caches remain
node-local for the first implementation; cross-node copying is outside scope.

The run manifest records the cache namespace, host path, container path,
cache-affecting provenance, and before/after file-count and byte snapshots.
No secret is stored in the cache manifest.

## Candidate-P warmup sequence

All four workers are candidate P nodes. Start them in the normal 1P3D topology
and pass the existing health and runtime-role gates. Start the router, but do
not start measurement helpers.

Use one immutable warmup profile on every candidate:

1. The first trace row labelled `long`, copied without experiment-only custom
   logit-processor metadata and with `max_tokens=1`.
2. The first trace row labelled `short`, transformed the same way.

Validate the tokenizer-reported prompt lengths as greater than 6,000 tokens
for long and between 500 and 1,000 tokens for short. Each warmup must return
HTTP 200, produce a non-empty first output, produce exactly one completion
token, and finish for reason `length`.

Warm node0 in its initial P role. Then warm node1, node2, and node3 serially:

1. Confirm the candidate is idle and currently D.
2. Drain it at the router, switch its worker runtime role to P, publish the P
   role at the router, and wait until all TP shards and the active event loop
   report P.
3. Drain node0 so the candidate is the only routable P.
4. Send the long and short warmups through the full router path. At least two
   remaining D workers stay routable for the one-token Decode tail.
5. Drain the candidate, switch it back to D, publish D at the router, wait for
   all shards and the event loop to report D, then undrain node0.

At most one original D is temporarily P. Each transition and request is
written to a timestamped warmup journal. On failure, stop scheduling new
warmups, attempt only the bounded role/topology restoration for the current
run, preserve evidence, and mark the run invalid. Do not relaunch under the
same RUN_ID.

## Post-warmup validity gate

After all eight warmup requests complete:

- flush KV/Radix cache directly on all four workers;
- require every flush response to succeed;
- require zero running and waiting requests on every worker;
- require node0 to report P and node1-node3 to report D on every TP shard and
  active event loop;
- require every worker's internal PD Flip FSM to remain `safe` with direction
  `none`, proving the warmup did not enter a state-machine transition;
- require the router to report exactly 1P3D with no draining workers;
- save the final worker and router snapshots;
- verify the persistent cache is readable and record its post-warmup snapshot;
- retain the full four-node cache provenance material plus manifest references
  to the before/after snapshot sets;
- assign every role or drain transition an action ID and retain full worker or
  router snapshots on both sides of the mutation.

Only then start the migration sampler, observer, controller, and measured
40-request trace.

## Evidence and acceptance criteria

Retain per-candidate long/short request records, the warmup action journal,
role snapshots before and after each transition, router snapshots, cache
provenance and size snapshots, flush responses, and timestamp-window worker and
router logs. Warmup records must be marked `measured=false`.

The diagnostic is valid only if all four nodes pass both Prefill warmups, all
four flushes pass, initial topology is restored, all 40 measured requests
finish uniquely with exactly 10,000 matching output tokens and reason
`length`, the controller reaches `role_flip_complete`, final topology is 2P2D,
raw event timing is complete, and teardown/postflight health checks pass.

Primary diagnostic interpretation focuses on the first-wave TTFT timeline and
the promoted worker's first measured Prefill after D-to-P. Disappearance of the
staircase supports missing candidate-P process warmup as the cause; it does not
identify a specific compiler or kernel without profiler evidence.
