# PD Flip DeepSeek-V3.1 Trace40 Design

## Goal

Adapt the existing four-node PD Flip experiment from a small TP1 model to the
real `/models/deepseek_v3.1_terminus` checkpoint on four 8xH20-3e nodes. Keep
the intended migration ownership contract:

```text
[0, B)   original Prefill node
[B, C0)  source Decode node
[C0, C1) source Decode delta
```

Run a 40-request trace in which every request has a distinct Prompt and emits
exactly 10,000 generated tokens. The generated token is forced to one verified
single-character token, so later experiments control output duration through a
single `max_tokens` setting rather than rewriting prompts.

The first live target remains 2P2D. Each logical P or D worker occupies one
complete eight-GPU node.

## Confirmed environment

- Four worker nodes, each with 8 NVIDIA H20-3e GPUs and roughly 140 GiB GPU
  memory per device.
- The model is present as `/models/deepseek_v3.1_terminus`.
- The checkpoint architecture is `DeepseekV3ForCausalLM` with 61 layers,
  FP8 weights, MLA, and 256 routed experts.
- The model context limit is 163,840 tokens.
- The existing PD Flip experiment and controller were validated with one
  scheduler/DP rank per worker, not DeepSeek DP Attention.

## Deployment topology

Each node launches one SGLang worker endpoint with:

```text
--model-path /models/deepseek_v3.1_terminus
--tp-size 8
--dp-size 8
--enable-dp-attention
--enable-custom-logit-processor
```

The checkpoint is already FP8, so no `--quantization fp8` argument is added.
DeepGEMM kernels are precompiled before the measured run. The first startup and
kernel compilation are warm-up operations and are excluded from experiment
timings.

The four HTTP endpoints still represent the experiment's four role-bearing
workers. Internally, however, each endpoint has eight DP Attention scheduler
ranks. A request's KV belongs to one routed DP rank rather than to an
undifferentiated node-wide KV pool.

This design intentionally adapts PD Flip to DP8 instead of running a TP8/DP1
shortcut. The experiment is meant to represent the real DeepSeek serving
layout.

## DeepSeek MLA KV layout

DeepSeek-V3.1 uses MLA rather than Qwen-style per-head MHA KV. SGLang's
`MLATokenToKVPool` exposes the latent KV and RoPE components through the pool's
contiguous buffer interface, so the page transfer mechanism remains usable.
PD Flip must treat the pool layout as an opaque, versioned transfer contract.

For this checkpoint, the approximate unquantized KV footprint is:

```text
(kv_lora_rank 512 + qk_rope_head_dim 64) * 2 bytes * 61 layers
= 70,272 bytes/token
~= 68.6 KiB/token
```

At a page size of 64 tokens, one full Prompt page is about 4.29 MiB before
transport metadata. Exact byte counts are taken from the runtime buffer
descriptors and recorded in artifacts; the estimate is not used for allocation
or correctness.

Every migration manifest carries and validates:

- `page_size`;
- `kv_layout` (`mla` for this experiment);
- tensor dtype and element size;
- layer count and per-token shape;
- model/config fingerprint;
- source and target SGLang build revision.

A mismatch fails before receiver allocation. PD Flip never attempts to stitch
MHA and MLA layouts or two different model revisions.

## DP-rank ownership

The request identity must include all locations needed to find its KV:

- `source_decode_worker` and `source_decode_dp_rank`;
- `prefill_donor_worker` and `prefill_donor_dp_rank`;
- `target_decode_worker` and `target_decode_dp_rank`;
- transfer `source_tp_rank`/buffer shard identity where required by the
  backend;
- the original Prefill bootstrap identity and rooms.

The original Prefill DP rank is captured during normal P-to-D routing and
retained by the Decode request. The source Decode rank captures its own routed
DP rank. The controller chooses exactly one target DP rank with enough request
and KV capacity, and both donors transfer into that same rank's staged request.

Node URL alone is not sufficient identity. Missing, ambiguous, or conflicting
rank metadata is a typed preflight failure.

All per-rank scheduler operations filter manifests by the rank named in each
request. A rank must reject a request assigned to another rank rather than
silently returning an empty result. The worker-level controller aggregates the
eight rank responses and requires exactly one owner for each request and
exactly one successful target receiver.

## Dual-source migration chain

For each request:

- `P = len(origin_input_ids)`;
- `B = floor(P / page_size) * page_size`;
- `C0` is source Decode committed KV at the initial snapshot;
- `C1` is committed KV after quiescence and delta capture.

The target Decode rank allocates one held request and three staged logical
ranges:

```text
Prefill donor:  [0, B)
Source D base:  [B, C0)
Source D delta: [C0, C1), including protocol-required overlap
```

The target does not perform target-local HiCache prefix matching in this mode.
The original Prefill rank restores complete Prompt pages from its L1/L2/L3 as
needed and sends `[0, B)`. The source Decode rank sends the Prompt boundary page
and existing Decode KV in `[B, C0)`, then sends the final delta after quiescing.

Both donors address the selected target DP rank. Existing code that derives the
destination solely from local `tp_rank` is replaced by explicit manifest-based
destination rank resolution.

The request remains non-runnable until all required ranges are ready. Target
validation proves:

- every declared range has arrived;
- all physical slots are valid;
- ranges cover `[0, C1)` exactly once by logical owner;
- page and layout metadata agree;
- the target request is still held;
- sampling state is complete.

Only then does one atomic commit publish `req_to_token_pool[:C1]`. Activation,
source release, and role transition follow commit. Any failure aborts staged
target state and leaves the source request authoritative.

## 40-request workload

### Prompt shape

The trace remains 40 requests and preserves the existing long/short mixture:

- 20 requests target approximately 10,000 input characters;
- 20 requests target approximately 1,000 input characters;
- all 40 request bodies are different.

Each user message begins with a deterministic high-entropy request nonce and
then uses request-specific content and block ordering. Chat-template tokens may
still be common, but no substantive user Prompt prefix is shared across two
requests. A DeepSeek tokenizer preflight reports pairwise common-prefix token
length and rejects accidental reuse above the configured chat-template
allowance.

The unique nonce and content are deterministic from the trace seed, so the
exact trace is reproducible.

### Single output-length control

`max_tokens` has one source of truth, exposed by the experiment launcher as:

```text
TRACE_MAX_TOKENS=10000
```

Trace construction copies this value into the top-level trace record and the
OpenAI request body. A validation pass rejects disagreement. Profile-specific
hard-coded output lengths do not override it. Future runs change only
`TRACE_MAX_TOKENS`.

For the accepted run:

```text
40 requests * 10,000 generated tokens = 400,000 generated tokens
```

### Deterministic repeated-character generation

A prompt instruction alone is insufficient: the model may emit punctuation,
another character, or EOS. The servers therefore enable SGLang custom logit
processors, and the request uses a built-in forced-single-token processor.

Before trace generation, the DeepSeek tokenizer validates a configured visible
character:

```text
encode(character, add_special_tokens=False) -> exactly one token ID
decode([token_id]) -> exactly the same character
```

The selected `forced_token_id` and `forced_text` are recorded in trace metadata.
On every Decode step, the processor sets all logits to negative infinity except
that token. Requests also use:

```text
temperature = 0
ignore_eos = true
no stop strings or stop token IDs
stream = true
```

Generation therefore ends only when `len(output_ids) == max_tokens`.

### Sampling-state migration

The current PD Flip manifest does not preserve `Req.custom_logit_processor`,
and its JSON-safe sampling serializer drops nested dictionaries such as
`SamplingParams.custom_params`. That would remove the forced-token constraint
after target adoption.

The migration manifest is extended to preserve and validate:

- `custom_logit_processor`;
- JSON-safe nested `sampling_params.custom_params`;
- `forced_token_id` and `forced_text`;
- `ignore_eos`;
- `max_new_tokens` and already-generated output length;
- stop settings and all other generation-affecting sampling fields.

The scheduler-injected `custom_params["__req__"]` object reference is never
serialized. The target `Req` constructor recreates that back-reference so the
processor observes the adopted target request rather than the source object.

The target reconstructs this state before activation. It rejects a missing or
changed processor, invalid forced token, or an output budget inconsistent with
the source. The target continues the original total budget; it does not restart
a fresh 10,000-token budget after migration.

## Trace and artifact behavior

The scheduled trace records, per request:

- request ID, nonce, prompt kind, characters, and actual token count;
- `max_tokens` and forced token metadata;
- arrival offset and SLO thresholds;
- original Prefill worker/rank;
- source and target Decode worker/rank after routing;
- completion tokens, finish reason, error, and migration session.

Storing 400,000 repeated visible characters in every derived report is
unnecessary. Raw HTTP stream evidence is retained according to the existing
artifact policy, while the compact result stores:

- exact output-token count;
- first and last output samples;
- an incremental content hash;
- count of tokens differing from `forced_token_id`.

The acceptance report does not infer token count from character count.

Timeouts are separated into server startup, workload completion, migration
phase, and post-run health windows. The old small-model 900-second assumption
is not treated as a DeepSeek-V3.1 completion guarantee. A no-migration baseline
first measures 40x10,000 completion time, after which the measured PD Flip run
uses a bounded timeout with explicit margin and reports the chosen value.

## Controller changes

The controller currently rejects multiple DP ranks. That guard is replaced by
DP-aware orchestration:

1. Query every worker endpoint and collect all eight scheduler-rank states.
2. Build a request-to-owner map and reject zero or multiple owners.
3. Select target DP ranks using request capacity, free KV pages, and role state.
4. Group Prefill donor calls by worker and DP rank.
5. Group source Decode operations by worker and DP rank.
6. Prepare target receivers on their declared target DP ranks.
7. Aggregate base-ready, quiesce, delta-ready, validate, commit, and activate
   barriers across all involved ranks.
8. Do not advance a worker role transition until every participating rank has
   reached the required barrier or the session has aborted cleanly.

Worker-level status includes both aggregate counts and per-rank detail. A
partial DP8 success is not reported as a successful migration.

## L3 retention

The original-P donor contract from the Prefill-donor design remains in force.
The dedicated Mooncake store is enlarged and reset before the run, Prefill uses
write-through, and donor transfer waits for verified backup coverage.

This provides a capacity-bound experiment guarantee, not an absolute
request-level Mooncake pin. Acceptance fails on any required donor miss,
eviction, Put failure, or incomplete restored range. Store capacity is sized
from the tokenized trace and runtime MLA bytes-per-token with headroom, then
validated against host memory before launch.

## Observability

In addition to the existing full-chain timestamps, record per request and per
rank:

- tokenizer start/end and Prompt token count;
- P L1/L2/L3 match and restore coverage;
- P donor restore start/end and sent bytes;
- source D base snapshot and transfer start/end;
- SLO sample that crossed the threshold and the controller decision time;
- source quiesce request/acknowledgement;
- delta snapshot and transfer start/end;
- target receiver completion for each range;
- target validation, atomic commit, activation, and first post-migration token;
- source release and role-flip barriers;
- output token count and forced-token mismatch count.

Every event includes monotonic time, wall-clock time, request ID, session ID,
worker, DP rank, and phase. Reports align clocks using the existing node clock
preflight and keep raw events for independent reconstruction.

## Test strategy

Tests are added before production changes and cover:

1. DeepSeek MLA pool metadata and model fingerprint validation.
2. Manifest round-trip of source/P/target DP rank identity.
3. Per-rank filtering selects exactly one source and target owner.
4. Both donor streams address one declared target DP rank.
5. DP8 aggregation rejects missing, duplicate, and partial-rank results.
6. Existing DP1 behavior remains unchanged when DP Attention is disabled.
7. Trace generation creates exactly 40 distinct Prompts with the configured
   20/20 long/short mix.
8. One `TRACE_MAX_TOKENS` value controls both trace and request body.
9. Tokenizer preflight accepts only a visible one-token/one-character value.
10. Forced-token processing masks every other vocabulary entry.
11. Migration manifest round-trips custom processor and nested custom params.
12. A migrated request continues the remaining total output budget.
13. Output validation detects early EOS, wrong finish reason, wrong token count,
    or any different generated token.
14. Strict Prefill-donor ownership and atomic stitch tests remain passing under
    MLA page metadata.

## Execution sequence

The live rollout is staged:

1. Run unit and controller simulation tests locally.
2. Precompile DeepGEMM and launch one DeepSeek worker per node without workload.
3. Verify model fingerprint, TP8/DP8 topology, rank health, Mooncake access, and
   forced-token tokenizer selection on all nodes.
4. Run one short request with a small `TRACE_MAX_TOKENS` to validate exact
   repeated-token behavior.
5. Run a small no-flip multi-rank smoke test.
6. Run a small PD Flip test proving P-prefix/source-D-delta ownership and
   sampling-state continuity.
7. Reset L3 and run the 40x10,000 no-migration baseline.
8. Reset L3 and run the measured 40x10,000 PD Flip experiment.
9. Validate artifacts, generate the full timing report, and stop only the
   experiment-owned processes.

Failure at any stage stops progression to the larger run. It does not fall back
to target-local prefix matching or source-full Prompt transfer in strict mode.

## Acceptance criteria

- DeepSeek-V3.1 runs on all four 8xH20 nodes with TP8/DP8 Attention.
- All worker ranks report the same model and KV-layout fingerprint.
- The trace contains exactly 40 distinct Prompts and the intended long/short
  distribution.
- Every request has `max_tokens == 10000` in both trace metadata and request
  body.
- Every request finishes with exactly 10,000 completion tokens and a length
  finish reason.
- Every generated token equals the verified forced token before and after
  migration.
- Every migrated request records P ownership `[0, B)`, source D ownership
  `[B, C0)`, and source D delta through `C1` on one target DP rank.
- No target-prefix substitution, source-full Prompt fallback, donor miss,
  Mooncake eviction, invalid slot, or partially committed request occurs.
- SLO trigger, migration stages, first post-migration token, and role-flip
  completion are reconstructable from raw timestamps.
- All four workers remain healthy through the post-run observation window.

## Non-goals

- Adding a production Mooncake pin or lease API.
- Changing normal HiCache prefix reuse outside strict Prefill-donor PD Flip.
- Recomputing missing original-P Prompt KV.
- Enabling speculative decoding in the first correctness run.
- Optimizing DeepSeek throughput before the DP8 migration path is proven
  correct.
- Treating Prompt instructions alone as proof of deterministic output length or
  content.
