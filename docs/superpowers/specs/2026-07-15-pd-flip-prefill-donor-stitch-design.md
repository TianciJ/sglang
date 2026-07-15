# PD Flip Prefill Donor Stitch Design

## Goal

Change PD Flip migration so the target Decode worker does not decide Prompt KV
ownership by matching its own HiCache. For every migrated request, the worker
that originally executed Prefill supplies all complete Prompt pages, while the
source Decode worker supplies the Prompt/Decode boundary page, the remaining
committed Decode KV, and the migration delta.

The first live acceptance target is the existing four-node, 40-request
interleaved trace.

## Current mismatch

The current target migration path calls `_match_prefix_and_lock(req)` on the
target Decode worker. The resulting target-local HiCache hit length `H` controls
the source transfer range `[H, C0)`. A shared Mooncake hit can contain data
originally written by any worker, and the source Decode fallback can transfer
most or all of the Prompt. Consequently, a successful migration does not prove
that Prompt KV was supplied by the request's original Prefill worker.

## Range definitions

For each request:

- `P = len(origin_input_ids)` is the Prompt token length.
- `page_size` is the KV allocator page size.
- `B = floor(P / page_size) * page_size` is the last complete Prompt-page
  boundary.
- `C0` is the source Decode committed-KV length at initial migration snapshot.
- `C1` is the committed-KV length after source quiescence and delta capture.

The required ownership is:

```text
[0, B)   original Prefill donor; complete Prompt pages
[B, C0)  source Decode; Prompt boundary page plus committed Decode KV
[C0, C1) source Decode; migration delta
```

If `P` is page aligned, `B == P` and the source Decode range contains no Prompt
tokens. If `P` is not page aligned, the source Decode sends the complete page
containing `P`. The Prompt tokens in that boundary page were originally
computed by Prefill and transferred to source Decode during normal PD
disaggregation, but source Decode is their direct donor during migration.

This boundary rule avoids duplicate target pages and keeps every transfer
page-aligned.

## Compatibility and activation

Add an explicit Prefill-donor migration mode rather than changing the existing
target-HiCache stitch behavior globally. The existing behavior remains the
default for deployments that do not enable the new mode. The four-node
experiment enables Prefill-donor mode on every worker and controller.

In Prefill-donor mode:

- target Decode must not call target-local radix/HiCache prefix matching for a
  migrated request;
- source Decode must not use target hit length to choose its initial transfer
  range;
- full-source Decode fallback is not allowed to mask a missing Prefill donor
  page;
- any missing Prefill donor page fails the migration session before ownership
  cutover.

## Donor identity and manifest

The PD router already injects the selected Prefill worker's `bootstrap_host`,
`bootstrap_port`, and `bootstrap_room` into both sides of the original request.
The source Decode `Req` therefore retains the original Prefill transfer
identity.

The migration manifest must preserve and expose:

- original `bootstrap_host` and `bootstrap_port` as the Prefill donor identity;
- original `bootstrap_room` for provenance only;
- `prompt_len = P`;
- `prefill_donor_end = B`;
- `source_decode_start = B`;
- separate migration bootstrap rooms for the Prefill donor transfer, source
  Decode base transfer, and source Decode delta transfer.

The controller resolves `bootstrap_host` to the corresponding worker HTTP URL
from the router worker registry. It must reject missing or ambiguous mappings
instead of selecting an arbitrary Prefill worker.

## Target preparation

The target prepares one held request with staged coverage for `[0, C0)` but no
target prefix ownership.

It creates two independent receiver states:

1. Prefill receiver with target pages covering `[0, B)`.
2. Source Decode receiver with target pages covering `[B, C0)`.

Both receiver states write into the same held request's staged
`req_to_token_pool` mapping. The ranges are disjoint and page aligned, so they
do not race on physical target pages. `B == 0` creates no Prefill receiver;
`B == C0` creates no source base receiver.

The target remains non-runnable until both required receivers report success.
Prepare-time validation checks each staged range independently. Formal mapping
commit and activation remain separate phases.

## Original Prefill donor operation

The original Prefill worker exposes an authenticated PD Flip donor operation.
The controller sends the request identity, `origin_input_ids`, `[0, B)` range,
target transfer endpoint, and dedicated migration room.

The donor operation reconstructs a held lookup request and performs radix and
HiCache matching on the original Prefill worker, not on target Decode. It may
use local L1/L2 data or restore complete pages from Mooncake L3 into temporary
GPU pages. It may send only after the local usable coverage is at least `B`.

The donor then sends the complete page range `[0, B)` through the existing KV
transfer backend. Temporary restore pages, locks, metadata buffers, and sender
state are released after success or abort. The operation does not generate a
new token and does not publish a user-visible response.

If the original Prefill worker can restore fewer than `B` tokens, it reports a
typed `prefill_donor_incomplete` failure containing the expected and restored
lengths. Strict experiment mode aborts the migration; it does not request
Prompt KV from source Decode.

## Source Decode operation

Source Decode continues to own and execute the request during preparation.
Its initial sender uses the fixed page-aligned range `[B, C0)`, independent of
all target cache state.

After both base transfers are ready, the controller quiesces the selected
source requests and captures `C1`. Source Decode sends the existing delta
range, including the page-aligned overlap required by the delta protocol. Since
the overlap is entirely inside the source-Decode-owned side of the boundary,
it does not change Prompt donor ownership.

Source requests are released only after target commit and activation succeed.
Abort keeps or restores source ownership.

## Atomic validation and commit

Before commit, the target validates:

- `0 <= B <= P <= C0 <= C1`;
- Prefill donor transfer covers exactly `[0, B)` when `B > 0`;
- source Decode base transfer covers exactly `[B, C0)` when `B < C0`;
- source Decode delta reaches `C1`;
- all staged slot indices used by `[0, C1)` are valid;
- every token position is covered once by its declared logical owner;
- both base receiver states and the delta receiver are successful;
- the target request is still held and has never entered a runnable queue.

Commit publishes the complete mapping atomically, rechecks
`req_to_token_pool[:C1]`, and changes the request to `ready_to_activate`.
Activation then adopts the request into target Decode. No partially committed
request is runnable.

## L3 capacity and retention contract

The live experiment uses the dedicated Mooncake store on `cloud-099` with:

```text
MOONCAKE_GLOBAL_SEGMENT_SIZE=64gb
```

Workers continue to use `MOONCAKE_GLOBAL_SEGMENT_SIZE=0`, so only the dedicated
store contributes capacity. Prefill workers keep HiCache `write_through`
enabled.

Mooncake currently exposes no request-level key pin through the SGLang storage
interface. This experiment therefore provides a capacity-bound retention
contract rather than an absolute production lease:

- reset the dedicated store immediately before the run;
- use 64 GB, well below the storage node's currently available host memory;
- wait for/verify Prefill L3 backup before donor transfer;
- require original Prefill restore coverage `>= B` for every migrated request;
- record Put failures, eviction count, allocated bytes, and donor misses;
- fail acceptance if any store eviction, Put failure, or donor miss occurs.

A production request-level Mooncake lease or dedicated non-evictable namespace
is outside this experiment's scope.

## Controller sequence

For one migration batch, the controller performs:

```text
1. source D: create manifests and retain ownership
2. target D: allocate held two-range receivers; no prefix match
3. original P donor(s): restore and send [0, B)
4. source D: send [B, C0)
5. target D: report both base ranges ready
6. source D: quiesce and send delta through C1
7. target D: validate and atomically commit [0, C1)
8. target D: activate requests
9. source D: release migrated requests
10. controller: continue observation, second migration, and D-to-P role flip
```

Requests in one batch may have different original Prefill donors. The
controller groups donor calls by resolved Prefill worker but preserves
per-request rooms and results.

## Observability

Per-request status and experiment artifacts must record:

- `prompt_len`, `prefill_donor_end`, `source_decode_start`, `C0`, and `C1`;
- Prefill donor host, expected pages, restored pages, sent bytes, L3 hit length,
  restore duration, and transfer duration;
- source Decode base pages/bytes and delta pages/bytes;
- target receiver status for both base ranges and delta;
- `target_prefix_match_skipped=true`;
- final provenance ranges and stitch mode;
- exact failure phase and typed error.

These fields must make it impossible to label target-HiCache reuse or
source-full fallback as a successful Prefill-donor run.

## Test strategy

Automated tests are written before production changes and cover:

1. Boundary calculation for zero, aligned, and unaligned Prompt lengths.
2. Manifest preservation of original Prefill host/port and fixed `B` ranges.
3. Target preparation does not invoke `_match_prefix_and_lock` in donor mode.
4. Target creates disjoint Prefill `[0, B)` and source `[B, C0)` receivers.
5. Source Decode sends `[B, C0)`, never target-derived `[H, C0)`.
6. Original Prefill full L1/L2/L3 restore sends `[0, B)` successfully.
7. Incomplete Prefill restore fails without source-full fallback.
8. Target commit waits for both base transfers and delta.
9. Abort releases target staged pages and Prefill temporary restore resources.
10. Controller resolves the correct original P and preserves phase ordering.
11. Existing target-HiCache mode remains unchanged when donor mode is off.

## Four-node acceptance

Run the saved 40-request long/short interleaved trace with initial roles
`1P3D`, then perform the existing progressive migration and `D -> P` flip.
Acceptance requires:

- 40/40 requests complete with no request error;
- every migrated request records P coverage `[0, B)` and D coverage `[B, C0)`;
- target prefix matching is skipped for every migrated request;
- no source-full Prompt fallback occurs;
- no Prefill donor miss, Mooncake Put failure, or Mooncake eviction occurs;
- target mappings contain no invalid slot through `C1` after commit;
- target Decode continues generation after activation;
- source worker completes the second migration and switches to Prefill;
- all four workers remain healthy after the existing post-run idle window;
- artifacts contain per-stage latency for P restore, P transfer, D base transfer,
  delta, target validation, commit, and activation.

## Non-goals

- Adding a general production Mooncake pin/lease API.
- Recomputing Prompt KV when original Prefill L3 data is absent.
- Allowing target Decode cache hits to replace the original Prefill donor.
- Changing normal non-PD-Flip prefix reuse behavior.
- Removing the existing target-HiCache stitch mode for other deployments.
