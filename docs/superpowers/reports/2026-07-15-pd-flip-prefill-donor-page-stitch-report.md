# PD Flip Prefill Donor Page Stitch Experiment

Date: 2026-07-15

## Outcome

The four-node PD Flip experiment completed successfully with the requested KV
provenance:

- the original Prefill node supplied complete Prompt pages from `[0, B)`;
- the source Decode node supplied the boundary page and Decode KV from `[B, C0)`;
- the target Decode node skipped target-local HiCache prefix matching;
- no source-full fallback was attempted;
- the source Decode node completed the runtime role switch to Prefill.

The acceptance run used Qwen3-8B, `page_size=64`, and a dedicated 64 GB
Mooncake store. The raw artifact directory is
`/home/tiancij/pd-artifacts/20260715T172500Z-prefill-donor-page64-trace40`
on cloud-099. Its archive is
`/home/tiancij/20260715T172500Z-prefill-donor-page64-trace40.tar.gz`.

## Exact migrated range

The controller migrated request `76426fa481bb44fda6af4340a2f51c07`.

| Quantity | Value |
| --- | ---: |
| Prompt length `P` | 1974 tokens |
| Page size | 64 tokens |
| Donor boundary `B = floor(P / 64) * 64` | 1920 tokens |
| Source snapshot `C0` | 2336 tokens |
| Original Prefill transfer | `[0, 1920)`, 30 pages, 283,115,520 bytes |
| Source Decode transfer | `[1920, 2336)`, 7 pages, 66,060,288 bytes |
| Prompt remainder in the source boundary page | 54 tokens |

The original Prefill restore hit was exactly 1920 tokens. The source page count
was `ceil((2336 - 1920) / 64) = 7`, so the first source page was the mixed
boundary page containing Prompt tokens `[1920, 1974)` and subsequent Decode
tokens. The target reported `target_prefix_match_skipped=true` and provenance
`prefill_donor_and_source_decode`.

## Acceptance checks

- workload requests: 40/40 completed;
- request errors: 0;
- migration outcome: committed;
- controller result: success, `source switched to prefill`;
- Prefill donor misses: 0;
- source-full fallback attempts: 0;
- Mooncake Put failures: 0;
- Mooncake evictions: 0;
- invalid KV index reports: 0;
- invariant failures in the acceptance worker lifetimes: 0;
- all workers idle after the run with 4096/4096 request slots free;
- final roles: node0=P, node1=D, node2=P, node3=D;
- router healthy and Mooncake segment size confirmed as 64 GB.

## Failure found and fixed during validation

The first donor run failed before transfer completion because the independent
`pd_flip_prefill_donor_session` owned one synthetic request-pool slot that the
strict invariant checker did not count. The observed equation was
`available=4095, session_held=0, pd_flip_held=0, total=4096`.

Commit `7d11ee363` adds active donor requests to the existing PD Flip invariant
accounting without relaxing the invariant or changing allocation/release
semantics. The regression tests first reproduced the missing method/accounting,
then passed after the fix. The subsequent page-size-1 and page-size-64 full
chain runs both completed successfully.

## Chain exercised

```text
original P HiCache (L1/L2/L3) -- restore hit >= B -- send pages [0, B) ----+
                                                                         |
source D live KV --------------------------- send pages [B, C0) ----------+--> target D slots
source D post-quiesce delta ---------------- send pages [C0, C1) --------+        |
                                                                                  +--> commit mapping
                                                                                  +--> activate request
                                                                                  +--> source D becomes P
```

The target allocated one destination mapping, received the two base ranges
from distinct senders, received any post-quiesce delta from the source Decode
node, validated the complete mapping, and only then committed and activated the
request.
