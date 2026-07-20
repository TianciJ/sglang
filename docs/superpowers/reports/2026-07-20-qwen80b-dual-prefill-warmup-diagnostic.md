# Qwen80B dual-prefill warm-up diagnostic

## Technical summary

The valid dual-warm-up run completed all 40 requests with no request errors,
exactly 10,000 completion tokens per request, and `finish_reason=length`.
Client-observed TTFT attainment was 40/40. The startup anomaly seen after a
long-only warm-up did not recur: `qwen80b-01`, the first short request, fell
from 7.600432 seconds to 0.126503 seconds.

The strongest within-run evidence is the short warm-up itself. A 647-token
short warm-up took 5.726731 seconds client-observed TTFT and 5,719.61 ms in the
P worker's prefill `forward_duration`. After all worker KV caches were flushed,
the formal 647-token `qwen80b-01` request took 0.126503 seconds client-observed
TTFT and 112.30 ms in P prefill forward. Its `cached_input_len` was zero. This
supports a one-time process-local short-shape initialization path rather than a
prefix-cache benefit or a stable property that makes short prompts slower.

The server logs do not expose a matching named JIT or kernel-compilation event.
Attributing the one-time prefill forward cost to a specific Triton kernel would
therefore be an inference, not a verified result.

## Validity

- Valid run: `20260720T034620Z-upstream-qwen80b-dualwarm-r2`.
- Runner fix commit: `6ab6eda62`.
- Image: `tiancij/sglang-upstream:v0.5.15-clean`, image ID
  `sha256:7dd92779d739364d79af34af65815ddc14e567728e5256f65ac922367161213e`.
- Model: `Qwen3-Next-80B-A3B-Instruct`; model fingerprint
  `2f95d2baa94271b4a5b214aad71bafa4f78ef34d3d2107404d9e188bc86c88b1`.
- Trace SHA256:
  `c5dbbf75c997dfc5d67a18251082f2f246d6c055eb4af5040fbe147f49f4ce5d`.
- Topology: four nodes, GPUs 0-3 per node, TP 4, DP 1, static 1P3D.
- All 60 files listed by the run-owned `INVENTORY.txt` passed local SHA256
  verification after artifact transfer.
- Independent checks found 40 unique request records, 40 completed responses,
  40 outputs of 10,000 tokens, 40 `length` finishes, 40 TTFT passes, and zero
  error rows.
- Exact run-owned containers were removed. GPUs 0-3 were released and
  `nvidia-smi` remained healthy on all four nodes.

The preceding attempt,
`20260720T033150Z-upstream-qwen80b-dualwarm-r1`, is forensic only. Both warm-ups
completed, but formal traffic never started because host Python 3.6.8 lacks
`datetime.fromisoformat`. The runner now keeps the datetime objects directly
and enables Bash ERR-trap inheritance so function-local failures trigger
run-owned teardown.

## Workload and measurement boundary

The serialized trace contains 40 interleaved long and short requests. Requests
arrive every 0.5 seconds, with an additional three-second pause after every ten
requests. Long prompts tokenize to 6,403 tokens and short prompts to 647 tokens.
Every request generates 10,000 natural output tokens with `ignore_eos=true`.

Before measurement, the runner sends two short smoke requests, one natural
10,000-token output probe, one long prefill warm-up, and one short prefill
warm-up. It then flushes KV cache on all four workers. The measured state is
therefore process-warm and KV-cold.

TTFT is client-observed `first_token_time - start_time`. Token-normalized TPOT
is `(last nonempty output event - first nonempty output event) /
(completion_tokens - 1)`. Neither metric is GPU kernel time. Internal stage
claims use server-side `ReqTimeStats` separately. Reported percentiles use the
nearest-rank definition, `ceil(p * n)` on sorted observations.

## Formal-run results

| Metric | Dual warm-up |
|---|---:|
| Requests | 40 |
| TTFT mean | 0.149889 s |
| TTFT P95 | 0.186504 s |
| TTFT maximum | 0.280499 s |
| TTFT attainment | 100% |
| Short TTFT P95 | 0.126503 s |
| Long TTFT P95 | 0.198811 s |
| Token-normalized TPOT mean | 0.014157 s |
| Token-normalized TPOT P95 | 0.014939 s |
| TPOT attainment | 100% |

## Diagnostic comparison

The long-only and dual-warm-up runs have matching image, model fingerprint,
trace hash, topology, GPU allocation, TP/DP settings, ports, and RoCE settings.
For the first ten requests, mean TTFT changed from 5.361982 seconds to 0.166148
seconds. `qwen80b-01` changed from 7.600432 seconds to 0.126503 seconds.

In the long-only run, the completion points for `qwen80b-01` through
`qwen80b-09` cluster around 8.1-8.75 seconds after the first actual request
start. That pattern is consistent with requests waiting behind the first short
shape's one-time slow path. In the dual-warm-up run, all first-ten TTFT values
are between 0.104199 and 0.280499 seconds.

This comparison is diagnostic, not a publishable performance A/B. There is only
one run per policy, the helper runner revisions differ, and the manifests do
not record the helper-code revision. The observed TPOT difference must not be
attributed to the prefill warm-up without matched repetitions.

## Recommended next steps

1. Run at least three matched repetitions of long-only and long-plus-short
   warm-up from one identical runner revision and record that revision in every
   manifest.
2. Keep the post-warm-up KV flush so runtime initialization is not confounded
   with prefix-cache reuse.
3. Add explicit Triton/JIT or first-kernel-call instrumentation around P prefill
   to identify the operation responsible for the 5.72-second first-short cost.
4. Sweep representative prompt lengths to determine whether other prefill
   shapes or chunk boundaries have independent first-use costs.

## Evidence locations

- Valid run root:
  `pd-flip-artifacts/20260720T034620Z-upstream-qwen80b-dualwarm-r2/`
- Client event ledger: `raw/slo_ledger.jsonl`
- Request metrics: `raw/upstream_baseline/request_metrics.jsonl`
- Per-output TPOT data: `raw/upstream_baseline/tpot_tokens.csv`
- Worker logs: `logs/node0.docker.log` through `logs/node3.docker.log`
- Warm-up records: `smoke/long-prefill-warmup.json` and
  `smoke/short-prefill-warmup.json`
- Reproducible TTFT timeline:
  `report/request_prefill_timeline.png`
- Portable diagnostic report: `analysis/report.html`
- Forensic failed attempt:
  `pd-flip-artifacts/20260720T033150Z-upstream-qwen80b-dualwarm-r1/`
