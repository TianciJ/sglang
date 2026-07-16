# Qwen3-Next 80B PD Flip A/B Performance Experiment Implementation Plan

> **For Codex:** Execute this plan task-by-task with tests first. Do not run the four-node live experiment as part of implementation; stop after local tests and dry-run validation unless the user separately authorizes the live run.

**Goal:** Add a safe, reproducible quick-validation harness comparing stock static SGLang `1P3D` with PD Flip progressive `1P3D -> 2P2D` on `Qwen3-Next-80B-A3B-Instruct`, preserving detailed raw timing, SLO, prefix-hit, and migration evidence.

**Architecture:** Generate one immutable 40-request trace, replay it identically in baseline and state-machine modes, collect client SSE timing plus existing SGLang request-time logs and PD Flip status journals, normalize them into one event schema, and regenerate comparison reports entirely from raw artifacts. The baseline uses an observer-only SLO monitor. The state-machine mode enables runtime role switching and full source-D migration, including Mamba/GDN state, while explicitly disabling Prefill Donor and HiCache stitch.

**Tech stack:** Python 3, `unittest`, Bash, SGLang native/OpenAI HTTP APIs, Docker, SSH, Mooncake, standard-library JSON/CSV/XML generation.

---

## Task 1: Build the deterministic Qwen80B trace

**Files:**

- Create: `scripts/playground/disaggregation/pd_flip_qwen80b_trace.py`
- Create: `test/srt/test_pd_flip_qwen80b_trace.py`
- Reuse: `scripts/playground/disaggregation/pd_flip_prepare_trace.py`

**Step 1: Write failing construction tests**

Test `build_qwen80b_trace(run_nonce, model, forced_token_id, forced_text)` for:

- exactly 40 records with stable request IDs;
- exactly 20 `short` prompts near 1,000 Chinese characters and 20 `long` prompts near 10,000 characters;
- deterministic short/long interleaving;
- the run/request nonce at the first user-content position;
- no shared user-content prefix after the nonce;
- `stream=true`, `max_tokens=10000`, `ignore_eos=true`, and the serialized forced-token processor on every request;
- short TTFT SLO 2 seconds, long TTFT SLO 5 seconds, and TPOT SLO 0.05 seconds;
- wave offsets `0..4.5`, `7.5..12.0`, `15.0..19.5`, and `22.5..27.0` in 0.5-second increments.

Run:

```bash
python -m unittest test.srt.test_pd_flip_qwen80b_trace -v
```

Expected: FAIL because the module does not exist.

**Step 2: Implement the trace builder and CLI**

Implement pure helpers:

```python
def build_prompt(*, request_index: int, prompt_kind: str, run_nonce: str) -> str: ...
def build_qwen80b_trace(*, run_nonce: str, model: str, forced_token_id: int,
                       forced_text: str, custom_logit_processor: str) -> list[dict]: ...
def write_trace(trace: Sequence[dict], output: Path, manifest: Path) -> dict: ...
```

The CLI accepts `--run-nonce`, `--model`, `--forced-token-id`, `--forced-text`, `--custom-logit-processor`, `--output`, and `--manifest`. Use fixed Chinese source paragraphs and deterministic repetition/truncation; do not call a model or network service. Store the SHA-256 of the canonical JSONL in the manifest.

**Step 3: Run tests and commit**

```bash
python -m unittest test.srt.test_pd_flip_qwen80b_trace -v
git add scripts/playground/disaggregation/pd_flip_qwen80b_trace.py test/srt/test_pd_flip_qwen80b_trace.py
git commit -m "feat(pd-flip): add deterministic qwen80b trace"
```

## Task 2: Capture stock request-path evidence without changing the hot path

**Files:**

- Modify: `scripts/playground/disaggregation/pd_flip_trace_replay.py`
- Create: `scripts/playground/disaggregation/pd_flip_req_timing.py`
- Create: `test/srt/test_pd_flip_req_timing.py`
- Modify: `test/srt/test_pd_flip_trace_replay.py`

**Step 1: Write failing parser and replay tests**

Add fixtures for Prefill and Decode log lines of the form:

```text
ReqTimeStats(rid=..., bootstrap_room=..., input_len=..., cached_input_len=..., output_len=..., type=Prefill): ...
ReqTimeStats(rid=..., bootstrap_room=..., input_len=..., cached_input_len=..., output_len=..., type=Decode): ...
```

Test that normalization produces request events for bootstrap, queue, forward/compute, transfer, and completion, preserving worker, role, request ID, bootstrap room, prompt length, cached tokens, transfer bytes/speed, source log file, and source line. Test monotonic ordering and explicit `missing` fields rather than fabricated zero-duration phases.

Extend replay fixtures so each SSE chunk may contain usage/prompt-cache data. Verify the response record preserves:

- client scheduled/send/first-token/last-token wall and monotonic timestamps;
- `prompt_tokens`, `cached_tokens`, device/host/storage cache details, and `prefix_hit_ratio`;
- compact output integrity evidence;
- every TPOT interval in the raw interval JSONL.

Run:

```bash
python -m unittest test.srt.test_pd_flip_req_timing test.srt.test_pd_flip_trace_replay -v
```

Expected: FAIL on missing parser and fields.

**Step 2: Implement request-time log normalization**

Implement:

```python
def parse_req_time_stats_line(line: str, *, worker: str, log_timestamp: float | None) -> dict | None: ...
def normalize_req_time_stats(rows: Iterable[dict]) -> list[dict]: ...
def join_request_path(*, client_rows: Sequence[dict], prefill_rows: Sequence[dict],
                      decode_rows: Sequence[dict]) -> tuple[list[dict], list[dict]]: ...
```

Do not infer unavailable router-internal sub-phases. Report the measured client-to-worker boundary as `router_and_dispatch`, and label it as a combined interval. Preserve raw lines so later router instrumentation can refine it without changing the schema.

**Step 3: Extend replay raw records**

Capture final streaming usage metadata and cache details when present. Add wall-clock epoch timestamps alongside monotonic durations and store one raw JSONL row per request plus one per token interval. Keep existing output files backward compatible.

**Step 4: Run tests and commit**

```bash
python -m unittest test.srt.test_pd_flip_req_timing test.srt.test_pd_flip_trace_replay -v
git add scripts/playground/disaggregation/pd_flip_req_timing.py scripts/playground/disaggregation/pd_flip_trace_replay.py test/srt/test_pd_flip_req_timing.py test/srt/test_pd_flip_trace_replay.py
git commit -m "feat(pd-flip): normalize stock request path timings"
```

## Task 3: Make SLO observation reusable for baseline and state-machine modes

**Files:**

- Modify: `scripts/playground/disaggregation/pd_flip_trace_slo.py`
- Modify: `scripts/playground/disaggregation/pd_flip_controller.py`
- Create: `scripts/playground/disaggregation/pd_flip_slo_observer.py`
- Modify: `test/srt/test_pd_flip_trace_slo_monitor.py`
- Create: `test/srt/test_pd_flip_slo_observer.py`

**Step 1: Write failing trigger-contract tests**

Use a synthetic ledger to verify:

- a 10-second rolling window;
- TTFT eligibility after 10 samples;
- TPOT eligibility after 100 intervals;
- entry only below 90% and recovery at or above 95%;
- the first violating request/interval, exact crossing timestamp, poll-detection timestamp, and poll lag;
- observer mode writes events but never calls drain, migrate, role-switch, or router mutation methods;
- controller mode consumes the same snapshot and trigger decision as observer mode.

**Step 2: Extract a pure decision API**

Add immutable result data:

```python
@dataclass(frozen=True)
class SLOSnapshot: ...

def evaluate_slo_window(rows, *, now, window_seconds, enter_threshold,
                        recover_threshold, min_ttft_samples,
                        min_tpot_intervals) -> SLOSnapshot: ...
```

Have `TraceSLOMonitor` and the controller call this function. Do not duplicate threshold logic.

**Step 3: Implement observer CLI**

The observer tails the ledger at 250 ms, writes JSONL snapshots and a final JSON summary, exits after all 40 requests are terminal, and has no admin API arguments.

**Step 4: Run tests and commit**

```bash
python -m unittest test.srt.test_pd_flip_trace_slo_monitor test.srt.test_pd_flip_slo_observer -v
git add scripts/playground/disaggregation/pd_flip_trace_slo.py scripts/playground/disaggregation/pd_flip_controller.py scripts/playground/disaggregation/pd_flip_slo_observer.py test/srt/test_pd_flip_trace_slo_monitor.py test/srt/test_pd_flip_slo_observer.py
git commit -m "feat(pd-flip): add read-only trace slo observer"
```

## Task 4: Normalize PD Flip migration evidence, including Mamba state

**Files:**

- Modify: `scripts/playground/disaggregation/pd_flip_migration_measure.py`
- Create: `test/srt/test_pd_flip_qwen80b_migration_measure.py`
- Reuse: `python/sglang/srt/managers/scheduler.py`
- Reuse: `python/sglang/srt/disaggregation/common/conn.py`

**Step 1: Write failing migration normalization tests**

Build source/target status fixtures containing `StateType.MAMBA`, `source_transfer_bytes`, `delta_transfer_bytes`, phase timestamps, 50%/remaining request batches, and final role publication. Verify:

- ordinary KV and auxiliary Mamba state are both declared in the transfer contract;
- aggregate transfer bytes equal the backend's combined KV-plus-state count;
- unknown per-component bytes are reported as `null`, not guessed;
- request base/delta/validation/commit/activation phases are joined;
- exactly two batches are present, the first is `ceil(source_active * 0.5)`, and the observation interval is at least 3.0 seconds;
- Prefill Donor and HiCache stitch evidence is absent and their flags are false.

**Step 2: Add normalized migration rows**

Extend measurement output with:

```text
state_types, includes_mamba_state, combined_transfer_bytes,
kv_component_bytes, mamba_component_bytes, byte_breakdown_available,
batch_index, observation_started_at, observation_finished_at,
validation_at, commit_at, activation_at
```

The existing Mooncake transfer metric counts KV and all state indices. Unless the backend exposes a trustworthy split, retain only the combined byte total and explicitly mark the per-component split unavailable.

**Step 3: Run tests and commit**

```bash
python -m unittest test.srt.test_pd_flip_qwen80b_migration_measure test.srt.test_pd_flip_timeline_measurements -v
git add scripts/playground/disaggregation/pd_flip_migration_measure.py test/srt/test_pd_flip_qwen80b_migration_measure.py
git commit -m "feat(pd-flip): report qwen hybrid migration evidence"
```

## Task 5: Generate the paired report entirely from raw artifacts

**Files:**

- Create: `scripts/playground/disaggregation/pd_flip_ab_report.py`
- Create: `test/srt/test_pd_flip_ab_report.py`

**Step 1: Write failing report tests**

Create small baseline/state-machine raw fixtures and verify generation of:

- `summary.json`;
- `request_comparison.csv`;
- `stage_timings.csv`;
- `slo_timeseries.csv`;
- `migration_timings.csv`;
- `report.md`;
- `timeline.svg`.

Test short/long and full-run splits, four attainment definitions, trigger request and timestamp, pre-trigger/migration/observation/post-switch windows, client and stock-path timings in both modes, PD Flip-only timings only in state-machine mode, and invalid-run suppression of winner conclusions.

**Step 2: Implement pure aggregation and validity checks**

Implement:

```python
def load_run_artifacts(run_dir: Path) -> dict: ...
def validate_pair(baseline: dict, state_machine: dict) -> list[dict]: ...
def build_comparison(baseline: dict, state_machine: dict) -> dict: ...
def write_report(comparison: dict, output_dir: Path) -> None: ...
```

Validity must compare trace hash, model fingerprint, code hash, GPU allocation, SLO configuration, 40 terminal requests, 10,000-token integrity, baseline static topology, and state-machine two-batch/final-role evidence. Generate SVG with stable standard-library XML, no plotting dependency.

**Step 3: Run tests and commit**

```bash
python -m unittest test.srt.test_pd_flip_ab_report -v
git add scripts/playground/disaggregation/pd_flip_ab_report.py test/srt/test_pd_flip_ab_report.py
git commit -m "feat(pd-flip): add raw-backed ab performance report"
```

## Task 6: Add the safe four-node A/B runner

**Files:**

- Create: `experiments/pd_flip_qwen80b_ab.sh`
- Create: `experiments/pd_flip_qwen80b_ab.env.example`
- Create: `test/srt/test_pd_flip_qwen80b_ab_runner.py`
- Modify: `scripts/playground/disaggregation/pd_flip_docker/run_worker.sh`

**Step 1: Write failing static and dry-run tests**

Test that the env defaults specify Qwen3-Next 80B, TP=4, 1P3D, 40 requests, 10,000 tokens, 0.5-second spacing, 7.5-second wave starts, 3-second observation, 50% first migration, and the agreed SLO thresholds.

Test runner actions `preflight`, `prepare`, `baseline`, `state-machine`, `compare`, and `run`. In `DRY_RUN=1`, assert:

- no SSH, Docker, copy, download, process kill, or external filesystem mutation occurs;
- baseline launch omits state-machine/runtime-role/HiCache/Prefill-Donor flags;
- state-machine launch enables only state-machine and runtime-role flags;
- both modes use identical model, TP, GPU set, trace hash, and standard request-time logging;
- workers start sequentially and router starts only after four health/role checks;
- baseline starts the read-only observer;
- state-machine starts `monitor-progressive` with ratio `0.5`, observation `3`, poll `0.25`, min TTFT `10`, and min TPOT `100`;
- container names and artifact directories differ by mode;
- no command contains `docker restart`, automatic model copy/download, broad process matching, or forced kill;
- missing model nodes are printed as a list and fail preflight;
- secrets are redacted from printed commands and manifests.

**Step 2: Generalize worker launch arguments**

Update `run_worker.sh` so DP attention is conditional (`ENABLE_DP_ATTENTION`, default preserving current behavior), GPU IDs are passed via Docker configuration, request-time logging is conditional, and baseline feature flags can all be disabled. Preserve existing DeepSeek runner behavior by default.

**Step 3: Implement runner orchestration**

The runner must:

1. snapshot code/image/model/GPU/port/clock/node health;
2. prepare the immutable trace once;
3. launch baseline workers sequentially, then router, observer, telemetry, and replay;
4. verify all 40 requests terminate and flush artifacts;
5. pause admission, drain, gracefully stop owned containers, and verify teardown/host health;
6. launch the state-machine mode with different owned names;
7. run controller, telemetry, replay, and migration sampling;
8. verify the two migration batches and final 2P2D topology;
9. gracefully stop owned processes; and
10. regenerate the paired report.

Every remote mutation must target an exact owned container or PID recorded in the run manifest. On abnormal teardown, preserve artifacts and abort before the second model load.

**Step 4: Run tests and commit**

```bash
python -m unittest test.srt.test_pd_flip_qwen80b_ab_runner -v
bash -n experiments/pd_flip_qwen80b_ab.sh scripts/playground/disaggregation/pd_flip_docker/run_worker.sh
git add experiments/pd_flip_qwen80b_ab.sh experiments/pd_flip_qwen80b_ab.env.example scripts/playground/disaggregation/pd_flip_docker/run_worker.sh test/srt/test_pd_flip_qwen80b_ab_runner.py
git commit -m "feat(pd-flip): add safe qwen80b ab experiment runner"
```

## Task 7: Full local verification and operator handoff

**Files:**

- Modify if needed: `docs/superpowers/specs/2026-07-17-qwen80b-pd-flip-ab-performance-design.md`
- Create: `docs/runbooks/pd_flip_qwen80b_ab.md`

**Step 1: Run the focused suite**

```bash
python -m unittest \
  test.srt.test_pd_flip_qwen80b_trace \
  test.srt.test_pd_flip_req_timing \
  test.srt.test_pd_flip_trace_replay \
  test.srt.test_pd_flip_trace_slo_monitor \
  test.srt.test_pd_flip_slo_observer \
  test.srt.test_pd_flip_qwen80b_migration_measure \
  test.srt.test_pd_flip_timeline_measurements \
  test.srt.test_pd_flip_ab_report \
  test.srt.test_pd_flip_qwen80b_ab_runner -v
```

**Step 2: Run adjacent PD Flip regression tests**

```bash
python -m unittest discover -s test/srt -p 'test_pd_flip_*.py' -v
```

If the entire adjacent suite is too environment-specific, record every skipped or unavailable test and run all pure-Python tests individually. Do not report them as passing without evidence.

**Step 3: Run syntax and dry-run checks**

```bash
bash -n experiments/pd_flip_qwen80b_ab.sh
bash -n scripts/playground/disaggregation/pd_flip_docker/run_worker.sh
DRY_RUN=1 RUN_ID=local-plan-check ENV_FILE=experiments/pd_flip_qwen80b_ab.env.example \
  bash experiments/pd_flip_qwen80b_ab.sh run
```

Verify the dry-run creates only its configured temporary/local artifact tree and emits no secret.

**Step 4: Write the operator runbook**

Document:

- model availability preflight and the expected initial failure while only node102 has the model;
- exact A/B actions and artifact layout;
- safe stop behavior and forbidden restart/kill operations;
- how to inspect validity before reading performance conclusions;
- how to regenerate the report from raw data; and
- the explicit limitation that this run measures full source-D hybrid-state migration, not Prefill Donor stitching.

**Step 5: Final verification commit**

```bash
git add docs/runbooks/pd_flip_qwen80b_ab.md docs/superpowers/specs/2026-07-17-qwen80b-pd-flip-ab-performance-design.md
git commit -m "docs(pd-flip): add qwen80b ab experiment runbook"
```

Do not stage unrelated untracked reports or `pd-flip-artifacts/` directories.
