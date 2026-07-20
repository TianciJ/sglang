# Qwen80B Long-Prefill Warmup Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and execute one reproducible long-Prompt warmup diagnostic in the clean upstream Qwen80B baseline runner.

**Architecture:** Reuse the first long request body from the immutable trace, reduce only its generation length to one token, record client and wall-clock evidence, capture timestamp-window Docker logs, flush KV, and run the unchanged measured trace. Keep all new artifacts under the unique run directory.

**Tech Stack:** Bash, embedded Python 3, pytest, Docker logs, SGLang OpenAI-compatible streaming API.

## Global Constraints

- Use `tiancij/sglang-upstream:v0.5.15-clean` and the frozen trace hash `c5dbbf75c997dfc5d67a18251082f2f246d6c055eb4af5040fbe147f49f4ce5d`.
- Do not alter the measured 40-request trace or mix warmup events into raw measured rows.
- Flush all four workers after warmup; abort rather than silently measuring with unproven cache state.
- Stop only exact run-owned containers and retain full unfiltered logs.

---

### Task 1: Runner contract test

**Files:**
- Modify: `test/srt/test_pd_upstream_qwen80b_runner.py`

**Interfaces:**
- Consumes: runner source as text through `source()`.
- Produces: a failing test that specifies trace-derived long warmup, timing artifacts, cache flush ordering, and independent warmup log windows.

- [ ] Add a test asserting `long-prefill-warmup.json`, trace-body reuse, `max_tokens = 1`, UTC timestamps, `warmup-node0.docker.log`, and warmup-before-flush-before-measure ordering.
- [ ] Run `pytest -q test/srt/test_pd_upstream_qwen80b_runner.py` and verify the new test fails because the artifacts and request do not exist.

### Task 2: Minimal runner implementation

**Files:**
- Modify: `experiments/pd_upstream_qwen80b_baseline.sh`
- Modify: `docs/runbooks/pd_upstream_qwen80b_baseline.md`

**Interfaces:**
- Consumes: `${RUN_DIR}/trace/trace.jsonl`, router endpoint, private admin key, exact worker/router names.
- Produces: `${RUN_DIR}/smoke/long-prefill-warmup.json` and `${RUN_DIR}/logs/warmup-{node0,node1,node2,node3,router}.docker.log`.

- [ ] In embedded Python, load the first trace row, copy `body`, remove `custom_params`, set `max_tokens=1`, stream it, and record UTC/monotonic timing plus integrity fields.
- [ ] Slice each container's timestamped Docker output using the recorded UTC start/end with a small boundary margin, without deleting or replacing full logs.
- [ ] Keep cache flush after the warmup and before `measure`; fail the run if the warmup or flush evidence is incomplete.
- [ ] Update the runbook artifact and lifecycle descriptions.
- [ ] Run `pytest -q test/srt/test_pd_upstream_qwen80b_runner.py` and verify all tests pass.
- [ ] Run `bash -n experiments/pd_upstream_qwen80b_baseline.sh` and the runner `dry-run` command.

### Task 3: Four-node execution and diagnosis

**Files:**
- Runtime output: `/root/tiancij-upstream-baseline-runs/<RUN_ID>/`

**Interfaces:**
- Consumes: tested runner, private environment, fixed trace, idle verified nodes.
- Produces: one valid 40-request diagnostic run and a log-backed warm-versus-cold comparison.

- [ ] Run the checked-in read-only preflight on all four nodes with a new `RUN_ID`.
- [ ] Sync only tested runner/helper/runbook files required by the controller and record their hashes.
- [ ] Execute the checked-in `run` lifecycle and communicate progress at readiness, warmup, trace, validation, and teardown boundaries.
- [ ] Verify warmup and flush evidence, 40/40 request integrity, zero errors, manifest validity, checksums, teardown, free ports, and healthy GPUs.
- [ ] Compare formal request TTFT groups `0-9`, `10-19`, and `20-39` with the three retained cold runs; quote direct log evidence separately from inferred causes.
