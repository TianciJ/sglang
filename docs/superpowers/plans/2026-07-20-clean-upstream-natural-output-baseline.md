# Clean Upstream Natural-Output Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute one valid 40-request clean-upstream Qwen80B 1P3D baseline without a custom logit processor.

**Architecture:** Freeze a new natural-output trace before deployment, keep workers and router entirely inside the immutable clean image, and run the modified repository only in the no-GPU client/report helper. Validate token counts from OpenAI usage and compute request TPOT from client first/last output anchors normalized by completion-token count.

**Tech Stack:** Bash, Python standard library, pytest, Docker, SGLang v0.5.15, Mooncake, four H20 nodes.

## Global Constraints

- Use exact run-owned names and a new `RUN_ID`; never stop unrelated resources.
- Use image ID `sha256:7dd92779d739364d79af34af65815ddc14e567728e5256f65ac922367161213e`.
- Use Qwen3-Next-80B-A3B-Instruct, GPUs 0-3, TP4, DP1, 1P3D, `mlx5_bond_0`, IPv6, GID 3.
- Preserve all failed attempts as forensic runs and redact secrets.

---

### Task 1: Freeze a natural-output trace

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_prepare_trace.py`
- Modify: `test/srt/test_pd_flip_prepare_trace.py`
- Create: `pd-flip-artifacts/qwen80b-trace40-natural/trace.jsonl`
- Create: `pd-flip-artifacts/qwen80b-trace40-natural/manifest.json`

**Interfaces:**
- Produces: `apply_natural_output_contract(row, max_tokens, model)` and a 40-row trace with no forced-output fields.

- [ ] Write tests proving natural mode preserves prompts/timing and removes only processor/forced fields.
- [ ] Run the focused tests and observe the expected failure before implementation.
- [ ] Implement the natural output contract and CLI mode.
- [ ] Generate the trace once, validate 40 unique rows, and record source/new SHA256.
- [ ] Re-run focused tests and verify the frozen trace contents.

### Task 2: Measure token-normalized client TPOT

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_trace_replay.py`
- Modify: `test/srt/test_pd_flip_trace_replay.py`

**Interfaces:**
- Produces: `token_normalized_tpot_s`, `tpot_metric_source`, and separately named stream-event gap metrics.

- [ ] Write a test with 10 usage tokens and three non-empty SSE events that expects TPOT to use first/last anchors divided by nine.
- [ ] Run it and confirm failure because the token-normalized metric is absent.
- [ ] Implement the metric without treating SSE events as tokens.
- [ ] Run the complete trace-replay test file.

### Task 3: Update runner and report validity

**Files:**
- Modify: `experiments/pd_upstream_qwen80b_baseline.sh`
- Modify: `scripts/playground/disaggregation/pd_upstream_baseline_report.py`
- Modify: `test/srt/test_pd_upstream_qwen80b_runner.py`
- Modify: `test/srt/test_pd_upstream_baseline_report.py`
- Modify: `docs/runbooks/pd_upstream_qwen80b_baseline.md`

**Interfaces:**
- Consumes: the new trace/hash and token-normalized TPOT fields.
- Produces: a runner with no custom processor flag, a 10k probe, dynamic event counts, and a report labeling stream gaps correctly.

- [ ] Write failing tests for the new hash, absence of custom processor, 10k probe, dynamic evidence counts, and report terminology.
- [ ] Remove portable-processor replay and the server custom-processor flag.
- [ ] Add the unmeasured 10k probe before cold-cache establishment.
- [ ] Validate 40 completed requests, usage token count, finish reason, timestamps, errors, and dynamic evidence counts.
- [ ] Update report summaries and runbook terminology.
- [ ] Run all runner/report/replay tests plus `bash -n` and `git diff --check`.

### Task 4: Deploy and execute the experiment

**Files:**
- Runtime artifacts under `/root/tiancij-upstream-baseline-runs/<RUN_ID>`.

**Interfaces:**
- Consumes: checked-in runner, frozen trace, private environment, clean router artifact.
- Produces: validated raw evidence, teardown evidence, and TTFT/TPOT report.

- [ ] Sync only tested files and frozen trace to `/root/sglang` on cloud-099.
- [ ] Generate a new formal `RUN_ID` and rerun read-only preflight.
- [ ] Run start, smoke, 10k probe, cold-cache gate, and one 40-request measurement.
- [ ] On failure, preserve forensic evidence and gracefully stop exact run-owned containers.
- [ ] On success, validate all raw gates, collect logs, stop, verify ports/GPUs, and generate the report.
