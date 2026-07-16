# PD Flip DeepSeek Four-Node Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy and validate the real DeepSeek-V3.1 TP8/DP8 PD Flip chain on four H20 nodes, then produce a measured 40x10,000-token experiment artifact.

**Architecture:** Parameterize the existing container launcher for DeepSeek, add non-mutating and startup preflights, and advance through one-request, small multi-rank, baseline, and full PD Flip gates. Keep process ownership labels and raw clocks/events so cleanup and the final report are unambiguous.

**Tech Stack:** Bash, Docker, SSH, SGLang, DeepGEMM, Mooncake, Python analysis scripts, four 8xH20-3e nodes.

## Global Constraints

- Model path is `/models/deepseek_v3.1_terminus`.
- Each role worker uses all eight local GPUs with TP8/DP8 Attention.
- The checkpoint is FP8 and is not launched with `--quantization fp8`.
- The first correctness run does not enable speculative decoding.
- The full measured trace is 40 distinct Prompts and 10,000 output tokens per request.
- The Mooncake store is reset before baseline and measured runs and must report no eviction or required donor miss.
- Only experiment-owned processes/containers may be stopped.

---

### Task 1: DeepSeek worker launch contract

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_docker/run_worker.sh`
- Modify: `scripts/playground/disaggregation/pd_flip_docker/env.example`
- Modify: `experiments/pd_flip_trace40_full_chain.env.example`
- Modify: `test/srt/test_pd_flip_experiment_script.py`

**Interfaces:**
- Consumes: `MODEL_PATH`, `TP_SIZE`, `DP_SIZE`, `ENABLE_DP_ATTENTION`, and `ENABLE_CUSTOM_LOGIT_PROCESSOR`.
- Produces: identical TP8/DP8 DeepSeek launch commands for all four role workers.

- [ ] **Step 1: Add failing launch-argument assertions**

```python
def test_worker_launches_deepseek_dp_attention(self):
    script = Path("scripts/playground/disaggregation/pd_flip_docker/run_worker.sh").read_text()
    self.assertIn('MODEL_PATH="${MODEL_PATH:-/models/deepseek_v3.1_terminus}"', script)
    self.assertIn('TP_SIZE="${TP_SIZE:-8}"', script)
    self.assertIn('DP_SIZE="${DP_SIZE:-8}"', script)
    self.assertIn("--enable-dp-attention", script)
    self.assertIn("--enable-custom-logit-processor", script)
    self.assertNotIn("--quantization fp8", script)
```

- [ ] **Step 2: Run the launcher test**

Run: `python -m pytest test/srt/test_pd_flip_experiment_script.py -k worker_launches_deepseek -v`

Expected: FAIL on missing DeepSeek defaults.

- [ ] **Step 3: Parameterize worker arguments**

```bash
MODEL_PATH="${MODEL_PATH:-/models/deepseek_v3.1_terminus}"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-8}"
WORKER_ARGS+=(--model-path "${MODEL_PATH}" --tp-size "${TP_SIZE}" --dp-size "${DP_SIZE}")
[[ "${ENABLE_DP_ATTENTION:-1}" == "1" ]] && WORKER_ARGS+=(--enable-dp-attention)
[[ "${ENABLE_CUSTOM_LOGIT_PROCESSOR:-1}" == "1" ]] && WORKER_ARGS+=(--enable-custom-logit-processor)
```

Preserve the existing PD Flip, Prefill donor, HiCache write-through, Mooncake, admin authentication, and bootstrap arguments.

- [ ] **Step 4: Add environment examples**

```bash
MODEL_PATH=/models/deepseek_v3.1_terminus
TP_SIZE=8
DP_SIZE=8
ENABLE_DP_ATTENTION=1
ENABLE_CUSTOM_LOGIT_PROCESSOR=1
TRACE_MAX_TOKENS=10000
TRACE_FORCED_TEXT=字
WORKLOAD_TIMEOUT_SECONDS=7200
MEASUREMENT_DURATION_SECONDS=7200
```

- [ ] **Step 5: Run script tests and ShellCheck**

Run: `python -m pytest test/srt/test_pd_flip_experiment_script.py -v`

Run: `shellcheck scripts/playground/disaggregation/pd_flip_docker/run_worker.sh experiments/pd_flip_trace40_full_chain.sh`

Expected: PASS.

- [ ] **Step 6: Commit launch configuration**

```bash
git add scripts/playground/disaggregation/pd_flip_docker/run_worker.sh scripts/playground/disaggregation/pd_flip_docker/env.example experiments/pd_flip_trace40_full_chain.env.example test/srt/test_pd_flip_experiment_script.py
git commit -m "feat: launch deepseek pd flip workers with dp8"
```

### Task 2: Four-node preflight and DeepGEMM warm-up

**Files:**
- Modify: `experiments/pd_flip_trace40_full_chain.sh`
- Modify: `test/srt/test_pd_flip_trace40_full_chain_runner.py`

**Interfaces:**
- Produces: captured node inventory, model hashes, clock state, store capacity, rank health, and an explicit precompile action.
- Consumes: environment from Task 1.

- [ ] **Step 1: Add failing preflight contract tests**

```python
def test_deepseek_preflight_checks_all_rank_and_model_inputs(self):
    script = Path("experiments/pd_flip_trace40_full_chain.sh").read_text()
    for text in ("nvidia-smi", "config.json", "MODEL_PATH", "dp_rank", "model_fingerprint", "compile_deep_gemm"):
        self.assertIn(text, script)
```

- [ ] **Step 2: Run the focused runner test**

Run: `python -m pytest test/srt/test_pd_flip_trace40_full_chain_runner.py -k deepseek_preflight -v`

Expected: FAIL on missing checks.

- [ ] **Step 3: Capture read-only hardware and model inventory**

For each host, write into `${RUN_DIR}/preflight/<node>/`:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
sha256sum "${MODEL_PATH}/config.json"
stat -c '%n %s %y' "${MODEL_PATH}/config.json"
free -b
df -B1 "${MODEL_PATH}"
```

Reject fewer than eight healthy GPUs, absent model config, mismatched config hashes, insufficient store host memory, or clock offset beyond the experiment threshold.

- [ ] **Step 4: Add a separate precompile action**

```bash
python3 -m sglang.compile_deep_gemm --model "${MODEL_PATH}" --tp 8 --trust-remote-code
```

Run it once per node inside the same image and mounted model/cache paths used by workers. Save logs and exit codes. Do not include this duration in measured startup or request timing.

- [ ] **Step 5: Validate eight runtime rank responses after launch**

Query runtime role, migration status, capacity, and model/KV fingerprint endpoints. Require rank set `{0,1,2,3,4,5,6,7}` and identical model fingerprint on all nodes.

- [ ] **Step 6: Run dry-run and script tests**

Run: `DRY_RUN=1 ENV_FILE=experiments/pd_flip_trace40_full_chain.env.example bash experiments/pd_flip_trace40_full_chain.sh preflight`

Expected: commands for four nodes, model checks, precompile action, and DP8 status checks are printed without mutation.

Run: `python -m pytest test/srt/test_pd_flip_trace40_full_chain_runner.py -v`

Expected: PASS.

- [ ] **Step 7: Commit preflight**

```bash
git add experiments/pd_flip_trace40_full_chain.sh test/srt/test_pd_flip_trace40_full_chain_runner.py
git commit -m "feat: preflight deepseek four node experiment"
```

### Task 3: Gated smoke and baseline actions

**Files:**
- Modify: `experiments/pd_flip_trace40_full_chain.sh`
- Modify: `test/srt/test_pd_flip_trace40_full_chain_runner.py`

**Interfaces:**
- Produces actions `smoke-output`, `smoke-dp8`, `smoke-flip`, and `baseline`.
- Consumes: runtime and workload implementations from the other two plans.

- [ ] **Step 1: Add action-dispatch tests**

```python
def test_runner_exposes_deepseek_gates(self):
    script = Path("experiments/pd_flip_trace40_full_chain.sh").read_text()
    for action in ("smoke-output", "smoke-dp8", "smoke-flip", "baseline"):
        self.assertIn(action, script)
```

- [ ] **Step 2: Run and confirm actions are absent**

Run: `python -m pytest test/srt/test_pd_flip_trace40_full_chain_runner.py -k exposes_deepseek_gates -v`

Expected: FAIL.

- [ ] **Step 3: Implement `smoke-output`**

Prepare one unique Prompt with `TRACE_MAX_TOKENS=32`, send it without migration, and require 32 completion tokens, length finish, zero forced-token mismatches, and healthy DP rank statuses.

- [ ] **Step 4: Implement `smoke-dp8`**

Prepare eight requests with `TRACE_MAX_TOKENS=128`, verify routing touches more than one DP rank, and require 8/8 completions without migration or invalid-slot errors.

- [ ] **Step 5: Implement `smoke-flip`**

Prepare four requests with `TRACE_MAX_TOKENS=512`, trigger one strict Prefill-donor migration, and require P `[0,B)`, source D `[B,C0)`, delta through `C1`, one target DP rank, sampling-state continuity, and first post-migration token evidence.

- [ ] **Step 6: Implement `baseline`**

Reset the dedicated store, prepare the full 40x10,000 trace, disable migration trigger while retaining identical routing/workload settings, capture completion duration and per-rank load, and write `${RUN_DIR}/baseline/acceptance.json`.

- [ ] **Step 7: Run script tests and dry-run every action**

Run:

```bash
python -m pytest test/srt/test_pd_flip_trace40_full_chain_runner.py -v
for action in smoke-output smoke-dp8 smoke-flip baseline; do DRY_RUN=1 ENV_FILE=experiments/pd_flip_trace40_full_chain.env.example bash experiments/pd_flip_trace40_full_chain.sh "$action"; done
```

Expected: PASS and no real SSH/Docker mutation in dry-run mode.

- [ ] **Step 8: Commit gated actions**

```bash
git add experiments/pd_flip_trace40_full_chain.sh test/srt/test_pd_flip_trace40_full_chain_runner.py
git commit -m "feat: gate deepseek pd flip experiment rollout"
```

### Task 4: Full acceptance validator and report inputs

**Files:**
- Create: `scripts/playground/disaggregation/pd_flip_deepseek_acceptance.py`
- Create: `test/srt/test_pd_flip_deepseek_acceptance.py`
- Modify: `experiments/pd_flip_trace40_full_chain.sh`

**Interfaces:**
- Consumes: run directory containing effective trace, request metrics, migration events, per-rank statuses, store metrics, and clocks.
- Produces: `acceptance.json`, `acceptance.md`, and nonzero exit on any failed invariant.

- [ ] **Step 1: Write failing acceptance tests with a minimal valid fixture**

```python
def test_acceptance_requires_exact_output_and_ownership(tmp_path):
    run = build_valid_run_fixture(tmp_path, requests=40, max_tokens=10000)
    result = validate_run(run)
    assert result["passed"] is True
    corrupt_completion_tokens(run, rid="trace-0003", value=9999)
    result = validate_run(run)
    assert result["passed"] is False
    assert "trace-0003" in result["failed_requests"]
```

- [ ] **Step 2: Run and confirm module import failure**

Run: `python -m pytest test/srt/test_pd_flip_deepseek_acceptance.py -v`

Expected: FAIL because the validator does not exist.

- [ ] **Step 3: Implement trace/output validation**

Load exactly 40 effective rows, require unique Prompts, matching 10,000-token budgets, 40 completed metrics, length finish, exact completion count, one forced token ID, zero mismatch count, and no request error.

- [ ] **Step 4: Implement migration and store validation**

For each migrated RID, require P/source/delta ranges, one target rank, valid mapping commit before activation, first post-migration token, no target prefix substitution/fallback, and no donor/store failure or eviction.

- [ ] **Step 5: Implement worker/rank and timeline validation**

Require four healthy workers, DP rank set 0-7 per worker, identical fingerprints, monotonic per-request phase order, captured SLO trigger event, role-flip completion, and post-run health window.

- [ ] **Step 6: Write machine-readable and Markdown results**

```python
def write_results(result, output_dir):
    (output_dir / "acceptance.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output_dir / "acceptance.md").write_text(render_markdown(result))
```

Exit 0 only when `result["passed"] is True`.

- [ ] **Step 7: Add validator to baseline and measured run completion**

Invoke it before packaging artifacts. A failed validator keeps raw artifacts, marks the run failed, and prevents a success report.

- [ ] **Step 8: Run acceptance and runner tests**

Run: `python -m pytest test/srt/test_pd_flip_deepseek_acceptance.py test/srt/test_pd_flip_trace40_full_chain_runner.py -v`

Expected: PASS.

- [ ] **Step 9: Commit acceptance tooling**

```bash
git add scripts/playground/disaggregation/pd_flip_deepseek_acceptance.py test/srt/test_pd_flip_deepseek_acceptance.py experiments/pd_flip_trace40_full_chain.sh
git commit -m "feat: validate deepseek pd flip acceptance"
```

### Task 5: Live four-node execution and cleanup

**Files:**
- Create: `docs/superpowers/reports/2026-07-16-pd-flip-deepseek-trace40-report.md`
- Raw artifacts: configured `ARTIFACT_ROOT/<run-id>/` on host0.

**Interfaces:**
- Consumes: completed code from all three plans and the four-node environment.
- Produces: baseline and PD Flip artifacts, acceptance output, timing chart, and experiment report.

- [ ] **Step 1: Verify no unrelated active experiment owns the nodes**

Run read-only container/process/GPU inventory on all four nodes. Continue only when experiment container names and labels are free and no unowned process would be stopped or oversubscribed.

- [ ] **Step 2: Run preflight and DeepGEMM precompile**

Run: `ENV_FILE=/home/tiancij/trace40-full-chain.env bash experiments/pd_flip_trace40_full_chain.sh preflight`

Expected: all four nodes, 32 GPUs, model hashes, store, clocks, and code hashes pass.

- [ ] **Step 3: Run the three smoke gates in order**

Run `smoke-output`, then `smoke-dp8`, then `smoke-flip`. Inspect each acceptance file before proceeding. Any failure stops progression and preserves artifacts.

- [ ] **Step 4: Run the no-migration baseline**

Run the `baseline` action and record elapsed time. Set the measured-run timeout to the baseline duration plus the explicit migration and safety margins captured in the run manifest.

- [ ] **Step 5: Reset L3 and run the full measured experiment**

Run: `ENV_FILE=/home/tiancij/trace40-full-chain.env bash experiments/pd_flip_trace40_full_chain.sh run`

Expected: 40/40 exact completions, strict dual-source ownership, successful role flip, and passing acceptance output.

- [ ] **Step 6: Package raw data and generate the report**

Include the effective trace, raw stream evidence, SLO ledger, controller events, per-rank statuses, Mooncake metrics, clock captures, acceptance outputs, full-stage timing table, and an SVG/PNG timeline whose labels include measured duration per segment.

- [ ] **Step 7: Stop only experiment-owned containers**

Use the run ID/label recorded in the manifest. Re-list processes before removal and leave unrelated containers, sessions, and data untouched.

- [ ] **Step 8: Verify post-cleanup node state**

Capture final Docker, process, GPU, and store state. Confirm no experiment-owned process remains and no unrelated process changed.

- [ ] **Step 9: Commit the final report, not bulky raw archives**

```bash
git add docs/superpowers/reports/2026-07-16-pd-flip-deepseek-trace40-report.md
git commit -m "docs: report deepseek trace40 pd flip experiment"
```

Reference raw artifact paths and checksums in the report; do not add multi-gigabyte archives to Git.
