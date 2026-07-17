# Clean Upstream Qwen80B Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a checked-in, safe runner and report pipeline for one clean upstream SGLang v0.5.15 Qwen3-Next-80B-A3B-Instruct 1P3D run using the retained 40-request trace and producing independently verifiable TTFT/TPOT results.

**Architecture:** A dedicated Bash runner performs read-only four-node preflight, builds the official router only from source embedded in the clean image, launches exact run-owned Mooncake/worker/router resources, drives the existing trace replayer from a non-GPU helper, validates raw evidence, and tears down only resources owned by the run. A dependency-light Python reporter validates the event-level artifact contract and generates JSON, CSV, Markdown, and SVG summaries without participating in the inference data plane.

**Tech Stack:** Bash, Docker, SSH, upstream SGLang v0.5.15, Mooncake, Python 3 standard library, pytest.

## Global Constraints

- Worker and router image must be `tiancij/sglang-upstream:v0.5.15-clean` with ID `sha256:7dd92779d739364d79af34af65815ddc14e567728e5256f65ac922367161213e`.
- Worker and router containers must not mount host code over `/sgl-workspace/sglang` or inject a host `PYTHONPATH`.
- Trace must be `pd-flip-artifacts/qwen80b-trace40-source/trace.jsonl` with SHA256 `82da848d68c9662a7aaaf76deb547b1d8cc6c4f562586f0d60dd212bc114e964`.
- Fixed configuration is four nodes, GPUs `0,1,2,3`, TP 4, DP 1, `1P3D`, Mooncake, active `mlx5_bond_0` with per-node IPv6 identity and GID index 3, memory fraction 0.88, and one measured run.
- Never use `docker restart`, `pkill`, `killall`, `kill -9`, wildcard container matching, or stop a resource not proven to belong to the current `RUN_ID`.
- The six state-machine, runtime-role-switch, HiCache, radix-cache, and Prefill-donor flags listed in the design are forbidden.
- Valid output requires 40 completed unique requests, 10,000 matching forced tokens per request, finish reason `length`, zero errors, 400,040 ledger records, and 399,960 TPOT interval data rows.
- The final report must label TTFT/TPOT as client-observed streaming timings and state that one run does not establish statistical significance.

---

### Task 1: Raw Artifact Validator and TTFT/TPOT Reporter

**Files:**
- Create: `scripts/playground/disaggregation/pd_upstream_baseline_report.py`
- Create: `test/srt/test_pd_upstream_baseline_report.py`

**Interfaces:**
- Consumes: `generate_report(run_dir: pathlib.Path) -> dict`, reading `manifest.json`, `raw/upstream_baseline/request_metrics.jsonl`, `raw/upstream_baseline/errors.jsonl`, `raw/slo_ledger.jsonl`, and `raw/upstream_baseline/tpot_tokens.csv`.
- Produces: `report/summary.json`, `report/request_metrics.csv`, `report/report.md`, `report/ttft_scatter.svg`, and `report/tpot_scatter.svg`; exits nonzero on any acceptance-gate violation.

- [ ] **Step 1: Write failing unit tests for a valid synthetic run**

Create fixtures with two requests and scaled expected counts, then assert `summarize_requests()` returns exact TTFT mean/P95/SLO attainment and token TPOT P50/P95/P99/max values. Assert SVG files contain both stable request IDs and Markdown explicitly says `client-observed`.

- [ ] **Step 2: Run the focused tests and verify the import fails**

Run: `python -m pytest -q test/srt/test_pd_upstream_baseline_report.py`

Expected: FAIL because `pd_upstream_baseline_report.py` does not exist.

- [ ] **Step 3: Implement parsing, validation, quantiles, tables, and SVG output**

Implement standard-library JSONL/CSV readers, nearest-rank percentile interpolation matching the existing experiment reports, invariant checks with explicit messages, XML escaping for SVG labels, and atomic report directory creation. The CLI is:

```bash
python3 scripts/playground/disaggregation/pd_upstream_baseline_report.py \
  --run-dir /path/to/run
```

- [ ] **Step 4: Add failure tests for count and integrity mismatches**

Cover duplicate request IDs, nonempty errors, wrong completion count, wrong finish reason, wrong forced output match, ledger count mismatch, TPOT interval count mismatch, and missing provenance fields.

- [ ] **Step 5: Run the focused tests**

Run: `python -m pytest -q test/srt/test_pd_upstream_baseline_report.py`

Expected: all tests PASS.

- [ ] **Step 6: Commit the independently testable reporter**

```bash
git add scripts/playground/disaggregation/pd_upstream_baseline_report.py test/srt/test_pd_upstream_baseline_report.py
git commit -m "feat: report clean upstream baseline timings"
```

### Task 2: Safe Clean-Upstream Runner Contract

**Files:**
- Create: `experiments/pd_upstream_qwen80b_baseline.sh`
- Create: `experiments/pd_upstream_qwen80b_baseline.env.example`
- Create: `test/srt/test_pd_upstream_qwen80b_runner.py`

**Interfaces:**
- Consumes: private `ENV_FILE`; fixed trace; SSH aliases `cloud-099` through `cloud-102`; model path; clean image; explicit Mooncake settings.
- Produces: subcommands `preflight`, `build-router`, `prepare`, `start`, `smoke`, `measure`, `collect-stop`, `report`, `run`, and `dry-run`, plus a run directory containing a redacted effective manifest.

- [ ] **Step 1: Write static contract tests before the runner exists**

Parse the Bash source and assert it contains exact image/trace hashes, exact run-owned naming, bounded health polling, concurrent worker starts, exact-name graceful stops, mount/argument inspection, 40/400040/399960 validation, and a failure trap. Assert it does not contain forbidden flags, `docker restart`, `pkill`, `killall`, `kill -9`, or a worker/router host-code mount.

- [ ] **Step 2: Run the static contract tests and verify they fail**

Run: `python -m pytest -q test/srt/test_pd_upstream_qwen80b_runner.py`

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement configuration parsing and read-only preflight**

Require a non-placeholder secret from the private environment, generate or accept a unique `RUN_ID`, and inspect all four nodes for SSH, image ID, model completeness/fingerprint, GPU allocation, driver health, ports, disk, mounts, RDMA/GID, clocks, existing containers/processes, and exact owned-name collisions. Preflight writes evidence but changes no remote state.

- [ ] **Step 4: Implement official router build and provenance**

Run a no-GPU build container from the clean image without a source mount, compile `/sgl-workspace/sglang/experimental/sgl-router`, copy the resulting binary into an owned artifact directory, and write its SHA256 plus image ID. The runtime router mounts only this binary and the tokenizer/model read-only.

- [ ] **Step 5: Implement run preparation and launch**

Verify and copy the fixed trace, create all artifact directories, start only explicitly configured run-owned Mooncake services/namespace, and start four clean workers concurrently using exact names. Mount only the model, RDMA devices, and run-specific configuration; then use bounded `/health` checks before starting the official router with four static worker URLs.

- [ ] **Step 6: Implement smoke, cold-cache gate, and measurement**

Send two unique smoke requests, validate streaming completion, call the verified upstream `/flush_cache` endpoint on all four workers, and record responses. If any flush cannot be proven, stop and relaunch only the exact run-owned workers before measurement. Run `pd_flip_trace_replay.py replay` in a non-GPU helper using mode `upstream_baseline`, the fixed trace, concurrency 40, and the existing raw ledger/output contract.

- [ ] **Step 7: Implement validation, collection, and safe teardown**

Validate all request and raw row invariants before accepting the run. Capture Docker inspect/logs, worker request-time logs, router/Mooncake/helper logs, after-status, and checksums. Stop only exact run-owned containers with long graceful timeouts, verify ports are free and GPU/driver health remains good, then invoke the reporter. Preserve failures under the same run ID as forensic evidence and never overwrite them.

- [ ] **Step 8: Add a deterministic dry-run output**

`dry-run` must print redacted, fully expanded worker/router/replay commands and resource names without SSH mutation or revealing secrets. Tests assert the clean image supplies worker code and that host code is mounted only into the external helper.

- [ ] **Step 9: Run shell syntax and runner contract tests**

Run:

```bash
bash -n experiments/pd_upstream_qwen80b_baseline.sh
python -m pytest -q test/srt/test_pd_upstream_qwen80b_runner.py
```

Expected: syntax check exits 0 and all tests PASS.

- [ ] **Step 10: Commit the runner**

```bash
git add experiments/pd_upstream_qwen80b_baseline.sh experiments/pd_upstream_qwen80b_baseline.env.example test/srt/test_pd_upstream_qwen80b_runner.py
git commit -m "feat: orchestrate clean upstream Qwen80B baseline"
```

### Task 3: Operator Runbook and End-to-End Local Verification

**Files:**
- Create: `docs/runbooks/pd_upstream_qwen80b_baseline.md`
- Modify: `experiments/pd_upstream_qwen80b_baseline.sh`
- Test: `test/srt/test_pd_upstream_qwen80b_runner.py`

**Interfaces:**
- Consumes: runner subcommands and environment schema from Task 2.
- Produces: a command-by-command operator procedure, failure interpretation, artifact inventory, and recovery rules.

- [ ] **Step 1: Write the runbook with exact operator sequence**

Document private env creation, `preflight`, `build-router`, `dry-run`, `run`, forensic failure handling, exact-name manual inspection, report interpretation, raw timing boundary, and the prohibition on treating a single run as a statistically reliable comparison.

- [ ] **Step 2: Add runbook references and artifact inventory checks**

Make the runner write `INVENTORY.txt` with SHA256 checksums and add static tests requiring every report/raw/provenance/log path named by the runbook to be produced by the runner.

- [ ] **Step 3: Run all focused verification**

Run:

```bash
bash -n experiments/pd_upstream_qwen80b_baseline.sh
python -m pytest -q \
  test/srt/test_pd_upstream_baseline_report.py \
  test/srt/test_pd_upstream_qwen80b_runner.py
ENV_FILE=experiments/pd_upstream_qwen80b_baseline.env.example \
  bash experiments/pd_upstream_qwen80b_baseline.sh dry-run
```

Expected: shell syntax succeeds, all tests pass, and dry-run exits 0 after printing redacted commands without contacting nodes.

- [ ] **Step 4: Review the diff for purity and ownership**

Run:

```bash
git diff --check
git diff -- experiments/pd_upstream_qwen80b_baseline.sh experiments/pd_upstream_qwen80b_baseline.env.example scripts/playground/disaggregation/pd_upstream_baseline_report.py test/srt/test_pd_upstream_baseline_report.py test/srt/test_pd_upstream_qwen80b_runner.py docs/runbooks/pd_upstream_qwen80b_baseline.md
```

Expected: no whitespace errors; no unrelated files; no secret; no host-code mount in worker/router; no unsafe stop primitive.

- [ ] **Step 5: Commit the verified runbook and final integration**

```bash
git add docs/runbooks/pd_upstream_qwen80b_baseline.md experiments/pd_upstream_qwen80b_baseline.sh test/srt/test_pd_upstream_qwen80b_runner.py
git commit -m "docs: add clean upstream baseline runbook"
```
