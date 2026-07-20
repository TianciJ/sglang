# PD Flip Candidate-P Prefill Warmup Implementation Plan

> **For Codex:** Follow test-driven development and the repository performance
> experiment policy. Do not operate the cluster until local tests and dry-run
> validation pass.

**Goal:** Persist compilation artifacts and warm the long/short Prefill paths
inside every candidate P worker before running the Qwen80B state-machine trace.

**Architecture:** Add a small Python warmup orchestrator that controls the
existing worker/router admin surfaces, targets one candidate P at a time, sends
two validated warmup requests through the router, restores 1P3D, flushes KV,
and writes event-level evidence. Extend the checked-in runner and worker launch
wrapper to mount a provenance-keyed host cache and invoke this gate before any
measurement helper.

**Tech stack:** Bash, Python standard library, unittest/pytest, existing SGLang
PD runtime-role and router admin APIs.

---

### Task 1: Lock the runner contract with failing tests

**Files:**
- Modify: `test/srt/test_pd_flip_qwen80b_ab_runner.py`
- Create: `test/srt/test_pd_flip_candidate_prefill_warmup.py`

Add assertions for a persistent cache mount/env contract, a state-machine-only
candidate warmup gate ordered after router readiness and before measurement,
all four candidate names, long/short selection, max_tokens=1, role restoration,
four-worker cache flush, and final 1P3D validation. Run the focused tests and
confirm they fail because the new helper and runner behavior do not exist.

### Task 2: Implement the candidate warmup orchestrator

**Files:**
- Create: `scripts/playground/disaggregation/pd_flip_candidate_prefill_warmup.py`
- Test: `test/srt/test_pd_flip_candidate_prefill_warmup.py`

Implement pure trace/profile selection and topology-validation helpers first,
then the bounded HTTP orchestration and JSONL evidence writer. Run focused
tests after each minimal implementation step.

### Task 3: Add provenance-keyed persistent caches

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_docker/run_worker.sh`
- Modify: `experiments/pd_flip_qwen80b_ab.sh`
- Modify: `experiments/pd_flip_qwen80b_ab.env.example`
- Test: `test/srt/test_pd_flip_qwen80b_ab_runner.py`

Compute one cache namespace from recorded execution provenance, create the
node-local directory during preparation, pass it through each private node env,
and mount/redirect supported compiler/JIT cache roots. Save pre/post cache
snapshots and the namespace in the mode manifest.

### Task 4: Integrate the pre-measurement gate

**Files:**
- Modify: `experiments/pd_flip_qwen80b_ab.sh`
- Modify: `docs/runbooks/pd_flip_qwen80b_ab.md`
- Test: `test/srt/test_pd_flip_qwen80b_ab_runner.py`

Invoke the helper only for state-machine diagnostic runs after worker/router
readiness. Require all warmup and restoration artifacts before starting the
sampler, observer, controller, or trace replay. Preserve failure evidence and
use existing exact-name teardown.

### Task 5: Verify locally and publish the implementation

Run focused unit tests, Bash syntax checks, the runner dry-run, and relevant PD
Flip tests. Inspect the diff for unrelated files, commit only intended paths on
main, and push the exact revision used for the cluster run.

### Task 6: Preflight, deploy, and run one state-machine diagnostic

Run the full read-only four-node preflight. Stop only exact run-owned remnants
if present and authorized; otherwise stop and report ownership conflicts.
Deploy the committed revision, use a new RUN_ID, prepare/reuse the frozen trace,
run state-machine only, validate all warmup and measured gates, collect raw
artifacts, and gracefully tear down exact run-owned resources.

### Task 7: Analyze and report

Generate the request TTFT timeline, TTFT/TPOT summaries, SLO-trigger request and
time, migration phase timing, warmup timings per node/shape, promoted-P first
measured Prefill timing, cache snapshots, validity statement, limitations, and
a raw-artifact inventory with checksums.

