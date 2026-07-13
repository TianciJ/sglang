# PD Flip Dual-Source Atomic Stitch Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `prepare_only` dual-source migration validate staged L1/L3/suffix coverage before atomic commit, then validate the formal request mapping after commit.

**Architecture:** Keep the target request held throughout Prepare. Split the current `_pd_flip_target_stitch_ready` semantics into a staged-coverage validator and a committed-mapping validator. Reuse the existing HiCache commit operation only during target Commit, preserve full-source fallback for Prepare failures, and abort the batch if the post-commit mapping is inconsistent.

**Tech Stack:** Python, SGLang scheduler/HiCache, NumPy-backed AST unit tests, pytest.

## Global Constraints

- Work directly on `main`; do not create a feature branch or official PR.
- Preserve two-phase target ownership: `transferred_held → ready_to_activate → active`.
- Do not activate a target request during Prepare or Commit.
- Preserve the existing `H=0` source-Decode full fallback.
- Do not start the four-node cluster before the scheduled 09:00–10:00 experiment window.
- Keep the fixed 40-request long/short interleaved trace and raw/log retention for live validation.

## File Structure

- Modify `python/sglang/srt/managers/scheduler.py`: staged coverage validation, committed mapping validation, and target pump/commit ordering.
- Modify `test/srt/test_pd_flip_hicache_stitch.py`: regression tests for the deferred L3 mapping gap and staged segment failures.
- Modify `test/srt/test_pd_flip_atomic_batch.py`: post-commit mapping validation and atomic batch abort tests.
- Modify `docs/superpowers/plans/2026-07-13-pd-flip-dual-source-atomic-stitch-fix.md`: mark completed steps during execution.

---

### Task 1: Validate staged dual-source coverage during Prepare

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py:4520-4755`
- Test: `test/srt/test_pd_flip_hicache_stitch.py:300-590`

**Interfaces:**
- Consumes: `entry["mooncake_hit_len"]`, `entry["target_committed_len"]`, `decode_req.prefix_match`, `decode_req.hicache_restored_kv_indices`, and `req_to_token_pool` suffix mappings.
- Produces: `_pd_flip_target_stitch_ready(entry) -> bool`, where `True` means staged coverage is complete even if the formal `[L1,H)` mapping is not committed yet.

- [x] **Step 1: Write the deferred-mapping regression test**

Extend `_target_entry` so its prefix match carries `l1_prefix_len`, `prefix_indices`, and `hicache_restored_kv_indices`. Add a test whose formal table contains zeros only in `[5,32)` while the staged restore indices cover that range:

```python
def test_target_prepare_accepts_staged_l3_indices_before_atomic_commit():
    entry = _target_entry(h=32, p=64, c0=100, l1=5)
    kv_indices = np.arange(1, 101, dtype=np.int64)
    kv_indices[5:32] = 0

    assert _target_stitch_ready(entry, kv_indices=kv_indices) is True
```

- [x] **Step 2: Run the regression test and confirm the current implementation fails**

Run:

```powershell
& 'C:\Users\Tianci J\anaconda3\python.exe' -m pytest test/srt/test_pd_flip_hicache_stitch.py::test_target_prepare_accepts_staged_l3_indices_before_atomic_commit -q
```

Expected: FAIL with `uninitialized KV indices remain after stitch`.

- [x] **Step 3: Add staged segment validation**

Keep `_pd_flip_target_stitch_ready` as the Prepare-facing method, but replace its whole-table check with explicit segment checks:

```python
def _pd_flip_invalid_kv_positions(self, kv_indices) -> List[int]:
    invalid_mask = kv_indices <= 0
    if not bool(invalid_mask.any().item()):
        return []
    values = (
        kv_indices.detach().cpu().tolist()
        if hasattr(kv_indices, "detach")
        else kv_indices.tolist()
    )
    return [index for index, value in enumerate(values) if int(value) <= 0]
```

Within `_pd_flip_target_stitch_ready(entry)`:

```python
decode_req = entry["decode_req"]
req = decode_req.req
prefix_match = getattr(decode_req, "prefix_match", None)

if hit_len == 0:
    suffix_indices = self.req_to_token_pool.req_to_token[
        req.req_pool_idx, :committed_len
    ]
    invalid_suffix = self._pd_flip_invalid_kv_positions(suffix_indices)
    if invalid_suffix:
        raise RuntimeError(
            "migration target source fallback coverage failed: "
            f"{len(invalid_suffix)} invalid KV indices for rid={req.rid}, "
            f"positions={invalid_suffix[:16]}"
        )
    return True

if prefix_match is None:
    raise RuntimeError("migration target HiCache restore failed: prefix match is missing")

l1_len = int(prefix_match.l1_prefix_len)
if not 0 <= l1_len <= hit_len:
    raise RuntimeError(
        "migration target HiCache restore failed: "
        f"invalid L1 boundary L1={l1_len}, H={hit_len}"
    )

l1_indices = prefix_match.prefix_indices
if len(l1_indices) != l1_len:
    raise RuntimeError("migration target HiCache restore failed: L1 length mismatch")

restore_indices = getattr(decode_req, "hicache_restored_kv_indices", None)
expected_restore = hit_len - l1_len
if expected_restore:
    if restore_indices is None or len(restore_indices) != expected_restore:
        raise RuntimeError(
            "migration target HiCache restore failed: "
            f"restored index length mismatch, expected={expected_restore}, "
            f"actual={0 if restore_indices is None else len(restore_indices)}"
        )
```

Validate positive indices independently for L1, staged restore, and the formal suffix `[H,C0)`. Error positions for restore must be reported with the `L1` offset; suffix errors must be reported with the `H` offset. All `H>0` staged failures must retain the `migration target HiCache restore failed` prefix so the existing pump requests full-source fallback.

- [x] **Step 4: Add staged failure tests**

Add focused tests:

```python
def test_target_prepare_rejects_short_staged_restore_indices():
    entry = _target_entry(h=32, p=64, c0=100, l1=5)
    entry["decode_req"].hicache_restored_kv_indices = np.arange(
        101, 127, dtype=np.int64
    )
    with pytest.raises(RuntimeError, match="expected=27, actual=26"):
        _target_stitch_ready(entry)


def test_target_prepare_rejects_invalid_staged_restore_indices():
    entry = _target_entry(h=32, p=64, c0=100, l1=5)
    entry["decode_req"].hicache_restored_kv_indices[0] = 0
    with pytest.raises(RuntimeError, match=r"positions=\[5\]"):
        _target_stitch_ready(entry)


def test_target_prepare_rejects_invalid_source_suffix_indices():
    entry = _target_entry(h=32, p=64, c0=100, l1=5)
    kv_indices = np.arange(1, 101, dtype=np.int64)
    kv_indices[40] = 0
    with pytest.raises(RuntimeError, match=r"positions=\[40\]"):
        _target_stitch_ready(entry, kv_indices=kv_indices)


def test_target_full_fallback_validates_complete_source_mapping():
    entry = _target_entry(h=0, p=64, c0=100, l1=0)
    kv_indices = np.arange(1, 101, dtype=np.int64)
    kv_indices[8] = 0
    with pytest.raises(RuntimeError, match=r"positions=\[8\]"):
        _target_stitch_ready(entry, kv_indices=kv_indices)
```

Each test must assert the exact failing segment and bounded absolute position sample.

- [x] **Step 5: Run the HiCache stitch test file**

Run:

```powershell
& 'C:\Users\Tianci J\anaconda3\python.exe' -m pytest test/srt/test_pd_flip_hicache_stitch.py -q
```

Expected: all tests pass, including the new deferred-mapping regression.

- [x] **Step 6: Commit Task 1**

```powershell
git add python/sglang/srt/managers/scheduler.py test/srt/test_pd_flip_hicache_stitch.py
git commit -m "fix(pd-flip): validate staged dual-source coverage"
```

---

### Task 2: Validate the formal mapping after atomic Commit

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py:2380-2435,4520-4585,4750-4780`
- Test: `test/srt/test_pd_flip_atomic_batch.py:90-245`
- Test: `test/srt/test_pd_flip_hicache_stitch.py:300-590`

**Interfaces:**
- Consumes: `_pd_flip_target_commit_hicache_restore(decode_req)` and staged-ready target entries.
- Produces: `_pd_flip_target_committed_mapping_ready(entry) -> bool`, which requires every formal mapping in `[0,C0)` to be valid.

- [x] **Step 1: Write the post-commit validation tests**

Update `target_scheduler()` with a default no-op validator and add:

```python
def test_target_commit_validates_each_formal_mapping(self):
    scheduler = target_scheduler({"r0": "transferred_held", "r1": "transferred_held"})
    checked = []
    scheduler._pd_flip_target_committed_mapping_ready = (
        lambda entry: checked.append(entry["decode_req"].req.rid) or True
    )

    out = Scheduler.commit_pd_flip_migration_target(
        scheduler,
        PDFlipMigrationTargetCommitReq(session_id="s", rids=["r0", "r1"]),
    )

    self.assertTrue(out.success)
    self.assertEqual(checked, ["r0", "r1"])
```

Add a second test where the validator raises on `r1`; assert both entries become `aborted`, no request enters `waiting_queue`, and the result is unsuccessful.

- [x] **Step 2: Run the new atomic test and confirm it fails**

Run:

```powershell
& 'C:\Users\Tianci J\anaconda3\python.exe' -m pytest test/srt/test_pd_flip_atomic_batch.py::TestAtomicTargetHandoff::test_target_commit_validates_each_formal_mapping -q
```

Expected: FAIL because `commit_pd_flip_migration_target` does not call the validator.

- [x] **Step 3: Implement the committed mapping validator**

Move the old whole-table logic into a dedicated method:

```python
def _pd_flip_target_committed_mapping_ready(self, entry: Dict[str, Any]) -> bool:
    req = entry["decode_req"].req
    committed_len = int(entry["target_committed_len"])
    kv_indices = self.req_to_token_pool.req_to_token[
        req.req_pool_idx, :committed_len
    ]
    invalid_positions = self._pd_flip_invalid_kv_positions(kv_indices)
    if invalid_positions:
        raise RuntimeError(
            "migration target committed KV mapping is incomplete: "
            f"{len(invalid_positions)} invalid indices for rid={req.rid}, "
            f"positions={invalid_positions[:16]}"
        )
    return True
```

- [x] **Step 4: Enforce the validator at both commit sites**

In `commit_pd_flip_migration_target`, after each non-drop HiCache commit:

```python
self._pd_flip_target_commit_hicache_restore(entry["decode_req"])
self._pd_flip_target_committed_mapping_ready(entry)
```

In `_pd_flip_target_pump_transfer`, for non-`prepare_only` sessions:

```python
self._pd_flip_target_commit_hicache_restore(decode_req)
self._pd_flip_target_committed_mapping_ready(entry)
```

Do not call the formal validator during `prepare_only`; those entries remain `transferred_held` until the explicit Commit request.

- [x] **Step 5: Run atomic and stitch tests**

Run:

```powershell
& 'C:\Users\Tianci J\anaconda3\python.exe' -m pytest test/srt/test_pd_flip_atomic_batch.py test/srt/test_pd_flip_hicache_stitch.py -q
```

Expected: all tests pass; no target request is scheduled before Activate.

- [x] **Step 6: Commit Task 2**

```powershell
git add python/sglang/srt/managers/scheduler.py test/srt/test_pd_flip_atomic_batch.py test/srt/test_pd_flip_hicache_stitch.py
git commit -m "fix(pd-flip): verify mapping after atomic commit"
```

---

### Task 3: Regression verification and four-node experiment handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-07-13-pd-flip-dual-source-atomic-stitch-fix.md`
- Verify: `python/sglang/srt/managers/scheduler.py`
- Verify: `test/srt/test_pd_flip_hicache_stitch.py`
- Verify: `test/srt/test_pd_flip_atomic_batch.py`

**Interfaces:**
- Consumes: completed Tasks 1 and 2.
- Produces: tested `main` commits and a precise live-experiment acceptance checklist.

- [x] **Step 1: Run syntax and whitespace checks**

```powershell
& 'C:\Users\Tianci J\anaconda3\python.exe' -m py_compile `
  python/sglang/srt/managers/scheduler.py `
  test/srt/test_pd_flip_hicache_stitch.py `
  test/srt/test_pd_flip_atomic_batch.py
git diff --check
```

Expected: exit code 0 with no syntax or whitespace errors.

- [x] **Step 2: Run the focused regression suite**

```powershell
& 'C:\Users\Tianci J\anaconda3\python.exe' -m pytest `
  test/srt/test_pd_flip_hicache_stitch.py `
  test/srt/test_pd_flip_atomic_batch.py `
  test/srt/test_pd_flip_reconciliation.py `
  test/srt/test_pd_flip_progressive_controller.py `
  test/srt/test_pd_flip_progressive_workload.py -q
```

Expected: all collected tests pass. If the local environment cannot import an optional SGLang dependency, run the AST-backed stitch and atomic files locally and record the exact collection blocker; do not restart the stopped cluster merely to satisfy local collection.

- [x] **Step 3: Review the final diff against the design**

Confirm:

```text
Prepare: staged L1 + staged L3 + formal suffix validation
Commit: HiCache binding + formal [0,C0) validation
Activate: only after ready_to_activate
Fallback: H>0 staged failure requests source full transfer
H=0: full-source validation remains independent of HiCache
```

Verified on 2026-07-13:

- `py_compile` and `git diff --check`: passed.
- Stitch and atomic regression files: `74 passed, 6 skipped`.
- Reconciliation/progressive regression files excluding the pre-existing stale exact-dict assertion: `157 passed, 1 deselected`.
- The deselected test expects an older request-measurement dictionary and omits fields already added by commit `150f713ef`; the dual-source implementation diff does not modify that status export or test.
- Final whole-change review: approved after adding stitch-disabled immediate and prepare-to-commit control-flow regressions.

- [x] **Step 4: Mark this plan complete and commit the plan update**

```powershell
git add docs/superpowers/plans/2026-07-13-pd-flip-dual-source-atomic-stitch-fix.md
git commit -m "docs(pd-flip): record dual-source stitch implementation"
```

- [x] **Step 5: Push tested `main` to the fork**

```powershell
git push origin main
```

Expected: `origin/main` points to the same commit as local `main`; no PR is created.

- [ ] **Step 6: Prepare tomorrow's 09:00–10:00 acceptance run**

Before starting containers, compare `scheduler.py` and `decode.py` content across local `main` and all four bind-mounted host repos. Then run the existing 40-request full-chain script. Accept only if `full_prefix_stitch` commits without `fallback_required`, 40/40 requests complete, node1 reaches Prefill, node2 continues Decode, and all workers remain healthy after at least 60 seconds idle.
