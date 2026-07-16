# PD Flip DeepSeek DP8 Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make strict Prefill-donor PD Flip correctly migrate DeepSeek MLA requests between TP8/DP8-Attention workers.

**Architecture:** Carry explicit P/source-D/target-D DP-rank identity and opaque KV-layout fingerprints in every manifest, execute each operation only on its owning scheduler rank, and aggregate DP8 barriers in the controller. Both donor streams write disjoint ranges into one selected target DP rank, followed by existing atomic validation/commit.

**Tech Stack:** Python 3, SGLang scheduler and disaggregation backends, Mooncake/HiCache, unittest/pytest, HTTP admin controller.

## Global Constraints

- Each role-bearing worker uses `--tp-size 8 --dp-size 8 --enable-dp-attention`.
- Strict ownership is P `[0,B)`, source D `[B,C0)`, and source D delta through `C1`.
- Target-local prefix matching and source-full Prompt fallback are forbidden in strict mode.
- Both donors must address one declared target DP rank.
- MLA transfer layout is opaque and must match exactly across source and target.
- A partial DP8 response is a failure, not a successful worker-level operation.
- Existing DP1 behavior remains compatible when DP Attention is disabled.

---

### Task 1: Rank and KV-layout manifest contract

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `test/srt/test_pd_flip_prefill_donor.py`
- Modify: `test/srt/test_pd_flip_atomic_batch.py`

**Interfaces:**
- Produces: manifest keys `source_decode_dp_rank`, `prefill_donor_dp_rank`, `target_decode_dp_rank`, `source_tp_rank`, `page_size`, `kv_layout`, and `model_fingerprint`.
- Consumes: scheduler parallel state, request routing fields, and KV pool runtime shape.

- [ ] **Step 1: Write a failing manifest metadata test**

```python
def test_manifest_carries_rank_and_mla_layout(self):
    req = self.make_req()
    req.routed_dp_rank = 5
    req.disagg_prefill_dp_rank = 2
    manifest = self.scheduler._pd_flip_build_migration_manifest(req)
    self.assertEqual(manifest["source_decode_dp_rank"], 5)
    self.assertEqual(manifest["prefill_donor_dp_rank"], 2)
    self.assertEqual(manifest["source_tp_rank"], self.scheduler.ps.tp_rank)
    self.assertEqual(manifest["kv_layout"], "mla")
    self.assertGreater(manifest["page_size"], 0)
    self.assertTrue(manifest["model_fingerprint"])
```

- [ ] **Step 2: Run the focused test**

Run: `python -m pytest test/srt/test_pd_flip_prefill_donor.py -k rank_and_mla_layout -v`

Expected: FAIL on missing keys.

- [ ] **Step 3: Add runtime identity helpers**

```python
def _pd_flip_attn_dp_rank(self) -> int:
    return int(getattr(self.ps, "attn_dp_rank", getattr(self.ps, "dp_rank", 0)) or 0)

def _pd_flip_kv_layout_metadata(self) -> Dict[str, Any]:
    allocator = self.token_to_kv_pool_allocator
    pool = allocator.get_kvcache()
    layout = "mla" if pool.__class__.__name__ == "MLATokenToKVPool" else "mha"
    return {
        "page_size": int(allocator.page_size),
        "kv_layout": layout,
        "kv_pool_class": pool.__class__.__name__,
        "model_fingerprint": self._pd_flip_model_fingerprint(),
    }
```

Build the fingerprint from model path/config values that determine KV shape and dtype using sorted JSON plus SHA-256.

- [ ] **Step 4: Add strict target compatibility validation**

```python
def _pd_flip_validate_layout(self, manifest):
    local = self._pd_flip_kv_layout_metadata()
    for key in ("page_size", "kv_layout", "kv_pool_class", "model_fingerprint"):
        if manifest.get(key) != local[key]:
            raise ValueError(f"PD Flip KV layout mismatch for {key}: source={manifest.get(key)!r} target={local[key]!r}")
```

- [ ] **Step 5: Run manifest and atomic tests**

Run: `python -m pytest test/srt/test_pd_flip_prefill_donor.py test/srt/test_pd_flip_atomic_batch.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the manifest contract**

```bash
git add python/sglang/srt/managers/scheduler.py test/srt/test_pd_flip_prefill_donor.py test/srt/test_pd_flip_atomic_batch.py
git commit -m "feat: identify pd flip dp ranks and kv layout"
```

### Task 2: Per-rank ownership filtering

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `python/sglang/srt/managers/io_struct.py`
- Create: `test/srt/test_pd_flip_dp_rank_ownership.py`

**Interfaces:**
- Produces: `_pd_flip_manifests_for_rank(manifests, field) -> list[dict]` and rank-aware operation output containing `dp_rank` and handled request IDs.
- Consumes: manifest rank fields from Task 1.

- [ ] **Step 1: Write rank-filter tests**

```python
def test_rank_filter_selects_only_local_owner():
    scheduler = make_scheduler(attn_dp_rank=3)
    manifests = [
        {"rid": "a", "source_decode_dp_rank": 3},
        {"rid": "b", "source_decode_dp_rank": 6},
    ]
    assert [m["rid"] for m in scheduler._pd_flip_manifests_for_rank(manifests, "source_decode_dp_rank")] == ["a"]

def test_rank_filter_rejects_missing_rank_in_dp8():
    scheduler = make_scheduler(attn_dp_rank=3, attn_dp_size=8)
    with pytest.raises(ValueError, match="source_decode_dp_rank"):
        scheduler._pd_flip_manifests_for_rank([{"rid": "a"}], "source_decode_dp_rank")
```

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest test/srt/test_pd_flip_dp_rank_ownership.py -v`

Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement strict DP8 filtering with DP1 compatibility**

```python
def _pd_flip_manifests_for_rank(self, manifests, field):
    local_rank = self._pd_flip_attn_dp_rank()
    dp_size = int(getattr(self.server_args, "dp_size", 1) or 1)
    selected = []
    for manifest in manifests:
        rank = manifest.get(field)
        if rank is None:
            if dp_size != 1:
                raise ValueError(f"{field} is required when dp_size={dp_size}")
            rank = 0
        if int(rank) == local_rank:
            selected.append(manifest)
    return selected
```

Apply it to source start/base/delta/finish, Prefill donor, target prepare/delta/commit/activate/abort, and migration status operations using the appropriate ownership field.

- [ ] **Step 4: Include rank evidence in operation results**

Extend the migration status/output schema with `dp_rank`, `handled_rids`, and `ignored_rids`; ensure ignored requests are not counted as failures on a non-owner rank.

- [ ] **Step 5: Run rank and existing DP1 tests**

Run: `python -m pytest test/srt/test_pd_flip_dp_rank_ownership.py test/srt/test_pd_flip_controller.py test/srt/test_pd_flip_status_route.py -v`

Expected: PASS.

- [ ] **Step 6: Commit rank ownership**

```bash
git add python/sglang/srt/managers/scheduler.py python/sglang/srt/managers/io_struct.py test/srt/test_pd_flip_dp_rank_ownership.py
git commit -m "feat: execute pd flip on owning dp ranks"
```

### Task 3: Explicit target-rank transfer routing

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `test/srt/test_pd_flip_prefill_donor.py`
- Modify: `test/srt/test_pd_flip_active_decode_handoff.py`

**Interfaces:**
- Produces: `_pd_flip_dest_ranks(manifest) -> list[int]` used by P donor, source base, and source delta sender construction.
- Consumes: `target_decode_dp_rank` from Task 1.

- [ ] **Step 1: Write failing sender routing tests**

```python
@pytest.mark.parametrize("sender_kind", ["prefill", "source_base", "source_delta"])
def test_sender_uses_manifest_target_rank(sender_kind):
    scheduler = make_scheduler(local_tp_rank=1)
    sender = build_sender(scheduler, sender_kind, {"target_decode_dp_rank": 6})
    assert sender.dest_tp_ranks == [6]
```

- [ ] **Step 2: Run tests and observe local-rank routing**

Run: `python -m pytest test/srt/test_pd_flip_prefill_donor.py test/srt/test_pd_flip_active_decode_handoff.py -k manifest_target_rank -v`

Expected: FAIL with destination `[self.ps.tp_rank]`.

- [ ] **Step 3: Implement target rank resolution**

```python
def _pd_flip_dest_ranks(self, manifest):
    rank = manifest.get("target_decode_dp_rank")
    if rank is None:
        if int(getattr(self.server_args, "dp_size", 1) or 1) == 1:
            return [int(self.ps.tp_rank)]
        raise ValueError("target_decode_dp_rank is required for DP Attention migration")
    return [int(rank)]
```

Replace every strict-mode sender call site that currently passes `[self.ps.tp_rank]` with the manifest-derived value. Do not change normal non-PD-Flip disaggregation routing.

- [ ] **Step 4: Add a negative split-target test**

```python
def test_dual_sources_reject_different_target_ranks():
    with pytest.raises(ValueError, match="same target_decode_dp_rank"):
        validate_dual_source_targets(prefill_rank=2, source_rank=5)
```

- [ ] **Step 5: Run dual-source regression tests**

Run: `python -m pytest test/srt/test_pd_flip_prefill_donor.py test/srt/test_pd_flip_active_decode_handoff.py test/srt/test_pd_flip_hicache_stitch.py -v`

Expected: PASS.

- [ ] **Step 6: Commit explicit routing**

```bash
git add python/sglang/srt/managers/scheduler.py test/srt/test_pd_flip_prefill_donor.py test/srt/test_pd_flip_active_decode_handoff.py
git commit -m "fix: route pd flip donors to target dp rank"
```

### Task 4: DP8 controller aggregation and target selection

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_controller.py`
- Create: `test/srt/test_pd_flip_dp8_controller.py`
- Modify: `test/srt/test_pd_flip_controller.py`

**Interfaces:**
- Produces: `_index_dp_responses(body) -> dict[int, dict]`, `_request_owner_map(responses, field)`, and `select_target_dp_rank(statuses, required_pages) -> int`.
- Consumes: per-rank output from Task 2.

- [ ] **Step 1: Write aggregation failure tests**

```python
def test_index_dp_responses_requires_unique_ranks():
    with pytest.raises(RuntimeError, match="duplicate dp_rank"):
        _index_dp_responses([{"dp_rank": 1}, {"dp_rank": 1}])

def test_request_owner_map_requires_exactly_one_owner():
    responses = [
        {"dp_rank": 0, "handled_rids": ["r1"]},
        {"dp_rank": 1, "handled_rids": ["r1"]},
    ]
    with pytest.raises(RuntimeError, match="multiple owners"):
        _request_owner_map(responses, "handled_rids")

def test_target_selection_uses_free_kv_capacity():
    statuses = [{"dp_rank": 0, "free_kv_pages": 2}, {"dp_rank": 1, "free_kv_pages": 20}]
    assert select_target_dp_rank(statuses, required_pages=8) == 1
```

- [ ] **Step 2: Run and confirm the current DP1 guard fails**

Run: `python -m pytest test/srt/test_pd_flip_dp8_controller.py -v`

Expected: FAIL because helpers do not exist and multi-rank status is rejected.

- [ ] **Step 3: Replace `_require_single_dp_runtime_status` with rank indexing**

```python
def _index_dp_responses(body):
    items = body if isinstance(body, list) else [body]
    indexed = {}
    for item in items:
        status = item.get("status") if isinstance(item.get("status"), dict) else item
        rank = status.get("dp_rank")
        if rank is None:
            rank = 0 if len(items) == 1 else None
        if rank is None or int(rank) in indexed:
            raise RuntimeError(f"missing or duplicate dp_rank: {rank}")
        indexed[int(rank)] = item
    return indexed
```

- [ ] **Step 4: Implement deterministic target selection**

Filter ranks by runtime role, admission state, request capacity, and `free_kv_pages >= required_pages`; choose the candidate with the largest free page count, breaking ties by lowest rank. Write `target_decode_dp_rank` into every request manifest before target prepare.

- [ ] **Step 5: Implement all-rank barriers**

For base-ready, delta-ready, commit, activate, source release, and role flip, validate every participating `(worker, dp_rank)` response and include missing ranks/RIDs in the thrown error. Abort all prepared ranks if any barrier fails.

- [ ] **Step 6: Run controller tests**

Run: `python -m pytest test/srt/test_pd_flip_dp8_controller.py test/srt/test_pd_flip_controller.py test/srt/test_pd_flip_state_machine.py -v`

Expected: PASS.

- [ ] **Step 7: Commit DP8 orchestration**

```bash
git add scripts/playground/disaggregation/pd_flip_controller.py test/srt/test_pd_flip_dp8_controller.py test/srt/test_pd_flip_controller.py
git commit -m "feat: orchestrate pd flip across dp8 workers"
```

### Task 5: Per-rank observability and atomic validation

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `scripts/playground/disaggregation/pd_flip_migration_measure.py`
- Modify: `test/srt/test_pd_flip_timeline_measurements.py`
- Modify: `test/srt/test_pd_flip_atomic_batch.py`

**Interfaces:**
- Produces: per-event `worker`, `dp_rank`, range, byte count, layout fingerprint, and phase timestamps.
- Consumes: rank-aware session state from Tasks 1-4.

- [ ] **Step 1: Add failing timeline schema assertions**

```python
def test_ranked_migration_event_has_reconstructable_identity():
    event = make_event()
    for field in ("request_id", "session_id", "worker", "dp_rank", "phase", "epoch_ns", "mono_ns"):
        assert field in event
```

- [ ] **Step 2: Run timeline tests**

Run: `python -m pytest test/srt/test_pd_flip_timeline_measurements.py -v`

Expected: FAIL on missing rank identity.

- [ ] **Step 3: Add rank/layout fields to scheduler measurements**

Every P restore/send, source base/delta, target receive/validate/commit/activate, and source release event includes local DP rank, declared target rank, logical range, actual slots, bytes, and model fingerprint.

- [ ] **Step 4: Strengthen pre-commit validation**

Validate `0 <= B <= P <= C0 <= C1`, exact logical ownership, valid staged slots, identical layout fingerprints, all required receiver states, and one target DP rank before publishing `req_to_token_pool[:C1]`.

- [ ] **Step 5: Update measurement flattening**

Preserve per-rank arrays instead of summing away identity. Worker-level totals are derived fields; raw event rows remain one request/rank/phase each.

- [ ] **Step 6: Run timing and atomic tests**

Run: `python -m pytest test/srt/test_pd_flip_timeline_measurements.py test/srt/test_pd_flip_atomic_batch.py test/srt/test_pd_flip_migration_measure.py -v`

Expected: PASS.

- [ ] **Step 7: Commit observability**

```bash
git add python/sglang/srt/managers/scheduler.py scripts/playground/disaggregation/pd_flip_migration_measure.py test/srt/test_pd_flip_timeline_measurements.py test/srt/test_pd_flip_atomic_batch.py
git commit -m "feat: record dp rank migration timeline"
```

### Task 6: Runtime verification gate

**Files:**
- Modify: no production files unless a verification failure requires a focused correction.

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: local evidence that DP8/MLA behavior is ready for four-node smoke testing.

- [ ] **Step 1: Run the new DP8 and donor suites**

Run:

```bash
python -m pytest test/srt/test_pd_flip_dp_rank_ownership.py test/srt/test_pd_flip_dp8_controller.py test/srt/test_pd_flip_prefill_donor.py test/srt/test_pd_flip_atomic_batch.py -v
```

Expected: PASS.

- [ ] **Step 2: Run broad PD Flip regression tests**

Run: `python -m pytest test/srt/test_pd_flip_*.py -v`

Expected: PASS; hardware-only tests may be separately marked but no new CPU test is skipped.

- [ ] **Step 3: Run syntax and formatting checks on touched files**

Run:

```bash
python -m compileall python/sglang/srt/managers/scheduler.py scripts/playground/disaggregation/pd_flip_controller.py scripts/playground/disaggregation/pd_flip_migration_measure.py
pre-commit run --files python/sglang/srt/managers/scheduler.py python/sglang/srt/managers/io_struct.py scripts/playground/disaggregation/pd_flip_controller.py scripts/playground/disaggregation/pd_flip_migration_measure.py
```

Expected: PASS.

- [ ] **Step 4: Commit verification corrections if present**

```bash
git add -u
git commit -m "test: verify pd flip dp8 runtime"
```
