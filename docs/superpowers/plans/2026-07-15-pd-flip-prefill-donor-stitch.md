# PD Flip Prefill Donor Stitch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in PD Flip mode where original Prefill workers send all complete Prompt pages `[0,B)`, source Decode sends `[B,C0)` and delta, and target Decode skips local prefix matching.

**Architecture:** Preserve the original Prefill bootstrap identity in the migration manifest, prepare two page-aligned receivers on target Decode, and add a Prefill-donor control operation that restores the original Prompt prefix through the Prefill worker's own HiCache before sending it. The controller coordinates donor, source, and target as a three-party atomic migration and rejects donor misses instead of falling back to a source-full Prompt copy.

**Tech Stack:** Python, SGLang scheduler and tokenizer control plane, Mooncake/HiCache, existing PD disaggregation KV sender/receiver APIs, pytest, Bash four-node runner.

## Global Constraints

- `B = floor(P / page_size) * page_size`, where `P = len(origin_input_ids)`.
- Original Prefill supplies exactly `[0,B)`; source Decode supplies exactly `[B,C0)` and the delta through `C1`.
- Target Decode does not call `_match_prefix_and_lock` in Prefill-donor mode.
- Missing Prefill donor coverage fails the session; source-full Prompt fallback is forbidden in this mode.
- Existing target-HiCache stitch behavior remains unchanged unless the new mode is enabled.
- The first implementation supports the DeepSeek V3.1 MLA experiment and rejects non-empty auxiliary `state_types` in Prefill-donor mode.
- The dedicated Mooncake store uses `MOONCAKE_GLOBAL_SEGMENT_SIZE=64gb`; workers contribute zero bytes.
- Existing 152 MB untracked experiment artifacts are never staged by implementation commits.

---

### Task 1: Protocol flag, fixed boundary, and manifest provenance

**Files:**
- Modify: `python/sglang/srt/server_args.py`
- Modify: `python/sglang/srt/managers/io_struct.py`
- Modify: `python/sglang/srt/managers/scheduler.py`
- Create: `test/srt/test_pd_flip_prefill_donor.py`

**Interfaces:**
- Produces `ServerArgs.enable_pd_flip_prefill_donor: bool` and CLI flag `--enable-pd-flip-prefill-donor`.
- Produces `Scheduler._pd_flip_prefill_donor_boundary(prompt_len: int, page_size: int) -> int`.
- Extends source/target requests with `prefill_donor_mode: bool = False`.
- Produces manifest fields `prefill_donor_host`, `prefill_donor_port`, `prompt_len`, `prefill_donor_end`, `source_decode_start`, and `prefill_donor_bootstrap_room`.

- [ ] **Step 1: Write failing boundary and manifest tests**

```python
@pytest.mark.parametrize(
    ("prompt_len", "page_size", "expected"),
    [(0, 64, 0), (63, 64, 0), (64, 64, 64), (1974, 64, 1920)],
)
def test_prefill_donor_boundary_uses_complete_prompt_pages(
    prompt_len, page_size, expected
):
    assert Scheduler._pd_flip_prefill_donor_boundary(prompt_len, page_size) == expected


def test_source_manifest_preserves_original_prefill_identity():
    req = make_req(
        origin_input_ids=list(range(1974)),
        bootstrap_host="192.168.0.42",
        bootstrap_port=8998,
        bootstrap_room=101,
    )
    scheduler = make_scheduler(page_size=64, donor_enabled=True)
    manifest = scheduler._pd_flip_build_migration_manifest(req)
    scheduler._pd_flip_apply_prefill_donor_manifest(req, manifest)
    assert manifest["prefill_donor_host"] == "192.168.0.42"
    assert manifest["prefill_donor_port"] == 8998
    assert manifest["prompt_len"] == 1974
    assert manifest["prefill_donor_end"] == 1920
    assert manifest["source_decode_start"] == 1920
    assert manifest["prefill_donor_bootstrap_room"] != manifest["migration_bootstrap_room"]
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest test/srt/test_pd_flip_prefill_donor.py -q
```

Expected: FAIL because the donor flag, boundary helper, request fields, and manifest helper do not exist.

- [ ] **Step 3: Add the minimal protocol surface**

Add request fields:

```python
@dataclass
class PDFlipMigrationSourceStartReq(BaseReq):
    session_id: Optional[str] = None
    target_url: Optional[str] = None
    rids: Optional[List[str]] = None
    include_waiting: bool = False
    prefill_donor_mode: bool = False


@dataclass
class PDFlipMigrationTargetPrepareReq(BaseReq):
    session_id: Optional[str] = None
    source_url: Optional[str] = None
    manifests: List[Dict[str, Any]] = field(default_factory=list)
    adopt_on_success: bool = False
    prepare_only: bool = False
    adopt_on_commit: bool = True
    prefill_donor_mode: bool = False
```

Add the boundary helper and derive a dedicated donor room from the base migration room using a non-overlapping DP-size multiple:

```python
@staticmethod
def _pd_flip_prefill_donor_boundary(prompt_len: int, page_size: int) -> int:
    page_size = max(1, int(page_size))
    return (max(0, int(prompt_len)) // page_size) * page_size


def _pd_flip_prefill_donor_room_for_req(self, req: Req) -> int:
    dp_size = max(1, int(getattr(self.server_args, "dp_size", 1)))
    return self._pd_flip_migration_room_for_req(req) + dp_size * (1 << 30)
```

Only call `_pd_flip_apply_prefill_donor_manifest` when `recv_req.prefill_donor_mode` is true, and reject it unless the server flag is enabled.

- [ ] **Step 4: Run focused tests and existing stitch boundary tests**

Run:

```bash
python -m pytest test/srt/test_pd_flip_prefill_donor.py test/srt/test_pd_flip_hicache_stitch.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add python/sglang/srt/server_args.py python/sglang/srt/managers/io_struct.py python/sglang/srt/managers/scheduler.py test/srt/test_pd_flip_prefill_donor.py
git commit -m "feat(pd-flip): define prefill donor protocol"
```

### Task 2: Target two-range receive without target prefix matching

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `test/srt/test_pd_flip_prefill_donor.py`

**Interfaces:**
- Produces `_pd_flip_prepare_target_donor_entry(manifest, source_host)` with separate `prefill_decode_req` and source `decode_req` receivers.
- Produces `_pd_flip_target_prealloc_donor_ranges(entry)` and `_pd_flip_target_donor_ranges_ready(entry)`.
- Stores `prefill_metadata_index` and `metadata_index` independently.

- [ ] **Step 1: Write failing target preparation tests**

```python
def test_target_donor_mode_skips_target_prefix_match():
    scheduler = make_target_scheduler(donor_enabled=True)
    scheduler.disagg_decode_prealloc_queue._match_prefix_and_lock = Mock(
        side_effect=AssertionError("target prefix match must be skipped")
    )
    entry = make_target_entry(prompt_len=1974, committed_len=2993, page_size=64)
    scheduler._pd_flip_target_prealloc_donor_ranges(entry)
    assert entry["target_prefix_match_skipped"] is True
    assert entry["prefill_received_start"] == 0
    assert entry["prefill_received_end"] == 1920
    assert entry["source_transfer_start"] == 1920
    assert entry["source_transfer_end"] == 2993


def test_target_donor_commit_waits_for_both_receivers():
    entry = ready_donor_entry(prefill_poll=KVPoll.Success, source_poll=KVPoll.Transferring)
    assert not scheduler._pd_flip_target_donor_ranges_ready(entry)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest test/srt/test_pd_flip_prefill_donor.py -q
```

Expected: FAIL because donor receiver preparation does not exist and current target code invokes `_match_prefix_and_lock`.

- [ ] **Step 3: Implement disjoint target receivers**

In donor mode:

```python
B = int(manifest["prefill_donor_end"])
C0 = int(manifest["kv_committed_len"])
dst_kv_indices = queue._pre_alloc(
    req, prefix_len=0, total_prefix_len=0, fill_len_override=C0
)
formal = self.req_to_token_pool.req_to_token[req.req_pool_idx, :C0]
prefill_pages = kv_to_page_indices(formal[:B].cpu().numpy(), page_size)
source_pages = kv_to_page_indices(formal[B:C0].cpu().numpy(), page_size)
```

Create the source receiver from `source_url` and `migration_bootstrap_room`.
Create the Prefill receiver from `prefill_donor_host`, `prefill_donor_port`, and
`prefill_donor_bootstrap_room`. Send metadata with `decode_prefix_len=0` to the
Prefill sender and `decode_prefix_len=B` to source Decode. Reject donor mode
when the KV manager reports non-empty `state_types`.

Keep donor-mode pump logic separate from the current target-HiCache pump branch
so disabled mode remains byte-for-byte behavior-compatible.

- [ ] **Step 4: Validate staged coverage and cleanup**

The donor-mode ready check must require:

```python
assert 0 <= B <= P <= C0
assert prefill_range == (0, B)
assert source_range == (B, C0)
assert not invalid(req_to_token_pool[req.req_pool_idx, :C0])
assert prefill_receiver_success or B == 0
assert source_receiver_success or B == C0
```

Abort and success cleanup must clear/abort both receivers and free both metadata
indices exactly once.

- [ ] **Step 5: Run target tests and regressions**

Run:

```bash
python -m pytest test/srt/test_pd_flip_prefill_donor.py test/srt/test_pd_flip_atomic_batch.py test/srt/test_pd_flip_hicache_stitch.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add python/sglang/srt/managers/scheduler.py test/srt/test_pd_flip_prefill_donor.py
git commit -m "feat(pd-flip): prepare dual donor target ranges"
```

### Task 3: Original Prefill restore-and-send control operation

**Files:**
- Modify: `python/sglang/srt/managers/io_struct.py`
- Modify: `python/sglang/srt/managers/tokenizer_control_mixin.py`
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `python/sglang/srt/entrypoints/http_server.py`
- Modify: `test/srt/test_pd_flip_prefill_donor.py`
- Modify: `test/srt/test_pd_flip_active_decode_handoff.py`

**Interfaces:**
- Produces `PDFlipPrefillDonorStartReq`, `PDFlipPrefillDonorStatusReq`, and `PDFlipPrefillDonorAbortReq`.
- Produces authenticated endpoints `/pd_flip/migration/prefill-donor/start`, `/status`, and `/abort`.
- Produces scheduler methods `start_pd_flip_prefill_donor`, `get_pd_flip_prefill_donor_status`, and `abort_pd_flip_prefill_donor`.

- [ ] **Step 1: Write failing full-hit, L3-restore, miss, and cleanup tests**

```python
def test_prefill_donor_sends_only_complete_prompt_pages_after_restore():
    scheduler, entry = make_prefill_donor_scheduler(hit_len=1974, prompt_len=1974)
    scheduler._pd_flip_pump_prefill_donor_entry(entry)
    assert entry["prefill_donor_restore_hit_len"] >= 1920
    assert entry["prefill_donor_transfer_start"] == 0
    assert entry["prefill_donor_transfer_end"] == 1920
    entry["sender"].send.assert_called_once()


def test_prefill_donor_incomplete_hit_fails_without_source_fallback():
    scheduler, entry = make_prefill_donor_scheduler(hit_len=1856, prompt_len=1974)
    scheduler._pd_flip_pump_prefill_donor_entry(entry)
    assert entry["phase"] == "failed"
    assert entry["error_type"] == "prefill_donor_incomplete"
    assert entry["expected_restore_len"] == 1920
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest test/srt/test_pd_flip_prefill_donor.py test/srt/test_pd_flip_active_decode_handoff.py -q
```

Expected: FAIL because donor control objects, routes, and scheduler lifecycle do not exist.

- [ ] **Step 3: Implement Prefill donor session lifecycle**

Use a separate `pd_flip_prefill_donor_session` so donor service does not replace
the worker's source/target migration session. For every manifest:

1. Build a synthetic request containing only `origin_input_ids[:B]`.
2. Call the local `disagg_decode_prealloc_queue._match_prefix_and_lock(req)`.
3. Require `decode_prefix_len >= B`.
4. Trim L1/L2/L3 lengths to exactly `B`, call `_pre_alloc`, and start HiCache
   prefetch.
5. Pump local restore until READY and commit staged restore indices into the
   synthetic request mapping.
6. Create a sender from `_pd_flip_get_source_kv_manager()` using
   `prefill_donor_bootstrap_room`.
7. After target metadata arrives, send page indices for `[0,B)`.
8. Record sender metrics, then release synthetic request KV, radix locks,
   metadata buffers, and sender state.

The session terminal states are `prefill_donor_transferred`,
`prefill_donor_failed`, and `prefill_donor_aborted`.

- [ ] **Step 4: Add control-plane forwarding and authenticated routes**

Add the three request classes to the scheduler dispatch table, tokenizer
communicator methods, HTTP imports, and `ADMIN_REQUIRED` routes. Status accepts
`session_id` as a query parameter, matching existing migration status behavior.

- [ ] **Step 5: Run donor and control-plane tests**

Run:

```bash
python -m pytest test/srt/test_pd_flip_prefill_donor.py test/srt/test_pd_flip_active_decode_handoff.py test/srt/test_pd_flip_admin_auth.py -q
python -m py_compile python/sglang/srt/managers/scheduler.py python/sglang/srt/managers/tokenizer_control_mixin.py python/sglang/srt/entrypoints/http_server.py
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add python/sglang/srt/managers/io_struct.py python/sglang/srt/managers/tokenizer_control_mixin.py python/sglang/srt/managers/scheduler.py python/sglang/srt/entrypoints/http_server.py test/srt/test_pd_flip_prefill_donor.py test/srt/test_pd_flip_active_decode_handoff.py
git commit -m "feat(pd-flip): send prompt pages from original prefill"
```

### Task 4: Three-party controller coordination and provenance measurements

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_controller.py`
- Modify: `scripts/playground/disaggregation/pd_flip_migration_measure.py`
- Modify: `test/srt/test_pd_flip_progressive_controller.py`
- Modify: `test/srt/test_pd_flip_migration_accounting.py`
- Modify: `test/srt/test_pd_flip_progressive_contract.py`

**Interfaces:**
- Adds `PDClusterConfig.prefill_donor_mode: bool = False` and CLI `--prefill-donor-mode`.
- Produces `_resolve_prefill_donor_groups(manifests) -> Dict[str, List[manifest]]` by exact URL-host match against configured nodes.
- Adds controller phases `prefill_donor_start_intent`, `prefill_donor_started`, and `prefill_donor_transferred`.

- [ ] **Step 1: Write failing controller-order and ambiguity tests**

```python
def test_atomic_batch_runs_original_prefill_donor_before_base_completion():
    controller, client, source, target = make_donor_controller()
    controller._execute_atomic_batch(
        source=source,
        target=target,
        session_id="donor-session",
        rids=["r0"],
        include_waiting=False,
        next_fsm_phase="observing",
        records=[],
    )
    assert client.steps.index("target_prepare") < client.steps.index("prefill_donor_start")
    assert client.steps.index("prefill_donor_ready") < client.steps.index("source_delta")
    assert "source_fallback" not in client.steps


def test_donor_host_resolution_rejects_ambiguous_nodes():
    controller = make_controller_with_duplicate_host_nodes()
    with pytest.raises(RuntimeError, match="ambiguous original Prefill donor"):
        controller._resolve_prefill_donor_groups([donor_manifest("10.0.0.1")])
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest test/srt/test_pd_flip_progressive_controller.py test/srt/test_pd_flip_migration_accounting.py -q
```

Expected: FAIL because donor grouping, posts, waits, and measurements do not exist.

- [ ] **Step 3: Implement controller handshake**

When donor mode is enabled:

1. POST source start with `prefill_donor_mode=true`.
2. POST target prepare with `prefill_donor_mode=true`.
3. Group returned manifests by exact configured worker URL hostname matching
   `prefill_donor_host`.
4. POST each original P `/pd_flip/migration/prefill-donor/start`.
5. Poll every donor status together with source and target base status.
6. Treat any donor miss/failure as terminal; abort donor, target, and source.
7. Do not enter existing source-full fallback handshake in donor mode.
8. Continue delta, commit, activation, source release, observation, and role flip
   in the existing order.

Persist donor URLs and phases in the recovery journal so a crash before cutover
can abort all three parties.

- [ ] **Step 4: Add measurement fields**

Emit per request:

```text
prompt_len
prefill_donor_end
source_decode_start
prefill_donor_host
prefill_donor_restore_hit_len
prefill_donor_pages
prefill_donor_transfer_bytes
prefill_donor_restore_seconds
prefill_donor_transfer_seconds
source_base_pages
source_base_transfer_bytes
target_prefix_match_skipped
provenance_mode
```

- [ ] **Step 5: Run controller and accounting tests**

Run:

```bash
python -m pytest test/srt/test_pd_flip_progressive_controller.py test/srt/test_pd_flip_migration_accounting.py test/srt/test_pd_flip_progressive_contract.py test/srt/test_pd_flip_reconciliation.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add scripts/playground/disaggregation/pd_flip_controller.py scripts/playground/disaggregation/pd_flip_migration_measure.py test/srt/test_pd_flip_progressive_controller.py test/srt/test_pd_flip_migration_accounting.py test/srt/test_pd_flip_progressive_contract.py
git commit -m "feat(pd-flip): coordinate prefill donor migration"
```

### Task 5: Four-node configuration, full verification, and live experiment

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_docker/reset_store_remote.sh`
- Modify: `scripts/playground/disaggregation/pd_flip_docker/run_worker.sh`
- Modify: `scripts/playground/disaggregation/pd_flip_docker/run_controller.sh`
- Modify: `scripts/playground/disaggregation/pd_flip_docker/env.example`
- Modify: `scripts/playground/disaggregation/pd_flip_docker/windows_four_node.ps1`
- Modify: relevant Docker contract tests under `test/srt/`

**Interfaces:**
- `MOONCAKE_STORE_SEGMENT_SIZE` defaults to `64gb` in the PD Flip experiment.
- `ENABLE_PD_FLIP_PREFILL_DONOR=1` adds the worker flag.
- `PD_FLIP_PREFILL_DONOR_MODE=1` adds the controller flag.

- [ ] **Step 1: Write failing configuration contract tests**

Assert the store reset script uses `${MOONCAKE_STORE_SEGMENT_SIZE:-64gb}`, workers
enable the donor flag only when configured, and the controller passes
`--prefill-donor-mode` only when configured.

- [ ] **Step 2: Run configuration tests and verify RED**

Run:

```bash
python -m pytest test/srt/test_pd_flip_progressive_contract.py test/srt/test_pd_flip_experiment_script.py -q
```

Expected: FAIL because the new environment controls are absent and store size is hardcoded to 4 GB.

- [ ] **Step 3: Implement 64 GB store and experiment flags**

Change the store launch to:

```bash
MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_STORE_SEGMENT_SIZE:-64gb}"
```

Keep worker `MOONCAKE_GLOBAL_SEGMENT_SIZE=0`. Add the donor worker/controller
flags through the existing environment-controlled argument arrays.

- [ ] **Step 4: Run the complete local verification set**

Run:

```bash
python -m pytest \
  test/srt/test_pd_flip_prefill_donor.py \
  test/srt/test_pd_flip_hicache_stitch.py \
  test/srt/test_pd_flip_atomic_batch.py \
  test/srt/test_pd_flip_progressive_controller.py \
  test/srt/test_pd_flip_migration_accounting.py \
  test/srt/test_pd_flip_progressive_contract.py \
  test/srt/test_pd_flip_reconciliation.py \
  test/srt/test_pd_flip_active_decode_handoff.py \
  -q
python -m py_compile \
  python/sglang/srt/managers/scheduler.py \
  python/sglang/srt/managers/tokenizer_control_mixin.py \
  python/sglang/srt/entrypoints/http_server.py \
  scripts/playground/disaggregation/pd_flip_controller.py \
  scripts/playground/disaggregation/pd_flip_migration_measure.py
git diff --check
```

Expected: all tests PASS, both compile commands exit 0, and `git diff --check` emits no errors.

- [ ] **Step 5: Commit the verified implementation**

```bash
git add python/sglang scripts/playground/disaggregation test/srt docs/superpowers/plans/2026-07-15-pd-flip-prefill-donor-stitch.md
git commit -m "feat(pd-flip): migrate kv from prefill and decode donors"
```

Use explicit paths or inspect `git status` first so old experiment artifacts are not staged.

- [ ] **Step 6: Deploy without disturbing unrelated experiments**

Check all four workers and active tmux/container processes first. Stop/restart only
the named PD Flip worker/router/controller sessions owned by this experiment.
Sync the committed code, reset the dedicated Mooncake store at 64 GB, and start
the four workers and router with donor mode enabled.

- [ ] **Step 7: Run the saved 40-request trace**

Run the existing trace40 full-chain runner with a new run ID. Capture worker,
router, controller, Mooncake, timeline, status, and measurement artifacts.

- [ ] **Step 8: Verify live acceptance from raw evidence**

Require:

```text
completed_requests = 40/40
request_errors = 0
target_prefix_match_skipped = true for every migrated RID
prefill donor coverage = [0,B) for every migrated RID
source base coverage = [B,C0) for every migrated RID
source full Prompt fallback count = 0
prefill donor misses = 0
Mooncake Put failures = 0
Mooncake evictions = 0
post-commit invalid indices = 0
all four workers healthy after idle window
source Decode completes D -> P role flip
```

If any requirement fails, preserve raw artifacts and report the first failing
phase without claiming the chain succeeded.

- [ ] **Step 9: Commit the experiment report and push `main`**

Stage only the concise report, manifest, measurement summary, and code changes;
do not stage raw logs or compressed archives unless explicitly requested.
Run a final `git status -sb`, push `main` to `origin/main`, and record the pushed
commit hash in the report.
