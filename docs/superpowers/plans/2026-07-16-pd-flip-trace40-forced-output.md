# PD Flip Trace40 Forced Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible 40-request long/short trace whose distinct Prompts each generate exactly the configured number of copies of one verified DeepSeek token.

**Architecture:** Add a reusable forced-single-token logit processor, inject and validate it while preparing the scheduled trace inside the SGLang container, and make `TRACE_MAX_TOKENS` the experiment's only output-length setting. Preserve the processor and nested custom parameters through PD Flip adoption, and validate output by token IDs/count rather than visible character count.

**Tech Stack:** Python 3, unittest/pytest, PyTorch, Hugging Face tokenizer, SGLang custom logit processors, Bash, OpenAI-compatible streaming API.

## Global Constraints

- The accepted trace has exactly 40 requests: 20 with 10,000-character Prompts and 20 with 1,000-character Prompts.
- All substantive user Prompt prefixes are distinct; shared chat-template tokens are allowed only up to a measured allowance.
- `TRACE_MAX_TOKENS=10000` is the sole output-length control for the accepted run.
- The forced visible character must encode to exactly one token and decode to the same character with `/models/deepseek_v3.1_terminus`.
- Requests use `temperature=0`, `ignore_eos=true`, no stop conditions, and streaming.
- Migration continues the original total token budget and preserves the forced-token processor.
- The replay script retains its standard-library-only import behavior.

---

### Task 1: Forced-single-token sampling primitive

**Files:**
- Modify: `python/sglang/srt/sampling/custom_logit_processor.py`
- Modify: `test/registered/unit/sampling/test_custom_logit_processor.py`

**Interfaces:**
- Consumes: `custom_param_list: list[dict]` containing `forced_token_id`.
- Produces: `ForcedSingleTokenLogitProcessor(CustomLogitProcessor)` and its inherited `to_str()` serialization.

- [ ] **Step 1: Write failing unit tests for per-row forcing and invalid IDs**

```python
from sglang.srt.sampling.custom_logit_processor import ForcedSingleTokenLogitProcessor


class TestForcedSingleTokenLogitProcessor(CustomTestCase):
    def test_forces_each_batch_row_to_its_declared_token(self):
        logits = torch.arange(20, dtype=torch.float32).reshape(2, 10)
        result = ForcedSingleTokenLogitProcessor()(
            logits,
            [{"forced_token_id": 3}, {"forced_token_id": 7}],
        )
        self.assertEqual(result[0, 3].item(), 0.0)
        self.assertEqual(result[1, 7].item(), 0.0)
        self.assertTrue(torch.isneginf(result[0, [0, 1, 2, 4, 5, 6, 7, 8, 9]]).all())
        self.assertTrue(torch.isneginf(result[1, [0, 1, 2, 3, 4, 5, 6, 8, 9]]).all())

    def test_rejects_out_of_range_token_id(self):
        with self.assertRaisesRegex(ValueError, "forced_token_id"):
            ForcedSingleTokenLogitProcessor()(torch.zeros(1, 4), [{"forced_token_id": 4}])
```

- [ ] **Step 2: Run the focused tests and confirm the import fails**

Run: `python -m pytest test/registered/unit/sampling/test_custom_logit_processor.py -k ForcedSingleToken -v`

Expected: FAIL because `ForcedSingleTokenLogitProcessor` is not defined.

- [ ] **Step 3: Implement strict row-wise masking**

```python
class ForcedSingleTokenLogitProcessor(CustomLogitProcessor):
    """Allow exactly one configured token for each request in a batch."""

    def __call__(self, logits, custom_param_list=None):
        if custom_param_list is None or len(custom_param_list) != logits.shape[0]:
            raise ValueError("forced_token_id parameters must match batch size")
        for batch_idx, params in enumerate(custom_param_list):
            token_id = params.get("forced_token_id") if isinstance(params, dict) else None
            if not isinstance(token_id, int) or not 0 <= token_id < logits.shape[-1]:
                raise ValueError(f"invalid forced_token_id for batch row {batch_idx}: {token_id}")
            logits[batch_idx, :] = -float("inf")
            logits[batch_idx, token_id] = 0.0
        return logits
```

- [ ] **Step 4: Run sampling tests**

Run: `python -m pytest test/registered/unit/sampling/test_custom_logit_processor.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the sampling primitive**

```bash
git add python/sglang/srt/sampling/custom_logit_processor.py test/registered/unit/sampling/test_custom_logit_processor.py
git commit -m "feat: add forced single token sampling"
```

### Task 2: Trace generation with one max-token control and unique Prompts

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_trace_replay.py`
- Modify: `test/srt/test_pd_flip_trace_replay.py`

**Interfaces:**
- Consumes: `build_trace(..., max_tokens: Optional[int], forced_text: Optional[str], forced_token_id: Optional[int])`.
- Produces: 40 distinct records with matching top-level/body `max_tokens` and forced-token metadata in `custom_params`.

- [ ] **Step 1: Add failing trace-contract tests**

```python
def test_trace40_has_one_output_budget_and_distinct_prompts(self):
    trace = build_trace(
        num_requests=40,
        interval_seconds=0.5,
        model="deepseek_v3.1_terminus",
        seed=7,
        short_chars=1000,
        long_chars=10000,
        short_count=20,
        long_count=20,
        max_tokens=10000,
        forced_text="字",
        forced_token_id=1234,
    )
    prompts = [row["body"]["messages"][0]["content"] for row in trace]
    self.assertEqual(len(set(prompts)), 40)
    self.assertTrue(all(row["max_tokens"] == 10000 for row in trace))
    self.assertTrue(all(row["body"]["max_tokens"] == 10000 for row in trace))
    self.assertTrue(all(row["body"]["ignore_eos"] is True for row in trace))
    self.assertTrue(all(row["body"]["custom_params"]["forced_token_id"] == 1234 for row in trace))
    self.assertTrue(all(row["body"]["custom_params"]["forced_text"] == "字" for row in trace))
```

- [ ] **Step 2: Run the contract test and confirm the new arguments fail**

Run: `python -m pytest test/srt/test_pd_flip_trace_replay.py::PDFlipTraceReplayTest::test_trace40_has_one_output_budget_and_distinct_prompts -v`

Expected: FAIL with unexpected keyword argument `max_tokens`.

- [ ] **Step 3: Add optional output controls to `build_trace`**

```python
def build_trace(
    *,
    num_requests,
    interval_seconds,
    model,
    seed,
    temperature=0.0,
    stream=True,
    short_chars=None,
    long_chars=None,
    short_count=None,
    long_count=None,
    max_tokens=None,
    forced_text=None,
    forced_token_id=None,
):
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if (forced_text is None) != (forced_token_id is None):
        raise ValueError("forced_text and forced_token_id must be provided together")
```

In the record loop, select `output_tokens = max_tokens if max_tokens is not None else profile.max_tokens`, set it in both record locations, prepend `f"[request-nonce:{seed:08x}-{index:04d}]\n"` before fitting the Prompt length, and add:

```python
body["ignore_eos"] = True
body["stop"] = None
if forced_token_id is not None:
    body["custom_params"].update(
        forced_token_id=int(forced_token_id),
        forced_text=forced_text,
    )
```

- [ ] **Step 4: Expose the values on the generate CLI**

```python
generate.add_argument("--max-tokens", type=int, default=None)
generate.add_argument("--forced-text", default=None)
generate.add_argument("--forced-token-id", type=int, default=None)
```

Pass all three arguments into `build_trace` in `main()`.

- [ ] **Step 5: Run all trace replay tests**

Run: `python -m pytest test/srt/test_pd_flip_trace_replay.py -v`

Expected: PASS.

- [ ] **Step 6: Commit trace generation**

```bash
git add scripts/playground/disaggregation/pd_flip_trace_replay.py test/srt/test_pd_flip_trace_replay.py
git commit -m "feat: parameterize trace40 output budget"
```

### Task 3: Tokenizer preflight and scheduled-trace injection

**Files:**
- Modify: `scripts/playground/disaggregation/pd_flip_prepare_trace.py`
- Create: `test/srt/test_pd_flip_prepare_trace.py`

**Interfaces:**
- Consumes: `tokenizer_path`, `forced_text`, and `max_tokens`.
- Produces: `resolve_forced_token(tokenizer, text) -> int`, a scheduled trace containing `custom_logit_processor`, and manifest token metadata.

- [ ] **Step 1: Write tokenizer and trace validation tests**

```python
class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [77] if text == "字" else [1, 2]

    def decode(self, ids):
        return "字" if ids == [77] else "wrong"


def test_resolve_forced_token_requires_one_round_trip_token():
    assert resolve_forced_token(FakeTokenizer(), "字") == 77
    with pytest.raises(ValueError, match="exactly one token"):
        resolve_forced_token(FakeTokenizer(), "两个")


def test_apply_output_contract_updates_both_budgets_and_processor():
    row = {"max_tokens": 1, "body": {"max_tokens": 1, "custom_params": {}}}
    apply_output_contract(row, 10000, "字", 77, "serialized")
    assert row["max_tokens"] == row["body"]["max_tokens"] == 10000
    assert row["body"]["custom_logit_processor"] == "serialized"
    assert row["body"]["custom_params"]["forced_token_id"] == 77
```

- [ ] **Step 2: Run tests and confirm helper imports fail**

Run: `python -m pytest test/srt/test_pd_flip_prepare_trace.py -v`

Expected: FAIL because the helpers do not exist.

- [ ] **Step 3: Implement tokenizer round-trip validation**

```python
def resolve_forced_token(tokenizer, forced_text: str) -> int:
    token_ids = tokenizer.encode(forced_text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError("forced_text must encode to exactly one token")
    if tokenizer.decode(token_ids) != forced_text:
        raise ValueError("forced_text must decode to exactly the same character")
    return int(token_ids[0])
```

- [ ] **Step 4: Implement one scheduled-row output contract**

```python
def apply_output_contract(row, max_tokens, forced_text, forced_token_id, processor):
    body = row.setdefault("body", {})
    custom = body.setdefault("custom_params", {})
    row["max_tokens"] = int(max_tokens)
    body["max_tokens"] = int(max_tokens)
    body["temperature"] = 0.0
    body["ignore_eos"] = True
    body["stop"] = None
    body["custom_logit_processor"] = processor
    custom["forced_text"] = forced_text
    custom["forced_token_id"] = int(forced_token_id)
```

At CLI execution time, import `AutoTokenizer` and `ForcedSingleTokenLogitProcessor`, resolve the token once, call `apply_output_contract` for all rows, and record the text, ID, and max-token value in the schedule manifest.

- [ ] **Step 5: Add CLI flags**

```python
parser.add_argument("--max-tokens", type=int, required=True)
parser.add_argument("--forced-text", required=True)
parser.add_argument("--tokenizer-path", required=True)
```

- [ ] **Step 6: Validate the complete scheduled trace**

Extend `_validate_trace` to require 40 unique request IDs and Prompts, equal top-level/body budgets, `ignore_eos is True`, one forced token ID, and one processor string across all rows.

- [ ] **Step 7: Run preparation and replay tests**

Run: `python -m pytest test/srt/test_pd_flip_prepare_trace.py test/srt/test_pd_flip_trace_replay.py -v`

Expected: PASS.

- [ ] **Step 8: Commit scheduled-trace preparation**

```bash
git add scripts/playground/disaggregation/pd_flip_prepare_trace.py test/srt/test_pd_flip_prepare_trace.py
git commit -m "feat: validate forced token trace contract"
```

### Task 4: Preserve generation state through PD Flip

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `test/srt/test_pd_flip_active_decode_handoff.py`
- Modify: `test/srt/test_pd_flip_atomic_batch.py`

**Interfaces:**
- Consumes: `Req.custom_logit_processor`, `SamplingParams.custom_params`, and `max_new_tokens`.
- Produces: a JSON-safe manifest round trip and an adopted target request with identical generation behavior.

- [ ] **Step 1: Write a failing manifest round-trip test**

```python
def test_manifest_preserves_forced_sampling_state(self):
    req = self.make_req(max_new_tokens=10000)
    req.custom_logit_processor = "serialized-forced-processor"
    req.sampling_params.custom_params = {"forced_token_id": 77, "forced_text": "字"}
    manifest = self.scheduler._pd_flip_build_migration_manifest(req)
    rebuilt = self.scheduler._pd_flip_manifest_to_req(manifest, "source-host")
    self.assertEqual(rebuilt.custom_logit_processor, "serialized-forced-processor")
    self.assertEqual(rebuilt.sampling_params.custom_params["forced_token_id"], 77)
    self.assertEqual(rebuilt.sampling_params.max_new_tokens, 10000)
```

- [ ] **Step 2: Run the focused handoff test**

Run: `python -m pytest test/srt/test_pd_flip_active_decode_handoff.py -k forced_sampling_state -v`

Expected: FAIL because the manifest omits the processor and nested custom params.

- [ ] **Step 3: Replace the shallow JSON-safe conversion with recursive conversion**

```python
@staticmethod
def _pd_flip_json_safe_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [Scheduler._pd_flip_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): Scheduler._pd_flip_json_safe_value(item)
            for key, item in value.items()
            if str(key) != "__req__"
        }
    raise TypeError(f"unsupported PD Flip manifest value: {type(value).__name__}")
```

Use it for sampling parameters, deliberately omit the scheduler-injected
`custom_params["__req__"]` back-reference, and add `custom_logit_processor` as
a top-level manifest string. The target `Req` constructor recreates `__req__`
with the new target request object.

- [ ] **Step 4: Reconstruct the target request with the processor**

Pass `custom_logit_processor=manifest.get("custom_logit_processor")` to `Req(...)` in `_pd_flip_manifest_to_req`. Before activation, assert that forced-token metadata is present when the processor is present and that `len(output_ids) <= max_new_tokens`.

- [ ] **Step 5: Add an atomic-commit rejection test for missing processor state**

```python
def test_target_rejects_forced_token_manifest_without_processor(self):
    manifest = self.forced_manifest()
    manifest["custom_logit_processor"] = None
    output = self.prepare_target([manifest])
    self.assertFalse(output.success)
    self.assertIn("custom_logit_processor", output.message)
```

- [ ] **Step 6: Run PD Flip handoff tests**

Run: `python -m pytest test/srt/test_pd_flip_active_decode_handoff.py test/srt/test_pd_flip_atomic_batch.py -v`

Expected: PASS.

- [ ] **Step 7: Commit sampling-state migration**

```bash
git add python/sglang/srt/managers/scheduler.py test/srt/test_pd_flip_active_decode_handoff.py test/srt/test_pd_flip_atomic_batch.py
git commit -m "fix: preserve sampling state across pd flip"
```

### Task 5: Experiment launcher and compact output evidence

**Files:**
- Modify: `experiments/pd_flip_trace40_full_chain.sh`
- Modify: `experiments/pd_flip_trace40_full_chain.env.example`
- Modify: `scripts/playground/disaggregation/pd_flip_trace_replay.py`
- Modify: `test/srt/test_pd_flip_experiment_script.py`
- Modify: `test/srt/test_pd_flip_trace_replay.py`

**Interfaces:**
- Consumes: `TRACE_MAX_TOKENS`, `TRACE_FORCED_TEXT`, and `MODEL_PATH`.
- Produces: validated effective trace, increased phase-specific timeouts, and compact per-request hash/sample/token-count evidence.

- [ ] **Step 1: Add failing launcher contract assertions**

```python
def test_deepseek_trace_contract_is_parameterized(self):
    script = Path("experiments/pd_flip_trace40_full_chain.sh").read_text()
    self.assertIn('TRACE_MAX_TOKENS="${TRACE_MAX_TOKENS:-10000}"', script)
    self.assertIn("--max-tokens '${TRACE_MAX_TOKENS}'", script)
    self.assertIn("--forced-text '${TRACE_FORCED_TEXT}'", script)
    self.assertIn("--tokenizer-path '${MODEL_PATH}'", script)
    self.assertNotIn("--timeout-seconds 900", script)
```

- [ ] **Step 2: Run the launcher test and confirm it fails**

Run: `python -m pytest test/srt/test_pd_flip_experiment_script.py -k deepseek_trace_contract -v`

Expected: FAIL on the missing environment contract.

- [ ] **Step 3: Add launcher settings and preflight validation**

```bash
TRACE_MAX_TOKENS="${TRACE_MAX_TOKENS:-10000}"
TRACE_FORCED_TEXT="${TRACE_FORCED_TEXT:-字}"
WORKLOAD_TIMEOUT_SECONDS="${WORKLOAD_TIMEOUT_SECONDS:-7200}"
MEASUREMENT_DURATION_SECONDS="${MEASUREMENT_DURATION_SECONDS:-7200}"
```

Pass these into `pd_flip_prepare_trace.py`, replace fixed 900-second workload/measurement values, and assert every effective row has the configured budget and forced-token fields.

- [ ] **Step 4: Add incremental output evidence to replay**

While parsing stream deltas, update `hashlib.sha256`, retain only the first and last 32 visible characters in compact metrics, and count completion tokens from stream token events/usage. Add fields `output_sha256`, `output_first`, `output_last`, `expected_completion_tokens`, and `completion_token_match`.

- [ ] **Step 5: Add replay validation tests**

```python
def test_compact_output_evidence_does_not_store_full_repetition(self):
    evidence = build_output_evidence(["字"] * 10000, expected_tokens=10000)
    self.assertEqual(evidence["completion_tokens"], 10000)
    self.assertTrue(evidence["completion_token_match"])
    self.assertLessEqual(len(evidence["output_first"]), 32)
    self.assertLessEqual(len(evidence["output_last"]), 32)
    self.assertNotIn("output_text", evidence)
```

- [ ] **Step 6: Run workload and script tests**

Run: `python -m pytest test/srt/test_pd_flip_trace_replay.py test/srt/test_pd_flip_experiment_script.py -v`

Expected: PASS.

- [ ] **Step 7: Generate and inspect a local 40-row trace**

Run:

```bash
python scripts/playground/disaggregation/pd_flip_trace_replay.py generate --output-dir /tmp/pd-flip-trace40 --model deepseek_v3.1_terminus --num-requests 40 --short-chars 1000 --long-chars 10000 --short-count 20 --long-count 20 --max-tokens 10000 --forced-text 字 --forced-token-id 77
```

Expected: 40 JSONL rows, 40 unique Prompt bodies, and `max_tokens=10000` in both locations. Token ID 77 here is fixture data only; live preparation resolves the actual DeepSeek token.

- [ ] **Step 8: Commit experiment integration**

```bash
git add experiments/pd_flip_trace40_full_chain.sh experiments/pd_flip_trace40_full_chain.env.example scripts/playground/disaggregation/pd_flip_trace_replay.py test/srt/test_pd_flip_experiment_script.py test/srt/test_pd_flip_trace_replay.py
git commit -m "feat: run trace40 with configurable forced output"
```

### Task 6: Workload verification gate

**Files:**
- Modify: `docs/superpowers/specs/2026-07-16-pd-flip-deepseek-v31-trace40-design.md` only if verification exposes a contract correction.

**Interfaces:**
- Consumes: all Task 1-5 deliverables.
- Produces: evidence that the workload layer is ready for DP8 migration integration.

- [ ] **Step 1: Run the focused CPU suite**

Run:

```bash
python -m pytest test/registered/unit/sampling/test_custom_logit_processor.py test/srt/test_pd_flip_prepare_trace.py test/srt/test_pd_flip_trace_replay.py test/srt/test_pd_flip_active_decode_handoff.py test/srt/test_pd_flip_atomic_batch.py test/srt/test_pd_flip_experiment_script.py -v
```

Expected: PASS with no skipped new contract tests.

- [ ] **Step 2: Run the existing PD Flip regression subset**

Run:

```bash
python -m pytest test/srt/test_pd_flip_hicache_stitch.py test/srt/test_pd_flip_prefill_donor.py test/srt/test_pd_flip_state_machine.py -v
```

Expected: PASS.

- [ ] **Step 3: Inspect the diff for hard-coded competing output budgets**

Run: `rg -n "max_tokens.*(96|256|768|4096|10000)|timeout-seconds 900" scripts/playground/disaggregation experiments/pd_flip_trace40_full_chain.sh`

Expected: profile defaults may remain for generic traces, but the Trace40 path has one `TRACE_MAX_TOKENS` override and no 900-second measured-run timeout.

- [ ] **Step 4: Commit any verification-only corrections**

```bash
git add -u
git commit -m "test: verify trace40 forced output contract"
```
