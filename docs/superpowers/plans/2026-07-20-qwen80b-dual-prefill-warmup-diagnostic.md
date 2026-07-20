# Qwen80B Dual-Prefill Warmup Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the clean-upstream runner's single long-Prompt warmup with sequential representative long and short Prefill warmups, then execute one valid four-node diagnostic run and report whether the request-01 TTFT barrier disappears.

**Architecture:** Keep the existing checked-in `1P3D` lifecycle and frozen 40-request trace unchanged. Generalize the embedded warmup client to select the first trace row of each requested `prompt_kind`, run `long` then `short` with one output token each, persist separate client evidence, capture one log window spanning both warmups, flush all workers' KV cache, and only then start measured replay.

**Tech Stack:** Bash orchestration, embedded Python 3 standard library HTTP/SSE client, pytest source-contract tests, Docker, SSH, SGLang upstream clean image, Mooncake PD transfer, JSON/JSONL artifacts.

## Global Constraints

- Use `tiancij/sglang-upstream:v0.5.15-clean` with image ID `sha256:7dd92779d739364d79af34af65815ddc14e567728e5256f65ac922367161213e`.
- Reuse trace SHA256 `c5dbbf75c997dfc5d67a18251082f2f246d6c055eb4af5040fbe147f49f4ce5d`, 40 requests, and 10,000 completion tokens per request.
- Keep `1P3D`, TP 4, DP 1, GPUs `0,1,2,3`, `mlx5_bond_0`, GID index 3, and memory fraction 0.88.
- Warmups are sequential `long` then `short`, each generates exactly one token, and neither contributes to measured metrics.
- Flush KV/Radix cache only after both warmups. If any flush fails, mark the attempt forensic and stop; do not relaunch under the same run ID.
- Never stop or reuse unowned resources. Never use `docker restart`, wildcard process killing, `pkill`, `killall`, `kill -9`, or `docker rm -f`.
- Do not print, commit, or bundle the admin key; load it through the existing private `ENV_FILE`/`ADMIN_API_KEY_FILE` contract.

---

### Task 1: Specify the dual-warmup runner contract

**Files:**
- Modify: `test/srt/test_pd_upstream_qwen80b_runner.py`
- Test: `test/srt/test_pd_upstream_qwen80b_runner.py`

**Interfaces:**
- Consumes: the current runner as UTF-8 text through `source()`.
- Produces: a failing source-contract test that requires two trace-selected warmup kinds, two records, correct ordering, validation bounds, and flush-before-measure behavior.

- [ ] **Step 1: Replace the single-warmup source-contract test**

Replace `test_smoke_warms_trace_long_prompt_before_flush_and_captures_log_window` with assertions equivalent to:

```python
def test_smoke_warms_long_and_short_trace_prompts_before_one_flush():
    text = source()
    assert 'warmup_kinds = ("long", "short")' in text
    assert 'next(row for row in trace_rows if row["prompt_kind"] == prompt_kind)' in text
    assert 'f"{prompt_kind}-prefill-warmup.json"' in text
    assert 'assert prompt_tokens > 6000' in text
    assert 'assert 500 <= prompt_tokens <= 1000' in text
    assert '"measured": False' in text
    assert '"kv_cache_flushed_after": True' in text
    assert 'warmup-node${index}.docker.log' in text
    assert 'warmup-router.docker.log' in text

    long_record = text.index('"long-prefill-warmup.json"')
    short_record = text.index('"short-prefill-warmup.json"')
    flush = text.index("flush_cache", short_record)
    measure = text.index("measure\n", text.index("run_all()"))
    assert long_record < short_record < flush < measure
```

- [ ] **Step 2: Add an assertion that failed flush cannot silently relaunch**

Add:

```python
def test_dual_warmup_flush_failure_is_forensic_instead_of_relaunching():
    text = source()
    assert "post-warmup cache flush failed" in text
    assert "stop_inference" in text
    flush_failure = text.index("post-warmup cache flush failed")
    measure = text.index("measure\n", text.index("run_all()"))
    assert flush_failure < measure
```

The runner may still contain relaunch logic for unrelated pre-warmup smoke gates, but the post-dual-warmup flush branch must return nonzero rather than call `start_all`.

- [ ] **Step 3: Run the focused test and confirm it fails for the missing short warmup**

Run:

```powershell
& 'C:\Users\Tianci J\anaconda3\python.exe' -m pytest test/srt/test_pd_upstream_qwen80b_runner.py -q
```

Expected: at least the new dual-warmup assertions fail because `short-prefill-warmup.json` and `warmup_kinds` are absent.

### Task 2: Implement and document sequential long/short warmups

**Files:**
- Modify: `experiments/pd_upstream_qwen80b_baseline.sh`
- Modify: `docs/runbooks/pd_upstream_qwen80b_baseline.md`
- Test: `test/srt/test_pd_upstream_qwen80b_runner.py`

**Interfaces:**
- Consumes: `${RUN_DIR}/trace/trace.jsonl`, the existing router URL, and the private helper environment.
- Produces: `${RUN_DIR}/smoke/long-prefill-warmup.json`, `${RUN_DIR}/smoke/short-prefill-warmup.json`, a shared warmup log window, and a proven KV-cold/process-warm boundary before `measure`.

- [ ] **Step 1: Read the full trace and select one row per kind**

Replace the single `trace_row` selection with:

```python
with open(trace_path, encoding="utf-8") as handle:
    trace_rows = [json.loads(line) for line in handle if line.strip()]
warmup_kinds = ("long", "short")
selected_rows = {
    prompt_kind: next(
        row for row in trace_rows if row["prompt_kind"] == prompt_kind
    )
    for prompt_kind in warmup_kinds
}
```

- [ ] **Step 2: Extract one reusable streamed warmup function**

Inside the embedded Python block, implement:

```python
def run_prefill_warmup(prompt_kind, trace_row):
    warmup_body = dict(trace_row["body"])
    warmup_body.pop("custom_params", None)
    warmup_body["max_tokens"] = 1
    warmup_body["stream"] = True
    warmup_body["stream_options"] = {"include_usage": True}
    request = urllib.request.Request(
        url,
        data=json.dumps(warmup_body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
        },
    )
    started_utc = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    first_output_utc = None
    first_output_monotonic = None
    completion_tokens = None
    prompt_tokens = None
    finish_reason = None
    response_status = None
    with urllib.request.urlopen(request, timeout=600) as response:
        response_status = response.status
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or {}
            if usage.get("prompt_tokens") is not None:
                prompt_tokens = int(usage["prompt_tokens"])
            if usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])
            choices = event.get("choices") or []
            if choices and choices[0].get("finish_reason") is not None:
                finish_reason = choices[0]["finish_reason"]
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or delta.get("reasoning_content") or ""
                if content and first_output_monotonic is None:
                    first_output_monotonic = time.monotonic()
                    first_output_utc = datetime.now(timezone.utc)
    finished_monotonic = time.monotonic()
    finished_utc = datetime.now(timezone.utc)
    assert response_status == 200, response_status
    assert first_output_monotonic is not None and first_output_utc is not None
    assert completion_tokens == 1, completion_tokens
    assert finish_reason == "length", finish_reason
    return {
        "trace_request_id": trace_row["request_id"],
        "trace_prompt_kind": prompt_kind,
        "trace_prompt_chars": trace_row.get("prompt_chars"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "response_status": response_status,
        "started_utc": started_utc.isoformat(),
        "first_output_utc": first_output_utc.isoformat(),
        "finished_utc": finished_utc.isoformat(),
        "ttft_s": first_output_monotonic - started_monotonic,
        "total_duration_s": finished_monotonic - started_monotonic,
        "measured": False,
        "kv_cache_flushed_after": True,
    }
```

The returned dictionary must add `trace_prompt_kind=prompt_kind` and preserve the current UTC, monotonic, status, usage, finish-reason, and non-measured fields.

- [ ] **Step 3: Validate and persist both warmups sequentially**

Call the helper exactly in tuple order:

```python
warmup_results = {}
for prompt_kind in warmup_kinds:
    result = run_prefill_warmup(prompt_kind, selected_rows[prompt_kind])
    prompt_tokens = result["prompt_tokens"]
    if prompt_kind == "long":
        assert prompt_tokens > 6000, prompt_tokens
    else:
        assert 500 <= prompt_tokens <= 1000, prompt_tokens
    warmup_results[prompt_kind] = result
    with open(
        f"{run_dir}/smoke/{prompt_kind}-prefill-warmup.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
```

Compute the retained log window from the long result's start minus two seconds through the short result's finish plus two seconds.

- [ ] **Step 4: Make post-warmup flush failure terminal**

Replace the existing relaunch branch after the warmup log capture with:

```bash
if [[ "${flush_ok}" != "1" ]]; then
  echo "post-warmup cache flush failed; preserving forensic run without relaunch" >&2
  return 1
fi
ssh "${host}" "printf '%s\n' 'process warm; KV cold after successful dual-prefill warmup flush' > '${RUN_DIR}/smoke/cold-state.txt'"
```

This preserves the experiment's independent variable instead of erasing compiled state with a relaunch.

- [ ] **Step 5: Update the runbook artifact and validity contract**

Document both warmup records, long-then-short ordering, one-token outputs, one shared timestamp log window, post-warmup KV flush, and the rule that a failed flush invalidates the run without relaunch.

- [ ] **Step 6: Run static and focused verification**

Run:

```powershell
bash -n experiments/pd_upstream_qwen80b_baseline.sh
& 'C:\Users\Tianci J\anaconda3\python.exe' -m pytest test/srt/test_pd_upstream_qwen80b_runner.py -q
bash -lc "RUN_ID=dual-warmup-dry ENV_FILE='/mnt/c/Users/Tianci J/Desktop/sglang/experiments/pd_upstream_qwen80b_baseline.env.example' experiments/pd_upstream_qwen80b_baseline.sh dry-run"
```

Expected: Bash syntax succeeds, all runner tests pass, and dry-run prints the fixed `1P3D` configuration without contacting nodes or exposing a key.

- [ ] **Step 7: Commit the implementation intentionally**

Run:

```powershell
git add -- experiments/pd_upstream_qwen80b_baseline.sh test/srt/test_pd_upstream_qwen80b_runner.py docs/runbooks/pd_upstream_qwen80b_baseline.md
git diff --cached --check
git commit -m "test: warm qwen80b long and short prefill shapes"
```

### Task 3: Execute one owned four-node diagnostic run

**Files:**
- Read: `experiments/pd_upstream_qwen80b_baseline.env.example`
- Execute: `experiments/pd_upstream_qwen80b_baseline.sh`
- Create remotely: `${REMOTE_ARTIFACT_ROOT}/${RUN_ID}/`

**Interfaces:**
- Consumes: the committed runner revision, configured private `ENV_FILE`, four reachable nodes, the clean image, complete identical model files, and the frozen trace.
- Produces: one uniquely named valid or forensic remote artifact directory with complete run evidence and exact run-owned teardown.

- [ ] **Step 1: Generate and record a unique run ID**

Run in WSL from the repository:

```bash
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-upstream-qwen80b-dualwarm-r1"
test -n "${ENV_FILE:-}" && test -r "${ENV_FILE}"
printf 'RUN_ID=%s\n' "$RUN_ID"
printf '%s\n' "$RUN_ID" > pd-flip-artifacts/.latest-dualwarm-run-id
```

Expected: the run ID ends in `upstream-qwen80b-dualwarm-r1`; the private environment is readable without printing its contents.

- [ ] **Step 2: Run the read-only preflight**

Run:

```bash
RUN_ID="$RUN_ID" ENV_FILE="$ENV_FILE" bash experiments/pd_upstream_qwen80b_baseline.sh preflight
```

Expected: all four SSH checks, image/model fingerprints, GPU ownership, ports, driver, disk, mounts, RoCE GID, and clock checks pass. If any node or resource is unavailable, stop without loading the model.

- [ ] **Step 3: Execute the complete checked-in lifecycle once**

Run:

```bash
RUN_ID="$RUN_ID" ENV_FILE="$ENV_FILE" bash experiments/pd_upstream_qwen80b_baseline.sh run 2>&1 | tee "${RUN_ID}.controller.log"
```

Expected: four workers become healthy, router becomes healthy, long and short warmups pass, all four flushes pass, 40 formal requests complete, exact run-owned containers stop gracefully, and report validation marks the manifest `valid`.

- [ ] **Step 4: Preserve failures without retrying under the same ID**

If the command fails, do not issue a second `run`. Confirm exact-name ownership and execute only:

```bash
RUN_ID="$RUN_ID" ENV_FILE="$ENV_FILE" bash experiments/pd_upstream_qwen80b_baseline.sh collect-stop
```

Expected: the failed directory remains `forensic`; no row is merged into valid data.

### Task 4: Collect, validate, and report the diagnostic result

**Files:**
- Create locally: `pd-flip-artifacts/${RUN_ID}/`
- Create: `docs/superpowers/reports/2026-07-20-qwen80b-dual-prefill-warmup-diagnostic.md`
- Read for comparison: `pd-flip-artifacts/20260720T023903Z-upstream-qwen80b-longwarm-r1/`

**Interfaces:**
- Consumes: the complete remote artifact directory and retained long-only run.
- Produces: a local evidence mirror, independent validity checks, a request-00-11 timing comparison, and a bounded root-cause conclusion.

- [ ] **Step 1: Copy the complete artifact directory without deleting remote evidence**

Run from PowerShell:

```powershell
$runId = (Get-Content 'pd-flip-artifacts/.latest-dualwarm-run-id' -Raw).Trim()
rsync -a --protect-args "cloud-099:/root/tiancij-upstream-baseline-runs/$runId/" "pd-flip-artifacts/$runId/"
```

Expected: the local directory contains the manifest, inventory, trace, raw ledger, request metrics, responses, errors, logs, status, warmup records, and report.

- [ ] **Step 2: Validate checksums and experiment gates**

Run:

```powershell
$runId = (Get-Content 'pd-flip-artifacts/.latest-dualwarm-run-id' -Raw).Trim()
wsl.exe bash -lc "cd '/mnt/c/Users/Tianci J/Desktop/sglang/pd-flip-artifacts/$runId' && sha256sum -c INVENTORY.txt"
& 'C:\Users\Tianci J\anaconda3\python.exe' -c "import json,pathlib; p=pathlib.Path(r'pd-flip-artifacts/$runId'); m=json.loads((p/'manifest.json').read_text()); s=json.loads((p/'report/summary.json').read_text()); assert m['validity']=='valid'; assert s['completed_requests']==40; assert s['errors']==0"
```

Expected: every retained checksum passes, the manifest is valid, and 40 requests completed with zero errors.

- [ ] **Step 3: Recompute the request-00-11 timing comparison**

Parse both runs' `raw/upstream_baseline/request_metrics.jsonl` and calculate for each row:

```text
relative_start = start_monotonic - first_start_monotonic
absolute_first_token = relative_start + ttft_s
```

Report long-only versus dual-warmup TTFT for requests 00-11, with special emphasis on request 01 and whether requests 02-09 still complete around one shared barrier.

- [ ] **Step 4: Correlate the warmups and first formal wave with P/D logs**

Extract Prefill-side `ReqTimeStats`, `Prefill batch`, Decode-side `transfer_duration`, router latency, and any explicit compile/JIT/Triton/autotune line. Treat Decode `transfer_duration` as end-to-end KV readiness waiting unless an independent network-only interval is available.

- [ ] **Step 5: Write the diagnostic report and commit it**

The report must lead with validity, state that client TTFT is not GPU Prefill time, state whether the request-01 barrier disappeared, distinguish observed evidence from kernel-level inference, disclose that this is one diagnostic run, and link the raw warmup records, logs, metrics, manifest, and inventory.

Run:

```powershell
git add -- docs/superpowers/reports/2026-07-20-qwen80b-dual-prefill-warmup-diagnostic.md
git diff --cached --check
git commit -m "docs: report qwen80b dual prefill warmup diagnostic"
```
