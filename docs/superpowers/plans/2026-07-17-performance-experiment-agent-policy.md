# Performance Experiment Agent Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repository-root `AGENTS.md` that makes the successful 2026-07-17 Qwen80B run the operational reference and enforces safe, reproducible, evidence-backed behavior for every SGLang performance experiment.

**Architecture:** Keep repository-wide mandatory rules in one root file so they apply to all descendants. Link to the executable Qwen80B runner, its runbook, the approved design, and retained artifact inventory instead of duplicating command implementations; preserve the historical run facts and its code-hash limitation directly in the policy.

**Tech Stack:** Markdown, Git, PowerShell validation commands, existing Bash experiment runner and JSON manifests.

## Global Constraints

- The policy applies to every SGLang performance experiment, not only PD Flip.
- Never stop or modify another user's processes, containers, GPUs, ports, mounts, or files.
- Never use `docker restart`, wildcard process matching, `pkill`, `killall`, or `kill -9` for experiment orchestration.
- A performance comparison requires matching code, image, model fingerprint, trace, hardware allocation, tokenizer, generation contract, SLO, and sampling configuration.
- Preserve event-level timestamps from which TTFT and every TPOT interval can be recomputed; aggregate CSV files are derived evidence.
- The historical baseline and state-machine runs prove operational completion but are not a controlled A/B pair because their code hashes differ.

---

### Task 1: Create the repository-wide agent policy

**Files:**
- Create: `AGENTS.md`
- Reference: `docs/superpowers/specs/2026-07-17-performance-experiment-agent-policy-design.md`
- Reference: `experiments/pd_flip_qwen80b_ab.sh`
- Reference: `docs/runbooks/pd_flip_qwen80b_ab.md`
- Reference: `pd-flip-artifacts/pd-switch-raw-20260717/INVENTORY.txt`

**Interfaces:**
- Consumes: The approved design and recorded manifest values from the successful run.
- Produces: Repository-wide instructions automatically inherited by agents working below the repository root.

- [x] **Step 1: Verify the root policy does not already exist**

Run:

```powershell
Test-Path .\AGENTS.md
```

Expected: `False`.

- [x] **Step 2: Create the root policy**

Create `AGENTS.md` with these exact top-level sections:

```markdown
# SGLang repository agent instructions

## Performance experiment policy
## Mandatory safety and ownership rules
## Required experiment lifecycle
## Comparison validity
## Raw evidence and artifact contract
## Successful Qwen80B reference run
## PD Flip reference workflow
## Failure handling and teardown
## Performance claim gate
```

The content must require preflight, run-owned identifiers, bounded health gates,
matched paired comparisons, graceful teardown, event-level timestamp retention,
artifact validation, and separation of forensic attempts from valid runs. It
must record run ID `20260717T042000Z-qwen80b-ab-obs2-gpu0123-gid3`, baseline
code `420bb4ad9`, state-machine code `f25c090c4`, 40 requests, 10,000 output
tokens, `1P3D`, 50% first migration, 2-second observation, final `2P2D`, and
first trigger request `qwen80b-02`.

- [x] **Step 3: Check the Markdown patch**

Run:

```powershell
git diff --check -- AGENTS.md
```

Expected: exit code 0 with no output.

### Task 2: Validate policy facts and coverage

**Files:**
- Validate: `AGENTS.md`
- Validate against: `experiments/pd_flip_qwen80b_ab.sh`
- Validate against: `docs/runbooks/pd_flip_qwen80b_ab.md`
- Validate against: `pd-flip-artifacts/pd-switch-raw-20260717/INVENTORY.txt`

**Interfaces:**
- Consumes: The completed root policy.
- Produces: Evidence that its references, mandatory warnings, and recorded run facts are complete and non-secret.

- [x] **Step 1: Verify referenced repository paths**

Run:

```powershell
@(
  'experiments/pd_flip_qwen80b_ab.sh',
  'docs/runbooks/pd_flip_qwen80b_ab.md',
  'docs/superpowers/specs/2026-07-17-performance-experiment-agent-policy-design.md',
  'pd-flip-artifacts/pd-switch-raw-20260717/INVENTORY.txt'
) | ForEach-Object { if (-not (Test-Path $_)) { throw "Missing reference: $_" } }
```

Expected: exit code 0 with no output.

- [x] **Step 2: Verify required policy phrases**

Run:

```powershell
$required = @(
  'docker restart', 'pkill', 'killall', 'kill -9',
  '20260717T042000Z-qwen80b-ab-obs2-gpu0123-gid3',
  '420bb4ad9', 'f25c090c4', 'qwen80b-02',
  '10,000', '1P3D', '2P2D', '50%', '2 seconds',
  'slo_ledger.jsonl', 'TTFT', 'TPOT'
)
$text = Get-Content -Raw .\AGENTS.md
$required | ForEach-Object { if (-not $text.Contains($_)) { throw "Missing phrase: $_" } }
```

Expected: exit code 0 with no output.

- [x] **Step 3: Check for placeholders and likely secrets**

Run:

```powershell
$bad = @(('T' + 'BD'), ('T' + 'ODO'), ('PLACE' + 'HOLDER'), 'ADMIN_API_KEY=', 'Bearer ')
$text = Get-Content -Raw .\AGENTS.md
$bad | ForEach-Object { if ($text.Contains($_)) { throw "Forbidden text: $_" } }
```

Expected: exit code 1 with no matches.

- [x] **Step 4: Review the complete diff**

Run:

```powershell
git diff -- AGENTS.md docs/superpowers/plans/2026-07-17-performance-experiment-agent-policy.md
```

Expected: only the approved root policy and this implementation plan are shown.

- [x] **Step 5: Commit the policy and plan**

Run:

```powershell
git add -- AGENTS.md docs/superpowers/plans/2026-07-17-performance-experiment-agent-policy.md
git commit -m "docs: codify performance experiment agent policy"
```

Expected: one commit containing only `AGENTS.md` and the implementation plan.
