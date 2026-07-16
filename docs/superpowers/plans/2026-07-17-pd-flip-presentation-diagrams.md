# PD Flip Presentation Diagrams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a coherent, presentation-ready diagram set that explains the native fixed 1P3D SGLang flow and this repository's SLO-driven progressive 1P3D-to-2P2D flow from overview through implementation details.

**Architecture:** Build one editable HTML/SVG visualization source containing a shared node grammar and twelve slide-sized figures. Keep the overview figures topology-first, then reuse the same P1, D1/P2, D2, and D3 positions in six stage drill-downs and three mechanism drill-downs. Render and inspect the figures at 16:9 presentation dimensions before delivery.

**Tech Stack:** HTML fragments, inline SVG, theme-aware CSS, Codex visualization renderer, repository source references.

## Global Constraints

- Audience already understands Prefill/Decode disaggregation and KV Cache.
- Main flow uses the approved four-panel horizontal storyboard.
- The observation panel branches downward to rollback or continuation.
- Each detail figure distinguishes `复用 SGLang`, `扩展 SGLang`, and `我们新增`.
- Use the stable names P1, D1/P2, D2, and D3 and stable node positions throughout.
- Do not claim that native SGLang has no migration or KV-transfer capability; the missing piece is the complete SLO-driven online reconfiguration loop.
- Figures must remain legible on a 16:9 group-meeting slide.

---

### Task 1: Shared Visual Grammar and Baseline Figures

**Files:**
- Create: `C:/Users/Tianci J/.codex/visualizations/2026/07/16/019f6ba3-f18a-7001-b69b-b91e9ee5b31a/pd-flip-full-chain.html`

**Interfaces:**
- Consumes: approved design in `docs/superpowers/specs/2026-07-17-pd-flip-presentation-diagrams-design.md`.
- Produces: shared CSS classes for worker nodes, migration arrows, control arrows, ownership labels, capability labels, and slide navigation used by all later figures.

- [ ] **Step 1: Create the shared node and edge vocabulary**

Implement stable visual primitives for Prefill, Decode, draining, migrated request, control/observation, success, and rollback. Pair every color with a text or line-style cue.

- [ ] **Step 2: Draw Figure 1**

Draw native SGLang as Client → Router → P1 → D1/D2/D3 → output, with a concise note that roles and request ownership stay fixed during normal service.

- [ ] **Step 3: Draw Figure 2**

Draw the approved four-panel storyboard: normal 1P3D, first migration, observation, stable 2P2D. Add the downward rollback branch under observation and keep protocol details out of this figure.

- [ ] **Step 4: Render and inspect the overview figures**

Run the bundled visualization renderer against the HTML source. Expected: all four storyboard panels fit at 16:9 width, labels do not overlap, and the observation branch is readable without zooming.

- [ ] **Step 5: Commit the completed overview source if it is stored in the repository**

The thread-scoped visualization file is normally not committed. If a repository copy is requested later, commit only that explicit copy.

### Task 2: Six Stage Drill-Down Figures

**Files:**
- Modify: `C:/Users/Tianci J/.codex/visualizations/2026/07/16/019f6ba3-f18a-7001-b69b-b91e9ee5b31a/pd-flip-full-chain.html`

**Interfaces:**
- Consumes: shared primitives and overview figures from Task 1.
- Produces: six stage figures with a stable three-part structure: node change, reused SGLang capability, project extension/new logic.

- [ ] **Step 1: Add normal operation and SLO-trigger figure**

Show unchanged 1P3D topology beside role-level count aggregation, minimum sample gates, and the trigger predicate `prefill < threshold && decode >= threshold`.

- [ ] **Step 2: Add source/target and first-batch selection figure**

Show D1 source selection, D2/D3 target capacity, first-N running requests, worst-case KV reservation, ratio halving, and DP-rank partitioning.

- [ ] **Step 3: Add first migration figure**

Show D1 admission paused, selected requests moving to D2/D3, unselected requests continuing on D1, and references to double-source KV plus atomic batch transfer.

- [ ] **Step 4: Add observation and rollback figure**

Show roles still at 1P3D, D1 draining and decoding, fresh TTFT/TPOT samples, and the two primary outcomes: recover 1P3D or continue migration.

- [ ] **Step 5: Add remaining migration figure**

Show remaining running and eligible waiting requests leaving D1, followed by explicit emptiness checks for queues and migration sessions.

- [ ] **Step 6: Add runtime role-switch figure**

Show D1 Decode loop exit, runtime role mutation, Prefill loop redispatch, dual confirmation of `role` and `active_event_loop_role`, Router update, and admission recovery.

- [ ] **Step 7: Render and inspect all six stage figures**

Expected: every figure answers where requests are, where KV is, who owns requests, whether D1 accepts new traffic, and which capability is reused, extended, or new.

### Task 3: Three Mechanism Drill-Down Figures

**Files:**
- Modify: `C:/Users/Tianci J/.codex/visualizations/2026/07/16/019f6ba3-f18a-7001-b69b-b91e9ee5b31a/pd-flip-full-chain.html`

**Interfaces:**
- Consumes: stage narrative from Task 2.
- Produces: detailed KV, ownership/output, and failure-recovery figures.

- [ ] **Step 1: Draw the dual-source KV interval figure**

Render `[0,H)` as HiCache/Mooncake, `[H,C0)` as source initial transfer, and `[C0,C1)` as source delta. Include the zero-hit full-source fallback and activation coverage check.

- [ ] **Step 2: Draw ownership and output continuity**

Render `source active → target prepared/held → source finish → target activate`, mark the unique owner on each interval, and show output Relay deduplication by session, request ID, and sequence.

- [ ] **Step 3: Draw the failure and rollback matrix**

Cover capacity failure, first migration failure, observation recovery, Decode risk, second migration failure, Worker role-switch failure, and Router update failure. State final owner, D1 admission, Router drain state, and final/retry topology for every row.

- [ ] **Step 4: Render and inspect mechanism figures**

Expected: interval labels are unambiguous, there is never a time with two active owners, and every failure row has a safe destination.

### Task 4: Source Verification and Presentation QA

**Files:**
- Modify: `C:/Users/Tianci J/.codex/visualizations/2026/07/16/019f6ba3-f18a-7001-b69b-b91e9ee5b31a/pd-flip-full-chain.html`

**Interfaces:**
- Consumes: all twelve figures.
- Produces: verified final visualization with accurate source pointers and consistent terminology.

- [ ] **Step 1: Verify implementation claims against repository code**

Check controller stages against `scripts/playground/disaggregation/pd_flip_controller.py`, Worker migration and event-loop behavior against `python/sglang/srt/managers/scheduler.py`, local FSM semantics against `python/sglang/srt/disaggregation/flip_state_machine.py`, output Relay against `python/sglang/srt/managers/tokenizer_manager.py`, and flags against `python/sglang/srt/server_args.py`.

- [ ] **Step 2: Add concise source pointers**

Add file/function references to detail figures without putting code paths in the overview figures.

- [ ] **Step 3: Check terminology and attribution**

Expected: every implementation item is classified consistently as reused, extended, or new; P1/D1/P2/D2/D3 names remain stable; observation rollback does not imply that first-batch requests move back.

- [ ] **Step 4: Perform final visual QA**

Inspect at presentation width and narrow width. Expected: no clipped labels, overlapping arrows, horizontal scrolling, or unreadable text; first render shows a useful overview and navigation reaches every figure.

- [ ] **Step 5: Deliver the visualization**

Embed the completed visualization in the conversation and summarize the figure order and recommended main-deck versus appendix split.
