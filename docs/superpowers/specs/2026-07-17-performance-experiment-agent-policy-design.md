# Repository-wide performance experiment agent policy design

## Goal

Create a repository-root `AGENTS.md` that makes the successful 2026-07-17
Qwen80B baseline and PD Flip state-machine runs the default operational
reference for future performance work. The policy applies to every SGLang
performance experiment, while keeping PD Flip-specific parameters in a
separate reference section.

## Scope

The policy governs local and multi-node performance experiments started by an
agent in this repository. It covers resource ownership, preflight, model and
service startup, paired comparisons, workload replay, telemetry, artifact
retention, teardown, validity checks, and performance claims.

It does not freeze every future workload to Qwen80B or 40 requests. Experiments
may change model, topology, load, SLO, and duration when required by their
question, but they must retain the same safety, provenance, raw-event, and
validity disciplines.

## Source-of-truth hierarchy

The root policy will direct agents to use these sources in order:

1. The manifest and raw artifacts from the successful run, for facts about
   what actually ran.
2. `experiments/pd_flip_qwen80b_ab.sh`, for executable orchestration.
3. `docs/runbooks/pd_flip_qwen80b_ab.md`, for operator guidance.
4. The root `AGENTS.md`, for mandatory repository-wide rules.

If prose and executable behavior disagree, agents must inspect the current
runner and manifests instead of guessing. For example, the successful runner
launches the four workers concurrently and then health-gates each worker before
starting the router. The successful run explicitly set a two-second observation
period even though a script default may differ.

## Historical successful reference

Record the following run as an operational reference, not as a clean A/B
performance conclusion:

- Run ID: `20260717T042000Z-qwen80b-ab-obs2-gpu0123-gid3`.
- Model: `Qwen3-Next-80B-A3B-Instruct`.
- Four nodes, GPUs `0,1,2,3` per node, TP 4, DP 1.
- Initial topology: `1P3D`.
- Trace: 40 requests, interleaved short/long prefixes, 10,000 forced output
  tokens per request.
- SLO window: 10 seconds; enter threshold 0.90; recover threshold 0.95.
- State-machine policy: migrate 50%, observe for 2 seconds, then migrate the
  remainder and finish at `2P2D`.
- First observed trigger: request `qwen80b-02`.
- Both retained modes completed all 40 requests without request errors. The
  state-machine controller reached `role_flip_complete`.

The baseline manifest recorded code `420bb4ad9`, while the successful
state-machine manifest recorded `f25c090c4`. Therefore these retained runs prove
the two chains can complete, but their metric difference must not be presented
as a controlled A/B result. A publishable comparison requires rerunning both
modes from the same code, image, model fingerprint, trace, and hardware
allocation.

## Mandatory policy structure

The root `AGENTS.md` will contain the following sections.

### Safety and ownership

- Inspect all target nodes before mutation and identify foreign containers and
  processes.
- Stop only exact run-owned container names and PIDs recorded by the runner.
- Never use `docker restart`, wildcard process matching, `pkill`, `killall`, or
  `kill -9` as experiment orchestration.
- Never stop, move, or reuse another person's containers, ports, GPUs, model
  files, or mounts.
- Abort rather than stacking a second model load on an unhealthy or partially
  torn-down node.

### Preflight and startup

- Verify SSH, model completeness and fingerprint, image identity, repository
  revision, GPU allocation and driver health, ports, mounts, RDMA/GID selection,
  disk capacity, and required secrets without logging secret values.
- Generate a unique run ID and run-owned names before starting anything.
- Start workers concurrently only through the validated runner. Require every
  worker's health and expected-role check before starting the router.
- Treat HTTP 503 during startup as "not ready" only within a bounded health
  gate. Do not send the workload until all gates pass.

### Comparison design

- Use paired baseline and candidate runs with identical trace, model
  fingerprint, code, image, GPU set, topology inputs, tokenizer, generation
  contract, SLOs, and sampling configuration.
- Run baseline first, collect and gracefully tear it down, verify node health,
  and only then load the candidate mode.
- Never call a pair valid when provenance fields differ. Operational smoke
  success and comparative performance validity are separate outcomes.
- For PD Flip reproduction, explicitly set the 50%/2-second policy rather than
  relying on defaults.

### Raw evidence and artifacts

- Preserve per-output client receive timestamps or an event ledger from which
  every TTFT and TPOT can be recomputed. Aggregated `ttft.csv`, `tpot.csv`, and
  percentiles are not raw evidence.
- Preserve the trace, manifests, request metrics, responses, errors, worker and
  router logs, controller journal/result, observer snapshots, topology/status
  snapshots, migration samples, and normalized request-stage events.
- Record wall-clock timestamps for cross-node alignment and monotonic timestamps
  for duration calculations. Record timezone and clock-sync evidence.
- Redact credentials from bundles; never copy private environment files into
  shareable artifacts.

### Validation and claims

- Require the planned number of terminal requests, unique request IDs, expected
  output-token counts, matching forced output, expected finish reason, zero
  request errors, successful teardown, and healthy nodes.
- For state-machine runs, additionally require controller completion, expected
  migration policy, final topology evidence, and request-to-migration linkage.
- Mark failed or partial attempts as forensic artifacts. Never silently merge
  them with the final run.
- Do not claim a performance winner from one quick-validation pair. Repeat
  matched pairs after the chain is stable and report variation.

### Reuse rules

- Default to the current Qwen80B runner and runbook for four-node PD Flip work.
- Reuse the general orchestration and evidence requirements for every other
  performance test, changing only parameters justified by the experiment.
- If a required behavior is absent from the runner, update and test the runner
  before launching an ad hoc sequence on the nodes.

## Verification

After creating `AGENTS.md`, verify that:

1. It is at the repository root and therefore applies repository-wide.
2. Every referenced repository path exists.
3. The recorded manifest values match the retained artifacts.
4. The policy explicitly distinguishes event-level raw timestamps from derived
   TTFT/TPOT statistics.
5. It explicitly records the historical code-hash mismatch and forbids using
   that pair as a controlled A/B conclusion.
6. It contains no credentials or private environment values.

