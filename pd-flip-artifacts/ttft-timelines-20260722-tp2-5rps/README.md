# TTFT timeline figures: Qwen80B TP=2, 5 req/s

These figures use the final valid 40-request runs from 2026-07-22.

## Sources

- Baseline run: `20260722T161500-qwen80b-tp2-5rps-upstream`
  - Remote source: `/home/tiancij/pd-artifacts/20260722T161500-qwen80b-tp2-5rps-upstream/raw/upstream_baseline/request_metrics.jsonl`
  - Local copy: `baseline_request_metrics.jsonl`
  - SHA-256: `62d7da107968421f3abf1fe662acdf8e795eb61d2119130f8aabc6e7ec6d8c49`
- State-machine run: `20260722T165500-qwen80b-tp2-5rps-candidate-final-state`
  - Remote source: `/home/tiancij/pd-artifacts/20260722T165500-qwen80b-tp2-5rps-candidate-final-state/state_machine/raw/request_metrics.jsonl`
  - Local copy: `state_machine_request_metrics.jsonl`
  - SHA-256: `5d1ecaf58066b7dfe9600a2903b4bb103192ebe473ee16c36d890b1cc573bab6`

Both inputs contain 40 completed request-level records. The chart uses the measured client timestamps:

```text
arrival = start_monotonic - first_start_monotonic
prefill_complete = start_monotonic + ttft_s - first_start_monotonic
```

## Figures

- `baseline_ttft_timeline.png`: generated directly with the user-provided `request_prefill_timeline.py`.
- `state_machine_ttft_timeline.png`: generated directly with the same script.
- `baseline_vs_state_machine_ttft_timeline.png`: same encoding on a shared axis, rendered by `plot_ttft_comparison.py`.

The open marker is the actual client request start. The filled marker is first-token receipt (`start_monotonic + TTFT`), used here as the client-observed Prefill-completion boundary. It is not an internal GPU-kernel timestamp.
