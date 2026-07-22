# Qwen80B TP2 5 req/s clean-baseline versus PD Flip run

This run is a four-node, one-instance-per-node experiment. It deliberately does
not use single-node multi-instance placement.

## Frozen design

- Model: `Qwen3-Next-80B-A3B-Instruct`.
- Four physical nodes, GPUs `0,1` on each node, TP=2, DP=1.
- Baseline: clean upstream image, static `1P3D`.
- Candidate: custom image, initial `1P3D`, final `2P2D`.
- 40 alternating long/short requests with unique prompts and natural output.
- Every request uses `max_tokens=10000` and `ignore_eos=true`.
- Continuous arrival interval `0.2s` (5 req/s); last arrival at `7.8s`.
- Short TTFT SLO `0.25s`; long TTFT SLO `0.45s`; TPOT SLO `0.05s`.
- SLO window 10s, enter below 0.90, recover at 0.95, minimum 10 TTFT
  samples and 100 TPOT intervals.
- First migration ratio 0.5, observation period 2s, then the remainder.
- Trace SHA-256:
  `d82d0f7fc5b745f43a48d6d91451794887b4a3f2e5f049d6e7a30a38652c9508`.

The result is an end-to-end comparison between clean upstream and the custom
state-machine system. Because their images and code differ, it is not an
isolated estimate of state-machine overhead.

## Before the reservation window

Copy these templates to chmod-600 private files on the coordinator:

- `experiments/pd_upstream_qwen80b_baseline.env.example`
- `experiments/pd_flip_qwen80b_ab.env.example`
- `experiments/pd_flip_qwen80b_tp2_5rps_pair.env.example`

The mode env files must select the same model, GPU IDs, TP/DP, memory fraction,
RDMA device and GID. The pair env points to an existing private worker env that
contains `ADMIN_API_KEY`; the pair runner reads the key without printing it.

The selected GPUs must be free on all four nodes. A resource collision is a
hard preflight failure. Do not stop another owner's workload.

## Run at 15:00

On the coordinator:

```bash
cd /home/tiancij/sglang-pd-qwen80b
export PAIR_ID="$(date -u +%Y%m%dT%H%M%SZ)-qwen80b-tp2-5rps-r1"
export PAIR_ENV_FILE=/home/tiancij/qwen80b-tp2-5rps-pair.env
experiments/pd_flip_qwen80b_tp2_5rps_pair.sh preflight
experiments/pd_flip_qwen80b_tp2_5rps_pair.sh run
```

Use the same `PAIR_ID` for both commands. The `run` command repeats preflight,
runs and validates baseline first, then repeats resource preflight before
preparing or loading the state-machine group.

## Validity gates

The pair is accepted only when:

- both modes retain the exact frozen trace hash;
- baseline completes 40 unique requests with zero errors and a valid report;
- baseline teardown releases ports and GPUs before candidate preflight;
- candidate completes 40 requests with exactly 10,000 output tokens each;
- the observer records an SLO trigger;
- the controller succeeds with migration ratio 0.5 and observation time 2s;
- final topology is `2P2D`.

If no SLO trigger occurs, retain the candidate as a valid workload run but an
invalid state-transition experiment. Do not lower the threshold after seeing
candidate results and present the rerun as the same experiment.

The final pair summary is written under:

```text
/home/tiancij/pd-artifacts/<PAIR_ID>-pair/summary.json
```
