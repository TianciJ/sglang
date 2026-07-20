# Qwen80B 长 Prefill 暖机诊断报告

运行 ID：`20260720T023903Z-upstream-qwen80b-longwarm-r1`

## 结论

一次 6,403-token 长 Prompt 暖机确实消除了正式请求 `qwen80b-00`
上的首次长形状停顿，但没有消除整批首波延迟。原本由第一个长请求承担的
约 8 秒停顿，转移到了第一个未覆盖的 647-token 短 Prompt
`qwen80b-01`。因此现有证据支持“推理进程存在多个输入形状或执行路径的
首次运行成本”，不支持“只暖一次长 Prompt 就能完成全部暖机”。

日志没有打印 `compile`、`JIT`、`autotune` 或具体 kernel 名称，因此不能把
首次成本精确归因给 Triton 编译。Qwen3-Next 的 P 节点日志确认使用 Triton
GDN/linear-attention backend，但这只是相关上下文，不是因果证明。

## 实验有效性

- 干净 upstream SGLang v0.5.15 镜像、Qwen3-Next-80B-A3B-Instruct、
  四节点 GPU 0--3、TP4、静态 `1P3D`。
- 复用同一个冻结 40 请求 trace，SHA256
  `c5dbbf75c997dfc5d67a18251082f2f246d6c055eb4af5040fbe147f49f4ce5d`。
- 唯一有意变化：正式测量前复制 `qwen80b-00` 的请求体，将输出改为一个
  token，完成后清空四个 worker 的 KV cache。
- 暖机不进入 measured rows。四个 cache flush 均成功。
- 正式请求 40/40 完成，0 error，每条恰好 10,000 completion tokens，
  `finish_reason=length`，manifest 为 `valid`。
- teardown 后四节点本次 owned container 为 0，GPU 0--3 无 compute PID，
  worker/bootstrap 端口已释放。

这是一轮诊断实验。它可以验证停顿位置的迁移，但不足以给出具有统计意义的
整体性能结论。

## 暖机链路的直接日志证据

暖机客户端记录：

| 项目 | 结果 |
|---|---:|
| Prompt tokens | 6,403 |
| Completion tokens | 1 |
| Client TTFT | 1,013.61 ms |
| Client total duration | 1,013.72 ms |

同一个 `bootstrap_room=4129494293185510455` 的 P/D 日志为：

```text
P: input_len=6403, forward_duration=996.77ms, queue_duration=0.55ms
D: input_len=6403, transfer_duration=998.04ms
```

D 的 transfer duration 与 P forward 同步，说明 D 此时主要在等待 P 产生并
交付 KV，而不是单独发生约 1 秒的网络复制。

暖机日志窗口内没有 `compile`、`JIT`、`autotune`、CUDA/NCCL error、OOM
或 traceback 明文。Info 日志只能把时间定位在 P forward 内部，不能继续分解
到某个 kernel。

## 暖机后正式首波发生了什么

清 KV 后，第一个正式长请求已变快：

```text
qwen80b-00 / 6403 tokens
P forward_duration   = 212.96ms
D transfer_duration  = 213.97ms
Client TTFT           = 243.32ms
```

紧接着，第一个未覆盖的短形状出现停顿：

```text
qwen80b-01 / 647 tokens
P queue_duration      = 0.49ms
P forward_duration    = 7673.43ms
D transfer_duration   = 7588.16ms
Client TTFT           = 7600.43ms
```

`qwen80b-01` 完成 Prefill 时，P 日志已经显示 `#queue-req: 5`。因此后续首波
请求的高 TTFT 是这次 647-token forward 停顿产生的级联等待。稳定后，
647-token Prefill 约 94--102 ms，6,403-token Prefill 约 160--161 ms。

## 与三次冷跑对照

下表均为客户端首个非空输出事件的 TTFT，单位秒：

| 请求 | 冷 R1 | 冷 R2 | 冷 R3 | 长暖机 R1 |
|---|---:|---:|---:|---:|
| 00 long/6403 | 6.824 | 6.719 | 6.721 | **0.243** |
| 01 short/647 | 8.103 | 8.020 | 8.007 | **7.600** |
| 02 long/6458 | 7.710 | 7.632 | 7.617 | 7.209 |
| 03 short/647 | 7.211 | 7.133 | 7.118 | 6.707 |
| 04 long/6458 | 6.881 | 6.801 | 6.783 | 6.378 |
| 05 short/653 | 6.382 | 6.299 | 6.279 | 5.876 |
| 06 long/6458 | 6.064 | 5.986 | 5.967 | 5.559 |
| 07 short/647 | 5.560 | 5.484 | 5.470 | 5.057 |
| 08 long/6403 | 5.249 | 5.166 | 5.151 | 4.743 |
| 09 short/647 | 4.748 | 4.668 | 4.651 | 4.246 |

前十条 TTFT 均值从冷跑的 6.47/6.39/6.38 秒降至 5.36 秒。TTFT SLO
达成率从三次冷跑稳定的 75% 提升到 80%，但只有一次暖机轮次，不能把 5 个
百分点作为稳定性能收益。第 20--39 条均值仍为约 150 ms，说明变化集中在
进程启动后的首次形状覆盖阶段。

## 解释边界和下一步

已验证：

- 首次成本位于 P forward，不在客户端调度，也不是独立的 D 网络停顿。
- 暖过的 6,403-token 路径显著加速。
- 未暖的 647-token 路径随后承担约 7.67 秒首次成本，并阻塞后续请求。

较可能但尚未验证：

- Triton GDN/linear-attention、MoE、allocator/workspace 或其他按形状惰性
  初始化共同构成首次成本。

最小下一步是同一设计下在正式 trace 前依次暖 `6403`、`647`、`6458` 和
`653` 四个实际 Prompt-token 形状，全部完成后清 KV 再测。如果停顿继续转移，
应使用 Nsight Systems、PyTorch profiler 或更细的 forward-pass/NVTX 埋点，
而不是继续靠 info 日志猜具体 kernel。

## 证据位置

- 客户端暖机记录：`pd-flip-artifacts/20260720T023903Z-upstream-qwen80b-longwarm-r1/smoke/long-prefill-warmup.json`
- P 暖机窗口：`pd-flip-artifacts/20260720T023903Z-upstream-qwen80b-longwarm-r1/logs/warmup-node0.docker.log`
- 完整 P 日志：`pd-flip-artifacts/20260720T023903Z-upstream-qwen80b-longwarm-r1/logs/node0.docker.log`
- 请求级指标：`pd-flip-artifacts/20260720T023903Z-upstream-qwen80b-longwarm-r1/report/request_metrics.csv`
- 有效性摘要：`pd-flip-artifacts/20260720T023903Z-upstream-qwen80b-longwarm-r1/report/summary.json`
- 远端完整原始数据：`/root/tiancij-upstream-baseline-runs/20260720T023903Z-upstream-qwen80b-longwarm-r1`

本地日志和远端日志均已脱敏；run-owned secret env 文件已在 teardown 后删除并
重新生成 `INVENTORY.txt`。
