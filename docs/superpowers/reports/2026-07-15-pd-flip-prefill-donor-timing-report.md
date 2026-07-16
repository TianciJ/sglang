# PD Flip Prefill Donor 全链路计时实验

日期：2026-07-15（Asia/Shanghai）
运行 ID：`20260715T102321Z-prefill-donor-page64-timing-trace40`
代码版本：`180aceb826b8450f0068419c027f18f226155c18`
模型：Qwen3-8B，TP=1，`page_size=64`

## 技术结论

本轮实验成功跑通两批 Prefill-donor PD Flip：40/40 workload 请求完成、0 请求错误、3 个运行中请求被迁移、目标 D 全部跳过本地前缀匹配，且没有 fallback 或失败的 controller action。node2 最终从 Decode 切换为 Prefill。

第一个迁移批次从 controller 判定 SLO 风险到目标 D 激活请求共 **1,650.215 ms**。其中，实际 P/D KV 传输是几十毫秒量级；关键路径主要消耗在两个控制等待段：

- base 已到齐到 delta 开始：**986.282 ms**；
- delta 全部发完到 target commit ready：**446.387 ms**。

因此，本轮链路的首要延迟优化对象不是 P 或 D 的 page copy，而是 quiesce/delta 和 commit 前后的轮询、状态传播与编排等待。

SLO 切换由 `trace-0012` 打开最小样本门槛。该请求在 **18:24:07.810342** 产出首 token，TTFT 为 **113.441 ms**，超过 30 ms SLO；它是第 10 个 TTFT 样本。controller 在下一次轮询 **18:24:08.077589** 检测到 `0/10` Prefill TTFT 达标、`4362/4362` Decode interval 达标，随后进入 `prefill_risky_decode_healthy` 切换流程，检测滞后 **267.247 ms**。

## 两批迁移的关键路径

下表中的 P restore、P transfer、源 D base/delta 是 worker 内部 `exact_process` 时长。target base receive 和各控制间隔由 worker epoch 时间点计算。P 与 D 的 base 传输并行，不能把两者简单相加。

| 阶段 | 第一批：2 请求 | 最终批：1 请求 | 含义 |
| --- | ---: | ---: | --- |
| 触发判定 → base 开始 | 38.227 ms | — | controller 选择、drain、source start、target prepare |
| P HiCache lookup/restore，最慢请求 | 2.679 ms | 0.453 ms | P 端获得完整 `[0,B)` 可用映射 |
| P `[0,B)` 传输，最慢请求 | 29.947 ms | 24.134 ms | 原始 P → 目标 D |
| 源 D `[B,C0)` 传输，最慢请求 | 71.238 ms | 56.401 ms | 边界页与 Decode base |
| target base 接收窗口 | 90.294 ms | 163.872 ms | 最早 target receiver 开始到全批 held |
| base ready → delta start | 986.282 ms | 996.228 ms | quiesce 请求与 controller 轮询等待 |
| 源 D delta 传输，最慢请求 | 71.494 ms | 62.740 ms | 页对齐 delta 到 `C1` |
| delta complete → commit ready | 446.387 ms | 469.120 ms | source/target 状态轮询与 commit 协调 |
| commit ready → source finish | 6.392 ms | 6.599 ms | 源 D 释放所有权 |
| source finish → target activate | 10.720 ms | 15.004 ms | 目标请求进入调度队列 |
| base 开始 → activate | 1,611.988 ms | 1,713.563 ms | 单批迁移主链路 |
| SLO 判定 → 第一批 activate | **1,650.215 ms** | — | 从策略触发到请求接管 |

第一批激活发生在 **18:24:09.727804**。最终批 base 在 **18:24:19.854021** 开始、在 **18:24:21.567584** 激活。之后经过约 0.507 秒的 source idle 检查，node2 的 Prefill 角色在采样器中于 **18:24:22.117482** 首次确认。

## 逐请求 KV 范围与时间

| 批次 | Trace 请求 | Prompt `P` | 边界 `B` | `C0→C1` | P pages / bytes | P restore / transfer | D base pages / transfer | Delta bytes / transfer |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 第一批 | `trace-0002` | 1,974 | 1,920 | 2,174→2,271 | 30 / 283,115,520 | 2.679 / 29.947 ms | 4 / 71.238 ms | 28,311,552 / 19.840 ms |
| 第一批 | `trace-0009` | 189 | 128 | 341→438 | 2 / 18,874,368 | 0.520 / 29.820 ms | 4 / 70.662 ms | 18,874,368 / 71.494 ms |
| 最终批 | `trace-0006` | 1,974 | 1,920 | 3,927→4,042 | 30 / 283,115,520 | 0.453 / 24.134 ms | 32 / 56.401 ms | 28,311,552 / 62.740 ms |

三个请求的 P restore hit 都恰好覆盖 `B`；目标 D 三次都记录 `target_prefix_match_skipped=true` 和 provenance `prefill_donor_and_source_decode`。

本轮只能确认 donor 的整体 HiCache lookup/restore 时间，不能从现有字段中继续拆成 L1、L2、L3 各自耗时。donor 的 `l1_hit_tokens/l2_hit_tokens/l3_hit_tokens` 未导出，因此报告不把这些 restore 时间解释为“纯 L3 拉取延迟”。如果下一轮要单测 L3，需要导出 donor tier breakdown，并在请求 Prefill 后主动排除或驱逐 P 的 L1/L2 命中。

## SLO 阈值在哪里被触发

配置为：

```text
SLO attainment threshold = 0.99
minimum Prefill TTFT samples = 10
minimum Decode TPOT samples = 10
TTFT SLO = 30 ms
```

本轮不是“第 10 个请求让达标率从高于 99% 跌到低于 99%”。实际情况是：从第一个样本开始，Prefill TTFT attainment 就是 0%；controller 一直等待最小样本数达到 10。

| 时刻（Asia/Shanghai） | Prefill 样本 | Decode intervals | 事件 |
| --- | ---: | ---: | --- |
| 18:24:07.793542 | 0/8 | 2919/2919 | 风险已存在，但 Prefill 样本不足 10，不切换 |
| 18:24:07.800358 | 第 9 个 TTFT miss | — | `trace-0006` 首 token |
| 18:24:07.810342 | 第 10 个 TTFT miss | — | `trace-0012` 首 token，TTFT 113.441 ms |
| 18:24:08.077589 | 0/10 | 4362/4362 | controller 轮询发现门槛满足并进入 selecting |
| 18:24:08.108188 | — | — | 采样器首次观察到 node2 admission paused |

触发样本 `trace-0012` 的 upstream request ID 为 `682230cbfd8948e3b3f1e90f679e8302`。它只触发策略门槛，并不是第一批被迁移的请求；第一批迁移的是源 node2 上的 `trace-0002` 和 `trace-0009`，最终批迁移 `trace-0006`。

## 实验范围和测量口径

- 初始角色：node0=P，node1=D，node2=D，node3=D。
- 迁移方向：node2 Decode → node3 Decode；迁移完成后 node2 D→P。
- P donor：node0，严格提供完整 Prompt pages `[0,B)`。
- 源 D：node2，提供 `[B,C0)` 和页对齐 delta。
- 目标 D：node3，预分配同一请求映射下的两组 receiver，不执行 target-local HiCache prefix match。
- 专用 Mooncake store：64 GB；实验前重启 metadata/master/store 和四个 worker，使角色、GPU cache 与专用 store 处于新进程状态。
- 状态采样：50 ms 配置间隔；controller 自身 SLO monitor 以 250 ms 间隔轮询。
- 时间统一：请求记录提供同机 wall/monotonic 对，40 个样本的 offset 跨度为 0.715 微秒；四节点 NTP 偏差均小于 0.3 ms。跨节点比较使用 worker 导出的 epoch 时间，不直接比较不同主机的 monotonic 值。

## 完整性和稳健性检查

- workload：40/40 completed，0 error；ledger 90,359 行、40 个 final record、40 个 request metric record。
- controller：45 个 action 全部成功；outcome `committed`；最终消息 `source switched to prefill`。
- 迁移请求：3 个；2 个批次；所有 target prefix match 均跳过。
- fallback：0 个 fallback action，所有请求的 `fallback_attempted=false`。
- 本轮观测窗口内没有 donor miss、invalid KV index、Mooncake Put failure 或 eviction 记录。
- TPOT：40/40 请求的平均与 p95 均满足 SLO；interval attainment 为 99.9502%（90,234/90,279）。
- TTFT：0/40 满足 30 ms SLO。该阈值明显偏激进，因此本轮证明的是控制链路、时间采集和切换行为，不证明 30 ms 是合理的生产阈值。

已有 summarizer 的 `request_migration_join_count=0` 是 ID 口径问题：workload 使用 `trace-xxxx`，worker migration 使用 upstream RID。本报告通过 `request_metrics.jsonl.upstream_request_id` 做了确定性映射，得到 3 个实际迁移请求。派生表保留两套 ID，便于复核。

## 原始数据和可复现派生表

可共享的脱敏 raw 目录：

```text
pd-flip-artifacts/20260715T102321Z-prefill-donor-page64-timing-trace40-redacted/
```

可共享归档：

```text
pd-flip-artifacts/20260715T102321Z-prefill-donor-page64-timing-trace40-redacted.tar.gz
SHA256 4a09f70d28f90caaa659d69f27a90f90ea047462a0c0f6d8b23c37d3b15a0d70
```

原始远端归档保留在 cloud-099：

```text
/home/tiancij/20260715T102321Z-prefill-donor-page64-timing-trace40.tar.gz
SHA256 f7d5757953b9e38461738dba13eef89cb9e372040d1aa311a4964456c23b50c8
```

原始 worker 日志包含启动时打印的 `admin_api_key`，因此不应直接共享。脱敏包只替换 4 个 worker 日志中的凭据文本，共 12 处；时间、请求、metrics 和状态记录没有改动，`REDACTION_MANIFEST.json` 的 residual count 为 0。

最重要的派生数据：

- `analysis/slo_trigger.json`：阈值、前一轮询、触发轮询、第 10 个请求和检测滞后；
- `analysis/slo_trigger_contributors.csv`：前 10 个 TTFT 样本；
- `analysis/request_stage_timings.csv`：3 个迁移请求逐阶段精确时间；
- `analysis/batch_stage_timings.csv`：两批迁移关键路径；
- `analysis/analysis_manifest.json`：输入文件大小、SHA256 和完整性结论；
- `report/controller_actions.csv`：45 个 controller API action 的调用耗时；
- `metrics/events.jsonl`：16,129 个 50 ms 级原始采样事件；
- `workload/trace_slo_ledger.jsonl`：90,359 个请求级 SLO 状态事件。

派生脚本为 `docs/superpowers/reports/scripts/analyze_pd_flip_prefill_donor_timing.py`；脱敏脚本为 `docs/superpowers/reports/scripts/redact_pd_flip_artifacts.py`。

## 建议的下一轮实验

1. 将 source delta 的 quiesce/status 等待从约 1 秒轮询改成更短间隔或事件通知，再复测 `base ready → delta start`。
2. 将 delta completion 到 commit 的状态传播从约 0.45 秒轮询改成事件或更短间隔。
3. 在 P donor measurement 中导出 L1/L2/L3 token breakdown、每层 restore start/end 和实际 L3 bytes。
4. 做一轮强制 L1/L2 miss、只允许 L3 restore 的实验，单独回答“L3 获取本身需要多久”。
5. 用贴近生产的 TTFT SLO 重跑；当前 30 ms 使 40/40 TTFT miss，只适合稳定触发状态机，不适合评价服务质量。

## 仍需回答的问题

- target final batch base receiver 为什么是 163.872 ms，而 P/D sender 分别只有 24.134/56.401 ms；需要进一步拆 receiver bootstrap、metadata 和 completion acknowledgment。
- 1 秒级 quiesce 等待有多少来自固定 polling interval，有多少来自请求到安全边界的真实等待。
- 在强制 L3 restore 后，P donor restore 与 P→target page transfer 是否仍能与源 D base transfer充分重叠。

本报告未生成定量趋势图：只有 2 个迁移批次和 3 个迁移请求，画折线或分布图会产生虚假的趋势感；逐阶段精确表更适合本次审计口径。
