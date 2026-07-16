# PD Flip Prefill Donor 全流程时延实验报告

运行 ID：`20260715T114406Z-prefill-donor-full-timeline-ignore-eos-trace40`

## 结论

本轮完整跑通了两批实体 KV 迁移和最终 D→P 角色切换：40/40 请求完成、0 错误、Controller 45 个动作全部成功，最终结果为 `committed`，node2 从 Decode 切换为 Prefill。

从打开第 10 个 Prefill SLO 样本门槛的请求首 token，到 Controller 完成角色切换并退出，共 **14,556.702 ms**。其中正式的 10 秒观察窗口实测 **10,005.432 ms**，是全流程最大阶段。

## 实验条件

- 模型：Qwen3-8B，TP=1，page size=64。
- 初始拓扑：node0=P，node1/node2/node3=D。
- 迁移：node2 → node3；node0 是 Prefill donor。
- 首次迁移比例：50%。
- SLO 阈值：99%；Prefill/Decode 最小样本数均为 10。
- 观察窗口：10 秒；Controller 轮询周期：250 ms；worker 状态采样周期：50 ms。
- 为确保 10 秒观察结束后源 D 仍有真实请求可迁移，本轮 workload 设置 `ignore_eos=true`、`max_tokens=4096`。状态机参数和 KV 链路没有改变，但该设置会提高 Decode 负载，因此本轮用于链路分段测量，不应直接当作生产 workload 延迟。

## SLO 触发点

- 触发请求：`trace-0012`，short prompt。
- 首 token：2026-07-15 19:44:37.819399（Asia/Shanghai）。
- TTFT：44.440 ms，阈值 30 ms，未达标。
- 该请求成为第 10 个 Prefill 样本并打开最小样本门槛。
- Controller 决策：2026-07-15 19:44:38.004796。
- 请求首 token → Controller 决策：**185.396 ms**。
- 触发时 Prefill：4/10 达标（40%）；Decode：4554/4554 达标（100%）。

## 全流程关键路径

| 阶段 | 时长 |
|---|---:|
| 第 10 个 SLO 样本首 token → Controller 决策 | 185.396 ms |
| Controller 决策 → 第一批 Base 开始 | 39.399 ms |
| 第一批 Base 接收窗口 | 29.037 ms |
| 第一批 Base ready → Delta 开始 | 1,059.606 ms |
| 第一批 Delta 传输窗口 | 122.774 ms |
| 第一批 Delta 完成 → Commit ready | 405.488 ms |
| 第一批 Commit ready → 源 D finish | 6.008 ms |
| 第一批源 D finish → 目标 D activate | 11.286 ms |
| 正式 SLO 观察窗口 | 10,005.432 ms |
| 观察决策 → 第二批 Base 开始 | 166.053 ms |
| 第二批 Base 接收窗口 | 29.128 ms |
| 第二批 Base ready → Delta 开始 | 1,084.182 ms |
| 第二批 Delta 传输窗口 | 45.755 ms |
| 第二批 Delta 完成 → Commit ready | 493.426 ms |
| 第二批 Commit ready → 源 D finish | 7.562 ms |
| 第二批源 D finish → 目标 D activate | 14.335 ms |
| 第二批 activate → Controller 完成退出 | 851.833 ms |

观察期结束时，Fresh SLO window 为 Prefill 5/30（16.67%）、Decode 81837/81846（99.989%），因此状态机判定 `prefill_risk_persisted`，进入第二批迁移。

## 两批 KV 数据路径细分

P donor 与源 D Base 并行，下面的 worker `exact_process` 时间不能相加；目标 D 的 Base 接收窗口是另外一个端到端窗口。

| 数据操作 | 第一批 | 第二批 |
|---|---:|---:|
| P：L3 → L1 restore，最慢请求 | 0.477 ms | 0.443 ms |
| P → 目标 D：完整页传输，最慢请求 | 27.203 ms | 28.668 ms |
| 源 D → 目标 D：Base/边界页传输，最慢请求 | 86.487 ms | 94.042 ms |
| 目标 D：Base receive window | 29.037 ms | 29.128 ms |
| 源 D → 目标 D：Delta，最慢请求 | 122.359 ms | 45.755 ms |

第一批迁移 2 个请求：`trace-0001` 和 `trace-0003`。第二批迁移 1 个请求：`trace-0011`。所有请求均记录 `target_prefix_match_skipped=true`，即目标 D 没有自行做前缀匹配；完整 prompt 页来自 Prefill donor，边界页和 Decode KV 来自源 D。

## 角色切换尾部

第二批目标 D 激活后，Controller 的可见动作耗时如下：

| 动作 | 时长 |
|---|---:|
| `post_migration_idle_assertion` | 502.619 ms |
| `set_source_runtime_role` | 1.375 ms |
| `wait_source_prefill_loop` | 1.177 ms |
| `refresh_router_source_role` | 0.351 ms |
| `resume_source_admission` | 1.267 ms |
| `router_undrain_source` | 0.291 ms |

这些 ActionRecord 合计 507.080 ms。第二批 activate 到 Controller 容器退出的墙钟窗口为 851.833 ms，剩余约 344.753 ms 包括 HTTP 返回之后的状态传播、结果序列化和容器退出，当前日志没有继续拆分。

## 复测过程与数据选择

本次共保留五轮数据，避免只保留成功样本：

1. `20260715T113112Z-prefill-donor-full-timeline-trace40`：40/40 成功，但观察后源 D 已空，第二批为空。
2. `20260715T113653Z-prefill-donor-full-timeline-ratio25-trace40`：命令行前置变量被环境文件覆盖，实际仍是默认参数；第二批为空。
3. `20260715T113907Z-prefill-donor-full-timeline-min20-trace40`：同样被环境文件覆盖；第二批为空。
4. `20260715T114147Z-prefill-donor-full-timeline-complete-trace40`：有效使用 25%/30 样本，但观察期新样本不足，Controller 按设计保留源 D 为 Decode，没有最终 commit。
5. `20260715T114406Z-prefill-donor-full-timeline-ignore-eos-trace40`：恢复正式 50%/10 样本/10 秒参数，仅延长 Decode 请求，完整跑通两批迁移和角色切换；本报告采用该轮。

## 数据与可复算性

- 全流程逐段数据：`analysis/full_timeline.csv`。
- 两批关键阶段：`analysis/batch_stage_timings.csv`。
- 每请求 KV 页和传输数据：`analysis/request_stage_timings.csv`。
- SLO 触发请求：`analysis/slo_trigger.json`、`analysis/slo_trigger_contributors.csv`。
- 原始 worker/controller/measurement 数据保存在完整归档中；对外共享应使用 redacted 目录或 redacted tar.gz。

时钟同步偏差小于 0.4 ms。KV 阶段优先使用 worker 内部 epoch 和 `exact_process`；SLO 触发使用 request ledger 与 Controller snapshot；角色尾部使用 Controller ActionRecord 和 runner timeline。50 ms 轮询样本仅作状态佐证，没有代替 worker 精确时间点。
