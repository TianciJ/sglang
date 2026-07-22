# Qwen80B 单节点四实例 PD Flip 运行记录

## 结论

2026-07-22 的可行性实验已证明：一台 8×H20 节点可以同时运行四个
`Qwen3-Next-80B-A3B-Instruct` 实例，每个实例使用 TP=2，并在四实例之间完成
真实的 `1P3D -> 2P2D` PD Flip。

这是一轮链路可行性实验，不是与四节点部署的匹配 A/B，也不能据此发布性能优劣结论。

## 成功运行

```text
RUN_ID: 20260722T094000Z-qwen80b-single-node-mi
节点: cloud-099 / 192.168.0.42
实例: mi0=GPU0,1; mi1=GPU2,3; mi2=GPU4,5; mi3=GPU6,7
初始拓扑: 1P3D
迁移: mi2 -> mi3，50% + 2 秒观察 + 剩余请求
最终拓扑: 2P2D
Trace SHA-256: d82d0f7fc5b745f43a48d6d91451794887b4a3f2e5f049d6e7a30a38652c9508
Artifact: /home/tiancij/pd-artifacts/20260722T094000Z-qwen80b-single-node-mi
```

接受门全部通过：40/40 请求完成、零错误、每请求 10,000 token、
`finish_reason=length`、controller `success=true`、`role_flip_complete`、最终 `2P2D`，
teardown 后本次容器为 0、8 张 GPU 均回到 4 MiB、driver 正常。

## 观测指标

指标均为客户端流式事件时间，不是 GPU kernel 时间。

| 指标 | 结果 |
|---|---:|
| TTFT 平均 | 263.39 ms |
| TTFT 中位数 | 277.05 ms |
| TTFT P95 | 467.29 ms |
| TTFT SLO 达成率 | 87.5% (35/40) |
| TPOT interval 平均 | 23.08 ms |
| TPOT interval P95 | 44.39 ms |
| TPOT interval SLO 达成率 | 99.596% (396711/398320) |
| 40 请求执行时间 | 237.70 s |

observer 的第一次风险判定关联 `qwen80b-00`；controller 在自己的最小样本和轮询边界
满足后，以 `qwen80b-09` 的触发快照开始执行。两者是不同采样边界，不应混为同一个时间点。

关键迁移阶段（单调时钟）：

| 阶段 | 相邻阶段耗时 |
|---|---:|
| router drain -> source admission paused | 67.37 ms |
| admission paused -> migration started | 161.03 ms |
| migration started -> target prepared | 36.93 ms |
| target prepared -> first KV progress | 336.22 ms |
| first KV progress -> first batch complete | 623.21 ms |
| first batch complete -> role commit | 6.674 s |
| router drain -> role commit 总计 | 7.899 s |

最后一段包含固定 2 秒观察、第二批迁移、等待 source idle 和角色提交，不能解释为纯 KV
传输时间。

## 推荐入口

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-qwen80b-single-node-mi" \
ENV_FILE=/absolute/path/to/private.env \
SGLANG_REPO=/home/tiancij/sglang-pd-qwen80b \
NODE_IP=192.168.0.42 \
MOONCAKE_HOST=fd03:4514:80:6241::1 \
TRACE_SOURCE=/absolute/path/to/frozen/trace.jsonl \
TRACE_MANIFEST_SOURCE=/absolute/path/to/frozen/manifest.json \
TRACE_SHA256=d82d0f7fc5b745f43a48d6d91451794887b4a3f2e5f049d6e7a30a38652c9508 \
bash experiments/pd_flip_qwen80b_single_node_multi_instance.sh run
```

先执行同样参数的 `preflight` 或 `prepare`。`prepare` 只生成并 source 校验四份实例 env，
不会加载模型。

## 已踩坑与固定修复

1. `EXTRA_SGLANG_ARGS` 和 `WORKER_URLS` 是含空格的 env 值，写 env 时必须 `%q` 转义；
   runner 会在模型加载前重新 source 四份 env 做断言。
2. sampler 不响应默认 SIGTERM；runner 对精确命名的本次 sampler 发送 SIGINT 并等待退出，
   不使用强杀。
3. worker 使用 `docker -e ADMIN_API_KEY` 继承导出的环境变量，不把密钥值放入 Docker argv。
4. SGLang 仍可能在 `server_args` 和 runtime-role status 中记录 admin key；收尾必须脱敏所有
   日志，私有 `env/` 不进入共享 artifact。成功运行的非 env 文件已扫描 166 个，密钥命中为 0。
5. `INVENTORY.txt` 含除私有 `env/` 外的 166 个文件 SHA-256；本轮 artifact 约 2.0 GiB，
   主要来自 50 ms migration sampler 的事件级快照。

## 扩展到四节点 16 实例

单节点资源模型已验证，四节点可按每节点四个 TP=2 实例扩展为 16 个角色。正式扩展前必须
实现通用实例表 runner，并重新定义多 source/multi-target 的切换策略；不能简单并行启动四个
互不协调的 controller。

本轮结束时 `cloud-100` 的物理 GPU4 仍被他人 Mooncake benchmark 容器占用，因此没有加载
完整 16 实例，也没有停止或复用该资源。资源释放后应先做 16-worker 只启动/健康/路由门，
再运行新的 40 请求 RUN_ID。不要把本轮单节点结果描述成已经完成 16 实例集群验证。
