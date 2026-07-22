# Qwen80B TP=2、5 req/s PD Flip 成功实验流程

本文档是四节点 `Qwen3-Next-80B-A3B-Instruct` 性能实验的成功复现基线。以后执行同类实验，应从本文档指定的 runner、配置模板和有效性门开始，不要重新手写 Docker、router、controller 或清理命令。

可执行脚本和实际 manifest 的优先级高于本文档。如果脚本参数发生变化，应先更新测试和本文档，再运行集群实验。

## 1. 实验回答的问题

比较以下两个端到端系统：

- Baseline：干净的 upstream SGLang，固定 `1P3D`，不启用状态机。
- Candidate：我们的 SGLang，初始 `1P3D`；SLO 触发后先迁移源 D 上 50% 的请求，观察 2 秒，再迁移剩余请求，最终切换为 `2P2D`。

这是“干净 upstream 系统”和“我们的状态机系统”的端到端对比。两组镜像和代码不同，因此结果不能解释为纯粹的状态机函数开销。每组只跑一轮时，只能作为链路验证和初步性能结果，不能作为统计显著的最终结论。

## 2. 已验证成功的固定配置

| 项目 | 配置 |
|---|---|
| 节点 | `cloud-099`、`cloud-100`、`cloud-101`、`cloud-102` |
| 业务 IPv4 | `192.168.0.42`、`192.168.0.40`、`192.168.0.39`、`192.168.0.41` |
| 每节点 GPU | `0,1` |
| TP / DP | `TP=2`、`DP=1` |
| 模型 | `Qwen3-Next-80B-A3B-Instruct` |
| 模型目录 | `/models/Qwen3-Next-80B-A3B-Instruct` |
| 初始拓扑 | node0=P，node1=D，node2=D，node3=D |
| 切换关系 | node2 为源 D，node3 为迁移目标；完成后 node2 变为 P |
| 正式请求数 | 40 |
| 到达间隔 | 连续每 `0.2s` 一个请求，即 `5 req/s` |
| Prompt | 长短交错、每个请求前缀不同，降低前缀复用 |
| 输出 | 自然输出，`max_tokens=10000`、`ignore_eos=true` |
| TTFT SLO | 长请求 `0.45s`，短请求 `0.25s` |
| TPOT SLO | `0.05s` |
| SLO 窗口 | 10 秒 |
| 进入 / 恢复阈值 | `0.90` / `0.95` |
| 最少证据 | 10 个 TTFT 样本、100 个 TPOT interval |
| 第一次迁移 | 50% |
| 观察期 | 2 秒 |
| RDMA | `mlx5_bond_1`，GID index `3`，IPv6 Mooncake 地址 |
| HiCache stitching | 关闭 |
| Prefill donor | 关闭 |

本次成功 trace 已冻结：

```text
pd-flip-artifacts/qwen80b-trace40-5rps-slo025-045/trace.jsonl
SHA-256: d82d0f7fc5b745f43a48d6d91451794887b4a3f2e5f049d6e7a30a38652c9508
```

除非实验问题明确要求改变 workload，否则 baseline 和 candidate 必须复用同一个序列化 trace 和同一个 SHA-256。

## 3. 唯一推荐入口

使用以下两个脚本：

```text
experiments/pd_flip_qwen80b_tp2_5rps_pair.sh
experiments/pd_flip_qwen80b_ab.sh
```

完整 baseline + candidate 配对实验只调用 pair runner：

```bash
PAIR_ENV_FILE=/absolute/path/to/private-pair.env \
PAIR_ID="$(date -u +%Y%m%dT%H%M%SZ)-qwen80b-tp2-5rps" \
bash experiments/pd_flip_qwen80b_tp2_5rps_pair.sh run
```

不要手工重写四个 worker、router、observer 或 controller 的启动命令。不要用 `docker restart`。

## 4. 一次性准备三个私有配置文件

在协调节点复制模板：

```bash
cp experiments/pd_upstream_qwen80b_baseline.env.example /home/tiancij/qwen80b-upstream-tp2-private.env
cp experiments/pd_flip_qwen80b_ab.env.example /home/tiancij/qwen80b-state-tp2-private.env
cp experiments/pd_flip_qwen80b_tp2_5rps_pair.env.example /home/tiancij/qwen80b-pair-tp2-private.env
chmod 600 /home/tiancij/qwen80b-*-tp2-private.env
```

管理密钥只能放在独立的 `chmod 600` 私有文件或环境变量中。pair 配置通过 `ADMIN_API_KEY_SOURCE_ENV` 指向该文件。不要把密钥提交到 Git、复制到共享 artifact，或在终端中打印出来。

### 4.1 两组必须相同的字段

pair runner 会拒绝以下字段不一致的实验：

- 模型路径、模型 ID 和模型指纹；
- GPU IDs、TP、DP、显存比例；
- worker、router、bootstrap 端口；
- IPv4 节点地址、Mooncake IPv6 地址；
- IB device、GID index、IPv6 开关；
- trace、输出参数和 SLO 参数。

upstream 使用 `WORKER_PORT`，candidate 使用 `PORT`，两者的值必须相同。

### 4.2 本次成功使用的关键值

```bash
GPU_IDS=0,1
TP_SIZE=2
DP_SIZE=1
MEM_FRACTION_STATIC=0.80
ROUTER_PORT=8000
IB_DEVICE=mlx5_bond_1
MC_GID_INDEX=3
MC_USE_IPV6=1
```

本次成功重跑使用 `BOOTSTRAP_PORT=18998`，原因是 node2 上的 `8998` 已被其他人的容器占用。以后不要无条件照抄 `18998`：应先检查四节点端口，选择四节点共同空闲的端口，并在 upstream、candidate 两份配置中使用同一个值。

Mooncake IPv6 地址应与所选 `mlx5_bond_1/GID 3` 一致：

```text
node0 fd03:4514:80:6241::1
node1 fd03:4514:80:7b81::1
node2 fd03:4514:80:6601::1
node3 fd03:4514:80:5f01::1
```

## 5. 开始前的人工只读检查

runner 会再次执行完整 preflight，但操作员开始前仍应确认资源归属：

```bash
for host in cloud-099 cloud-100 cloud-101 cloud-102; do
  ssh "$host" "hostname; date; docker ps --format '{{.Names}} {{.Status}}'; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | head -n 2; ss -ltn | grep -E ':(30000|8000|18998|8998) ' || true; show_gids | grep '^mlx5_bond_1' || true"
done
```

检查要点：

1. 四节点 SSH 都可用，时钟同步正常。
2. GPU 0、1 没有计算进程。
3. worker、router、bootstrap 端口没有被占用。
4. `/models/Qwen3-Next-80B-A3B-Instruct` 完整且四节点一致。
5. 两个镜像都存在，candidate 四节点镜像 ID 和代码哈希一致。
6. `nvidia-smi -L` 正常，没有 driver/Xid 错误。
7. `mlx5_bond_1` 和 GID 3 存在，地址与配置一致。
8. 现有容器、GPU、端口和进程的所有者已经确认。

不要停止名称不包含本次 `RUN_ID` 的容器。端口被别人占用时，换四节点共同空闲端口，不要停止对方实验。

## 6. 先运行不加载模型的验证

```bash
PAIR_ENV_FILE=/home/tiancij/qwen80b-pair-tp2-private.env \
PAIR_ID=local-qwen80b-tp2-check \
bash experiments/pd_flip_qwen80b_tp2_5rps_pair.sh validate
```

然后运行四节点 read-only preflight：

```bash
PAIR_ENV_FILE=/home/tiancij/qwen80b-pair-tp2-private.env \
PAIR_ID="$(date -u +%Y%m%dT%H%M%SZ)-qwen80b-tp2-preflight" \
bash experiments/pd_flip_qwen80b_tp2_5rps_pair.sh preflight
```

任何一项失败都不应加载模型。先保存输出并修复配置，不要连续重试模型加载。

## 7. 正式执行完整配对实验

建议记录唯一的 `PAIR_ID`：

```bash
export PAIR_ID="$(date -u +%Y%m%dT%H%M%SZ)-qwen80b-tp2-5rps"
export PAIR_ENV_FILE=/home/tiancij/qwen80b-pair-tp2-private.env

bash experiments/pd_flip_qwen80b_tp2_5rps_pair.sh run
```

如果需要脱离终端运行：

```bash
mkdir -p "/home/tiancij/pd-artifacts/${PAIR_ID}-launcher"
nohup env PAIR_ID="$PAIR_ID" PAIR_ENV_FILE="$PAIR_ENV_FILE" \
  bash experiments/pd_flip_qwen80b_tp2_5rps_pair.sh run \
  > "/home/tiancij/pd-artifacts/${PAIR_ID}-launcher/launcher.log" 2>&1 < /dev/null &
echo $! > "/home/tiancij/pd-artifacts/${PAIR_ID}-launcher/launcher.pid"
```

完整顺序固定为：

1. 验证 trace 和两组配置。
2. 对四节点执行 read-only preflight。
3. 先运行干净 upstream baseline。
4. 四个 baseline worker 并发加载；全部健康后才启动 upstream 镜像自带 router。
5. 发送同一份 40 请求 trace，验证 40/40、每个 10,000 token、`finish_reason=length`、0 错误。
6. 保存 baseline 原始证据，优雅停止精确容器名，确认端口、GPU、driver 正常。
7. 重新 preflight，再加载 candidate；baseline 和 candidate 不能重叠占用 GPU。
8. 四个 candidate worker 全部通过健康和初始角色门后启动 router。
9. 对 node0、node1、node2、node3 分别执行长、短 Prefill 暖机，共 8 个请求。
10. 暖机完成后清空四个 worker 的 KV cache，恢复并验证严格 `1P3D`；在此之前不启动正式测量。
11. 启动 observer、controller 和 50 ms migration sampler。
12. 发送正式 40 请求 trace。
13. controller 按“50% 迁移 → 2 秒观察 → 剩余迁移 → 角色切换”执行。
14. 验证 40/40、0 错误、最终 `2P2D`，再生成内部阶段和迁移汇总。
15. 归集日志并优雅停止本次精确容器名。
16. 运行配对 provenance 门并生成 pair summary。

模型和镜像已存在时，下载量为 0。成功记录中，单组 candidate 从启动到完成约 7 分钟，其中正式 40 请求约 179 秒；实际时间受共享存储、模型加载和集群负载影响。

## 8. 只补跑 candidate

只有在 baseline 已经完整有效、trace SHA 相同且配置仍一致时，才能复用 baseline：

```bash
export PAIR_ID="$(date -u +%Y%m%dT%H%M%SZ)-qwen80b-tp2-candidate"
export BASELINE_RUN_ID_OVERRIDE=<已验证的-baseline-run-id>
export PAIR_ENV_FILE=/home/tiancij/qwen80b-pair-tp2-private.env

bash experiments/pd_flip_qwen80b_tp2_5rps_pair.sh run-state-only
```

`run-state-only` 会先重新验证 baseline manifest 和 summary。不能用部分完成、请求数不足、trace 不同或 teardown 未确认的 baseline。

## 9. 运行时如何判断阶段

只查看本次精确 `RUN_ID`，不要使用模糊进程清理命令。

### 9.1 worker 是否正在加载

```bash
docker ps --filter name="$PAIR_ID" --format '{{.Names}} {{.Status}}'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | head -n 2
```

Qwen80B TP=2 的权重加载完成后，每张 GPU 的模型和运行时显存会明显增加。端口尚未开放但日志仍在正常加载时，不要重启容器。

### 9.2 暖机是否完成

```bash
test -s "/home/tiancij/pd-artifacts/<STATE_RUN_ID>/state_machine/warmup/summary.json" \
  && echo warmup-complete
tail -n 20 "/home/tiancij/pd-artifacts/<STATE_RUN_ID>/state_machine/warmup/warmup_events.jsonl"
```

有效暖机必须满足：

- `warmup_request_count=8`；
- 四个候选 P 都有 long/short 两个请求；
- 四节点 cache flush 成功；
- 所有队列为空；
- 角色恢复为 `1P3D`；
- 正式 ledger 在暖机门之后才开始写入。

### 9.3 正式请求进度

```bash
tail -n 1 "/home/tiancij/pd-artifacts/<STATE_RUN_ID>/state_machine/raw/slo_ledger.jsonl"
wc -l "/home/tiancij/pd-artifacts/<STATE_RUN_ID>/state_machine/raw/request_metrics.jsonl"
```

`request_metrics.jsonl` 可能在 replay 结束时一次性完成写入，因此它暂时不存在或为 0 行不代表死循环。实时 token 进度应查看 `slo_ledger.jsonl`。

### 9.4 状态机是否完成

最终必须同时满足：

- controller `success=true`；
- state trace 最后事件为 `reason=role_flip_complete`；
- `controller/final_router.json` 显示 `2P2D`；
- 第一次迁移比例为 `0.5`；
- 观察期为 `2.0s`。

## 10. 正式有效性门

实验结束后执行：

```bash
PAIR_ENV_FILE=/home/tiancij/qwen80b-pair-tp2-private.env \
PAIR_ID=<本次-pair-id> \
bash experiments/pd_flip_qwen80b_tp2_5rps_pair.sh report
```

成功时命令退出码为 0。只有以下条件全部满足，才能引用性能指标：

- baseline 和 candidate 各 40 个唯一请求；
- HTTP、客户端和 server 侧均无请求错误；
- 每个请求完成 10,000 token，`finish_reason=length`；
- trace SHA、模型、tokenizer、GPU、TP/DP、端口和网络 provenance 通过；
- observer 原始快照存在；
- controller 完成并得到最终 `2P2D`；
- token 级 ledger、request metrics、responses、errors 和 worker/router 日志存在；
- 四节点 teardown 完成，端口释放，GPU/driver 正常。

若旧 candidate manifest 使用过不含 `tokenizer.json` 的历史模型指纹公式，report 会保留原始字段，并额外写出：

```text
<PAIR_DIR>/model_fingerprint_reconciliation.json
```

只有历史公式、新统一公式和四节点实时文件同时一致时才允许通过。不要手工修改原 manifest 来“消除”不匹配。

## 11. Artifact 位置

pair runner 使用三个独立目录：

```text
${ARTIFACT_ROOT}/${BASELINE_RUN_ID}/
${ARTIFACT_ROOT}/${STATE_RUN_ID}/
${ARTIFACT_ROOT}/${PAIR_ID}-pair/
```

核心证据：

```text
trace/trace.jsonl
trace/manifest.json
manifest.json
raw/slo_ledger.jsonl
raw/request_metrics.jsonl
raw/*/responses.jsonl
raw/*/errors.jsonl
observer/snapshots.jsonl
observer/summary.json
controller/result.json
controller/final_router.json
raw/migration_events.jsonl
metrics/request_stage_events.jsonl
metrics/req_time_stats.jsonl
warmup/warmup_events.jsonl
warmup/summary.json
```

`TTFT` 和 `TPOT` 是客户端观察到的流式事件时间，不是 GPU kernel 时间。内部 Prefill、Decode、迁移阶段必须使用规范化的 server stage events，并在报告中注明时钟和测量边界。

## 12. 已解决且不能回退的坑

以下修复是这次成功流程的一部分：

1. 干净 baseline 直接使用 upstream 镜像内置 `sglang-router`，不再现场下载或编译 Rust router。
2. 从多行 worker env 中只读取实际 admin key，避免错误拼接导致 401。
3. candidate worker 同时传递 `MOONCAKE_LOCAL_HOSTNAME` 和同地址的 `SGLANG_HOST_IP`。当 `MC_USE_IPV6=1` 时，如果 bootstrap 仍公布 IPv4，会导致 `SocketHandShakePlugin` 地址解析失败并使 P→D KV 传输失败。
4. controller 最终拓扑写回使用兼容远端 shell 的引号形式，避免测量完成后因 `NameError: prefill` 导致收尾失败。
5. baseline 和 candidate 使用统一的模型指纹：`config.json`、`tokenizer.json` 和排序后的 safetensors 名称/大小。
6. 报告脚本兼容协调节点 Python 3.6，不使用 `statistics.fmean`。
7. baseline 有效后允许通过 `run-state-only` 使用新 RUN_ID 补跑 candidate，但必须先通过 baseline validity 和 provenance 检查。

推荐从提交 `851ac734d` 或包含等价/更新修复的版本开始。代码更新后必须重新运行 runner 单元测试和四节点代码哈希检查。

## 13. 失败处理

出现以下任一情况，应停止调度新请求、保留证据，并让 runner 只清理本次精确资源：

- 节点 SSH 丢失；
- driver/Xid 错误；
- 模型 shard 缺失；
- health gate 超时；
- measured run 出现 HTTP 503；
- Mooncake handshake 或 KV transfer 失败；
- controller 未完成；
- 请求数、token 数或输出完整性不满足；
- teardown 无法确认。

失败目录必须保留，不能覆盖。修复后使用新的 `RUN_ID` 重跑。不要执行：

```text
docker restart
pkill
killall
kill -9
docker rm -f
按名称子串批量停止容器
```

如果控制终端中断，先读取本次 launcher PID 和精确容器名。只对名称完整包含本次 `RUN_ID` 的容器执行长超时优雅停止：

```bash
docker stop --time 1800 <exact-run-owned-container-name>
```

随后确认本次端口释放、GPU 0/1 无计算 PID、`nvidia-smi -L` 正常，再开始新一轮。

## 14. 2026-07-22 成功记录

成功 pair 使用：

```text
Baseline RUN_ID: 20260722T161500-qwen80b-tp2-5rps-upstream
Candidate RUN_ID: 20260722T165500-qwen80b-tp2-5rps-candidate-final-state
Pair directory: 20260722T165500-qwen80b-tp2-5rps-candidate-final-pair
```

结果：

| 指标 | Baseline | State machine |
|---|---:|---:|
| 完成请求 | 40/40 | 40/40 |
| 错误 | 0 | 0 |
| TTFT 平均 | 284.14 ms | 238.93 ms |
| TTFT P95 | 484.73 ms | 353.17 ms |
| TTFT 达成率 | 90% | 92.5% |
| TPOT 平均 | 21.45 ms | 16.98 ms |
| TPOT P95 | 21.71 ms | 17.27 ms |
| TPOT 达成率 | 100% | 100% |
| 40 请求执行时间 | 224.82 s | 179.48 s |

Candidate 的首次 SLO 触发请求为 `qwen80b-02`，最终达到 `2P2D`。以上是一组端到端初步结果；需要多轮匹配重复实验后，才能报告稳定的性能差异和方差。
