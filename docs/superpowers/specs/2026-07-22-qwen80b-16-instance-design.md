# Qwen80B 四节点 16 实例 PD Flip 可行性实验设计

## 问题与拓扑

验证四台 8×H20 节点能否各运行四个 Qwen3-Next-80B-A3B-Instruct TP=2
实例，共 16 个独立 worker，并完成真实 `1P15D -> 2P14D` PD Flip。

- 每节点实例 GPU 对：`0,1`、`2,3`、`4,5`、`6,7`。
- 初始唯一 P：`h0i0`；其余 15 个实例为 D。
- 迁移源：`h2i2`；迁移目标：`h3i3`。
- 策略：迁移 50%，观察 2 秒，再迁移剩余请求，源 D 切换为 P。
- 负载：冻结的 40 请求、5 req/s trace，每请求 10,000 token。
- 16 个候选角色均做长、短 Prefill 暖机，共 32 个非测量请求；清空 KV 并恢复
  `1P15D` 后才开始正式负载。

本轮只验证 16 实例链路，不与之前四实例或四节点四实例数据做性能 A/B。

## 接受门

1. 四节点模型、镜像、代码、RDMA、时钟、端口和全部 32 张 GPU 通过 preflight。
2. 16 个 worker 的 GPU、worker 端口、bootstrap 端口和容器名互不冲突。
3. Router 初始严格为 `1P15D`，暖机后仍严格为 `1P15D` 且所有 FSM safe。
4. 40/40 请求完成、零错误、每请求 10,000 token、finish reason 为 length。
5. Controller 记录 SLO 触发、两阶段迁移、2 秒观察、role_flip_complete。
6. Router 与 worker 最终严格为 `2P14D`。
7. 精确 teardown 后 16 个本次 worker、router、helper 均退出，32 张 GPU 无本次进程，
   driver 正常；非私有 artifact 密钥扫描为 0。

## 自审批

设计批准，但模型加载必须等待所有 32 张 GPU 均无他人计算进程。安全策略禁止停止、
复用或干扰非本次 RUN_ID 的容器。资源门未通过时，只允许完成代码、测试和无模型检查。
