# Qwen80B 单节点多实例 PD Flip 可行性实验设计

## 实验问题

验证一台 8-GPU 节点能否同时运行四个互相隔离的 Qwen3-Next-80B-A3B-Instruct
实例（每实例 TP=2），并在同一节点内完成真实的 `1P3D -> 2P2D` PD Flip。

本轮是链路可行性实验，不与既有四节点结果比较，也不发布性能优劣结论。成功后，
同一实例描述表可扩展为四节点共 16 个实例；扩展实验必须重新做四节点资源审计。

## 固定配置

- 节点：一台经审计的空闲节点，首选 `cloud-099`。
- 实例：`mi0..mi3`，分别绑定 GPU `0,1`、`2,3`、`4,5`、`6,7`。
- 每实例：TP=2、DP=1、独立 worker 端口和 bootstrap 端口。
- 初始角色：`mi0=P`，`mi1=D`，`mi2=D`，`mi3=D`。
- 切换：`mi2` 的活动请求按 50% / 观察 2 秒 / 剩余请求迁移到 `mi3`，随后
  `mi2` 切换为 P，最终严格验证 `2P2D`。
- 负载：复用冻结的 40 请求 trace；到达间隔 0.2 秒，长短 prompt 交错，
  `max_tokens=10000`，TTFT SLO 分别为 0.45 秒和 0.25 秒。
- 缓存：四个候选角色均执行长、短 Prefill 暖机，清空 KV 后才开始测量。

## 隔离与证据

- 每次使用唯一 `RUN_ID`；容器、env、PID、日志和 artifact 均带完整 RUN_ID。
- 只停止精确匹配本次 RUN_ID 的容器；不使用 restart、pkill、killall 或强杀。
- 私钥仅从 `chmod 600` 的私有 env 文件读取，不进入命令参数、Git 或共享 artifact。
- 保留 trace/hash、manifest、客户端 token 事件、request metrics、响应/错误、
  worker/router/controller/observer/sampler 日志、迁移事件、初末拓扑和 teardown 状态。
- 启动前后验证端口、GPU compute PID、driver、RDMA GID 和节点 SSH。

## 接受标准

1. 四个 worker 均通过 health 和初始角色检查，router 严格显示 `1P3D`。
2. 四个实例暖机成功，KV 清理成功，测量前恢复严格 `1P3D`。
3. 40/40 请求完成、零错误，每个请求输出完整 10,000 token。
4. controller 记录 SLO 触发、两阶段迁移、2 秒观察和 `role_flip_complete`。
5. 最终 router 和 worker 一致显示 `2P2D`。
6. 精确 teardown 后所有本次端口释放，8 张 GPU 无计算进程，driver 正常。

## 自审批结论

**批准分阶段执行。** 单节点阶段不占用 `cloud-100` 上属于他人的 GPU4 容器，
且不会触碰任何非本次 RUN_ID 的资源。只有上述六项全部通过，才将 runner 扩展到
四节点 16 实例；若失败，保留独立 forensic 目录并停止继续加载。
