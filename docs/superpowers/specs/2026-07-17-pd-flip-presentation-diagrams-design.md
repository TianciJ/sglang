# PD Flip 全链路组会图组设计

**日期：** 2026-07-17  
**用途：** 组会汇报与系统实现学习  
**范围：** SGLang 固定 1P3D 基线，以及本仓库实现的 SLO 驱动 1P3D → 2P2D 完整链路

## 1. 目标

图组同时服务两个层次：

1. 主图让已经理解 PD 分离与 KV Cache 的听众快速理解动态调整的必要性、执行顺序、观察分支和最终拓扑。
2. 细节图帮助讲解者理解每个阶段的请求、KV、节点角色和所有权变化，并区分原生 SGLang 能力、本项目对既有能力的扩展以及完全新增的控制逻辑。

图组不把所有控制线、接口和异常分支放入同一张图。总图只负责建立主线；实现机制分别下钻。

## 2. 统一表达规则

### 2.1 节点命名

- `P1`：初始 Prefill Worker。
- `D1`：被选中准备转换角色的源 Decode Worker；完成后成为 `P2`。
- `D2`、`D3`：承接迁移请求的 Decode Worker。
- Router 和 Controller 只在其行为属于当前图的重点时出现。

### 2.2 视觉语义

- Prefill 节点：一种稳定填充。
- Decode 节点：另一种稳定填充。
- Draining 节点：虚线轮廓，表示不接收新请求但仍可执行已有请求。
- 迁移请求：单向实线箭头。
- 控制或观测：细虚线，仅在细节图中使用。
- 回退：从观察阶段向下分叉，不与成功主线争夺横向空间。

颜色不是唯一编码方式；节点内始终保留 `P/D/draining` 文字。

### 2.3 能力归属标签

每张细节图统一使用三类归属：

- **复用 SGLang：** 原有 Scheduler、PD Router、KV Pool、PD 传输、HiCache/Mooncake、Prefill/Decode 事件循环等能力。
- **扩展 SGLang：** 在既有 Worker、Scheduler、TokenizerManager 或 Router 中新增状态、接口和安全边界。
- **我们新增：** Controller FSM、SLO 决策、渐进迁移编排、观察与回退策略、事务式请求所有权交接等。

不得将“原生 SGLang 没有本项目的完整控制闭环”表述成“原生 SGLang 完全没有迁移或 KV 传输能力”。

## 3. 图组结构

### 图 1：原生 SGLang 固定 1P3D 基线

以节点拓扑展示客户端、Router、P1、D1/D2/D3 和输出路径。只传达：

- Worker 启动后角色固定。
- Router 完成正常 PD 路由。
- 请求进入 Decode 后通常由该 Worker 持有到结束。

### 图 2：动态 1P3D → 2P2D 主流程

采用已确认的 A 版“四格横向故事板”，四格均使用相同节点位置：

1. 正常 1P3D。
2. D1 draining，首批请求迁往 D2/D3，D1 继续执行剩余请求。
3. 角色仍为 1P3D，采集进入观察阶段后的新 TTFT/TPOT。
4. 剩余请求迁移完成，D1 热切换成 P2，最终为 2P2D。

观察格向下分叉：

- Prefill SLO 恢复、Decode SLO 变差或样本不足：D1 恢复 admission，稳定为 1P3D；首批已经迁移的请求不搬回。
- Prefill SLO 仍差且 Decode 健康：迁移 D1 剩余工作，继续角色切换。

主图不展示 API 名、KV 边界、提交协议或 Controller 的逐节点连线。

## 4. 六张阶段细节图

每张采用统一的三栏结构：本阶段节点变化、复用的 SGLang 能力、我们新增或扩展的实现。

### 阶段 1：正常运行与 SLO 触发

- 节点变化：没有拓扑变化，仍为 1P3D。
- 复用：正常 PD 请求链路、Worker 指标和请求执行结果。
- 新增/扩展：按角色聚合请求级 TTFT/TPOT 达标数；最小样本门槛；仅在 Prefill 不达标且 Decode 达标时触发 D→P 尝试。

### 阶段 2：源/目标与首批请求选择

- 节点变化：选出 D1 和承接节点 D2/D3，尚不转移所有权。
- 复用：Scheduler 的 running/waiting 队列、请求槽位、KV 容量信息。
- 新增/扩展：排除 draining 节点；按 running batch 顺序选择 first-N；对请求槽位与最坏情况 KV 容量做预检；容量不足时反复减半迁移比例；DP 场景按 rank 分区和编排。

### 阶段 3：首批请求迁移

- 节点变化：D1 停止接收新请求；选中请求迁往 D2/D3；未选请求继续在 D1 Decode。
- 复用：KV Pool、PD 传输后端、HiCache/Mooncake 和 Scheduler 请求对象。
- 新增/扩展：双源 KV 恢复；initial copy 与冻结后的 delta copy；目标 held 状态；整批原子 prepare/commit/finish/activate；跨 TokenizerManager 输出 Relay 与序号去重。

### 阶段 4：观察与回退

- 节点变化：角色仍为 1P3D；D1 保持 draining 并执行剩余请求。
- 复用：D1 的普通 Decode 执行能力与请求指标。
- 新增/扩展：进入观察时清空触发窗口；只使用 fresh samples；观察时间结束后分为恢复、Decode 风险、样本不足和 Prefill 风险持续四种判断；回退时恢复 D1 admission 和 Router 路由。

### 阶段 5：剩余请求迁移

- 节点变化：D1 的 remaining running 和 eligible waiting 请求迁往 D2/D3。
- 复用：Scheduler 队列管理和目标 Decode 执行。
- 新增/扩展：第二批原子迁移；迁移失败时保持 D1 为 Decode；成功后检查 running、waiting、PD 队列和 migration session 全部清空。

### 阶段 6：Decode → Prefill 热切换

- 节点变化：D1 的 Decode 事件循环退出并以 P2 身份进入 Prefill 事件循环；Router 最终形成 2P2D。
- 复用：已有 Prefill/Decode 事件循环和 Router 角色路由能力。
- 新增/扩展：Scheduler 外层事件循环重分发；仅在完全空闲时修改 runtime role；同时校验 `role` 与 `active_event_loop_role`；先确认 Worker，再更新 Router，最后恢复 admission 和取消 draining。

## 5. 三张机制下钻图

### 机制 A：双源 KV 拼接

用一条 token 区间带展示：

- `[0,H)`：HiCache/Mooncake 连续 Prompt KV。
- `[H,C0)`：源 Decode initial transfer。
- `[C0,C1)`：冻结边界前新增的 delta transfer。

同时展示零命中时由源 Decode 提供 `[0,C1)` 的 fallback。强调目标只有验证 `[0,C1)` 连续完整覆盖后才能激活请求。

### 机制 B：请求所有权与输出连续性

展示四个所有权节点：

`source active → target prepared/held → source finish → target activate`

在 source finish 之前源端是唯一执行和输出所有者；target activate 之后目标端是唯一所有者。输出使用 session、request ID 和递增 sequence 去重，并 Relay 回原请求链路。

### 机制 C：故障与回退矩阵

按失败位置列出：容量预检失败、首批迁移失败、观察恢复、Decode SLO 下降、第二批迁移失败、Worker 切换失败、Router 更新失败。每行明确：

- 请求最终所有者。
- D1 是否恢复 admission。
- Router 是否保持 draining。
- 最终拓扑是 1P3D、2P2D，还是处于可重试的安全中间态。

## 6. 讲解顺序

1. 用图 1 建立原生固定拓扑基线。
2. 用图 2 一次性讲清完整故事，不解释底层协议。
3. 按六个阶段逐页回答“当前哪个节点发生变化、复用了什么、增加了什么”。
4. 用三张机制图解释最难的 KV 连续性、请求唯一所有权和失败安全。
5. 最后总结：模型前向和基本 PD 能力主要复用 SGLang，创新集中在 SLO 驱动的在线重配置闭环及其一致性协议。

## 7. 验收标准

- 主图在不阅读代码名和接口名的情况下可按顺序讲完。
- 每个阶段都能回答请求在哪里、KV 在哪里、谁拥有请求以及节点是否接收新流量。
- 每项能力均被标为复用、扩展或新增，不使用含混的“我们实现了所有迁移能力”表述。
- 观察回退与继续切换在主图中清晰可见。
- 细节图与主图使用相同的 P1、D1/P2、D2、D3 命名和节点位置。
- 图组适合 16:9 幻灯片展示，并在附录保留实现深度。
