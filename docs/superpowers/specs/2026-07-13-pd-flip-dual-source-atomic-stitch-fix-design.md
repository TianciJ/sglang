# PD Flip 双源 KV 原子拼接修复设计

## 目标

修复 PD Flip 目标 Decode 在 `prepare_only` 阶段把尚未提交的 HiCache L3
恢复索引误判为 KV 空洞的问题，使以下三段能够在目标节点安全拼接：

```text
[0, L1)    目标 Decode 已有的 GPU prefix KV
[L1, H)    Mooncake L3 → Host L2 → GPU L1 恢复的 prefix KV
[H, C0)    源 Decode 发送的运行中 suffix KV
```

同时保留现有两阶段原子提交、批量所有权切换和源 Decode 全量 KV 回退。

## 已确认的根因

HiCache restore 完成时，GPU page 已经分配，恢复后的 page indices 保存在
`decode_req.hicache_restored_kv_indices` 中。它们只有在
`_commit_hicache_local_restore_to_req()` 执行时才会写入正式的
`req_to_token_pool[L1:H]`。

当前迁移顺序是：

```text
HiCache restore READY
  → _pd_flip_target_stitch_ready()
  → prepare_only 时保持 held
  → 稍后 atomic commit 才写入 L3 indices
```

但是 `_pd_flip_target_stitch_ready()` 提前检查了
`req_to_token_pool[:C0]`。因此在本次实验中：

```text
L1 = 5
H = 1974
H - L1 = 1969
```

区间 `[5,1974)` 尚未正式绑定，检查恰好报告 1969 个未初始化 index。
该错误只能证明检查时机错误，不能证明 Mooncake L3 数据传输失败。

## 设计原则

1. 原子提交约束请求所有权和可运行状态，不要求 Prepare 阶段的所有内部资源为空。
2. Prepare 阶段只验证临时资源能否组成完整 KV，不激活目标请求。
3. Commit 阶段才把 HiCache 恢复索引写入正式请求映射。
4. 正式映射写入后必须再次检查 `[0,C0)`，再允许目标进入可激活状态。
5. 任何真实的 prefix restore 覆盖失败继续触发源 Decode 全量 KV 回退。

## 方案比较

### 方案 A：Prepare 验证临时分段，Commit 正式绑定（采用）

Prepare 分别验证 L1、L3 restore buffer 和源 suffix，不要求
`req_to_token_pool[L1:H]` 已写入。Commit 时调用现有 HiCache commit，再检查
正式映射。

优点：保持现有两阶段所有权协议；Abort 前没有对外可见的目标请求；改动集中。
缺点：需要两套语义明确的检查函数。

### 方案 B：Prepare 提前写正式映射，请求继续 held

Prepare 时直接调用 HiCache commit，但不激活请求；Abort 时回滚映射、节点锁和
page ownership。

优点：可复用现有全映射检查。缺点：现有 commit 会修改 `last_node`、锁引用和
请求字段，批量失败后的回滚更复杂，风险更高。

### 方案 C：关闭双源拼接，永久使用源 Decode 全量传输

实现简单且当前已跑通，但不满足双源拼接需求，也无法利用远端 prefix KV。

## Prepare 阶段覆盖检查

将当前 `_pd_flip_target_stitch_ready()` 的职责改为“准备态覆盖检查”。检查内容：

1. 元数据边界合法：`0 <= H <= P <= C0`。
2. 源 suffix 覆盖必须严格等于 `[H,C0)`。
3. `prefix_match.l1_prefix_len` 必须在 `[0,H]` 内。
4. L1 的 `prefix_indices` 长度必须等于 L1，且所有 index 有效。
5. 当 `H > L1` 时：
   - HiCache restore 状态必须为 `READY`；
   - `hicache_restored_kv_indices` 必须存在；
   - 长度必须严格等于 `H-L1`；
   - 所有 index 必须有效。
6. 正式表中的源 suffix `req_to_token_pool[H:C0]` 必须全部有效。
7. Prepare 阶段允许正式表 `[L1:H]` 仍为未初始化值，因为该段尚未 commit。

检查成功后 entry 进入 `transferred_held`，请求不能进入 running batch。

## Commit 阶段正式映射检查

目标收到 atomic commit 后，对整批 entry 执行：

1. 再确认所有 entry 仍为 `transferred_held`。
2. 对需要 restore 的 entry 调用
   `_pd_flip_target_commit_hicache_restore()`。
3. 检查正式的 `req_to_token_pool[:C0]`：
   - 长度覆盖 `C0`；
   - 所有 index 均有效。
4. 全批通过后将 entry 设为 `ready_to_activate`。
5. 激活阶段才将请求加入目标 Decode 的可运行队列。

如果 Commit 内部检查失败，目标批次 Abort，释放目标临时资源；源仍是请求 owner，
不得释放源请求。Commit 前的 Prepare 已排除常规覆盖错误，因此 Commit 失败视为
内部一致性错误，不在同一 session 中继续叠加第三条恢复路径。

## 全源回退

以下 Prepare 错误继续进入现有 `fallback_required`：

- L3 restore 状态为 `FAILED`；
- restore indices 不存在或长度不足；
- restore indices 包含无效值；
- suffix 覆盖不是 `[H,C0)`；
- suffix 正式映射存在空洞。

Controller 随后重新创建 `H=0` 的目标接收 entry，由源 Decode 发送完整
`[0,C0)`。全源回退不依赖 HiCache restore。

## 状态与资源不变量

- `transferred_held`：目标拥有完整临时 KV 资源，但没有请求执行权。
- `ready_to_activate`：正式映射完整，目标仍未开始执行。
- `active`：目标获得执行权；随后源才可以释放请求。
- Prepare 或 Commit 失败时，源请求始终保留。
- 一个 entry 的有效 KV 覆盖只能是双源 `[0,H)+[H,C0)` 或全源 `[0,C0)`，不能混合残留。

## 测试设计

先增加能够复现当前错误的失败测试，再修改实现：

1. `prepare_only`，L1=5、L3=1969、正式 gap 为 0、临时 restore indices 有效：
   Prepare 必须成功进入 `transferred_held`。
2. restore indices 长度小于 `H-L1`：必须请求全源回退。
3. restore indices 含无效 index：必须请求全源回退。
4. suffix `[H,C0)` 含空洞：必须请求全源回退。
5. Commit 将 restore indices 写入 `[L1,H)` 后，全映射检查通过。
6. Commit 后仍有空洞：整批 Abort，不激活任何请求。
7. `H=0` 全源回退不要求 HiCache restore。
8. 多请求批次中任一 entry 未准备完成时，不允许部分 Commit。

## 四节点验收

明天 9–10 点复用固定 40 请求长短交错 trace，验收条件：

1. 日志显示 `full_prefix_stitch`，且不触发 `fallback_required`。
2. Prepare 记录 L1、L3 restore 和 suffix 三段长度，总覆盖为 `C0`。
3. Commit 后 `req_to_token_pool[:C0]` 无未初始化 index。
4. 目标 Decode 激活并继续生成，40/40 请求完成、0 error。
5. 源节点完成观察和第二轮迁移后切换为 Prefill。
6. 实验结束并空闲至少 60 秒后，四节点健康，无 scheduler exception、KV leak 或 invariant failure。
7. 若双源仍失败，保留全源回退，raw/log 必须能够区分 prefetch、load-back、临时
   indices、正式 commit 和 suffix transfer 的具体失败边界。

