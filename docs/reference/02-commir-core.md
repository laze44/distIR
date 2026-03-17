# Section 4 CommIR

## 4.1 定义与原语

### 设计出发点

论文认为：本地 tensor compiler 的 loop IR 本就包含并行语义（例如将 loop 绑定到硬件执行单元），因此可自然扩展到分布式场景。`CommIR` 在继承 loop-based 表达的基础上，引入远程内存访问语义。

### 计算原语（结构重写）

- `tile`：拆分循环并引入中间 buffer/层次化访问
- `join`：合并循环维度
- `reorder`：交换 loop 次序以优化局部性
- `patch`：把匹配子图替换成高性能微内核/外部算子

### 通信原语（分布注解）

- `parallelize(loop, mesh_level)`：将 loop 迭代映射到网络层级并行
- `shift(local_loop, parallel_loop)`：相对并行维度错位访问，形成异步/交错通信
- `shard(buffer, loop)`：按并行维对 buffer 分片
- `replicate(buffer, loop)`：在并行参与者间复制 buffer
- `async_collective_overlap(reduce, overlap_axis, stage_count)`：为 managed reduction 显式标注异步 collective 的 `start/wait` 生命周期与双缓冲流水

论文强调：这四个原语是“注解语义”，主要在 lowering 阶段解释，不在变换当下直接插通信调用。

## 4.2 远程内存访问语义

### 并行语义（Parallelize）

- `parallelize` 决定工作划分与设备映射。
- 默认初始化策略用于减少无谓远程访问：
  - 被并行 loop 索引的 buffer 倾向 `shard`
  - 未被并行 loop 索引的 buffer 倾向 `replicate`

这让每个 rank 尽量本地命中其工作集。

### 分层 mesh 语义

`parallelize` 显式携带 mesh 层级（如跨节点/节点内）。通过不同层级 loop 的组合，可表达分层共享与复制。例如某 buffer 在 inter-node 共享、在 intra-node 复制。

### 异步语义（Shift）

`shift` 通过 loop 索引错位让不同 rank 在不同时间步访问共享数据块：

- 减少热点争用
- 支持计算与通信重叠
- 支持多层 shift 组合（适配层次网络）

### Collective 推导（Shard/Replicate）

在 lowering 阶段，系统依据 buffer 的 `shard/replicate` 状态与读写语义，推导 collective：

- 读路径可能触发 AllGather/Broadcast
- 写路径可能触发 AllReduce
- 约简结果若仍是分片布局，可退化为 ReduceScatter

具体 collective 类型由算子语义（如 sum/product）与布局共同决定。

在 managed reduction 的 async overlap 路径中，lowering 会显式生成：

- collective `start`（例如 `dist.all_reduce(..., async_op=True)`）
- slot 复用前 `wait`
- loop 末尾 `drain wait`

从而替代旧版”在 `load_buffer(reduce_buf)` 时隐式触发阻塞 collective”的行为。

### Pipeline 合法化（Pipeline Legalization）

`async_collective_overlap` 策略不再仅依靠 `ReduceOp` 上的元数据注解来声明 overlap。在 estimation 和 codegen 之前，系统执行一次 **legalization pass**：

1. 将满足条件的 async managed reduction 转化为显式的 `ManagedReductionPipelineRegion` IR 节点
2. 合法化条件包括：
   - overlap axis 至少有 2 个 tile
   - collective 参与者 > 1
   - reduction buffer 的消费者可被 retime（仅有一条直接消费路径）
   - 无额外同迭代消费者强制提前 wait
3. 不满足条件的 async 候选被 fallback 为 `blocking_collective`

### 待退役 Tile 状态（Pending-Tile Retirement）

合法化后的 pipeline region 使用 `PendingTileDescriptor` 结构化描述符来跟踪在途 tile 身份：

- `slot_index`：pipeline slot 编号
- `tile_coords`：沿 overlap axis 的逻辑 tile 坐标
- `reduce_buffer`：正在执行 collective 的 reduction buffer
- `output_buffer` / `retire_indices`：退役时写入的目标位置

这替代了旧版 codegen 中的 ad-hoc 数组（`pending_j`, `works`）方式。

### Wait-at-Reuse / Drain 语义

合法化 pipeline 的等待策略：

- **wait-at-reuse**：仅在 slot 即将被新 tile 复用时 wait 前一个 tile 的 collective
- **drain**：overlap loop 退出时，对所有仍在途的 slot 执行 wait 并退役延迟消费者
- 完成后的 tile 通过 consumer store 写入最终输出（delayed retirement）

