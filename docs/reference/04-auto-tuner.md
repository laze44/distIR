# Section 6 Auto Tuner

## 目标

Auto-tuner 的优化目标是：

- 最小化分布式算子的端到端延迟
- 满足每个 worker 的内存容量约束

## 设计空间生成

搜索空间由 `CommIR` 变换原语及其参数组合构成。论文给出三条关键规则来降低复杂度。

### 规则 1：两阶段生成（先算后通）

1. 计算阶段：`tile/reorder/join`，确定本地 loop 结构与布局
2. 通信阶段：`parallelize/shift/shard/replicate` + managed reduction 通信策略（`blocking_collective` / `ring_overlap` / `async_collective_overlap`），确定分布式访问与远程内存行为

好处：
- 避免大量冗余或非法组合
- 本地计算候选可在多种并行策略间复用

### 规则 2：硬件 mesh 约束前置

候选生成显式受硬件网格约束：

- loop 的 tile 大小与 mesh 尺寸对齐
- 被 `parallelize` / `shift` 的 loop 长度需匹配对应 mesh 维度

**拓扑感知 Mesh 生成 (MeshShapePolicy)：**
不再直接基于 `world_size` 进行盲目的因子分解，而是引入 `MeshShapePolicy` 结合物理拓扑进行受控枚举。
- **物理域划分 (TopologySpec)**：将硬件定义为多个物理域（如 `inter_node` 和 `intra_node`），每个域具有特定的连接特性（如 `clique` 全连接或 `mesh2d` 网格）。
- **受控分解策略**：
    - `single_dim`：物理域映射为单一逻辑维度，不进行分解。
    - `rank_limited`：允许将物理域分解为有限数量（通常最多 2 个）的逻辑子轴，以平衡搜索覆盖度与复杂度。
- **域感知枚举**：逻辑 mesh 维度按物理域顺序拼接，严禁跨域混合（不再产生 `mixed_dims`），确保生成的 mesh 形状在物理上是高效可实现的。

配合 `reorder/join` 仍可保持较高覆盖度，同时明显缩小候选数。

### 拓扑感知搜索 (Topology-aware Search)

在搜索过程中，逻辑 mesh 的每个维度在生成时即携带明确的拓扑元数据。
- **元数据直接生成**：`inter_node_dims` 和 `intra_node_dims` 在构建 mesh 形状时确定，不再依赖事后推断。
- **通信成本对齐**：拓扑元数据直接指导 estimator 选择对应的物理链路参数（带宽/延迟），提高时延评估的准确性。
- **搜索空间压缩**：通过排除物理上不合理（如跨节点混合维）或冗余的 mesh 形状，在保证最优解覆盖的同时显著降低了待评估的候选总数。

### 逻辑分片因子 (Logical Shard Factors)

`MeshShapePolicy.enumerate_shapes()` 生成的 mesh 形状（如 `(8, 2)`、`(4, 4)`）是**搜索枚举工件**，它们是对物理拓扑的因子分解，用于遍历组合切分空间，**不是**物理拓扑本身的描述。

**`LogicalShardFactors`** 提供了每个 buffer 在每个物理域上的真实切分因子描述：
- 从搜索枚举 mesh 的每个维度，通过 `topology_metadata` 映射回对应的物理域（如 `inter_node`）
- 将每个 buffer 维度上的 shard spec 所引用的 mesh 维度的大小，按物理域汇聚成因子元组

**示例**：对于 `inter_node=16, intra_node=1` 的物理拓扑，搜索可能产生 mesh 形状 `(8, 2)`（两个维度均属于 `inter_node`）。若矩阵 `A` 的 shard spec 为 `[S(0), S(1)]`，则其逻辑分片因子为：

```
A: inter_node=(8, 2)   # dim 0 被 inter_node 分成 8 份，dim 1 被分成 2 份
```

这意味着 `A` 在 `inter_node` 域上总共被分成 16 份，但具体的分法是行方向 8 份、列方向 2 份。

**核心函数**：
- `compute_buffer_logical_shard_factors()` — 为单个 buffer 计算逻辑分片因子
- `compute_program_logical_shard_factors()` — 为 program 中的所有边界 buffer 计算因子
- `logical_shard_factor_for_dim()` — 计算单个 buffer 维度在指定物理域上的有效分片因子
- `program_satisfies_logical_factor_constraints()` — 基于逻辑因子进行约束匹配

### 规则 3：通信原语打包

实践中将 `parallelize/shift` 与对应 `shard/replicate` 打包生成。

理由：
- 仅做 shift 而不做合适分片通常不改变有效语义
- 打包可显著降低搜索复杂度
- 在论文评测工作负载中未观察到明显覆盖损失

## 评估与剪枝

### 延迟评估

每个候选都进行完整 lowering 并在真实硬件上 profile，以实测时延为准。
在理论估算路径中，`async_collective_overlap` 采用 tile 级流水模型（warmup / steady-state / drain）而非将 collective 全量视作阻塞。

**重要：async overlap 排名仅在合法化成功后生效。** `estimate_program()` 和 `generate_pytorch_code()` 在评估/生成前会自动调用 `prepare_pipeline()`，确保 legalization 总是先于 ranking 和 codegen 执行。estimator 仅使用 `ManagedReductionPipelineRegion` 中的合法化信息来计算 async pipeline overhead——不存在"信任裸 ReduceOp metadata"的 fallback 路径。未合法化的 async 候选统一退化为 `blocking_collective` 排名和 lowering。

**async 候选轴过滤**：search 阶段通过 `_overlap_axis_is_realizable()` 在源头过滤不可物化的 overlap axis（例如 `size <= min_block_size` 的 collapsed axis），避免生成虚假的 async 候选。当没有可物化的 overlap axis 时，候选保留为 `blocking_collective`。多个可选 axis 时优先选择外层（更大尺寸的）axis。

合法化流程：
1. legalization pass 检查 overlap axis tile 数、axis 可物化性、collective 参与者数、消费者可 retime 性
2. verifier 验证 pipeline region 不变量（每 slot 至多一个在途 work、wait-before-reuse、retire-after-wait、overlap axis 可物化为运行时循环、pipeline_scope_axis 已设置）
3. 仅通过验证的 region 才允许使用 async pipeline 估算和 async codegen
4. legalized pipeline region 替换原 `GridLoop.body` 中的 `ReduceOp + BufferLoad + BufferStore`，不再作为 loop 外的附加节点——消除重复 codegen 风险
5. estimator 使用 `materialized_overlap_axis` 的 tile count（而非 ReduceOp metadata 上的原始值）计算 pipeline overhead，确保估算与 codegen 使用相同的循环轴

如果 program 在首次 legalization 之后又经过会原地修改 axis 的 pass（例如 `eliminate_loops()`），`prepare_pipeline()` 会重新检查已存在 region 的 `materialized_overlap_axis` 是否仍能物化为真实循环；失效 region 会被降级回 `blocking_collective`。同时 estimator 对已经塌缩的 `materialized_overlap_axis` 会直接跳过 async pipeline overhead，避免沿用旧 `tile_count` 产生过期收益估算。

### 内存约束

基于 `CommIR` 的静态布局分析估算 per-worker footprint；超过容量的候选在早期剪枝。
对 `async_collective_overlap` 额外计入 reduction buffer 双缓冲（stage count）带来的 slot 内存开销。

## 结果特征（本节结尾）

论文在该节末给出经验结果：按上述规则，评测中单算子调优可在约 10 分钟内完成；同时也指出更复杂算子/拓扑下仍有进一步引入 cost model 或 ML predictor 的空间。

## Roofline Hardware Config Schema

理论性能估算使用 `config/*.json` 中的硬件参数，不再依赖运行时 profile。默认配置文件为
`config/h100.json`，字段要求如下：

- `name`: 非空字符串
- `compute.peak_tflops.bf16`: 正数
- `compute.peak_tflops.fp16`: 正数
- `compute.peak_tflops.fp32`: 正数
- `compute.peak_tflops.tf32`: 可选，若提供则必须为正数
- `memory.bandwidth_tb_per_s`: 正数
- `memory.capacity_gb`: 可选，若提供则必须为正数
- `interconnect.intra_node.bandwidth_gb_per_s`: 正数
- `interconnect.intra_node.latency_us`: 非负数
- `interconnect.inter_node.bandwidth_gb_per_s`: 正数
- `interconnect.inter_node.latency_us`: 非负数

通信估算会根据 mesh 维度和显式的拓扑元数据（如 `inter_node_dims`）自动选择 intra-node/inter-node 的带宽与延迟参数。
