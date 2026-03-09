# 2-step 多 GPU 计算图搜索策略（以 FFN 为例）

## 0. 目标与范围

本文定义一个分两阶段执行的搜索策略：

1. `step-1` 搜索计算图级别的 tensor layout plan；
2. `step-2` 在给定 layout plan 下搜索每个算子的 distributed kernel lowering。

这里的 `layout` 指张量在 `DeviceMesh` 上的分布方式，包括：

- mesh shape
- 每个 tensor 维度对应的 sharding spec

FFN 示例使用如下逻辑图：

`X -> {gate_gemm, up_gemm} -> silu_and_mul -> down_gemm -> Y`

本文采用以下约定：

- `step-1` 搜索的是 tensor layout plan，其中既包括 activation layout，也包括每个算子 weight tensor 的 layout。
- `step-2` 负责在给定 layout plan 下搜索算子内部 lowering。
- 若一期实现只搜索 3 个 GEMM kernel，需要明确假设 `silu_and_mul` 已经被融合，不作为独立 kernel 搜索。
- 若 `silu_and_mul` 没有被融合，它必须作为独立 graph node 进入 `step-1` 和 `step-2`。
- 若 FFN 不是搜索边界，`Y` 的 layout 需要由外层图固定，或者显式纳入 `step-1` 的状态。
- 在本项目优化中，默认每个 device 上的算子按顺序执行。
- 运行时允许 `compute stream` 与 `communication stream` 并发，但不考虑多个 `compute stream` 并发执行多个算子或多个 compute 段。

## step-1: 搜索 tensor layout plan

### 1. 图级状态定义

`step-1` 的状态同时包含：

- 图上的共享 activation layout node
- 每个算子本地的 weight layout node

对 FFN，可定义如下状态：

- `L_in`：FFN 输入 `X` 的 layout
- `L_mid`：`gate/up` 输出以及 `silu_and_mul` 输入输出的共享 layout
- `L_out`：`down` 输出 `Y` 的 layout
- `W_gate`：`gate_gemm` 中权重 `B` 的 layout
- `W_up`：`up_gemm` 中权重 `B` 的 layout
- `W_down`：`down_gemm` 中权重 `B` 的 layout

其中：

- activation layout 是图边状态，决定算子之间的衔接方式；
- weight layout 是算子节点状态，决定单个算子的局部执行形态。

如果一期实现把 FFN 当成局部子图，且 `L_out` 由外层固定，可以只搜索 `L_in` 和 `L_mid`；否则 `L_out` 必须保留。

### 2. layout 合法性约束

对 `step-1` 中的每个 tensor layout，都需要满足以下约束：

- world size 固定，不能改变全局 tensor shape；
- 每个 tensor 维度只能是 `replicate`，或者 shard 到一个非空 mesh-dim 子集；
- 同一个 mesh dim 不能同时分配给两个 tensor 维度；
- 被分到某个 tensor 维度上的 mesh-dim 乘积必须能整除该维的全局长度；
- 若存在拓扑约束，还要满足 `inter_node` / `intra_node` / `mixed` 等 placement 要求。

可将单个 tensor 维度的映射写成：

- `R`：该 tensor 维度在 mesh 上复制
- `S(d0, d1, ...)`：该 tensor 维度分片到若干 mesh dim

此外：

- activation layout 需要满足跨算子衔接约束，因为它决定 graph edge 上是否发生 reshard；
- weight layout 不承担跨算子衔接，但会影响单个算子的局部计算形态、局部归约方式和 operator-internal communication 代价。

### 3. 通信类型

在 `step-1` 中，通信分为两类：

- `operator-internal communication`
  - 为了让某个算子在给定 input/output/weight layout 下语义正确而必须执行的通信
  - 例如 partial reduction 之后的 all-reduce、reduce-scatter、all-gather、ring exchange
- `inter-operator reshard`
  - 上一个算子的输出 layout 和下一个算子的输入 layout 不一致时，为了衔接两个 layout 发生的转换

对 FFN 而言：

- weight layout 主要影响 `operator-internal communication`
- activation layout 同时影响 `operator-internal communication` 和 `inter-operator reshard`

### 4. step-1 与 step-2 的通信职责划分

`step-1` 只负责决定 kernel 边界上的 tensor layout，不负责决定具体通信实现。

对任意一条图边：

- 若上游 kernel 的输出 layout 与下游 kernel 的输入 layout 一致，则该边不存在独立的 `inter-operator reshard`
- 若二者不一致，则该边存在独立的 `inter-operator reshard`

因此，`step-1` 的职责是判断：

- kernel 的边界 layout 是什么
- 图边上是否还需要独立的 reshard

`step-2` 的职责是：

- 在固定输入/输出 layout 的前提下，决定算子内部如何通信
- 决定是否存在更优的 collective rewrite 或更优的通信排布

例如：

- 若 `gate/up` 的输出 `L_mid` 与 `down` 的输入 `L_mid` 一致，则 `step-2` 中不需要再处理独立的 edge reshard
- `gate` 或 `up` 内部究竟采用 all-reduce、reduce-scatter、ring exchange 还是其他等价实现，由 `step-2` 决定

### 5. step-1 的代价模型

`step-1` 只探索 tensor mapping 的最佳方式，不搜索细粒度 lowering，也不绑定具体通信实现。但为了避免把所有通信都当成可以完全重叠，`step-1` 仍然对单算子时间做粗粒度分解。

对每个算子，定义：

- `T_compute`
- `T_overlapable_comm`
- `T_blocking_comm`

并使用如下估计：

`T_op = T_blocking_comm + max(T_compute, T_overlapable_comm)`

其中：

- `T_compute` 表示该算子在当前 activation/weight layout 下的计算时间估计；
- `T_overlapable_comm` 表示理论上可以与 compute 并发的通信时间估计；
- `T_blocking_comm` 表示必须阻塞在关键路径上的通信时间估计。

#### 5.1 `T_compute` 的估算

`T_compute` 使用 roofline model 估算：

- 先根据当前 layout 推导该算子的 local problem size；
- 根据算子类型计算总 FLOPs；
- 根据 local tensor 访问量估算 memory traffic；
- 分别得到 compute-bound 时间和 memory-bound 时间；
- 取二者较大值作为 `T_compute`。

可写成：

- `T_compute_bound = FLOPs / PeakCompute`
- `T_memory_bound = BytesMoved / MemoryBandwidth`
- `T_compute = max(T_compute_bound, T_memory_bound)`

其中：

- `PeakCompute` 来自硬件配置；
- `BytesMoved` 由当前 local activation 和 local weight 的读写量决定。

#### 5.2 `T_overlapable_comm` 与 `T_blocking_comm` 的估算

`T_overlapable_comm` 和 `T_blocking_comm` 都通过通信模型估算，但按是否可能与 compute 重叠分开累计。

对每个抽象通信事件，先根据其 collective 类型、参与设备数、消息大小和链路类型估计通信时间：

- 参与设备数由当前 tensor layout 和 mesh 维度决定；
- 消息大小由当前 local tensor shape 和 dtype 决定；
- 链路类型根据 mesh dim 属于 `inter_node` 还是 `intra_node` 选择；
- 时间由带宽项和时延项组成。

可写成：

- `T_comm_event = T_latency + T_bandwidth`

然后将通信事件分为两类：

- `overlapable communication`
  - 例如非输出数据在 load 阶段触发的 ring exchange
  - 这类通信可以粗粒度地认为有机会与 compute 重叠
- `blocking communication`
  - 例如输出结果归约、输出 materialization、必须在 epilogue 完成的 collective
  - 这类通信位于关键路径上，不在 `step-1` 中视为可重叠

于是：

- `T_overlapable_comm = sum(T_comm_event for overlapable events)`
- `T_blocking_comm = sum(T_comm_event for blocking events)`

在 `step-1` 中，默认采用保守规则：

- 只有明显属于输入侧、非输出依赖链上的通信才记入 `T_overlapable_comm`
- 所有输出侧通信、归约结束通信、以及为了 materialize 最终输出 layout 所需的通信都记入 `T_blocking_comm`

这样可以避免把输出通信或 epilogue collective 错误地当作完全可重叠。

对每条图边，定义：

- 若上游输出和下游输入之间仍需独立 reshard，则 `T_edge = T_reshard`
- 若上游输出和下游输入 layout 一致，则 `T_edge = 0`

图级总时间写成：

`T_total = sum(T_op) + sum(T_edge)`

对 FFN，`gate` 和 `up` 在同一 device 上默认按顺序执行，因此两者时间直接求和。`step-1` 中不建模细粒度的 stream schedule，只在粗粒度上使用 `T_blocking_comm + max(T_compute, T_overlapable_comm)` 近似单算子时间。

### 6. step-1 的输出

`step-1` 不只输出单一最优解，而是输出：

- top-k 个 tensor layout plan；或
- activation/weight layout 组合的 Pareto 前沿

这样可以把更精细的 lowering 与通信实现决策留给 `step-2`。

## step-2: 在固定 layout plan 下搜索 distributed kernel lowering

### 1. 输入与目标

`step-2` 的输入是 `step-1` 产出的某个 layout plan，例如：

- `gate`: `L_in + W_gate -> L_mid`
- `up`: `L_in + W_up -> L_mid`
- `down`: `L_mid + W_down -> L_out`

`step-2` 的目标是在这些边界 layout 已固定的前提下，搜索每个算子的 lowering 方式，包括：

- loop tiling / split
- loop order
- mesh reshape 与 mesh-dim assignment
- ring / collective 的具体插入方式
- 通信放置在 load/store/reduce/epilogue 的位置

在该阶段中：

- 若相邻 kernel 的边界 layout 一致，则忽略独立的 edge reshard，把问题视为单纯的 distributed kernel 优化；
- 若边界 layout 不一致，则对应的 edge reshard 作为图级固定代价保留，不属于单 kernel 内部优化对象。

### 2. kernel 边界

kernel 边界由 lowering/fusion 策略决定，不必与前端图中的算子数一一对应：

- 一个前端算子可以被拆成多个 kernel；
- 多个前端算子也可以被融合成一个 kernel。

在 FFN 示例里，只有在明确采用以下约束时，才将一期问题写成 3 个 kernel：

- `gate_gemm`
- `up_gemm`
- `down_gemm`

同时默认 `silu_and_mul` 已经融合到相邻 kernel，或者当前版本暂不搜索该段。

### 3. kernel 内通信与输出 layout

每个 kernel 必须产出 `step-1` 指定的输出 layout。为满足该 layout，kernel 内可能：

- 需要插入通信；
- 不需要通信；
- 在 reduction 过程中边算边通信；
- 在 load 阶段执行 ring exchange；
- 在 epilogue 阶段执行 collective。

`step-2` 不改变 `step-1` 已经固定的输入/输出 layout，只优化“如何到达该 layout”。

### 4. step-2 的时间估计

kernel 级时间模型写成：

`T_kernel = T_compute + T_comm - T_overlap_in_kernel`

其中 `T_overlap_in_kernel` 只覆盖：

- 在具体 IR 中已经显式表达的 overlap
- 运行时可实现的 overlap

跨 kernel overlap 不在该公式中建模，应在图级调度模型里单独定义。

在本项目默认执行模型下，图级调度模型仍然遵循“每个 device 上算子顺序执行”的原则，因此不引入多 compute stream 并发；可建模的 overlap 仅包括：

- 同一 kernel 内的 `compute-communication overlap`
- 相邻阶段中可实现的 communication 重排或融合

### 5. step-2 的输出

`step-2` 最终输出：

- 每个 operator/segment 的最佳 lowering IR
- 对应的 kernel 时间估计
- 若存在多个 `step-1` plan，则输出全局重排后的最佳 plan

整个两阶段流程为：

1. 搜索 top-k tensor layout plan；
2. 在每个 layout plan 下搜索 operator lowering；
3. 重新评估图级总时间并选出最终方案。

## 建议采用的一期简化版本

如果先做一个可落地的一期版本，建议采用以下边界：

- 只搜索 FFN 的 3 个 GEMM；
- 默认 `silu_and_mul` 已融合或忽略；
- 每个 device 上算子按顺序执行，只允许 `compute-communication` 多 stream 并发；
- `step-1` 同时搜索 `L_in`、`L_mid`、`L_out` 以及 `W_gate`、`W_up`、`W_down`；
- `step-1` 输出 top-k layout plan，而不是单一最优解；
- `step-2` 只在给定边界 layout 下搜索算子内部 lowering；
- `step-1` 只决定边界 layout 与是否存在独立 edge reshard；
- `step-2` 在固定边界 layout 下优化算子内部通信实现。
