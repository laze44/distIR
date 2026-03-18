# 拓扑感知 Mesh Shape 搜索策略 (Topology-Aware Mesh Shape Policy)

## 背景与动机

在早期的搜索流程中，`enumerate_mesh_shapes(world_size)` 会对 `world_size` 进行任意因子分解。这种“盲目”的分解方式在多节点大规模搜索时会导致候选空间爆炸，并产生大量缺乏物理意义的逻辑形状。

例如，在 `inter_node=16, intra_node=1` 的 GEMM 场景下：
- 原始候选总数约为 44,060 个。
- 仅 `mesh_shape=(2, 2, 2, 2)` 这一个形状就贡献了 21,549 个候选（占比近 50%）。

此外，旧方法采用事后推断（post-hoc inference）来识别逻辑维度所属的物理域。这经常导致产生 `mixed_dims`（跨越物理边界的维度），增加了代价模型（Estimator）的不确定性。

为了在压缩搜索空间的同时保留必要的表达力，Mercury 引入了拓扑感知的 Mesh Shape 策略。

## 核心概念

新策略将物理硬件拓扑与逻辑切分方式解耦，通过以下核心抽象进行建模：

- **TopologySpec**: 描述硬件的物理拓扑，由多个物理域（Domain）组成。物理域的顺序决定了逻辑 Mesh 维度的排列顺序（通常为先 inter-node 后 intra-node）。
- **DomainSpec**: 描述单一物理互连域。
  - `kind`: `clique`（全连接，如 NVLink 或 RoCE 网路）或 `mesh2d`（物理 2D 拓扑）。
  - `size`: 域内的设备总数。
  - `factorization_policy`: 决定该物理域如何映射到逻辑维度。
- **MeshShapePolicy**: 最终传递给 `search()` 的配置对象，控制逻辑 Mesh Shape 的生成规则。

## 因子化策略 (Factorization Policy)

针对 `clique` 类型的物理域，提供受控的因子分解能力：

- **single_dim**: 该物理域仅映射为一个逻辑维度。适用于大多数 intra-node 场景。
- **rank_limited**: 允许物理域分解为最多 `max_virtual_dims` 个逻辑维度。
- **示例**: 当 `inter_node=16` 且使用 `rank_limited` (max=2) 时：
  - 允许生成 `(16,)`, `(8, 2)`, `(4, 4)` 等逻辑组合。
  - 禁止生成 `(2, 2, 2, 2)`，从而消除了高维组合爆炸。

## Mesh Shape 生成规则

`enumerate_topology_mesh_shapes()` 按照以下规则合并物理域的因子：

1. **按域拼接**: 逻辑维度严格按 `TopologySpec` 中定义的域顺序排列。
2. **禁止跨域混合**: 单个逻辑维度不能同时包含来自不同物理域的设备。
3. **确定性顺序**: 不允许通过重排物理域顺序来制造等价的逻辑 Mesh Shapes。

例如，`inter_node=4, intra_node=8` 且两者均为 `clique` 时：
- 若 inter 允许 `(2, 2)`，则可生成 `(2, 2, 8)`。
- 不允许生成 `(2, 4, 4)` 这种将 inter 与 intra 混合的形状。

## Topology Metadata

拓扑元数据（如 `inter_node_dims` 和 `intra_node_dims`）在 Mesh Shape 生成时即已确定，不再依赖事后推断。
- 每一个逻辑维度在创建之初就绑定了对应的物理域标签。
- 在新的策略下，`mixed_dims` 始终为空数组 `[]`，确保了硬件语义的纯净性。

## GEMM 默认配置

对于标准的 GEMM 搜索，`make_gemm_mesh_shape_policy()` 推荐以下默认配置：

- **Inter-node**: 使用 `rank_limited`，`max_virtual_dims=2`。这确保了 2D 矩阵在跨节点维度上可以表达 `(4, 1)`, `(2, 2)`, `(1, 4)` 等灵活布局。
- **Intra-node**: 使用 `single_dim`。将单节点内视为一个整体，有效控制搜索宽度。

## 与现有模块的兼容性

- **estimate.py**: 完全兼容。由于元数据直接生成且无 `mixed_dims`，Cost Model 的评估更加稳定。
- **mapping_constraints.py**: 完全兼容。约束检查逻辑现在基于更加确定的维度属性运行。
- **gemm_two_step_search.py**: 目前通过 `_normalize_topology_metadata` 保持兼容，未来计划将硬编码的 `_fixed_topology_metadata` 逻辑彻底迁移至本策略。

## 搜索入口集成

`search()` 和 `search_with_progress()` 现在接受可选的 `mesh_shape_policy` 参数：

```python
policy = make_gemm_mesh_shape_policy(inter_node=4, intra_node=8)
search(..., mesh_shape_policy=policy)
```

若未提供该参数，系统将回退到旧有的 `world_size` 任意因子分解模式，确保向下兼容。
