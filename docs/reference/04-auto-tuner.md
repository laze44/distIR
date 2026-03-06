# Section 6 Auto Tuner

## 目标

Auto-tuner 的优化目标是：

- 最小化分布式算子的端到端延迟
- 满足每个 worker 的内存容量约束

## 设计空间生成

搜索空间由 `CommIR` 变换原语及其参数组合构成。论文给出三条关键规则来降低复杂度。

### 规则 1：两阶段生成（先算后通）

1. 计算阶段：`tile/reorder/join`，确定本地 loop 结构与布局
2. 通信阶段：`parallelize/shift/shard/replicate`，确定分布式访问与远程内存行为

好处：
- 避免大量冗余或非法组合
- 本地计算候选可在多种并行策略间复用

### 规则 2：硬件 mesh 约束前置

候选生成显式受硬件网格约束：

- loop 的 tile 大小与 mesh 尺寸对齐
- 被 `parallelize` / `shift` 的 loop 长度需匹配对应 mesh 维度

配合 `reorder/join` 仍可保持较高覆盖度，同时明显缩小候选数。

### 规则 3：通信原语打包

实践中将 `parallelize/shift` 与对应 `shard/replicate` 打包生成。

理由：
- 仅做 shift 而不做合适分片通常不改变有效语义
- 打包可显著降低搜索复杂度
- 在论文评测工作负载中未观察到明显覆盖损失

## 评估与剪枝

### 延迟评估

每个候选都进行完整 lowering 并在真实硬件上 profile，以实测时延为准。

### 内存约束

基于 `CommIR` 的静态布局分析估算 per-worker footprint；超过容量的候选在早期剪枝。

## 结果特征（本节结尾）

论文在该节末给出经验结果：按上述规则，评测中单算子调优可在约 10 分钟内完成；同时也指出更复杂算子/拓扑下仍有进一步引入 cost model 或 ML predictor 的空间。
