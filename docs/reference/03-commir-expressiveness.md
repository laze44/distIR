# Section 5 CommIR's Expressiveness

## 核心结论

`CommIR` 不仅能复现已有并行策略，还能通过组合变换自动发现新模式，覆盖 Attention 与 GEMM 等典型算子。

## Attention 示例主线

论文用 attention 的简化形式（scaled accumulation）展示从基础到复杂模式的构造：

1. 从局部计算出发（I/J 维）
2. 并行化 I 维，得到 context-parallel 类模式
3. 通过 `shift` 引入异步数据流
4. 在分层设备拓扑下继续拆分与重排
5. 并行化/重排约简维，得到类似 TreeAttention 的模式

## 组合式新策略

与手工设计不同，`CommIR` 可以继续在“非传统轴”上自动组合变换：

- 对约简轴做 shift，使部分和在 worker 间并行传递
- 拆分 J 轴，实现多 worker 的局部约简协作
- 在不同 I/J 子轴上叠加 shift，避免昂贵 collective 的同时保持并行度

论文将这类“多轴 + 多层级 + 异步”组合视为人工难以稳定构造的复杂调度。

## 对其他算子的泛化

Section 5 还指出其表达能力可迁移到 GEMM 类场景：

- 通过拆分并并行化约简轴表达 TP 风格线性层
- 在外层 loop 上叠加 `shift` 以表达 AsyncTP 风格策略

因此 `CommIR` 的价值不止于单算子模板，而是提供统一可搜索的分布式设计空间。
