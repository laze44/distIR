# Section 3 Overview

## 目标

`Mercury` 面向多 GPU 分布式算子优化，核心思想是把计算、通信、内存管理统一到一个 loop-based IR（`CommIR`）中，并将远程 GPU 内存视作可显式调度的内存层级扩展。

## 端到端流程

Section 3 将系统分为四个阶段：

1. Parsing（DSL -> CommIR）
2. Transformation（基于原语的调度变换）
3. Code Generation（通信 + 本地计算 lowering）
4. Auto-Tuning（在约束下搜索最优分布式调度）

## DSL 设计要点

- 采用 Python 风格 DSL，降低使用门槛。
- 与传统 tensor DSL 相比，关键差异是显式 loop 符号与 loop 注解。
- 这些 loop 信息同时承载：
  - 本地计算结构
  - 并行维度
  - 数据分片/复制
  - 通信模式（如 shift）

因此 DSL 从一开始就暴露了“可分布式 lowering”的结构信息。

## CommIR 与变换调度

`CommIR` 保留 loop nest 层次，并通过计算原语与通信原语统一表达分布式执行意图：

- 计算原语：`tile`、`reorder`、`patch` 等
- 通信原语：`parallelize`、`shard`、`shift`、`replicate`

这些通信原语在 IR 阶段是“符号注解”，不直接生成通信代码；后续 lowering 才将其物化为 P2P 或 collective。

## Lowering 核心思路

代码生成拆成两步：

1. 先生成通信内核：根据 loop 变换与 buffer 注解推断 P2P/collective 形式。
2. 再生成本地计算内核：落到设备后端 IR（文中示例提到 TorchInductor），并可通过 `patch` 替换为高性能库（如 FlashAttention）。

## 调优入口（与 Section 6 对应）

Overview 中明确调优空间来源于 `CommIR` 变换：

- 先枚举本地计算 schedule
- 再叠加通信策略（如 `parallelize`、`shift`）
- 用 profile latency + 静态内存检查做筛选

这为 Section 6 的 auto-tuner 细节奠定了框架。
