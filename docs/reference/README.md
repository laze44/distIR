# Mercury Paper Docs (Overview to Auto Tuner)

本文档集基于论文 `sosp_mercury.pdf`，覆盖范围为：
- Section 3: Overview
- Section 4: CommIR
- Section 5: CommIR's Expressiveness
- Section 6: Auto Tuner

不包含 Section 7 及之后（Implementation/Evaluation/Conclusion）。

## 文档导航

- `01-overview.md`：Mercury 总体流程（DSL -> CommIR -> Lowering -> Tuning）
- `02-commir-core.md`：CommIR 的定义、原语语义与远程内存抽象
- `03-commir-expressiveness.md`：CommIR 在 Attention/GEMM 上的表达能力与模式组合
- `04-auto-tuner.md`：搜索空间构造、约束与优化目标

## 说明

- 内容是对对应章节的结构化转换与技术要点整理，不是逐句直译。
- 术语尽量与原文保持一致（如 `parallelize`、`shift`、`shard`、`replicate`、`CommIR`）。
