## 1. IR and lowering model

- [x] 1.1 Replace the implicit blocking collective trigger on managed-reduction buffer loads with an explicit async collective start/wait representation for overlap schedules.
- [x] 1.2 Add double-buffered reduction-buffer state and legality checks needed to represent slot reuse, drain, and fallback to blocking collectives.

## 2. Search-space expansion

- [x] 2.1 Extend managed-reduction search decisions to enumerate `blocking_collective`, existing ring overlap, and `async_collective_overlap` strategies.
- [x] 2.2 Add async-overlap legality filtering for overlap-axis selection, tile count, tensor-layout constraints, and double-buffer memory requirements.

## 3. Code generation and estimation

- [x] 3.1 Update PyTorch codegen to emit double-buffered async `all_reduce` start/wait ordering while preserving the current `shift`/ring lowering path.
- [x] 3.2 Update memory accounting and the theoretical cost model to estimate tile-level async overlap with warmup, steady-state, and drain costs.

## 4. Validation and documentation

- [x] 4.1 Add regression tests covering GEMM async-overlap search/codegen behavior and preserving existing attention ring candidates.
- [x] 4.2 Add estimator-focused tests that distinguish blocking collective, ring overlap, and async collective overlap candidates.
- [x] 4.3 Update `docs/reference` summaries so the documented search and communication model includes explicit async collective overlap for managed reductions.
