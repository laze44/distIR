## 1. IR and lowering model

- [ ] 1.1 Replace the implicit blocking collective trigger on managed-reduction buffer loads with an explicit async collective start/wait representation for overlap schedules.
- [ ] 1.2 Add double-buffered reduction-buffer state and legality checks needed to represent slot reuse, drain, and fallback to blocking collectives.

## 2. Search-space expansion

- [ ] 2.1 Extend managed-reduction search decisions to enumerate `blocking_collective`, existing ring overlap, and `async_collective_overlap` strategies.
- [ ] 2.2 Add async-overlap legality filtering for overlap-axis selection, tile count, tensor-layout constraints, and double-buffer memory requirements.

## 3. Code generation and estimation

- [ ] 3.1 Update PyTorch codegen to emit double-buffered async `all_reduce` start/wait ordering while preserving the current `shift`/ring lowering path.
- [ ] 3.2 Update memory accounting and the theoretical cost model to estimate tile-level async overlap with warmup, steady-state, and drain costs.

## 4. Validation and documentation

- [ ] 4.1 Add regression tests covering GEMM async-overlap search/codegen behavior and preserving existing attention ring candidates.
- [ ] 4.2 Add estimator-focused tests that distinguish blocking collective, ring overlap, and async collective overlap candidates.
- [ ] 4.3 Update `docs/reference` summaries so the documented search and communication model includes explicit async collective overlap for managed reductions.
