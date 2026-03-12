## Why

The current GEMM search and lowering path can express pipelined ring communication through `shift`, but managed reductions that lower to `all_reduce` remain blocking. This prevents the search space from representing tensor-parallel GEMM schedules that overlap the `all_reduce` of one output tile with the computation of later `n`-dimension tiles.

## What Changes

- Preserve the existing `shift`/ring communication path used by attention-style schedules.
- Add a new async collective overlap path for managed reductions so search can choose between blocking collective reduction, ring-based overlap, and async collective overlap.
- Introduce tiled double-buffered reduction scheduling so each output tile can compute on one buffer while a previous tile's `all_reduce` progresses on another buffer.
- Expand the search space to enumerate overlap-friendly tiling and communication choices for managed reductions, especially GEMM with `K`-sharded weights.
- Update the performance model to account for tile-level overlap, pipeline warmup/drain, and communication/computation running on separate streams.

## Capabilities

### New Capabilities
- `async-reduction-overlap`: Search, IR, and lowering support for overlapping managed reductions with asynchronous collective communication and double-buffered tiles.

### Modified Capabilities

None.

## Impact

- Affected compiler areas: `mercury/ir/`, `mercury/search/`, and `mercury/backend/pytorch/`.
- Likely touched concepts: managed reduction IR, communication lowering, search-space enumeration, and theoretical latency estimation.
- New tests will be needed for search result structure, code generation ordering, and overlap-aware cost modeling.
