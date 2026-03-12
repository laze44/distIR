## Why

The current two-step FFN flow can identify explicit edge reshard transitions during step-1, but step-2 still lowers only isolated operator kernels and does not carry those edge obligations into executable segment-level code generation. This produces plans that are valid in the cost model yet incomplete or misleading at export time, especially when `L_mid` and `down.A` differ and the required reshard should be absorbed into the consumer side.

## What Changes

- Add a step-2 segment-level fusion capability that can assign explicit edge reshard obligations to a consumer-side segment instead of treating them only as fixed graph-level penalties.
- Define logical boundary layouts and materialized kernel boundaries separately so step-1 continues to fix graph semantics while step-2 chooses where obligations are satisfied.
- Extend FFN step-2 planning to support the minimal fusion unit `layout-preserving chain + edge + consumer`, with `silu_and_mul + edge(L_mid -> down.A) + down_gemm` as the initial supported case.
- Update graph-edge reshard semantics so an explicit edge obligation remains part of the selected plan even when its execution is fused into a segment and no standalone reshard kernel is emitted.
- Clarify result export expectations so selected artifacts reflect the chosen step-2 segment ownership rather than unrelated per-operator rank-1 candidates.

## Capabilities

### New Capabilities
- `segment-edge-fusion`: Define how step-2 searches segment-level lowerings that absorb explicit edge reshard obligations, including ownership assignment and FFN consumer-side fusion.

### Modified Capabilities
- `boundary-layout-constrained-lowering`: Step-2 constraints now apply to logical boundary obligations and segment ownership, not only isolated operator-local lowerings.
- `graph-edge-reshard-costing`: Explicit edge reshard transitions must persist as semantic obligations through step-2 even when their execution is fused into a neighboring segment.

## Impact

- Affected docs: [`docs/changes/two_step_search.md`](/home/zehuali/workspace/mercury_artifact/docs/changes/two_step_search.md)
- Affected search modules: `mercury/search/ffn_two_step_search.py`, `mercury/search/ffn_graph_search.py`, and boundary-layout filtering utilities
- Affected IR/codegen surface: graph or segment-level lowering/result export paths for FFN step-2
- Affected tests: FFN two-step search, graph search, and example result export validation
