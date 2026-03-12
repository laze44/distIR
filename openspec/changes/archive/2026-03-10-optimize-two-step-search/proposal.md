## Why

The current two-step search flow still derives step-1 plans from fully lowered operator candidates, so kernel-internal split and ring decisions leak into the graph-level layout state. This prevents step-1 from acting as the documented layout-only search stage and blocks accurate graph-level costing, especially when adjacent kernels need explicit reshard transitions.

## What Changes

- Introduce a step-1 layout planning stage that searches exact logical tensor mappings for kernel boundary tensors under a fixed inter-node/intra-node topology.
- Define a boundary-layout representation that is independent from kernel-internal partitioning, loop tiling, and ring-based communication.
- Update step-1 cost estimation to derive local compute size, abstract communication obligations, and edge reshard costs directly from logical layouts instead of lowered IR communication nodes.
- Add a step-2 lowering stage that searches operator implementations under fixed boundary layout constraints while preserving freedom for kernel-internal split, ring, and collective choices.
- Align GEMM and FFN two-step search flows around the same boundary-plan and boundary-constrained-lowering model.

## Capabilities

### New Capabilities
- `logical-layout-plan-search`: Search exact logical tensor mappings for operator and graph boundaries without encoding kernel-internal lowering decisions into the plan state.
- `boundary-layout-constrained-lowering`: Re-run operator lowering search under fixed boundary layout constraints so step-2 can optimize internal split, tile, ring, and collective placement independently of step-1.
- `graph-edge-reshard-costing`: Represent layout mismatches between producer outputs and consumer inputs as explicit reshard transitions with graph-level cost accounting.

### Modified Capabilities
- None.

## Impact

- Affected code in `mercury/search/`, especially two-step search entry points, mapping constraints, and estimation logic.
- Affected IR/distributed layout metadata in `mercury/ir/` so logical boundary layout can be separated from execution-time partitioning.
- New or updated OpenSpec capability specs for layout planning, boundary-constrained lowering, and edge reshard costing.
- Updated tests and examples for GEMM and FFN two-step search behavior.
