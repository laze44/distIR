## 1. Boundary Layout Model

- [x] 1.1 Add logical boundary layout data structures for exact GEMM step-1 plans under fixed topology.
- [x] 1.2 Extend distributed buffer metadata to separate logical boundary layout from execution-time partitioning state.
- [x] 1.3 Add helpers to derive logical local shapes and compare logical boundary layouts without reading lowered execution artifacts.

## 2. GEMM Step-1 Planning

- [x] 2.1 Implement GEMM step-1 layout enumeration from problem shape, fixed topology, and mapping template constraints.
- [x] 2.2 Implement GEMM step-1 obligation-based cost estimation for local compute, input/output materialization, and reduction-finalize communication.
- [x] 2.3 Add plan ranking and top-k selection for GEMM logical layout plans using the step-1 total-cost formula.

## 3. GEMM Step-2 Lowering

- [x] 3.1 Implement boundary-layout-constrained GEMM lowering search that filters candidates by logical boundary contract rather than exact lowered state.
- [x] 3.2 Allow multiple lowerings to compete under one logical plan and rank them with the existing lowered-program estimator.
- [x] 3.3 Add a GEMM two-step search entry point and update the GEMM example flow to report selected step-1 plans and step-2 lowerings.

## 4. Graph Reshard Costing

- [x] 4.1 Implement explicit edge reshard transition objects and logical-layout-based edge cost estimation helpers.
- [x] 4.2 Integrate edge reshard costing into graph-level step-1 plan scoring without merging it into operator-internal communication cost.
- [x] 4.3 Update FFN step-1 planning to allow mismatched adjacent boundary layouts with explicit reshard penalties.

## 5. Validation and Migration

- [x] 5.1 Replace existing exact-layout tests that depend on lowered execution state with boundary-layout and plan-identity tests.
- [x] 5.2 Add GEMM tests for fixed-topology step-1 plan enumeration, obligation inference, and boundary-constrained step-2 rerun behavior.
- [x] 5.3 Add FFN graph tests covering explicit edge reshard transitions and step-1 ranking with operator-local plus edge costs.
