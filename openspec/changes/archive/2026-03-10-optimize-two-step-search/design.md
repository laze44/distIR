## Context

The current search pipeline generates fully lowered operator programs before any graph-level plan is formed. In practice, `tile_loop`, `parallelize`, and `shift` mutate buffer shape and sharding state during enumeration, so the current "layout" identity is derived from execution-time partitioning rather than kernel boundary semantics. This causes step-1 to rank lowered candidates instead of logical tensor mappings, prevents explicit edge reshard modeling between adjacent kernels, and makes step-2 exact matching overly restrictive because it matches internal lowering artifacts rather than boundary layout contracts.

The target workflow is a true two-step search:

- Step-1 runs under a fixed topology defined by the requested inter-node and intra-node counts.
- Step-1 searches only logical tensor mappings at kernel boundaries and estimates cost from those mappings.
- Step-2 re-runs lowering search under fixed boundary layout constraints and remains free to choose internal split, tile, ring, and collective placement.

The change must support GEMM first and provide a path to apply the same model to FFN graph search.

## Goals / Non-Goals

**Goals:**
- Separate logical boundary layout from execution-time partitioning in search state and IR metadata.
- Define a GEMM step-1 layout planner that enumerates exact logical A/W/B mappings under fixed topology.
- Estimate step-1 execution time from logical layouts using local compute size, abstract internal communication obligations, and explicit edge reshard costs.
- Define step-2 boundary-constrained lowering so multiple lowerings can satisfy the same step-1 plan.
- Reuse the same model for FFN by treating inter-operator layout mismatch as explicit reshard transitions instead of invalid plans.

**Non-Goals:**
- Redesign the full lowering IR or codegen backend in one step.
- Search topology shape itself; topology is fixed by the provided inter-node and intra-node counts.
- Model exact stream scheduling or exact collective implementation in step-1.
- Guarantee that every theoretically valid logical plan is immediately supported by the first step-2 implementation.

## Decisions

### 1. Step-1 plan objects are logical boundary plans, not lowered programs

Step-1 will not use `Program` objects as its output. It will emit exact logical layout plans that describe tensor mappings at kernel boundaries. For GEMM, the plan contains:

- fixed problem shape `(M, N, K)`
- fixed topology `(inter_node, intra_node)`
- exact logical layouts for `A`, `W`, and `B`
- a step-1 cost summary

This prevents kernel-internal split and ring choices from becoming part of the step-1 identity.

Alternatives considered:
- Keep step-1 based on lowered `Program` candidates and derive plan identity afterward.
  Rejected because lowered buffer state already encodes internal partitioning and reintroduces the current leak.

### 2. Logical layout and execution layout are stored separately

Each distributed buffer will carry two different layout views:

- logical boundary layout: immutable for a selected step-1 plan
- execution layout: mutable during step-2 lowering search

Global tensor shape is stored separately from execution-local buffer shape. Logical local shape is derived from global shape plus logical layout and is not stored as plan identity. Existing lowering passes may continue to mutate execution-local fields, but they must not overwrite logical boundary layout metadata.

Alternatives considered:
- Keep one `shard_spec` field and infer logical layout from the current lowered state.
  Rejected because step-2 lowering mutates exactly the state step-1 must treat as fixed.

### 3. Step-1 cost estimation is obligation-based

Step-1 will estimate cost using only logical layout, hardware configuration, and fixed topology. It will not read `RingComm`, `ReduceOp.comm`, tiled axis sizes, or other lowered IR details.

The estimator derives:

- local compute shape for each device
- input materialization obligations
- reduction-finalize obligations
- output materialization obligations
- edge reshard obligations between adjacent kernels

These obligations are converted into:

- `T_compute`
- `T_overlapable_comm`
- `T_blocking_comm`
- `T_edge`

and ranked with:

- `T_total = T_edge + T_blocking_comm + max(T_compute, T_overlapable_comm)`

Reduction-class communication is inferred from contraction-dimension sharding and output boundary requirements. Step-1 names communication by obligation class rather than by exact collective implementation.

Alternatives considered:
- Reuse `estimate_program()` for step-1 ranking.
  Rejected because it depends on lowered IR communication nodes and therefore mixes step-2 choices into step-1.
- Require step-1 to predict exact collective type.
  Rejected because collective choice belongs to step-2 lowering.

### 4. Layout mismatch between kernels is legal and costed as reshard

For graph search, a producer output layout differing from the consumer input layout is not an invalid plan. It creates an explicit edge transition with a reshard cost term. This allows step-1 to consider graph-level plans where kernels prefer different boundary layouts.

Alternatives considered:
- Restrict plans so adjacent kernels must share one exact activation layout.
  Rejected because it excludes legal plans and prevents the planner from trading off operator-local benefit against reshard cost.

### 5. Step-2 matches boundary layout contracts, not exact lowered state

Step-2 filtering will be based on logical boundary layout equality:

- same tensor identity
- same global shape
- same topology / mesh shape
- same logical shard mapping

Step-2 will not require exact equality of execution-local shapes, internal ring partitioning, or temporary materialization structure. This allows multiple lowerings to compete under the same step-1 plan.

Alternatives considered:
- Continue exact matching on current buffer shape and shard state.
  Rejected because it makes step-2 unable to explore different lowerings for one logical plan.

### 6. GEMM is the reference implementation; FFN follows the same model

The first implementation target is GEMM because its step-1 plan is a single-kernel boundary mapping problem with no graph edge transitions. Once the GEMM model is working, FFN step-1 can be expressed as:

- operator-local weight layouts
- operator input/output boundary layouts
- explicit edge reshard transitions between operators

This reduces risk by validating the representation split and cost model on the smallest useful case before applying them to FFN.

Alternatives considered:
- Redesign GEMM and FFN together in one implementation pass.
  Rejected because it increases debugging surface and makes cost-model validation harder.

## Risks / Trade-offs

- [Logical/execution metadata drift] -> Keep one canonical boundary-layout matcher and require lowering passes to mutate only execution-local fields.
- [Step-1 cost model may mis-rank plans] -> Use a conservative overlap policy in the first version and validate ranking on small GEMM cases before expanding.
- [Theoretical plans may not yet be supported by step-2] -> Treat unsupported step-2 outcomes as implementation gaps, not step-1 invalidity, and record them in tests.
- [Incremental migration complexity] -> Land GEMM-first interfaces and helpers before moving FFN two-step search to the new model.
- [Existing tests encode current exact-match behavior] -> Replace exact lowered-layout assertions with boundary-layout assertions where appropriate.

## Migration Plan

1. Add logical-layout plan data structures and boundary-layout matching helpers without removing existing search entry points.
2. Introduce GEMM step-1 layout enumeration and obligation-based cost estimation under fixed topology.
3. Add GEMM step-2 boundary-constrained rerun entry points and verify multiple lowerings can satisfy one logical plan.
4. Update GEMM tests and example flows to use the new two-step interface.
5. Extend the same boundary-plan model to FFN search, including explicit edge reshard costing.
6. Retire or simplify older exact-layout grouping paths once the new planner covers their use cases.

Rollback strategy:
- Keep the legacy search entry points available until GEMM and FFN parity is demonstrated.
- If ranking or matching regressions appear, fall back to the legacy single-stage candidate enumeration path while preserving new metadata fields.

## Open Questions

- Which theoretically legal GEMM layout combinations should be marked unsupported in the first step-2 release, and how should those failures surface in planner output?
- Should the first GEMM step-1 estimator treat all input-side communication as blocking, or introduce a bounded overlap factor from the start?
- How much of the current `TensorMappingConstraints` config format should be reused directly for exact step-1 plan serialization versus remaining a template-only input format?
