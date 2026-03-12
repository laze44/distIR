## Context

The current two-step FFN search stack splits graph planning and kernel lowering, but its executable model still assumes step-2 chooses isolated operator kernels. Step-1 can already expose `L_mid -> down.A` as an explicit edge reshard cost, yet step-2 has no way to own, lower, or export that obligation as part of a runnable segment. This creates a mismatch between graph-level plan semantics, step-2 selection, and the files exported for inspection.

The revised `docs/changes/two_step_search.md` establishes a clearer contract: step-1 fixes logical boundary obligations, while step-2 decides which obligations materialize as standalone boundaries and which are satisfied inside a segment. For FFN, the first supported fusion target is consumer-side ownership of `L_mid -> down.A` by the `silu_and_mul + down_gemm` chain.

Constraints:
- Existing GEMM/FFN step-2 search uses operator-local `Program` filtering keyed by logical A/B/C boundaries.
- Existing result export writes per-operator rank-1 candidates, not the selected two-step result.
- Existing backend communication support is strongest for operator-internal ring/all-reduce patterns; graph-edge reshard lowering needs an explicit ownership model before codegen can be made coherent.

## Goals / Non-Goals

**Goals:**
- Preserve step-1 as the owner of logical layout-plan semantics and explicit edge reshard obligations.
- Extend step-2 to search segment-level lowerings in addition to isolated operators.
- Support the minimal fusion unit `layout-preserving chain + edge + consumer`.
- Make FFN consumer-side fusion the first-class path for `silu_and_mul + edge(L_mid -> down.A) + down_gemm`.
- Ensure selected step-2 results and exported artifacts describe the same chosen ownership and lowering result.

**Non-Goals:**
- Generalize all edge-reshard fusion patterns in one step.
- Introduce arbitrary producer-side fusion for FFN gate/up branches.
- Solve all graph orchestration and codegen concerns for every model family beyond the initial FFN case.
- Redesign step-1 search identity away from logical layouts.

## Decisions

### 1. Keep step-1 plans semantic and make step-2 own materialization decisions
Step-1 remains responsible for logical boundary layouts and explicit edge obligations only. It does not choose whether an obligation is emitted as a standalone reshard segment or absorbed into a neighboring segment.

Rationale:
- This preserves the clean separation already documented in `two_step_search.md`.
- It avoids collapsing different lowerings into different step-1 plans.
- It keeps plan identity stable across future step-2 strategies.

Alternatives considered:
- Let step-2 rewrite `gate/up` logical outputs directly to `down.A`: rejected because it mutates the step-1 semantic contract and obscures where the edge obligation comes from.
- Forbid mismatched adjacent layouts altogether: rejected because it shrinks the search space and throws away graph plans that may become optimal once fusion is allowed.

### 2. Promote the step-2 search unit from operator to segment
Step-2 should search over a mix of isolated operators and fused segments. A segment owns one or more logical obligations and returns one executable lowering candidate plus a segment-level cost estimate.

Rationale:
- Edge fusion cannot be represented cleanly if step-2 only enumerates isolated operator kernels.
- Segment ownership gives one place to attach communication placement, materialized boundaries, and export metadata.

Alternatives considered:
- Represent fused edge handling only as a post-processing rewrite after selecting operator kernels: rejected because ownership affects eligibility, cost, and export, not just final serialization.

### 3. Define `layout-preserving chain + edge + consumer` as the initial fusion primitive
The first supported segment pattern is a layout-preserving chain followed by one explicit edge reshard obligation and its consumer kernel.

Rationale:
- This pattern matches the FFN `silu_and_mul + down` case directly.
- It avoids requiring fully general graph fusion in the first implementation.
- It lets the system preserve shared logical producer outputs while still absorbing the edge obligation.

Alternatives considered:
- Full arbitrary segment fusion: rejected as too large for the initial change.
- Edge-only standalone segments only: rejected because it keeps the current semantic/executable gap.

### 4. For FFN, assign `L_mid -> down.A` to the consumer side, not to gate/up
The initial FFN implementation will model the following ownership:
- `gate`: isolated segment producing logical `L_mid`
- `up`: isolated segment producing logical `L_mid`
- `down-chain`: consumer-side segment owning `silu_and_mul`, the explicit `L_mid -> down.A` obligation, and `down_gemm`

Rationale:
- Consumer-side ownership matches the layout-preserving chain structure.
- It avoids duplicating the same reshard work on both producer branches.
- It keeps `gate.C` and `up.C` semantically equal to `L_mid`.

Alternatives considered:
- Producer-side ownership on both branches: rejected because it risks duplicated communication and muddles the shared-midpoint semantics.

### 5. Export selected step-2 artifacts from the chosen plan, not from independent operator rankings
Result export must use the selected step-2 segments/programs and include ownership metadata for explicit edge obligations.

Rationale:
- The current export path can show files that contradict the selected layout plan.
- Segment ownership needs explicit visibility in debug artifacts and summaries.

Alternatives considered:
- Keep per-operator top-k export only: rejected because it is useful for debugging but insufficient as the canonical selected result.

## Risks / Trade-offs

- [Segment search increases state space] -> Mitigation: scope the first implementation to one explicit fusion primitive and one FFN consumer-side ownership pattern.
- [Existing `Program`-based APIs may not fit segment outputs cleanly] -> Mitigation: add a segment-level result wrapper before expanding backend codegen.
- [Cost attribution can drift if fused edge time is counted both as `T_edge` and inside segment time] -> Mitigation: require step-2 ownership to consume the edge obligation and remove standalone edge time from final plan totals when fused.
- [Export and tests may still implicitly assume one file per operator] -> Mitigation: update example outputs and tests to distinguish canonical selected artifacts from auxiliary per-operator candidate dumps.

## Migration Plan

1. Introduce spec changes and design contract for logical obligations, segment ownership, and FFN consumer-side fusion.
2. Refactor step-2 result structures so selected output can represent segments and explicit edge ownership.
3. Implement FFN-specific consumer-side fusion search and final-cost recomputation.
4. Update export paths and tests to serialize the selected step-2 result coherently.
5. Keep legacy per-operator candidate dumps only as non-canonical debug output during transition.

Rollback strategy:
- Retain the current operator-local step-2 path behind a simple fallback until segment selection stabilizes.
- If fusion ownership proves too disruptive, disable segment lowering and continue using explicit edge costs without code export claims of fusion.

## Open Questions

- Should the canonical selected artifact for FFN be one segment-level file for `down-chain`, or a small manifest plus referenced operator/chain files?
- Does the initial implementation need a dedicated segment IR node, or can it use an intermediate result object that still lowers to one or more existing `Program` objects?
- Should step-2 memoization keys include edge-ownership strategy explicitly, or derive it entirely from segment shape/obligation signatures?
