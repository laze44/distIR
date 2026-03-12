## Why

The current FFN step-1 planner collapses graph planning to the shared activation nodes `L_in` and `L_mid`, then chooses one per-operator argmin candidate underneath each pair. This hides legal combinations of `down.C` and the three weight layouts from step-1 ranking, underreports the true plan count, and conflicts with the intended two-step contract where graph plans should preserve operator boundary semantics before step-2 lowering.

## What Changes

- Change FFN step-1 plan identity so it explicitly preserves one logical boundary class for each operator: `gate`, `up`, and `down`.
- Keep `L_in` and `L_mid` as shared graph nodes, but project `L_out`, `W_gate`, `W_up`, and `W_down` from the selected operator boundary classes instead of deriving them after a hidden argmin.
- Expand step-1 statistics and reporting so plan counts reflect shared-layout counts, operator-boundary-class counts, and projected layout counts.
- Update FFN two-step tests and example summaries to validate the expanded step-1 state and the `down.C == L_out` boundary rule.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `logical-layout-plan-search`: FFN step-1 plans must preserve operator-local logical boundary classes and report projected `L_out` and weight layouts as part of the visible plan state.

## Impact

- Affects FFN step-1 planning and reporting in `mercury/search/ffn_two_step_search.py` and `example_ffn_ir.py`.
- Requires test updates in FFN two-step and example coverage.
- Does not change GEMM semantics, lowering IR, or code generation in this change.
