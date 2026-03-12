## Context

The current FFN step-1 implementation enumerates only the shared activation layouts `L_in` and `L_mid`, then picks the cheapest `gate`, `up`, and `down` candidate for each `(L_in, L_mid)` pair. The selected candidate's `layout_b` and `layout_c` are written back into the plan as `W_*` and `L_out`, but those fields are not part of step-1 state identity.

This creates two problems:

- The visible step-1 plan says it contains `L_out`, `W_gate`, `W_up`, and `W_down`, but these are post-hoc projections of hidden argmin choices rather than explicit search dimensions.
- Top-k retention happens after the hidden per-operator argmin, so step-2 never sees many legal combinations of operator boundary layouts.

For the concrete FFN case discussed here (`batch=1`, `seq_len=256`, `d_model=4096`, `d_ffn=4096`, mesh `(1, 4)`), the current implementation reports:

- `|L_in| = 12`
- `|L_mid| = 12`
- `total_plan_count = 144`

But the candidate pool actually induces:

- `|G| = 12` unique gate boundary classes
- `|U| = 12` unique up boundary classes
- `|D| = 12` unique down boundary classes

where each boundary class is one exact logical `(A, B, C)` triple. The intended full step-1 plan space is therefore:

```text
|P| = |L_in| * |L_mid| * |G| * |U| * |D|
    = 12 * 12 * 12 * 12 * 12
    = 248,832
```

This change adopts the user-confirmed boundary semantics:

```text
L_out is the down output boundary itself.
Therefore every valid plan must satisfy down.C == L_out.
```

That means `L_out` must be explicit in the user-visible plan, but it should be projected from the chosen `down` boundary class rather than treated as an independent free variable.

## Goals / Non-Goals

**Goals:**
- Define FFN step-1 plan identity using shared graph layouts plus explicit per-operator logical boundary classes.
- Preserve `L_in` and `L_mid` as graph-level coordination nodes while exposing `L_out`, `W_gate`, `W_up`, and `W_down` in the selected plan.
- Keep step-2 keyed by exact logical operator boundaries, not by hidden candidate selection.
- Expand step-1 statistics so they describe both the internal search state and the projected user-visible state.
- Keep the design compatible with the existing `down.C == L_out` contract.

**Non-Goals:**
- Change GEMM step-1 semantics in this change.
- Introduce a new outer-graph `L_out` reshard edge. This design explicitly disallows `down.C != L_out`.
- Redesign the step-2 lowering filter beyond feeding it the preserved operator boundary class.
- Model new communication formulas beyond the current operator-local estimate plus explicit edge reshard costs.

## Decisions

### 1. Canonical step-1 identity is `(L_in, L_mid, G_cls, U_cls, D_cls)`

The internal FFN step-1 state will no longer be represented as only the shared activation pair `(L_in, L_mid)`. The canonical state will be:

```text
P = (L_in, L_mid, G_cls, U_cls, D_cls)
```

where:

- `L_in` is the shared FFN input layout
- `L_mid` is the shared intermediate activation layout used to stitch `gate`, `up`, and `down`
- `G_cls` is one exact logical boundary class for `gate`
- `U_cls` is one exact logical boundary class for `up`
- `D_cls` is one exact logical boundary class for `down`

Each operator boundary class stores one exact logical triple:

```text
OpBoundaryClass(op) = (A_op, B_op, C_op)
```

This is the smallest state that preserves:

- the shared graph nodes that create edge-reshard tradeoffs
- the operator-local weight/output choices that are currently hidden
- the full boundary contract required by step-2

Alternatives considered:
- Represent plans only as projected fields `(L_in, L_mid, L_out, W_gate, W_up, W_down)`.
  Rejected because projected weights and `L_out` do not preserve the full operator boundary contract. In general, multiple operator boundary classes may share the same projected `B` or `C`.
- Treat `L_out` as an independent seventh search dimension.
  Rejected because the confirmed semantics require `down.C == L_out`, so an independent `L_out` axis would create invalid states that must be filtered out immediately.

### 2. `L_out` and `W_*` remain user-visible, but are projections rather than free axes

The visible `FFNLayoutPlan` should still expose:

- `L_in`
- `L_mid`
- `L_out`
- `W_gate`
- `W_up`
- `W_down`

but the projection rules become explicit:

```text
W_gate = B(G_cls)
W_up   = B(U_cls)
W_down = B(D_cls)
L_out  = C(D_cls)
```

No extra edge is introduced from `down.C` to `L_out`; they are the same logical boundary.

This preserves the user-facing summary format while making it honest about where those fields come from.

Alternatives considered:
- Remove `activation_layouts` and `weight_layouts` from the plan entirely and expose only boundary classes.
  Rejected because examples, debug summaries, and human inspection still benefit from explicit projected layout fields.

### 3. Introduce explicit boundary-class data structures and expand step-1 stats

The step-1 implementation should group candidate programs by exact logical `(A, B, C)` boundary signature before plan enumeration.

The design introduces one new internal concept:

```python
@dataclass(frozen=True)
class FFNOperatorBoundaryClass:
    operator_name: str
    layout_a: LogicalBoundaryLayoutSignature
    layout_b: LogicalBoundaryLayoutSignature
    layout_c: LogicalBoundaryLayoutSignature
    candidate_count: int
    representative_candidate_ids: Tuple[int, ...]
    best_step1_exec_time_ms: float
```

Notes:
- `representative_candidate_ids` exists only to preserve traceability back to raw candidate pools.
- `best_step1_exec_time_ms` is the minimum step-1 operator cost over all candidates in the class.
- Step-2 is still free to rerun and choose any lowering satisfying the class boundary constraints.

`FFNLayoutPlan` should retain the current projected dictionaries, but add one explicit field:

```python
boundary_classes: Dict[str, FFNOperatorBoundaryClass]
```

`FFNStep1LayoutStats` should be expanded so it reports both internal and projected counts:

```python
@dataclass(frozen=True)
class FFNStep1LayoutStats:
    unique_l_in_count: int
    unique_l_mid_count: int
    gate_boundary_class_count: int
    up_boundary_class_count: int
    down_boundary_class_count: int
    projected_l_out_count: int
    projected_w_gate_count: int
    projected_w_up_count: int
    projected_w_down_count: int
    total_plan_count: int
```

Projection counts are defined as:

```text
projected_l_out_count  = |{ C(d) | d in D }|
projected_w_gate_count = |{ B(g) | g in G }|
projected_w_up_count   = |{ B(u) | u in U }|
projected_w_down_count = |{ B(d) | d in D }|
```

Alternatives considered:
- Keep using raw candidate counts in step-1 stats.
  Rejected because raw candidate multiplicity reflects step-2 lowering diversity, not step-1 logical layout diversity.
- Replace `operator_layouts` immediately.
  Rejected because a migration period is simpler if plans expose both `boundary_classes` and the current projected `operator_layouts` view.

### 4. Step-1 enumeration is a full Cartesian product over shared layouts and operator boundary classes

After class grouping, step-1 enumerates:

```text
P_all = L_in_set × L_mid_set × G × U × D
```

where:

- `L_in_set = { gate.A classes } ∪ { up.A classes }`
- `L_mid_set = { gate.C classes } ∪ { up.C classes } ∪ { down.A classes }`
- `G`, `U`, `D` are the unique operator boundary-class sets

For one plan `p = (L_in, L_mid, g, u, d)`, the step-1 score is:

```text
T_step1(p) =
    T_gate(g)
  + T_up(u)
  + T_down(d)
  + T_edge(L_in, A(g))
  + T_edge(L_in, A(u))
  + T_edge(C(g), L_mid)
  + T_edge(C(u), L_mid)
  + T_edge(L_mid, A(d))
```

with:

```text
T_gate(g) = best_step1_exec_time_ms(g)
T_up(u)   = best_step1_exec_time_ms(u)
T_down(d) = best_step1_exec_time_ms(d)
```

There is intentionally no:

```text
T_edge(C(d), L_out)
```

because `L_out = C(d)` by definition.

For the current measured FFN case, the new total becomes:

```text
12 * 12 * 12 * 12 * 12 = 248,832
```

instead of the current:

```text
12 * 12 = 144
```

Alternatives considered:
- Continue performing a hidden argmin over all candidates for each `(L_in, L_mid)` pair.
  Rejected because it discards operator-boundary combinations before top-k retention.
- Enumerate raw candidate triples instead of grouped boundary classes.
  Rejected because it would inflate the search space with step-2-only multiplicity. In the measured case it would scale from `248,832` logical plans to billions of candidate-level combinations.

### 5. Step-2 should consume the selected boundary class directly

The current step-2 memoization key already uses exact logical operator boundary constraints. That contract should remain unchanged conceptually:

```text
constraints(op) = {
    "A": A(boundary_class(op)),
    "B": B(boundary_class(op)),
    "C": C(boundary_class(op)),
}
```

The selected plan passed into step-2 should therefore come from the preserved boundary class, not from an earlier hidden per-operator argmin.

This keeps step-2 semantics aligned with the real step-1 search state:

- step-1 chooses a boundary class
- step-2 reruns lowering under that exact boundary contract

Alternatives considered:
- Keep using the cheapest hidden step-1 candidate as the source of step-2 constraints.
  Rejected because that reintroduces the current loss of search-space visibility.

### 6. Reporting should separate internal search-state counts from projected plan fields

`summary.txt` and debugging output should no longer imply that step-1 counted only two layout dimensions. The report should distinguish:

```text
Step-1 Layout Plan Counts:
  unique L_in: ...
  unique L_mid: ...
  gate boundary classes: ...
  up boundary classes: ...
  down boundary classes: ...
  projected unique L_out: ...
  projected unique W_gate: ...
  projected unique W_up: ...
  projected unique W_down: ...
  total evaluated plans: ...
```

This makes the step-1 report match the actual plan identity while still exposing the user-visible layout fields they care about.

Alternatives considered:
- Report only boundary-class counts and remove projected layout counts.
  Rejected because projected counts remain useful for understanding the effective variety of visible FFN boundary layouts.

## Risks / Trade-offs

- [State space grows from 144 to 248,832 in the measured FFN case] -> Group by exact logical boundary class before enumeration so step-1 scales with logical plans rather than raw lowering candidates.
- [Projected fields can appear redundant with boundary classes] -> Keep a single canonical source of truth (`boundary_classes`) and define projections explicitly in one place.
- [Some future workloads may have multiple boundary classes that share the same `W_*` or `L_out`] -> Preserve boundary classes in plan identity and use projected counts only for reporting.
- [Existing tests currently encode the old `|L_in| * |L_mid|` formula] -> Replace those assertions with formulas derived from grouped boundary classes and add synthetic cases that break the old equivalence.

## Migration Plan

1. Introduce candidate grouping by exact logical `(A, B, C)` signature for `gate`, `up`, and `down`.
2. Expand the step-1 plan object so it stores canonical `boundary_classes` plus projected `activation_layouts` and `weight_layouts`.
3. Change step-1 enumeration from hidden per-operator argmin over `(L_in, L_mid)` to full enumeration over `(L_in, L_mid, G_cls, U_cls, D_cls)`.
4. Update step-1 stats and example summaries to report both boundary-class counts and projected layout counts.
5. Keep step-2 filtering keyed by exact logical operator boundaries, but source those boundaries from the preserved class in the selected plan.
6. After parity is confirmed, simplify any debug output or helper code that still assumes the old two-axis step-1 state.

Rollback strategy:
- Preserve the old stats and summary formatting behind a temporary compatibility path until the new tests are green.
- If the expanded enumeration proves too expensive in practice, add pruning on grouped boundary classes rather than returning to raw hidden argmin behavior.

## Test Plan

The change should update and extend FFN step-1 coverage in three layers.

### A. Update existing FFN two-step statistics tests

File:
- `tests/test_ffn_two_step_search.py`

Changes:
- Replace the current assertion:

```text
total_plan_count == |L_in| * |L_mid|
```

with:

```text
total_plan_count == |L_in| * |L_mid| * |G| * |U| * |D|
```

- Compute `|G|`, `|U|`, and `|D|` from unique logical `(A, B, C)` signatures in the candidate pools.
- Add assertions for:
  - `gate_boundary_class_count`
  - `up_boundary_class_count`
  - `down_boundary_class_count`
  - `projected_l_out_count`
  - `projected_w_gate_count`
  - `projected_w_up_count`
  - `projected_w_down_count`

### B. Preserve explicit `down.C == L_out` semantics

File:
- `tests/test_ffn_two_step_search.py`

Add coverage that:
- `selected_plan.activation_layouts["L_out"] == selected_plan.boundary_classes["down"].layout_c`
- no extra step-1 edge-cost term is introduced for `down.C -> L_out`
- step-2 still receives the same exact logical `A/B/C` constraints as the selected boundary class

### C. Add a synthetic regression where projected weights are not enough

File:
- `tests/test_ffn_two_step_search.py`

Add one mock-program test where two candidates share the same projected `W_gate` but have different `A` and/or `C` logical layouts. The test should assert that:

- they form two distinct `gate` boundary classes
- step-1 plan count reflects both classes
- top-k retention can keep both plans if their scores differ only via edge costs

This test is important because the real measured FFN case currently has a one-to-one mapping between `B` and `(A, C)`, which would otherwise hide the need for preserving full boundary classes.

### D. Update example summary expectations

File:
- `tests/test_example_ffn_ir.py`

Changes:
- replace the current `summary.txt` checks that only require `unique L_in` and `unique L_mid`
- assert the presence of:
  - `gate boundary classes:`
  - `up boundary classes:`
  - `down boundary classes:`
  - `projected unique L_out:`
  - `projected unique W_gate:`
  - `projected unique W_up:`
  - `projected unique W_down:`

### E. Keep current mismatch-with-edge-transition behavior

File:
- `tests/test_ffn_two_step_search.py`

The existing edge-transition test with one candidate per operator should still pass with the same total:

```text
|L_in| = 2
|L_mid| = 2
|G| = |U| = |D| = 1
total_plan_count = 4
```

This guards the migration from accidentally changing the meaning of `L_in`/`L_mid` edge reshards while expanding the hidden operator state.

## Open Questions

- Should the public example summary show only counts, or also print the selected boundary-class signatures explicitly in addition to `W_*` and `L_out`?
- Do we want one generic boundary-class helper reusable by FFN and any future graph searches, or keep the first implementation FFN-specific?
- If future outer-graph integration fixes `L_out`, should FFN step-1 treat it as an input constraint that restricts `D_cls`, or expose a separate constrained enumeration mode?
