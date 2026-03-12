## 1. Step-1 Plan Identity

- [x] 1.1 Add FFN step-1 data structures for operator boundary classes and expanded step-1 layout statistics
- [x] 1.2 Group per-operator FFN candidate programs by exact logical `(A, B, C)` boundary signature instead of using raw candidates directly in step-1
- [x] 1.3 Update FFN layout plan construction so canonical plan identity preserves `L_in`, `L_mid`, and the selected `gate`/`up`/`down` boundary classes while projecting `L_out`, `W_gate`, `W_up`, and `W_down`

## 2. Step-1 Enumeration And Ranking

- [x] 2.1 Replace the hidden per-operator argmin over `(L_in, L_mid)` with full enumeration over `(L_in, L_mid, G_cls, U_cls, D_cls)`
- [x] 2.2 Compute step-1 operator costs from grouped boundary classes and keep explicit edge-cost terms only for `L_in`/`L_mid` graph connections
- [x] 2.3 Enforce the `down.C == L_out` rule in plan construction and ensure no separate `down.C -> L_out` edge cost is created

## 3. Step-2 And Reporting Integration

- [x] 3.1 Feed step-2 logical layout constraints from the selected operator boundary classes rather than from hidden step-1 candidate choices
- [x] 3.2 Update `example_ffn_ir.py` summary generation to report boundary-class counts, projected layout counts, and the expanded visible plan state
- [x] 3.3 Keep plan ordering and memoization deterministic after introducing grouped boundary classes and projected layout fields

## 4. Verification

- [x] 4.1 Update `tests/test_ffn_two_step_search.py` to validate canonical plan counts, projected layout counts, and `down.C == L_out`
- [x] 4.2 Add a synthetic FFN regression test where the same projected weight layout maps to multiple distinct operator boundary classes
- [x] 4.3 Update `tests/test_example_ffn_ir.py` to check the new summary fields and step-1 reporting format
