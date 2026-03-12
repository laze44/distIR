## 1. Segment Planning Model

- [x] 1.1 Add step-2 data structures for explicit edge obligations, lowering ownership, and selected segment-level results
- [x] 1.2 Update FFN two-step planning APIs to distinguish logical boundary obligations from materialized segment boundaries
- [x] 1.3 Add memoization/signature helpers that treat edge-ownership strategy as part of step-2 segment selection

## 2. FFN Consumer-Side Fusion Search

- [x] 2.1 Extend FFN step-2 search to evaluate isolated operator lowerings and consumer-side `layout-preserving chain + edge + consumer` segment lowerings
- [x] 2.2 Implement the initial FFN ownership rule for `silu_and_mul + edge(L_mid -> down.A) + down_gemm`
- [x] 2.3 Recompute final step-2 plan cost so fused edge obligations are charged inside segment cost and not double-counted as standalone edge cost

## 3. Result Export And Introspection

- [x] 3.1 Update selected-result export to write the actual chosen step-2 segments/programs instead of unrelated per-operator rank-1 candidates
- [x] 3.2 Add summary/debug output that records explicit edge obligations and their final lowering ownership
- [x] 3.3 Keep auxiliary per-operator candidate dumps only as non-canonical debug artifacts with labels that distinguish them from the selected result

## 4. Validation

- [x] 4.1 Add or update unit tests for segment ownership, FFN consumer-side fusion eligibility, and fused-edge cost attribution
- [x] 4.2 Add or update example/output tests to verify exported selected artifacts match the chosen layout plan and ownership metadata
- [x] 4.3 Sync any implementation-facing documentation and comments with the new logical-boundary versus materialized-boundary contract
