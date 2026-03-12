## Context

`distIR` currently has two distinct communication behaviors for managed reductions. The `shift` primitive can rewrite loads/stores/reductions into `RingComm` events that codegen lowers as pipelined send/recv with explicit `commit()` and later `wait()`. In contrast, a managed reduction that remains collective-based is represented only as `ReduceOp.shard_dim`, and codegen emits the collective synchronously when the reduction buffer is later loaded.

This split is sufficient for attention-style ring overlap, but it does not represent tensor-parallel GEMM schedules where one output tile's collective reduction runs concurrently with compute on later tiles. The current estimator has the same blind spot: collective reductions are modeled as blocking, and overlap is only approximated for some ring load/store events.

The change must preserve the existing ring path because attention search already relies on it, while extending the compiler so managed reductions can also use an async collective overlap strategy.

## Goals / Non-Goals

**Goals:**
- Preserve the current `shift`/ring behavior and search coverage.
- Add an explicit async collective overlap path for managed reductions that currently lower to blocking `all_reduce`.
- Support double-buffered tile pipelines so compute and collective communication can progress on separate streams.
- Extend search so it can choose among blocking collective, ring overlap, and async collective overlap for the same reduction.
- Update the estimator so async overlap candidates are ranked using tile-level pipeline costs rather than fully blocking collective costs.

**Non-Goals:**
- Replacing the existing ring lowering used by attention schedules.
- Building a fully general stream scheduler for arbitrary IR nodes in this change.
- Supporting every collective pattern at once; the initial target is managed reductions that lower to `all_reduce`.
- Reworking the entire compute search space beyond the tiling and ordering needed to expose a legal overlap axis.

## Decisions

### 1. Keep ring overlap and async collective overlap as parallel strategies

The compiler will continue to treat `shift`/ring as a first-class communication family. Async collective overlap will be added as a separate managed-reduction strategy rather than replacing ring.

Why:
- Existing attention paths already depend on `shift` semantics and ring codegen.
- Ring and async `all_reduce` serve different hardware and schedule niches.
- This makes search responsible for choosing the strategy instead of baking in a global preference.

Alternatives considered:
- Unifying ring and collective under one opaque communication annotation. Rejected because it hides important lowering differences and would make the estimator less interpretable.
- Replacing ring with async collective. Rejected because it would regress schedules that depend on data circulation rather than collective reduction.

### 2. Introduce explicit async collective lifecycle nodes in IR/lowering

Managed reductions that use async overlap will no longer be represented only by `ReduceOp.shard_dim` plus an implicit blocking collective at reduction-buffer load time. The lowering path should instead materialize explicit communication lifecycle events, conceptually:

- start async collective on a completed tile buffer
- continue compute on a different tile buffer
- wait on the outstanding collective before buffer reuse or final writeback

The exact IR names can be decided during implementation, but the design requires explicit start/wait placement rather than a single ad-hoc collective trigger.

Why:
- Wait placement is the key scheduling decision; it cannot be expressed cleanly as a property on `ReduceOp`.
- Explicit lifecycle nodes allow the estimator and codegen to reason about in-flight collectives.
- This removes the current coupling where loading `reduce_buf` also triggers collective lowering.

Alternatives considered:
- Storing `async_op=True` and wait metadata directly on `ReduceOp`. Rejected because it still leaves collective placement implicit and tied to one producer node.
- Encoding overlap only in codegen heuristics. Rejected because search and estimation would not see the same schedule.

### 3. Use double-buffered tile pipelines as the initial overlap form

The initial async overlap strategy will require an overlap axis outside the managed reduction axis and will tile that axis to create a steady-state pipeline. For GEMM with `K`-sharded `B`, the expected first target is a tiled outer `J/N` axis:

```text
tile 0: compute -> all_reduce_start(slot 0)
tile 1: compute on slot 1 while slot 0 communicates
tile 2: wait/reuse slot 0, compute on slot 0 while slot 1 communicates
...
drain outstanding slots at the end
```

Two slots are the default stage count in this change.

Why:
- Double buffering is the minimum structure that enables real overlap without uncontrolled memory growth.
- It matches the user-observed GEMM case and keeps the initial search expansion bounded.
- It gives codegen a concrete stream and buffer ownership rule.

Alternatives considered:
- Single buffer with deferred wait. Rejected because correctness forces an early wait before reuse, eliminating overlap.
- N-way buffering from the start. Rejected because the search and memory model would grow too quickly before the basic async path is validated.

### 4. Expand search with a managed-reduction communication choice plus overlap parameters

Search will add a communication decision for each eligible managed reduction:

- `blocking_collective`
- `ring_overlap`
- `async_collective_overlap`

For `async_collective_overlap`, search will additionally enumerate:
- overlap axis, chosen from outer spatial axes that dominate the reduction-buffer consumption
- tile factor on that axis
- stage count, fixed to 2 in the initial version

Eligibility rules should keep the search bounded:
- collective participant count must be greater than 1
- there must be at least two tiles on the overlap axis
- available memory must fit double-buffered reduction storage
- the async path must not invalidate existing tensor-layout constraints

Why:
- The current search only sees axis-to-mesh assignment plus `shift`; it does not treat collective overlap as a selectable schedule family.
- Bounding the overlap choices prevents exponential growth while still exposing the GEMM opportunity.

Alternatives considered:
- Always enabling async overlap whenever a collective reduce exists. Rejected because some kernels will not have a profitable or legal overlap axis.
- Exhaustively enumerating any outer axis and arbitrary buffer count. Rejected because the search space would blow up before the estimator is ready.

### 5. Replace blocking collective accounting with a tile-pipeline cost model for async overlap

The estimator will keep separate models for:
- blocking collective reductions
- ring overlap
- async collective overlap

For async collective overlap, total time will be estimated as:
- warmup for the first tile(s)
- steady-state `num_tiles - 1` iterations dominated by `max(tile_compute, tile_collective)`
- drain time for outstanding collectives

The model must also include:
- launch/latency terms per tile
- double-buffer memory footprint
- reduced overlap when the final consumer forces an early wait

Why:
- The current model marks collective reductions as blocking and therefore cannot rank async overlap candidates correctly.
- A pipeline model is sufficient for search ranking without requiring full runtime profiling.

Alternatives considered:
- Reusing the existing one-shot overlap heuristic. Rejected because it does not model per-tile warmup/drain or wait-before-reuse behavior.
- Deferring estimation changes until after codegen support lands. Rejected because search would be biased against the new strategy.

## Risks / Trade-offs

- [Search-space growth] → Mitigation: constrain async overlap to eligible managed reductions, fixed double buffering, and outer axes with at least two legal tiles.
- [IR complexity from explicit start/wait nodes] → Mitigation: keep the new lifecycle limited to managed-reduction collectives instead of generalizing all communication at once.
- [Estimator drift from real runtime behavior] → Mitigation: add focused tests and compare against generated schedule structure; leave room for later calibration against profiling.
- [Memory overhead from double buffering] → Mitigation: integrate buffer-stage footprint into pruning so impossible candidates never reach codegen.
- [Behavioral regression for attention ring schedules] → Mitigation: preserve the existing ring path unchanged and treat async collective overlap as an additive option.

## Migration Plan

This change is additive. Existing programs, ring schedules, and blocking collective schedules remain valid. Implementation should proceed in stages:

1. Add explicit overlap-aware representation for async managed reductions.
2. Teach search to generate async overlap candidates behind clear legality checks.
3. Extend codegen to emit separate communication and compute streams with double-buffered reduction buffers.
4. Update the estimator and memory checks.
5. Add regression tests for search coverage, generated ordering, and estimate behavior.

If the async path is incorrect or underperforms, rollback consists of disabling the `async_collective_overlap` search branch while leaving the existing ring and blocking collective paths intact.

## Open Questions

- Should the first implementation support only `all_reduce`, or also reserve IR hooks for future `reduce_scatter + all_gather` decomposition?
- How should overlap-axis eligibility be defined for non-GEMM managed reductions that still use the same IR pattern?
- Do we want explicit stream objects in generated PyTorch code, or should the backend initially rely on async collectives plus default-stream compute ordering?
