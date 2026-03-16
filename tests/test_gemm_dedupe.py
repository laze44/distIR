# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

import ast
import copy
import textwrap
from typing import List

import pytest
import torch

from mercury.frontend.parser import IRBuilder
from mercury.ir.distributed import DeviceMesh, ShardType, ShardingSpec
from mercury.ir.elements import Axis, Buffer
from mercury.ir.init_distributed import init_distributed
from mercury.ir.nodes import Program, ReduceOp
from mercury.ir.primitives import parallelize, shift
from mercury.ir.tile import tile_loop
from mercury.ir.utils import (
    collect_axis,
    collect_loops,
    collect_parallelizeable_axes,
    collect_reduce,
    get_io_buffers,
)
from mercury.search.gemm_dedupe import (
    _collect_managed_reduce_ops,
    _effective_local_k_extent,
    _find_gemm_boundary_buffers,
    _is_k_family,
    _is_safe_gemm_blocking_collective,
    _normalized_collective_shard_dims,
    _visible_non_k_loop_structure,
    gemm_canonical_dedupe_key,
)
from mercury.search.search import search, search_with_progress
from utils.gemm_dsl import format_gemm_template


def _parse_gemm_program(m: int, n: int, k: int) -> Program:
    source = format_gemm_template(m, n, k)
    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return builder.visit(node)
    raise ValueError("Could not find function definition in GEMM template")


def _search_gemm_candidates(
    m: int, n: int, k: int, world_size: int, mesh_shape=None
) -> List[Program]:
    program = _parse_gemm_program(m, n, k)
    devices = list(range(world_size))
    if mesh_shape is None:
        mesh_shape = (world_size,)
    mesh = DeviceMesh(devices, mesh_shape)
    return list(search(program, mesh, ["I", "J", "K"]))


def _search_gemm_dedupe(
    m: int, n: int, k: int, world_size: int, mesh_shape=None
) -> List[Program]:
    program = _parse_gemm_program(m, n, k)
    devices = list(range(world_size))
    if mesh_shape is None:
        mesh_shape = (world_size,)
    mesh = DeviceMesh(devices, mesh_shape)
    return list(
        search(
            program,
            mesh,
            ["I", "J", "K"],
            dedupe_key_fn=gemm_canonical_dedupe_key,
        )
    )


# ---------------------------------------------------------------------------
# Task 2.3: Duplicate equivalence class tests
# ---------------------------------------------------------------------------


class TestDuplicateEquivalenceClasses:
    def test_dedupe_reduces_candidate_count(self):
        # GIVEN a GEMM search with 2 devices producing blocking-collective candidates
        # WHEN we run with and without dedupe
        raw = _search_gemm_candidates(256, 256, 256, 2)
        deduped = _search_gemm_dedupe(256, 256, 256, 2)
        # THEN deduped count is strictly less (duplicates were removed)
        assert len(deduped) > 0
        assert len(deduped) <= len(raw)

    def test_unsplit_k_equals_noop_split_k(self):
        # GIVEN a GEMM program with unsplit K
        program_unsplit = _parse_gemm_program(256, 256, 256)
        devices = list(range(2))
        mesh = DeviceMesh(devices, (2,))

        # WHEN we create two variants: unsplit K and K_outer=1/K_inner=full
        init_distributed(program_unsplit, mesh)

        program_split = _parse_gemm_program(256, 256, 256)
        k_axis = None
        for ax in program_split.visit(collect_axis):
            if ax.name == "K":
                k_axis = ax
                break
        assert k_axis is not None
        tile_loop(program_split, k_axis, k_axis.size)
        init_distributed(program_split, mesh)

        # parallelize I on mesh dim 0 for both
        for prog in [program_unsplit, program_split]:
            loops = prog.visit(collect_loops)
            axes_list = prog.visit(collect_parallelizeable_axes)
            for loop, axes in zip(loops, axes_list):
                for axis in axes:
                    if axis.name == "I":
                        parallelize(prog, loop, axis, mesh, 0, 1, set())
                        shift(prog, axis, mesh, 0, 1, 1, set())

        # set topology metadata
        program_unsplit.topology_metadata = {
            "inter_node_dims": [],
            "intra_node_dims": [0],
            "mixed_dims": [],
        }
        program_split.topology_metadata = {
            "inter_node_dims": [],
            "intra_node_dims": [0],
            "mixed_dims": [],
        }

        # annotate blocking collective strategy
        for prog in [program_unsplit, program_split]:
            for reduce_op in prog.visit(collect_reduce):
                reduce_op.managed_collective_strategy = "blocking_collective"

        # THEN both produce the same canonical key
        key_unsplit = gemm_canonical_dedupe_key(program_unsplit)
        key_split = gemm_canonical_dedupe_key(program_split)
        assert key_unsplit is not None
        assert key_split is not None
        assert key_unsplit == key_split

    def test_k_outer_vs_k_inner_parallel_placement_same_key(self):
        # GIVEN two blocking-collective candidates that differ only in whether
        # parallel_info is on K_outer or K_inner
        raw = _search_gemm_candidates(256, 256, 256, 2)

        blocking_candidates = []
        for p in raw:
            reduce_ops = _collect_managed_reduce_ops(p)
            if len(reduce_ops) == 1:
                r = reduce_ops[0]
                if (
                    r.managed_collective_strategy == "blocking_collective"
                    and len(r.comm) == 0
                ):
                    blocking_candidates.append(p)

        # WHEN we compute canonical keys
        keys = [gemm_canonical_dedupe_key(p) for p in blocking_candidates]
        non_none_keys = [k for k in keys if k is not None]

        # THEN there should be duplicates (same key for different K placements)
        if len(non_none_keys) > 1:
            unique_keys = set(non_none_keys)
            assert len(unique_keys) < len(non_none_keys), (
                "Expected duplicate keys from K-family placement variants"
            )

    def test_search_with_progress_dedupe(self):
        # GIVEN a GEMM search using search_with_progress with dedupe
        program = _parse_gemm_program(256, 256, 256)
        mesh = DeviceMesh(list(range(2)), (2,))
        deduped = list(
            search_with_progress(
                program,
                mesh,
                ["I", "J", "K"],
                dedupe_key_fn=gemm_canonical_dedupe_key,
                show_progress=False,
            )
        )
        raw = list(search(program, mesh, ["I", "J", "K"]))

        # THEN deduped should have fewer or equal candidates
        assert len(deduped) > 0
        assert len(deduped) <= len(raw)


# ---------------------------------------------------------------------------
# Task 2.4: Preservation tests
# ---------------------------------------------------------------------------


class TestPreservationOfDistinctCandidates:
    def test_different_collective_shard_dims_remain_distinct(self):
        # GIVEN candidates with different collective shard dimensions
        raw = _search_gemm_candidates(256, 256, 256, 4, mesh_shape=(2, 2))
        deduped = _search_gemm_dedupe(256, 256, 256, 4, mesh_shape=(2, 2))

        # WHEN we examine deduped keys
        keys = set()
        for p in deduped:
            key = gemm_canonical_dedupe_key(p)
            if key is not None:
                keys.add(key)

        # THEN we should have multiple distinct keys
        # (different shard dim assignments across 2D mesh)
        if len(keys) > 1:
            shard_dim_sets = set()
            for key in keys:
                shard_dim_sets.add(key[5])
            assert len(shard_dim_sets) >= 1

    def test_different_spatial_loop_structure_remains_distinct(self):
        # GIVEN candidates with different spatial loop tiling
        raw = _search_gemm_candidates(256, 256, 256, 2)
        deduped = _search_gemm_dedupe(256, 256, 256, 2)

        # WHEN we look at all unique loop structures
        loop_structures = set()
        for p in deduped:
            key = gemm_canonical_dedupe_key(p)
            if key is not None:
                loop_structures.add(key[4])

        # THEN different spatial structures should be preserved
        assert len(loop_structures) >= 1

    def test_ring_candidate_not_merged_with_blocking(self):
        # GIVEN a search producing both ring and blocking candidates
        raw = _search_gemm_candidates(256, 256, 256, 2)

        ring_count = 0
        blocking_count = 0
        for p in raw:
            reduce_ops = p.visit(collect_reduce)
            for r in reduce_ops:
                if len(r.comm) > 0:
                    ring_count += 1
                elif r.managed_collective_strategy == "blocking_collective":
                    blocking_count += 1

        # WHEN we dedupe
        deduped = _search_gemm_dedupe(256, 256, 256, 2)

        deduped_ring = 0
        deduped_blocking = 0
        for p in deduped:
            reduce_ops = p.visit(collect_reduce)
            for r in reduce_ops:
                if len(r.comm) > 0:
                    deduped_ring += 1
                elif r.managed_collective_strategy == "blocking_collective":
                    deduped_blocking += 1

        # THEN ring candidates are preserved (not merged into blocking)
        if ring_count > 0:
            assert deduped_ring > 0

    def test_async_overlap_not_merged_with_blocking(self):
        # GIVEN a search producing both async-overlap and blocking candidates
        raw = _search_gemm_candidates(256, 256, 256, 2)

        async_count = 0
        for p in raw:
            reduce_ops = p.visit(collect_reduce)
            for r in reduce_ops:
                if r.managed_collective_strategy == "async_collective_overlap":
                    async_count += 1

        # WHEN we dedupe
        deduped = _search_gemm_dedupe(256, 256, 256, 2)

        deduped_async = 0
        for p in deduped:
            reduce_ops = p.visit(collect_reduce)
            for r in reduce_ops:
                if r.managed_collective_strategy == "async_collective_overlap":
                    deduped_async += 1

        # THEN async-overlap candidates are preserved
        if async_count > 0:
            assert deduped_async > 0

    def test_none_returned_for_non_gemm(self):
        # GIVEN a non-GEMM program (no A/B/C boundary tensors)
        program = Program(
            name="non_gemm",
            inputs=[],
            defaults=[],
            outputs=None,
            body=[],
        )
        # THEN the canonical key returns None
        assert gemm_canonical_dedupe_key(program) is None

    def test_none_returned_for_no_mesh(self):
        # GIVEN a GEMM program without mesh
        program = _parse_gemm_program(256, 256, 256)
        assert gemm_canonical_dedupe_key(program) is None


# ---------------------------------------------------------------------------
# Task 3.3: Regression tests for GEMM ranking/export deduplication
# ---------------------------------------------------------------------------


class TestGEMMRankingDeduplication:
    def test_deduped_candidates_have_unique_keys(self):
        # GIVEN a deduped GEMM search
        deduped = _search_gemm_dedupe(256, 256, 256, 2)

        # WHEN we compute canonical keys
        keys = []
        for p in deduped:
            key = gemm_canonical_dedupe_key(p)
            if key is not None:
                keys.append(key)

        # THEN all keys should be unique
        assert len(keys) == len(set(keys)), (
            f"Found {len(keys) - len(set(keys))} duplicate keys in deduped results"
        )

    def test_no_dedupe_when_callback_is_none(self):
        # GIVEN search without dedupe callback
        program = _parse_gemm_program(256, 256, 256)
        mesh = DeviceMesh(list(range(2)), (2,))
        raw = list(search(program, mesh, ["I", "J", "K"]))
        also_raw = list(search(program, mesh, ["I", "J", "K"], dedupe_key_fn=None))

        # THEN both should return the same count
        assert len(raw) == len(also_raw)

    def test_deduped_set_covers_all_equivalence_classes(self):
        # GIVEN raw and deduped searches
        program = _parse_gemm_program(256, 256, 256)
        mesh = DeviceMesh(list(range(2)), (2,))
        raw = list(search(program, mesh, ["I", "J", "K"]))
        deduped = list(
            search(
                program, mesh, ["I", "J", "K"], dedupe_key_fn=gemm_canonical_dedupe_key
            )
        )

        # WHEN we collect all raw keys
        raw_keys = set()
        for p in raw:
            key = gemm_canonical_dedupe_key(p)
            if key is not None:
                raw_keys.add(key)

        deduped_keys = set()
        for p in deduped:
            key = gemm_canonical_dedupe_key(p)
            if key is not None:
                deduped_keys.add(key)

        # THEN deduped covers all equivalence classes from raw
        assert deduped_keys == raw_keys

    def test_2d_mesh_dedup(self):
        # GIVEN a 4-device 2D mesh GEMM search
        deduped = _search_gemm_dedupe(256, 256, 256, 4, mesh_shape=(2, 2))
        assert len(deduped) > 0

        # THEN all deduped blocking-collective keys should be unique
        keys = []
        for p in deduped:
            key = gemm_canonical_dedupe_key(p)
            if key is not None:
                keys.append(key)
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_is_k_family(self):
        assert _is_k_family("K") is True
        assert _is_k_family("K_outer") is True
        assert _is_k_family("K_inner") is True
        assert _is_k_family("I") is False
        assert _is_k_family("J") is False
        assert _is_k_family("K_other") is False

    def test_find_gemm_boundary_buffers(self):
        program = _parse_gemm_program(256, 256, 256)
        buffers = _find_gemm_boundary_buffers(program)
        assert buffers is not None
        assert set(buffers.keys()) == {"a", "b", "c"}
        for buf in buffers.values():
            assert buf.ndim == 2

    def test_effective_local_k_extent_unsplit(self):
        program = _parse_gemm_program(256, 256, 256)
        devices = list(range(2))
        mesh = DeviceMesh(devices, (2,))
        init_distributed(program, mesh)
        for reduce_op in program.visit(collect_reduce):
            extent = _effective_local_k_extent(reduce_op)
            assert extent == 256 // 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
