# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

import ast
import textwrap
from typing import List, Set, Tuple

import pytest

from mercury.frontend.parser import IRBuilder
from mercury.ir.distributed import DeviceMesh
from mercury.ir.nodes import Program
from mercury.ir.utils import get_io_buffers
from mercury.search.gemm_dedupe import gemm_canonical_dedupe_key
from mercury.search.search import enumerate_mesh_shapes, search, search_with_progress
from mercury.search.topology_policy import (
    DomainSpec,
    FlatMeshShapePolicy,
    MeshShapePolicy,
    TopologySpec,
    _factorize_rank_limited,
    _factorize_single_dim,
    _ordered_divisor_pairs,
    compute_buffer_logical_shard_factors,
    compute_program_logical_shard_factors,
    enumerate_domain_shapes,
    enumerate_topology_mesh_shapes,
    LogicalShardFactors,
    make_gemm_flat_mesh_shape_policy,
    make_gemm_mesh_shape_policy,
    make_gemm_topology_spec,
    topology_metadata_for_shape,
)


class TestOrderedDivisorPairs:
    def test_prime(self):
        assert _ordered_divisor_pairs(7) == [(7, 1)]

    def test_small_composite(self):
        pairs = _ordered_divisor_pairs(4)
        assert pairs == [(4, 1), (2, 2)]

    def test_sixteen(self):
        pairs = _ordered_divisor_pairs(16)
        assert pairs == [(16, 1), (8, 2), (4, 4)]

    def test_one(self):
        assert _ordered_divisor_pairs(1) == [(1, 1)]


class TestFactorizeSingleDim:
    def test_basic(self):
        assert _factorize_single_dim(4) == [(4,)]
        assert _factorize_single_dim(1) == [(1,)]

    def test_invalid(self):
        with pytest.raises(ValueError):
            _factorize_single_dim(0)


class TestFactorizeRankLimited:
    def test_size_4_max2(self):
        result = _factorize_rank_limited(4, max_virtual_dims=2)
        assert (4,) in result
        assert (2, 2) in result
        assert len(result) == 2

    def test_size_16_max2(self):
        result = _factorize_rank_limited(16, max_virtual_dims=2)
        assert (16,) in result
        assert (8, 2) in result
        assert (4, 4) in result
        assert len(result) == 3
        for t in result:
            product = 1
            for v in t:
                product *= v
            assert product == 16

    def test_no_2222_for_16(self):
        result = _factorize_rank_limited(16, max_virtual_dims=2)
        assert (2, 2, 2, 2) not in result

    def test_size_1(self):
        result = _factorize_rank_limited(1, max_virtual_dims=2)
        assert result == [(1,)]

    def test_prime(self):
        result = _factorize_rank_limited(7, max_virtual_dims=2)
        assert result == [(7,)]

    def test_max_virtual_dims_1(self):
        result = _factorize_rank_limited(16, max_virtual_dims=1)
        assert result == [(16,)]

    def test_canonicalized_descending(self):
        result = _factorize_rank_limited(12, max_virtual_dims=2)
        for t in result:
            if len(t) == 2:
                assert t[0] >= t[1], f"Not descending: {t}"

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            _factorize_rank_limited(0)

    def test_invalid_max_dims(self):
        with pytest.raises(ValueError):
            _factorize_rank_limited(4, max_virtual_dims=0)


class TestDomainSpec:
    def test_clique_defaults(self):
        d = DomainSpec(kind="clique", size=4)
        assert d.shape == (4,)

    def test_clique_with_matching_shape(self):
        d = DomainSpec(kind="clique", size=4, shape=(2, 2))
        assert d.size == 4
        assert d.shape == (2, 2)

    def test_clique_mismatched_shape(self):
        with pytest.raises(ValueError, match="product does not match"):
            DomainSpec(kind="clique", size=4, shape=(3, 2))

    def test_mesh2d(self):
        d = DomainSpec(kind="mesh2d", shape=(2, 4))
        assert d.size == 8
        assert d.shape == (2, 4)

    def test_mesh2d_no_shape(self):
        with pytest.raises(ValueError, match="2-tuple shape"):
            DomainSpec(kind="mesh2d")

    def test_unknown_kind(self):
        with pytest.raises(ValueError, match="Unknown domain kind"):
            DomainSpec(kind="ring")

    def test_unknown_policy(self):
        with pytest.raises(ValueError, match="Unknown factorization policy"):
            DomainSpec(kind="clique", size=4, factorization_policy="full")


class TestEnumerateDomainShapes:
    def test_clique_single_dim(self):
        d = DomainSpec(kind="clique", size=8, factorization_policy="single_dim")
        assert enumerate_domain_shapes(d) == [(8,)]

    def test_clique_rank_limited(self):
        d = DomainSpec(
            kind="clique",
            size=4,
            factorization_policy="rank_limited",
            max_virtual_dims=2,
        )
        result = enumerate_domain_shapes(d)
        assert (4,) in result
        assert (2, 2) in result

    def test_clique_size_1(self):
        d = DomainSpec(kind="clique", size=1)
        assert enumerate_domain_shapes(d) == [()]

    def test_mesh2d_preserves_shape(self):
        d = DomainSpec(kind="mesh2d", shape=(2, 4))
        assert enumerate_domain_shapes(d) == [(2, 4)]


class TestTopologySpec:
    def test_auto_labels_two_domains(self):
        t = TopologySpec(
            domains=[
                DomainSpec(kind="clique", size=4),
                DomainSpec(kind="clique", size=8),
            ]
        )
        assert t.domain_labels == ["inter_node", "intra_node"]

    def test_auto_labels_three_domains(self):
        t = TopologySpec(
            domains=[
                DomainSpec(kind="clique", size=2),
                DomainSpec(kind="clique", size=4),
                DomainSpec(kind="clique", size=8),
            ]
        )
        assert t.domain_labels == ["domain_0", "domain_1", "domain_2"]

    def test_custom_labels(self):
        t = TopologySpec(
            domains=[
                DomainSpec(kind="clique", size=4),
                DomainSpec(kind="clique", size=2),
            ],
            domain_labels=["fast", "slow"],
        )
        assert t.domain_labels == ["fast", "slow"]

    def test_label_count_mismatch(self):
        with pytest.raises(ValueError, match="must match"):
            TopologySpec(
                domains=[DomainSpec(kind="clique", size=4)],
                domain_labels=["a", "b"],
            )


class TestEnumerateTopologyMeshShapes:
    def test_inter16_intra1(self):
        topology = make_gemm_topology_spec(inter_node=16, intra_node=1)
        shapes = enumerate_topology_mesh_shapes(topology)
        assert (16,) in shapes
        assert (8, 2) in shapes
        assert (4, 4) in shapes
        assert (2, 2, 2, 2) not in shapes
        assert len(shapes) == 3

    def test_inter4_intra1(self):
        topology = make_gemm_topology_spec(inter_node=4, intra_node=1)
        shapes = enumerate_topology_mesh_shapes(topology)
        assert (4,) in shapes
        assert (2, 2) in shapes
        assert len(shapes) == 2

    def test_inter4_intra8_single(self):
        topology = make_gemm_topology_spec(
            inter_node=4,
            intra_node=8,
            intra_factorization="single_dim",
        )
        shapes = enumerate_topology_mesh_shapes(topology)
        assert (4, 8) in shapes
        assert (2, 2, 8) in shapes
        for shape in shapes:
            product = 1
            for v in shape:
                product *= v
            assert product == 32

    def test_inter1_intra1(self):
        topology = make_gemm_topology_spec(inter_node=1, intra_node=1)
        shapes = enumerate_topology_mesh_shapes(topology)
        assert shapes == [()]

    def test_no_cross_domain_mixing(self):
        topology = make_gemm_topology_spec(inter_node=4, intra_node=8)
        shapes = enumerate_topology_mesh_shapes(topology)
        for shape in shapes:
            product = 1
            for v in shape:
                product *= v
            assert product == 32
        assert (8, 4) not in shapes
        assert (2, 8, 2) not in shapes
        assert (2, 4, 4) not in shapes

    def test_empty_topology(self):
        topology = TopologySpec()
        assert enumerate_topology_mesh_shapes(topology) == [()]

    def test_products_match_world_size(self):
        topology = make_gemm_topology_spec(inter_node=16, intra_node=8)
        shapes = enumerate_topology_mesh_shapes(topology)
        for shape in shapes:
            product = 1
            for v in shape:
                product *= v
            assert product == 128, f"Shape {shape} product != 128"


class TestTopologyMetadataForShape:
    def test_inter16_single_dim(self):
        topology = make_gemm_topology_spec(inter_node=16, intra_node=1)
        metadata = topology_metadata_for_shape(topology, (16,))
        assert metadata["inter_node_dims"] == [0]
        assert metadata["intra_node_dims"] == []
        assert metadata["mixed_dims"] == []

    def test_inter4_two_dims(self):
        topology = make_gemm_topology_spec(inter_node=4, intra_node=1)
        metadata = topology_metadata_for_shape(topology, (2, 2))
        assert metadata["inter_node_dims"] == [0, 1]
        assert metadata["intra_node_dims"] == []
        assert metadata["mixed_dims"] == []

    def test_inter4_intra8(self):
        topology = make_gemm_topology_spec(inter_node=4, intra_node=8)
        metadata = topology_metadata_for_shape(topology, (4, 8))
        assert metadata["inter_node_dims"] == [0]
        assert metadata["intra_node_dims"] == [1]
        assert metadata["mixed_dims"] == []

    def test_inter4_factored_intra8(self):
        topology = make_gemm_topology_spec(inter_node=4, intra_node=8)
        metadata = topology_metadata_for_shape(topology, (2, 2, 8))
        assert metadata["inter_node_dims"] == [0, 1]
        assert metadata["intra_node_dims"] == [2]
        assert metadata["mixed_dims"] == []

    def test_inter1_intra1(self):
        topology = make_gemm_topology_spec(inter_node=1, intra_node=1)
        metadata = topology_metadata_for_shape(topology, ())
        assert metadata["inter_node_dims"] == []
        assert metadata["intra_node_dims"] == []
        assert metadata["mixed_dims"] == []

    def test_never_produces_mixed_dims(self):
        topology = make_gemm_topology_spec(inter_node=16, intra_node=8)
        for shape in enumerate_topology_mesh_shapes(topology):
            metadata = topology_metadata_for_shape(topology, shape)
            assert metadata["mixed_dims"] == [], f"mixed_dims non-empty for {shape}"


class TestMeshShapePolicy:
    def test_enumerate_shapes(self):
        policy = make_gemm_mesh_shape_policy(inter_node=16, intra_node=1)
        shapes = policy.enumerate_shapes()
        assert (16,) in shapes
        assert (8, 2) in shapes
        assert (4, 4) in shapes
        assert len(shapes) == 3

    def test_topology_metadata(self):
        policy = make_gemm_mesh_shape_policy(inter_node=4, intra_node=8)
        metadata = policy.topology_metadata_for_shape((2, 2, 8))
        assert metadata["inter_node_dims"] == [0, 1]
        assert metadata["intra_node_dims"] == [2]


class TestMakeGemmFactories:
    def test_default_policies(self):
        topology = make_gemm_topology_spec(inter_node=4, intra_node=2)
        inter = topology.domains[0]
        intra = topology.domains[1]
        assert inter.factorization_policy == "rank_limited"
        assert inter.max_virtual_dims == 2
        assert intra.factorization_policy == "single_dim"

    def test_custom_policies(self):
        topology = make_gemm_topology_spec(
            inter_node=8,
            intra_node=4,
            inter_factorization="single_dim",
            intra_factorization="rank_limited",
            intra_max_virtual_dims=3,
        )
        inter = topology.domains[0]
        intra = topology.domains[1]
        assert inter.factorization_policy == "single_dim"
        assert intra.factorization_policy == "rank_limited"
        assert intra.max_virtual_dims == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def _parse_gemm_program(m: int, n: int, k: int) -> Program:
    from utils.gemm_dsl import format_gemm_template

    source = format_gemm_template(m, n, k)
    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return builder.visit(node)
    raise ValueError("Could not find function definition")


def _extract_shard_specs(program: Program) -> Set[Tuple]:
    buffers = program.visit(get_io_buffers)
    specs = set()
    for buf in buffers:
        if buf.tensor in ("a", "b", "c") and buf.shard_spec is not None:
            spec_key = []
            for s in buf.shard_spec.specs:
                if isinstance(s, tuple):
                    spec_key.append(("S", tuple(int(d) for d in s[1])))
                else:
                    spec_key.append("R")
            specs.add((buf.tensor, tuple(spec_key)))
    return specs


class TestGEMMLayoutCoverage:
    """Phase 3: Verify policy preserves layout expressiveness for GEMM."""

    def test_inter4_layout_coverage(self):
        # GIVEN inter_node=4 clique with rank_limited policy
        # WHEN we search with the policy
        # THEN mesh shapes (4,) and (2,2) must both appear,
        # and layout assignments must cover (4,1), (2,2), (1,4) sharding patterns
        policy = make_gemm_mesh_shape_policy(inter_node=4, intra_node=1)
        shapes = policy.enumerate_shapes()
        assert (4,) in shapes
        assert (2, 2) in shapes

        program = _parse_gemm_program(256, 256, 256)
        devices = list(range(4))
        mesh = DeviceMesh(devices, (4,))

        candidates = list(
            search(
                program,
                mesh,
                ["I", "J", "K"],
                mesh_shape_policy=policy,
                dedupe_key_fn=gemm_canonical_dedupe_key,
            )
        )
        assert len(candidates) > 0

        all_shard_specs: Set[Tuple] = set()
        for cand in candidates:
            all_shard_specs.update(_extract_shard_specs(cand))

        a_shard_configs = {s for s in all_shard_specs if s[0] == "a"}
        b_shard_configs = {s for s in all_shard_specs if s[0] == "b"}
        c_shard_configs = {s for s in all_shard_specs if s[0] == "c"}
        assert len(a_shard_configs) > 1, "A should have multiple shard patterns"
        assert len(b_shard_configs) > 1, "B should have multiple shard patterns"
        assert len(c_shard_configs) > 1, "C should have multiple shard patterns"

    def test_inter16_no_4d_shapes(self):
        # GIVEN inter_node=16 with default GEMM policy
        # WHEN we enumerate shapes
        # THEN (2,2,2,2) must NOT appear
        policy = make_gemm_mesh_shape_policy(inter_node=16, intra_node=1)
        shapes = policy.enumerate_shapes()
        assert (2, 2, 2, 2) not in shapes
        for shape in shapes:
            assert len(shape) <= 2, f"Shape {shape} exceeds 2 dims"

    def test_inter16_candidate_count_regression(self):
        # GIVEN inter_node=16, intra_node=1
        # WHEN we search with vs without policy
        # THEN policy produces fewer candidates (no 4D explosion)
        program = _parse_gemm_program(256, 256, 256)
        devices = list(range(16))
        mesh = DeviceMesh(devices, (16,))

        policy = make_gemm_mesh_shape_policy(inter_node=16, intra_node=1)

        policy_candidates = list(
            search(
                program,
                mesh,
                ["I", "J", "K"],
                mesh_shape_policy=policy,
                dedupe_key_fn=gemm_canonical_dedupe_key,
            )
        )

        old_candidates = list(
            search(
                program,
                mesh,
                ["I", "J", "K"],
                dedupe_key_fn=gemm_canonical_dedupe_key,
            )
        )

        assert len(policy_candidates) > 0
        assert len(policy_candidates) < len(old_candidates), (
            f"Policy should reduce candidates: policy={len(policy_candidates)} "
            f"vs old={len(old_candidates)}"
        )

    def test_policy_topology_metadata_is_deterministic(self):
        # GIVEN candidates produced with policy
        # WHEN we check topology_metadata
        # THEN all candidates have non-None metadata with no mixed_dims
        policy = make_gemm_mesh_shape_policy(inter_node=4, intra_node=2)
        program = _parse_gemm_program(256, 256, 256)
        devices = list(range(8))
        mesh = DeviceMesh(devices, (4, 2))

        candidates = list(
            search(
                program,
                mesh,
                ["I", "J", "K"],
                mesh_shape_policy=policy,
            )
        )
        assert len(candidates) > 0

        for cand in candidates:
            meta = cand.topology_metadata
            assert meta is not None, "topology_metadata should be set by policy"
            assert "inter_node_dims" in meta
            assert "intra_node_dims" in meta
            assert meta["mixed_dims"] == [], (
                f"Policy should never produce mixed_dims, got {meta}"
            )

    def test_policy_preserves_multiple_layout_classes(self):
        # GIVEN policy for inter_node=4, intra_node=2
        # WHEN we search with dedupe
        # THEN at least 2 distinct dedupe keys exist (multiple layout classes)
        policy = make_gemm_mesh_shape_policy(inter_node=4, intra_node=2)
        program = _parse_gemm_program(256, 256, 256)
        devices = list(range(8))
        mesh = DeviceMesh(devices, (4, 2))

        candidates = list(
            search(
                program,
                mesh,
                ["I", "J", "K"],
                mesh_shape_policy=policy,
                dedupe_key_fn=gemm_canonical_dedupe_key,
            )
        )

        keys = set()
        for cand in candidates:
            key = gemm_canonical_dedupe_key(cand)
            if key is not None:
                keys.add(key)

        assert len(keys) >= 2, (
            f"Expected multiple distinct layout classes, got {len(keys)}"
        )

    def test_fallback_without_policy_still_works(self):
        # GIVEN search without policy (fallback path)
        # WHEN we run search
        # THEN it works exactly as before
        program = _parse_gemm_program(256, 256, 256)
        devices = list(range(4))
        mesh = DeviceMesh(devices, (4,))

        candidates_no_policy = list(search(program, mesh, ["I", "J", "K"]))
        assert len(candidates_no_policy) > 0

        for cand in candidates_no_policy:
            assert cand.topology_metadata is not None


class TestPhase4EstimatorAlignment:
    """Phase 4: Verify policy metadata is compatible with estimate.py."""

    def test_normalize_topology_metadata_passthrough(self):
        # GIVEN policy-generated metadata with all keys present
        # WHEN _normalize_topology_metadata receives a program with this metadata
        # THEN it passes through without alteration
        from mercury.search.estimate import _normalize_topology_metadata

        policy = make_gemm_mesh_shape_policy(inter_node=4, intra_node=2)
        program = _parse_gemm_program(256, 256, 256)
        devices = list(range(8))
        mesh = DeviceMesh(devices, (4, 2))

        candidates = list(
            search(program, mesh, ["I", "J", "K"], mesh_shape_policy=policy)
        )
        assert len(candidates) > 0

        for cand in candidates:
            normalized = _normalize_topology_metadata(cand, num_inter_dims=None)
            policy_meta = cand.topology_metadata
            assert normalized["inter_node_dims"] == sorted(
                policy_meta["inter_node_dims"]
            )
            assert normalized["mixed_dims"] == []

    def test_is_inter_mesh_dim_with_policy_metadata(self):
        # GIVEN policy metadata for inter_node=4, intra_node=2
        # WHEN _is_inter_mesh_dim checks each dim
        # THEN inter dims return True, intra dims return False
        from mercury.search.estimate import _is_inter_mesh_dim

        policy = make_gemm_mesh_shape_policy(inter_node=4, intra_node=2)
        for shape in policy.enumerate_shapes():
            meta = policy.topology_metadata_for_shape(shape)
            for dim in meta.get("inter_node_dims", []):
                assert _is_inter_mesh_dim(dim, meta) is True
            for dim in meta.get("intra_node_dims", []):
                assert _is_inter_mesh_dim(dim, meta) is False


class TestPhase4MappingConstraintsAlignment:
    """Phase 4: Verify policy metadata works with mapping_constraints.py."""

    def test_resolve_topology_tokens_inter(self):
        # GIVEN policy metadata for inter_node=4, intra_node=2 with shape (2, 2, 2)
        # WHEN resolving "inter_node" token
        # THEN returns the inter dims from policy metadata
        from mercury.search.mapping_constraints import (
            resolve_topology_tokens_from_metadata,
        )

        policy = make_gemm_mesh_shape_policy(inter_node=4, intra_node=2)
        meta = policy.topology_metadata_for_shape((2, 2, 2))
        resolved = resolve_topology_tokens_from_metadata(meta, ("inter_node",))
        assert resolved == tuple(sorted(meta["inter_node_dims"]))

    def test_resolve_topology_tokens_intra(self):
        # GIVEN policy metadata for inter_node=4, intra_node=2 with shape (4, 2)
        # WHEN resolving "intra_node" token
        # THEN returns the intra dims from policy metadata
        from mercury.search.mapping_constraints import (
            resolve_topology_tokens_from_metadata,
        )

        policy = make_gemm_mesh_shape_policy(inter_node=4, intra_node=2)
        meta = policy.topology_metadata_for_shape((4, 2))
        resolved = resolve_topology_tokens_from_metadata(meta, ("intra_node",))
        assert resolved == tuple(sorted(meta["intra_node_dims"]))

    def test_resolve_topology_tokens_mixed_empty(self):
        # GIVEN policy metadata (mixed_dims always empty)
        # WHEN resolving "mixed" token
        # THEN returns empty tuple
        from mercury.search.mapping_constraints import (
            resolve_topology_tokens_from_metadata,
        )

        policy = make_gemm_mesh_shape_policy(inter_node=16, intra_node=1)
        for shape in policy.enumerate_shapes():
            meta = policy.topology_metadata_for_shape(shape)
            resolved = resolve_topology_tokens_from_metadata(meta, ("mixed",))
            assert resolved == ()


class TestPhase4TwoStepAlignment:
    """Phase 4: Verify policy metadata matches _fixed_topology_metadata for default cases."""

    def test_single_dim_mesh_matches(self):
        # GIVEN mesh shape (N,) where N is the inter_node size
        # WHEN comparing _fixed_topology_metadata vs policy metadata
        # THEN for 1D meshes, _fixed_topology_metadata classifies dim 0 as intra
        #   while policy correctly classifies it as inter — this is a known
        #   divergence that will be resolved when two-step migrates to the policy
        from mercury.search.gemm_two_step_search import _fixed_topology_metadata

        for inter_node in (2, 4, 8, 16):
            policy = make_gemm_mesh_shape_policy(inter_node=inter_node, intra_node=1)
            shape = (inter_node,)
            fixed = _fixed_topology_metadata(shape)
            policy_meta = policy.topology_metadata_for_shape(shape)
            assert fixed["inter_node_dims"] == []
            assert fixed["intra_node_dims"] == [0]
            assert policy_meta["inter_node_dims"] == [0]
            assert policy_meta["intra_node_dims"] == []
            assert fixed["mixed_dims"] == policy_meta["mixed_dims"] == []

    def test_two_dim_inter_intra_matches(self):
        # GIVEN mesh shape (inter, intra)
        # WHEN comparing _fixed_topology_metadata vs policy metadata
        # THEN they produce equivalent results
        from mercury.search.gemm_two_step_search import _fixed_topology_metadata

        for inter_node, intra_node in [(4, 2), (8, 4), (2, 8)]:
            policy = make_gemm_mesh_shape_policy(
                inter_node=inter_node, intra_node=intra_node
            )
            shape = (inter_node, intra_node)
            fixed = _fixed_topology_metadata(shape)
            policy_meta = policy.topology_metadata_for_shape(shape)
            assert fixed["inter_node_dims"] == policy_meta["inter_node_dims"], (
                f"Mismatch for shape {shape}"
            )
            assert fixed["intra_node_dims"] == policy_meta["intra_node_dims"], (
                f"Mismatch for shape {shape}"
            )
            assert fixed["mixed_dims"] == policy_meta["mixed_dims"]


class TestLogicalShardFactors:
    """Unit tests for LogicalShardFactors and compute helpers."""

    def test_replicated_buffer_has_no_factors(self):
        # GIVEN a fully replicated buffer (all specs are R)
        specs = (("R", ()), ("R", ()))
        mesh_shape = (4, 2)
        metadata = {"inter_node_dims": [0], "intra_node_dims": [1], "mixed_dims": []}

        factors = compute_buffer_logical_shard_factors(specs, mesh_shape, metadata)
        assert factors.domain_factors == {}
        assert factors.to_summary() == "replicated"

    def test_single_dim_shard_on_inter_node(self):
        # GIVEN buffer sharded on dim 0 via mesh dim 0 (inter_node)
        specs = (("S", (0,)), ("R", ()))
        mesh_shape = (4, 2)
        metadata = {"inter_node_dims": [0], "intra_node_dims": [1], "mixed_dims": []}

        factors = compute_buffer_logical_shard_factors(specs, mesh_shape, metadata)
        assert factors.domain_factors == {"inter_node": (4,)}
        assert factors.total_factor("inter_node") == 4
        assert factors.total_factor("intra_node") == 1

    def test_two_dim_shard_on_inter_node(self):
        # GIVEN buffer sharded on dim 0 via mesh dim 0, dim 1 via mesh dim 1
        # WHEN mesh (8, 2) with both dims belonging to inter_node
        specs = (("S", (0,)), ("S", (1,)))
        mesh_shape = (8, 2)
        metadata = {"inter_node_dims": [0, 1], "intra_node_dims": [], "mixed_dims": []}

        factors = compute_buffer_logical_shard_factors(specs, mesh_shape, metadata)
        assert factors.domain_factors == {"inter_node": (8, 2)}
        assert factors.total_factor("inter_node") == 16

    def test_cross_domain_shard(self):
        # GIVEN buffer sharded dim 0 on inter (dim 0), dim 1 on intra (dim 1)
        specs = (("S", (0,)), ("S", (1,)))
        mesh_shape = (4, 2)
        metadata = {"inter_node_dims": [0], "intra_node_dims": [1], "mixed_dims": []}

        factors = compute_buffer_logical_shard_factors(specs, mesh_shape, metadata)
        assert factors.domain_factors == {"inter_node": (4,), "intra_node": (2,)}
        assert factors.total_factor("inter_node") == 4
        assert factors.total_factor("intra_node") == 2

    def test_multi_mesh_dim_on_one_tensor_dim(self):
        # GIVEN buffer with one tensor dim sharded across mesh dims 0 and 1
        # (both belonging to inter_node), e.g. S(0,1) on dim 0
        specs = (("S", (0, 1)), ("R", ()))
        mesh_shape = (4, 4)
        metadata = {"inter_node_dims": [0, 1], "intra_node_dims": [], "mixed_dims": []}

        factors = compute_buffer_logical_shard_factors(specs, mesh_shape, metadata)
        # Both mesh dims contribute to inter_node
        assert factors.domain_factors == {"inter_node": (4, 4)}
        assert factors.total_factor("inter_node") == 16

    def test_summary_format(self):
        lsf = LogicalShardFactors(domain_factors={"inter_node": (2, 8)})
        assert lsf.to_summary() == "inter_node=(2, 8)"

    def test_empty_factors_summary(self):
        lsf = LogicalShardFactors(domain_factors={})
        assert lsf.to_summary() == "replicated"

    def test_multi_domain_summary_sorted(self):
        lsf = LogicalShardFactors(
            domain_factors={"intra_node": (2,), "inter_node": (4,)}
        )
        summary = lsf.to_summary()
        # Should be sorted by domain label
        assert summary.index("inter_node") < summary.index("intra_node")

    def test_inter16_logical_factor_coverage(self):
        """Integration: inter_node=16 search produces diverse A tilings."""
        policy = make_gemm_mesh_shape_policy(inter_node=16, intra_node=1)
        program = _parse_gemm_program(256, 256, 256)
        devices = list(range(16))
        mesh = DeviceMesh(devices, (16,))

        candidates = list(
            search(
                program,
                mesh,
                ["I", "J", "K"],
                mesh_shape_policy=policy,
                dedupe_key_fn=gemm_canonical_dedupe_key,
            )
        )
        assert len(candidates) > 0

        a_factor_patterns: Set[Tuple[int, ...]] = set()
        for cand in candidates:
            logical_factors = compute_program_logical_shard_factors(cand)
            a_factors = logical_factors.get("A")
            if a_factors is not None:
                inter_tuple = a_factors.domain_factors.get("inter_node", ())
                if inter_tuple:
                    a_factor_patterns.add(inter_tuple)

        # With inter_node=16 and mesh shapes (16,), (8,2), (4,4),
        # we should see diverse A tiling patterns
        assert len(a_factor_patterns) >= 2, (
            f"Expected diverse A tiling patterns, got {a_factor_patterns}"
        )


class TestFlatMeshShapePolicy:
    """Tests for FlatMeshShapePolicy and the flat search path."""

    def test_flat_policy_enumerate_shapes_single_domain(self):
        policy = make_gemm_flat_mesh_shape_policy(inter_node=16, intra_node=1)
        shapes = policy.enumerate_shapes()
        assert shapes == [(16,)]

    def test_flat_policy_enumerate_shapes_two_domains(self):
        policy = make_gemm_flat_mesh_shape_policy(inter_node=4, intra_node=2)
        shapes = policy.enumerate_shapes()
        assert shapes == [(4, 2)]

    def test_flat_policy_is_flat_mesh_shape_policy(self):
        policy = make_gemm_flat_mesh_shape_policy(inter_node=16, intra_node=1)
        assert isinstance(policy, FlatMeshShapePolicy)
        assert isinstance(policy, MeshShapePolicy)

    def test_flat_topology_metadata_single_domain(self):
        policy = make_gemm_flat_mesh_shape_policy(inter_node=16, intra_node=1)
        # Virtual mesh (4, 4) -> both dims belong to inter_node
        meta = policy.topology_metadata_for_shape((4, 4))
        assert meta["inter_node_dims"] == [0, 1]
        assert meta["intra_node_dims"] == []
        assert meta["mixed_dims"] == []

    def test_flat_topology_metadata_two_domains(self):
        policy = make_gemm_flat_mesh_shape_policy(inter_node=4, intra_node=2)
        # Virtual mesh (2, 2, 2) -> first two dims are inter (2*2=4), third is intra
        meta = policy.topology_metadata_for_shape((2, 2, 2))
        assert meta["inter_node_dims"] == [0, 1]
        assert meta["intra_node_dims"] == [2]
        assert meta["mixed_dims"] == []

    def test_flat_topology_metadata_single_dim(self):
        policy = make_gemm_flat_mesh_shape_policy(inter_node=16, intra_node=1)
        meta = policy.topology_metadata_for_shape((16,))
        assert meta["inter_node_dims"] == [0]
        assert meta["intra_node_dims"] == []

    def test_flat_search_produces_candidates(self):
        policy = make_gemm_flat_mesh_shape_policy(inter_node=4, intra_node=1)
        program = _parse_gemm_program(256, 256, 256)
        devices = list(range(4))
        mesh = DeviceMesh(devices, (4,))

        candidates = list(
            search(
                program,
                mesh,
                ["I", "J", "K"],
                mesh_shape_policy=policy,
                dedupe_key_fn=gemm_canonical_dedupe_key,
            )
        )
        assert len(candidates) > 0

    def test_flat_search_all_have_topology_metadata(self):
        policy = make_gemm_flat_mesh_shape_policy(inter_node=4, intra_node=1)
        program = _parse_gemm_program(256, 256, 256)
        devices = list(range(4))
        mesh = DeviceMesh(devices, (4,))

        candidates = list(
            search(
                program,
                mesh,
                ["I", "J", "K"],
                mesh_shape_policy=policy,
            )
        )
        for cand in candidates:
            assert cand.topology_metadata is not None
            assert "inter_node_dims" in cand.topology_metadata
            assert cand.topology_metadata["mixed_dims"] == []

    def test_flat_search_with_fixed_b_constraint_inter16(self):
        """Core test: flat policy + B fixed factor=4,4 on inter_node=16."""
        from mercury.search.mapping_constraints import (
            load_tensor_mapping_constraints,
            program_satisfies_tensor_mapping_constraints,
        )

        policy = make_gemm_flat_mesh_shape_policy(inter_node=16, intra_node=1)
        program = _parse_gemm_program(256, 256, 256)
        devices = list(range(16))
        mesh = DeviceMesh(devices, (16, 1))

        constraints = load_tensor_mapping_constraints(
            "config/gemm_tensor_mapping_fixed_b_inter_node.json"
        )

        candidates = list(
            search(
                program,
                mesh,
                ["I", "J", "K"],
                tensor_mapping_constraints=constraints,
                mesh_shape_policy=policy,
                dedupe_key_fn=gemm_canonical_dedupe_key,
            )
        )

        assert len(candidates) > 0

        # All must satisfy constraints
        for cand in candidates:
            assert program_satisfies_tensor_mapping_constraints(cand, constraints)

        # A should have more than 1 inter_node factor pattern
        a_factor_patterns: Set[Tuple[int, ...]] = set()
        for cand in candidates:
            factors = compute_program_logical_shard_factors(cand)
            a_factors = factors.get("A")
            if a_factors is not None:
                inter_tuple = a_factors.domain_factors.get("inter_node", ())
                if inter_tuple:
                    a_factor_patterns.add(inter_tuple)

        assert len(a_factor_patterns) >= 2, (
            f"Flat policy should unlock more A tiling diversity, "
            f"but got only {a_factor_patterns}"
        )

    def test_flat_search_more_diverse_than_old_policy(self):
        """Flat policy should produce at least as diverse A tilings as old."""
        from mercury.search.mapping_constraints import (
            load_tensor_mapping_constraints,
        )

        program = _parse_gemm_program(256, 256, 256)
        devices = list(range(16))
        mesh = DeviceMesh(devices, (16, 1))

        constraints = load_tensor_mapping_constraints(
            "config/gemm_tensor_mapping_fixed_b_inter_node.json"
        )

        # Flat policy
        flat_policy = make_gemm_flat_mesh_shape_policy(inter_node=16, intra_node=1)
        flat_candidates = list(
            search(
                program,
                mesh,
                ["I", "J", "K"],
                tensor_mapping_constraints=constraints,
                mesh_shape_policy=flat_policy,
                dedupe_key_fn=gemm_canonical_dedupe_key,
            )
        )

        # Old policy
        old_policy = make_gemm_mesh_shape_policy(inter_node=16, intra_node=1)
        old_candidates = list(
            search(
                program,
                mesh,
                ["I", "J", "K"],
                tensor_mapping_constraints=constraints,
                mesh_shape_policy=old_policy,
                dedupe_key_fn=gemm_canonical_dedupe_key,
            )
        )

        # Collect A factor patterns
        def _collect_a_patterns(candidates):
            patterns = set()
            for cand in candidates:
                factors = compute_program_logical_shard_factors(cand)
                a_factors = factors.get("A")
                if a_factors is not None:
                    inter_tuple = a_factors.domain_factors.get("inter_node", ())
                    if inter_tuple:
                        patterns.add(inter_tuple)
            return patterns

        flat_a_patterns = _collect_a_patterns(flat_candidates)
        old_a_patterns = _collect_a_patterns(old_candidates)

        assert len(flat_a_patterns) >= len(old_a_patterns), (
            f"Flat should be at least as diverse: flat={flat_a_patterns} "
            f"vs old={old_a_patterns}"
        )
