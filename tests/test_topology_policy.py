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
    make_flat_mesh_shape_policy,
    make_mesh_shape_policy,
    make_topology_spec,
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
    def test_auto_labels_single_domain(self):
        t = TopologySpec(
            domains=[
                DomainSpec(kind="clique", size=4),
            ]
        )
        assert t.domain_labels == ["device"]

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
    def test_device16(self):
        topology = make_topology_spec(num_devices=16)
        shapes = enumerate_topology_mesh_shapes(topology)
        assert (16,) in shapes
        assert (8, 2) in shapes
        assert (4, 4) in shapes
        assert (2, 2, 2, 2) not in shapes
        assert len(shapes) == 3

    def test_device4(self):
        topology = make_topology_spec(num_devices=4)
        shapes = enumerate_topology_mesh_shapes(topology)
        assert (4,) in shapes
        assert (2, 2) in shapes
        assert len(shapes) == 2

    def test_device4_single_dim(self):
        topology = make_topology_spec(
            num_devices=4,
            factorization="single_dim",
        )
        shapes = enumerate_topology_mesh_shapes(topology)
        assert shapes == [(4,)]

    def test_device1(self):
        topology = make_topology_spec(num_devices=1)
        shapes = enumerate_topology_mesh_shapes(topology)
        assert shapes == [()]

    def test_products_match_world_size(self):
        topology = make_topology_spec(num_devices=128)
        shapes = enumerate_topology_mesh_shapes(topology)
        for shape in shapes:
            product = 1
            for v in shape:
                product *= v
            assert product == 128, f"Shape {shape} product != 128"

    def test_empty_topology(self):
        topology = TopologySpec()
        assert enumerate_topology_mesh_shapes(topology) == [()]


class TestTopologyMetadataForShape:
    def test_device16_single_dim(self):
        topology = make_topology_spec(num_devices=16)
        metadata = topology_metadata_for_shape(topology, (16,))
        assert metadata["device_dims"] == [0]

    def test_device4_two_dims(self):
        topology = make_topology_spec(num_devices=4)
        metadata = topology_metadata_for_shape(topology, (2, 2))
        assert metadata["device_dims"] == [0, 1]

    def test_device32(self):
        topology = make_topology_spec(num_devices=32)
        metadata = topology_metadata_for_shape(topology, (4, 8))
        assert metadata["device_dims"] == [0, 1]

    def test_device32_three_dims(self):
        topology = make_topology_spec(num_devices=32)
        metadata = topology_metadata_for_shape(topology, (2, 2, 8))
        assert metadata["device_dims"] == [0, 1, 2]

    def test_device1(self):
        topology = make_topology_spec(num_devices=1)
        metadata = topology_metadata_for_shape(topology, ())
        assert metadata["device_dims"] == []


class TestMeshShapePolicy:
    def test_enumerate_shapes(self):
        policy = make_mesh_shape_policy(num_devices=16)
        shapes = policy.enumerate_shapes()
        assert (16,) in shapes
        assert (8, 2) in shapes
        assert (4, 4) in shapes
        assert len(shapes) == 3

    def test_topology_metadata(self):
        policy = make_mesh_shape_policy(num_devices=8)
        metadata = policy.topology_metadata_for_shape((2, 2, 2))
        assert metadata["device_dims"] == [0, 1, 2]


class TestMakeTopologySpecFactories:
    def test_default_policies(self):
        topology = make_topology_spec(num_devices=8)
        device_domain = topology.domains[0]
        assert device_domain.factorization_policy == "rank_limited"
        assert device_domain.max_virtual_dims == 2

    def test_custom_policies(self):
        topology = make_topology_spec(
            num_devices=8,
            factorization="single_dim",
        )
        device_domain = topology.domains[0]
        assert device_domain.factorization_policy == "single_dim"


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

    def test_device4_layout_coverage(self):
        # GIVEN num_devices=4 clique with rank_limited policy
        # WHEN we search with the policy
        # THEN mesh shapes (4,) and (2,2) must both appear,
        # and layout assignments must cover (4,1), (2,2), (1,4) sharding patterns
        policy = make_mesh_shape_policy(num_devices=4)
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

    def test_device16_no_4d_shapes(self):
        # GIVEN num_devices=16 with default GEMM policy
        # WHEN we enumerate shapes
        # THEN (2,2,2,2) must NOT appear
        policy = make_mesh_shape_policy(num_devices=16)
        shapes = policy.enumerate_shapes()
        assert (2, 2, 2, 2) not in shapes
        for shape in shapes:
            assert len(shape) <= 2, f"Shape {shape} exceeds 2 dims"

    def test_device16_candidate_count_regression(self):
        # GIVEN num_devices=16
        # WHEN we search with vs without policy
        # THEN policy produces fewer candidates (no 4D explosion)
        program = _parse_gemm_program(256, 256, 256)
        devices = list(range(16))
        mesh = DeviceMesh(devices, (16,))

        policy = make_mesh_shape_policy(num_devices=16)

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
        # THEN all candidates have non-None metadata with device_dims
        policy = make_mesh_shape_policy(num_devices=8)
        program = _parse_gemm_program(256, 256, 256)
        devices = list(range(8))
        mesh = DeviceMesh(devices, (8,))

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
            assert "device_dims" in meta

    def test_policy_preserves_multiple_layout_classes(self):
        # GIVEN policy for num_devices=8
        # WHEN we search with dedupe
        # THEN at least 2 distinct dedupe keys exist (multiple layout classes)
        policy = make_mesh_shape_policy(num_devices=8)
        program = _parse_gemm_program(256, 256, 256)
        devices = list(range(8))
        mesh = DeviceMesh(devices, (8,))

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

        policy = make_mesh_shape_policy(num_devices=8)
        program = _parse_gemm_program(256, 256, 256)
        devices = list(range(8))
        mesh = DeviceMesh(devices, (8,))

        candidates = list(
            search(program, mesh, ["I", "J", "K"], mesh_shape_policy=policy)
        )
        assert len(candidates) > 0

        for cand in candidates:
            normalized = _normalize_topology_metadata(cand, num_inter_dims=None)
            policy_meta = cand.topology_metadata
            assert normalized["device_dims"] == sorted(
                policy_meta["device_dims"]
            )


class TestPhase4MappingConstraintsAlignment:
    """Phase 4: Verify policy metadata works with mapping_constraints.py."""

    def test_resolve_topology_tokens_device(self):
        # GIVEN policy metadata for num_devices=8 with shape (2, 2, 2)
        # WHEN resolving "device" token
        # THEN returns the device dims from policy metadata
        from mercury.search.mapping_constraints import (
            resolve_topology_tokens_from_metadata,
        )

        policy = make_mesh_shape_policy(num_devices=8)
        meta = policy.topology_metadata_for_shape((2, 2, 2))
        resolved = resolve_topology_tokens_from_metadata(meta, ("device",))
        assert resolved == tuple(sorted(meta["device_dims"]))


class TestPhase4TwoStepAlignment:
    """Phase 4: Verify policy metadata matches _fixed_topology_metadata for default cases."""

    def test_single_dim_mesh_matches(self):
        # GIVEN mesh shape (N,) where N is the num_devices
        # WHEN comparing _fixed_topology_metadata vs policy metadata
        # THEN both classify all dims as device dims
        from mercury.search.gemm_two_step_search import _fixed_topology_metadata

        for num_devices in (2, 4, 8, 16):
            policy = make_mesh_shape_policy(num_devices=num_devices)
            shape = (num_devices,)
            fixed = _fixed_topology_metadata(shape)
            policy_meta = policy.topology_metadata_for_shape(shape)
            assert fixed["device_dims"] == [0]
            assert policy_meta["device_dims"] == [0]

    def test_two_dim_mesh_matches(self):
        # GIVEN mesh shape (a, b) where a*b == num_devices
        # WHEN comparing _fixed_topology_metadata vs policy metadata
        # THEN they produce equivalent results (all dims are device dims)
        from mercury.search.gemm_two_step_search import _fixed_topology_metadata

        for num_devices, shape in [(8, (4, 2)), (32, (8, 4)), (16, (2, 8))]:
            policy = make_mesh_shape_policy(num_devices=num_devices)
            fixed = _fixed_topology_metadata(shape)
            policy_meta = policy.topology_metadata_for_shape(shape)
            assert fixed["device_dims"] == policy_meta["device_dims"], (
                f"Mismatch for shape {shape}"
            )


class TestLogicalShardFactors:
    """Unit tests for LogicalShardFactors and compute helpers."""

    def test_replicated_buffer_has_no_factors(self):
        # GIVEN a fully replicated buffer (all specs are R)
        specs = (("R", ()), ("R", ()))
        mesh_shape = (4, 2)
        metadata = {"device_dims": [0, 1]}

        factors = compute_buffer_logical_shard_factors(specs, mesh_shape, metadata)
        assert factors.domain_factors == {}
        assert factors.to_summary() == "replicated"

    def test_single_dim_shard_on_device(self):
        # GIVEN buffer sharded on dim 0 via mesh dim 0 (device)
        specs = (("S", (0,)), ("R", ()))
        mesh_shape = (4, 2)
        metadata = {"device_dims": [0, 1]}

        factors = compute_buffer_logical_shard_factors(specs, mesh_shape, metadata)
        assert factors.domain_factors == {"device": (4,)}
        assert factors.total_factor("device") == 4

    def test_two_dim_shard_on_device(self):
        # GIVEN buffer sharded on dim 0 via mesh dim 0, dim 1 via mesh dim 1
        # WHEN mesh (8, 2) with both dims belonging to device
        specs = (("S", (0,)), ("S", (1,)))
        mesh_shape = (8, 2)
        metadata = {"device_dims": [0, 1]}

        factors = compute_buffer_logical_shard_factors(specs, mesh_shape, metadata)
        assert factors.domain_factors == {"device": (8, 2)}
        assert factors.total_factor("device") == 16

    def test_cross_domain_shard_single_domain(self):
        # GIVEN buffer sharded dim 0 on device (dim 0), dim 1 on device (dim 1)
        specs = (("S", (0,)), ("S", (1,)))
        mesh_shape = (4, 2)
        metadata = {"device_dims": [0, 1]}

        factors = compute_buffer_logical_shard_factors(specs, mesh_shape, metadata)
        assert factors.domain_factors == {"device": (4, 2)}
        assert factors.total_factor("device") == 8

    def test_multi_mesh_dim_on_one_tensor_dim(self):
        # GIVEN buffer with one tensor dim sharded across mesh dims 0 and 1
        # (both belonging to device), e.g. S(0,1) on dim 0
        specs = (("S", (0, 1)), ("R", ()))
        mesh_shape = (4, 4)
        metadata = {"device_dims": [0, 1]}

        factors = compute_buffer_logical_shard_factors(specs, mesh_shape, metadata)
        # Both mesh dims contribute to device
        assert factors.domain_factors == {"device": (4, 4)}
        assert factors.total_factor("device") == 16

    def test_summary_format(self):
        lsf = LogicalShardFactors(domain_factors={"device": (2, 8)})
        assert lsf.to_summary() == "device=(2, 8)"

    def test_empty_factors_summary(self):
        lsf = LogicalShardFactors(domain_factors={})
        assert lsf.to_summary() == "replicated"

    def test_device16_logical_factor_coverage(self):
        """Integration: num_devices=16 search produces diverse A tilings."""
        policy = make_mesh_shape_policy(num_devices=16)
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
                device_tuple = a_factors.domain_factors.get("device", ())
                if device_tuple:
                    a_factor_patterns.add(device_tuple)

        # With num_devices=16 and mesh shapes (16,), (8,2), (4,4),
        # we should see diverse A tiling patterns
        assert len(a_factor_patterns) >= 2, (
            f"Expected diverse A tiling patterns, got {a_factor_patterns}"
        )


class TestFlatMeshShapePolicy:
    """Tests for FlatMeshShapePolicy and the flat search path."""

    def test_flat_policy_enumerate_shapes_single_domain(self):
        policy = make_flat_mesh_shape_policy(num_devices=16)
        shapes = policy.enumerate_shapes()
        assert shapes == [(16,)]

    def test_flat_policy_is_flat_mesh_shape_policy(self):
        policy = make_flat_mesh_shape_policy(num_devices=16)
        assert isinstance(policy, FlatMeshShapePolicy)
        assert isinstance(policy, MeshShapePolicy)

    def test_flat_topology_metadata_single_domain(self):
        policy = make_flat_mesh_shape_policy(num_devices=16)
        # Virtual mesh (4, 4) -> both dims belong to device
        meta = policy.topology_metadata_for_shape((4, 4))
        assert meta["device_dims"] == [0, 1]

    def test_flat_topology_metadata_single_dim(self):
        policy = make_flat_mesh_shape_policy(num_devices=16)
        meta = policy.topology_metadata_for_shape((16,))
        assert meta["device_dims"] == [0]

    def test_flat_search_produces_candidates(self):
        policy = make_flat_mesh_shape_policy(num_devices=4)
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
        policy = make_flat_mesh_shape_policy(num_devices=4)
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
            assert "device_dims" in cand.topology_metadata

    def test_flat_search_with_fixed_b_constraint_device16(self):
        """Core test: flat policy + B K-dim fixed factor=8 on device=16."""
        from mercury.search.mapping_constraints import (
            load_tensor_mapping_constraints,
            program_satisfies_tensor_mapping_constraints,
        )

        policy = make_flat_mesh_shape_policy(num_devices=16)
        program = _parse_gemm_program(256, 256, 256)
        devices = list(range(16))
        mesh = DeviceMesh(devices, (16,))

        constraints = load_tensor_mapping_constraints(
            "config/gemm_tensor_mapping_fixed_b_device.json"
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

        # A should have more than 1 device factor pattern
        a_factor_patterns: Set[Tuple[int, ...]] = set()
        for cand in candidates:
            factors = compute_program_logical_shard_factors(cand)
            a_factors = factors.get("A")
            if a_factors is not None:
                device_tuple = a_factors.domain_factors.get("device", ())
                if device_tuple:
                    a_factor_patterns.add(device_tuple)

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
        mesh = DeviceMesh(devices, (16,))

        constraints = load_tensor_mapping_constraints(
            "config/gemm_tensor_mapping_fixed_b_device.json"
        )

        # Flat policy
        flat_policy = make_flat_mesh_shape_policy(num_devices=16)
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
        old_policy = make_mesh_shape_policy(num_devices=16)
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
                    device_tuple = a_factors.domain_factors.get("device", ())
                    if device_tuple:
                        patterns.add(device_tuple)
            return patterns

        flat_a_patterns = _collect_a_patterns(flat_candidates)
        old_a_patterns = _collect_a_patterns(old_candidates)

        assert len(flat_a_patterns) >= len(old_a_patterns), (
            f"Flat should be at least as diverse: flat={flat_a_patterns} "
            f"vs old={old_a_patterns}"
        )
