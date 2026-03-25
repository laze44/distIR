# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Topology-aware mesh shape policy for domain-aware search enumeration.

Replaces blind world_size factorization in enumerate_mesh_shapes with
controlled per-domain factorization that separates physical topology
from logical sharding.

Physical topology vs search shapes vs logical shard factors
-----------------------------------------------------------
- **Physical topology** is defined by ``TopologySpec`` and describes the
  actual hardware domains (e.g. ``num_devices=16``).
- **Search enumeration shapes** (e.g. ``(8, 2)``, ``(4, 4)``) are
  factorised mesh shapes produced by ``MeshShapePolicy.enumerate_shapes()``
  to explore the combinatorial tiling space.  They are *not* physical
  topology descriptions.
- **Logical shard factors** (``LogicalShardFactors``) provide the true
  per-buffer, per-physical-domain tiling description by mapping each
  search-mesh dimension back to its owning physical domain.

Topology metadata (device_dims) is generated at shape-construction time
rather than post-hoc inferred.
"""

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def _ordered_divisor_pairs(n: int) -> List[Tuple[int, int]]:
    """Return all (a, b) with a * b == n, a >= b >= 1, sorted by a descending."""
    pairs: List[Tuple[int, int]] = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            pairs.append((n // i, i))
        i += 1
    return pairs


def _factorize_single_dim(size: int) -> List[Tuple[int, ...]]:
    if size < 1:
        raise ValueError(f"Domain size must be >= 1, got {size}")
    return [(size,)]


def _factorize_rank_limited(
    size: int,
    max_virtual_dims: int = 2,
) -> List[Tuple[int, ...]]:
    """Allow up to max_virtual_dims logical sub-dims for a clique domain.

    For max_virtual_dims=2: returns (size,) plus all 2-factor splits (a, b)
    with a >= b > 1 (canonicalized descending to avoid duplicate permutations).
    """
    if size < 1:
        raise ValueError(f"Domain size must be >= 1, got {size}")
    if max_virtual_dims < 1:
        raise ValueError(f"max_virtual_dims must be >= 1, got {max_virtual_dims}")

    results: List[Tuple[int, ...]] = [(size,)]

    if max_virtual_dims >= 2 and size > 1:
        for a, b in _ordered_divisor_pairs(size):
            if b > 1:
                results.append((a, b))

    return results


@dataclass
class DomainSpec:
    """Description of a single physical interconnect domain.

    Attributes:
        kind: "clique" (all-to-all) or "mesh2d" (physical 2-D mesh).
        size: Total devices in this domain.
        shape: Physical shape (derived from size for clique, explicit for mesh2d).
        factorization_policy: "single_dim" or "rank_limited".
        max_virtual_dims: Max logical sub-dims for rank_limited policy.
    """

    kind: str = "clique"
    size: int = 1
    shape: Optional[Tuple[int, ...]] = None
    factorization_policy: str = "single_dim"
    max_virtual_dims: int = 2

    def __post_init__(self) -> None:
        if self.kind not in ("clique", "mesh2d"):
            raise ValueError(f"Unknown domain kind: {self.kind!r}")
        if self.kind == "clique":
            if self.size < 1:
                raise ValueError(f"Clique domain size must be >= 1, got {self.size}")
            if self.shape is not None:
                import numpy as np

                if int(np.prod(self.shape)) != self.size:
                    raise ValueError(
                        f"Clique domain shape {self.shape} product does not match "
                        f"size {self.size}"
                    )
            else:
                self.shape = (self.size,)
        elif self.kind == "mesh2d":
            if self.shape is None or len(self.shape) != 2:
                raise ValueError("mesh2d domain requires a 2-tuple shape")
            import numpy as np

            self.size = int(np.prod(self.shape))
        if self.factorization_policy not in ("single_dim", "rank_limited"):
            raise ValueError(
                f"Unknown factorization policy: {self.factorization_policy!r}"
            )


@dataclass
class TopologySpec:
    """Ordered list of physical domains defining the device topology.

    Domain ordering determines logical mesh dim ordering: domains[0] dims
    come first, then domains[1], etc.  Default label for single-domain case
    is "device".
    """

    domains: List[DomainSpec] = field(default_factory=list)
    domain_labels: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.domain_labels) == 0 and len(self.domains) > 0:
            if len(self.domains) == 1:
                self.domain_labels = ["device"]
            else:
                self.domain_labels = [f"domain_{i}" for i in range(len(self.domains))]
        if len(self.domain_labels) != len(self.domains):
            raise ValueError(
                f"Number of domain labels ({len(self.domain_labels)}) must match "
                f"number of domains ({len(self.domains)})"
            )


@dataclass
class MeshShapePolicy:
    """Controls how search enumeration mesh shapes are generated from a TopologySpec.

    Main entry-point passed to ``search()``.  The shapes returned by
    ``enumerate_shapes()`` are *search enumeration artifacts* — factorised
    views of the physical topology used to explore the combinatorial tiling
    space.  They are **not** descriptions of the physical topology itself.
    Use ``LogicalShardFactors`` (computed via
    ``compute_buffer_logical_shard_factors``) to obtain the true per-buffer,
    per-physical-domain tiling interpretation.
    """

    topology: TopologySpec

    def enumerate_shapes(self) -> List[Tuple[int, ...]]:
        return enumerate_topology_mesh_shapes(self.topology)

    def topology_metadata_for_shape(
        self, mesh_shape: Tuple[int, ...]
    ) -> Dict[str, List[int]]:
        return topology_metadata_for_shape(self.topology, mesh_shape)


def enumerate_domain_shapes(domain: DomainSpec) -> List[Tuple[int, ...]]:
    """Enumerate allowed logical sub-dimension tuples for one domain."""
    if domain.kind == "mesh2d":
        if domain.shape is None:
            raise ValueError("mesh2d domain requires a shape")
        return [domain.shape]

    if domain.size == 1:
        return [()]

    if domain.factorization_policy == "single_dim":
        return _factorize_single_dim(domain.size)
    elif domain.factorization_policy == "rank_limited":
        return _factorize_rank_limited(domain.size, domain.max_virtual_dims)
    else:
        raise ValueError(
            f"Unknown factorization policy: {domain.factorization_policy!r}"
        )


def enumerate_topology_mesh_shapes(
    topology: TopologySpec,
) -> List[Tuple[int, ...]]:
    """Generate all allowed mesh shapes by combining per-domain factors.

    Leading dims come from domains[0], then domains[1], etc.  Inter and
    intra dims never intermix within a single logical dimension.
    """
    if len(topology.domains) == 0:
        return [()]

    per_domain_factors: List[List[Tuple[int, ...]]] = []
    for domain in topology.domains:
        per_domain_factors.append(enumerate_domain_shapes(domain))

    result: List[Tuple[int, ...]] = []
    for combo in itertools.product(*per_domain_factors):
        shape: Tuple[int, ...] = ()
        for domain_factors in combo:
            shape = shape + domain_factors
        result.append(shape)

    return result


def topology_metadata_for_shape(
    topology: TopologySpec,
    mesh_shape: Tuple[int, ...],
) -> Dict[str, List[int]]:
    """Build topology_metadata dict mapping logical dim indices to domain labels.

    Returns e.g. {"device_dims": [0, 1], "mixed_dims": []}.
    Generated deterministically from topology — no post-hoc inference.

    Dims are consumed left-to-right.  For each domain with ``size > 1``,
    a contiguous block of dims whose product equals ``domain.size`` is
    assigned to that domain.  Domains with ``size == 1`` contribute no dims.
    """
    dim_offset = 0
    label_to_dims: Dict[str, List[int]] = {}

    for domain, label in zip(topology.domains, topology.domain_labels):
        if domain.size <= 1:
            label_to_dims[label] = []
            continue
        # Consume as many dims as needed to cover this domain's size.
        product = 1
        start = dim_offset
        while product < domain.size and dim_offset < len(mesh_shape):
            product *= int(mesh_shape[dim_offset])
            dim_offset += 1
        if product != domain.size:
            raise ValueError(
                f"Cannot match domain '{label}' (size={domain.size}) "
                f"at offset {start} in mesh shape {mesh_shape}: "
                f"product of consumed dims is {product}"
            )
        label_to_dims[label] = list(range(start, dim_offset))

    metadata: Dict[str, List[int]] = {"mixed_dims": []}
    for label, dims in label_to_dims.items():
        metadata[f"{label}_dims"] = dims

    return metadata


def make_topology_spec(
    num_devices: int,
    factorization: str = "rank_limited",
    max_virtual_dims: int = 2,
) -> TopologySpec:
    """Create a TopologySpec for a single-domain clique topology."""
    domain = DomainSpec(
        kind="clique",
        size=num_devices,
        factorization_policy=factorization,
        max_virtual_dims=max_virtual_dims,
    )
    return TopologySpec(
        domains=[domain],
        domain_labels=["device"],
    )


def make_mesh_shape_policy(
    num_devices: int,
    factorization: str = "rank_limited",
    max_virtual_dims: int = 2,
) -> MeshShapePolicy:
    """Create a MeshShapePolicy for a single-domain clique topology."""
    topology = make_topology_spec(
        num_devices=num_devices,
        factorization=factorization,
        max_virtual_dims=max_virtual_dims,
    )
    return MeshShapePolicy(topology=topology)


@dataclass
class FlatMeshShapePolicy(MeshShapePolicy):
    """Flat mesh shape policy where each domain stays as a 1-D clique.

    Instead of factorising the domain size into multi-dimensional search
    meshes, the search engine enumerates **logical shard factor
    combinations** independently per axis and constructs a virtual mesh
    for each combination.  The ``enumerate_shapes()`` method returns
    only the flat ``(domain_size,)`` shape per domain; the actual
    virtual mesh shapes are built by ``search()`` via
    ``_enumerate_logical_factor_assignments()``.

    ``topology_metadata_for_virtual_shape`` maps *all* dims of a
    virtual mesh to the owning physical domain based on the domain
    sizes stored in the topology.
    """

    def enumerate_shapes(self) -> List[Tuple[int, ...]]:
        return enumerate_topology_mesh_shapes(self.topology)

    def topology_metadata_for_shape(
        self, mesh_shape: Tuple[int, ...]
    ) -> Dict[str, List[int]]:
        """Build topology metadata for a *virtual* mesh shape.

        For flat policies, virtual mesh shapes are constructed by the
        search engine from logical factor combinations.  Each virtual
        dim is assigned to the physical domain whose flat segment it
        belongs to, determined by consuming dims left-to-right according
        to domain order.
        """
        return self.topology_metadata_for_virtual_shape(mesh_shape)

    def topology_metadata_for_virtual_shape(
        self, virtual_shape: Tuple[int, ...]
    ) -> Dict[str, List[int]]:
        """Map every dim of *virtual_shape* to its owning physical domain.

        Dims are consumed left-to-right.  For each domain with
        ``size > 1``, a contiguous block of dims whose product equals
        ``domain.size`` is assigned to that domain.  Domains with
        ``size == 1`` contribute no dims.
        """
        dim_offset = 0
        label_to_dims: Dict[str, List[int]] = {}
        for domain, label in zip(
            self.topology.domains, self.topology.domain_labels
        ):
            if domain.size <= 1:
                label_to_dims[label] = []
                continue
            # Consume as many dims as needed to cover this domain's size.
            product = 1
            start = dim_offset
            while product < domain.size and dim_offset < len(virtual_shape):
                product *= int(virtual_shape[dim_offset])
                dim_offset += 1
            if product != domain.size:
                # Fallback: assign remaining dims to last domain
                label_to_dims[label] = list(range(start, len(virtual_shape)))
                dim_offset = len(virtual_shape)
            else:
                label_to_dims[label] = list(range(start, dim_offset))

        metadata: Dict[str, List[int]] = {"mixed_dims": []}
        for label, dims in label_to_dims.items():
            metadata[f"{label}_dims"] = dims
        return metadata


def make_flat_mesh_shape_policy(
    num_devices: int,
) -> FlatMeshShapePolicy:
    """Create a flat MeshShapePolicy where the domain is a 1-D clique.

    Unlike ``make_mesh_shape_policy``, this forces the domain to use
    ``single_dim`` factorisation so that ``enumerate_shapes()`` only
    returns flat shapes (e.g. ``(16,)`` instead of ``(8, 2)`` and
    ``(4, 4)``).  The search engine then uses
    ``_enumerate_logical_factor_assignments()`` to explore the full
    factor space independently per axis.
    """
    topology = make_topology_spec(
        num_devices=num_devices,
        factorization="single_dim",
    )
    return FlatMeshShapePolicy(topology=topology)


# ---------------------------------------------------------------------------
# Logical shard factors — the true per-buffer, per-domain tiling description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicalShardFactors:
    """Per-physical-domain shard factors for a single buffer.

    ``domain_factors`` maps a domain label (e.g. ``"device"``) to
    a tuple of per-tensor-dimension shard factors contributed by that
    domain.  Unsharded tensor dimensions are omitted (factor == 1).

    Example: for matrix ``A`` with shape ``[M, K]`` under search mesh
    ``(8, 2)`` where both dims belong to ``device``, and the buffer
    is sharded ``S(0)`` on dim 0 and ``S(1)`` on dim 1, the logical
    factors would be ``{"device": (8, 2)}``, meaning the device
    domain contributes a factor of 8 along dim 0 and 2 along dim 1.
    """

    domain_factors: Dict[str, Tuple[int, ...]]

    def total_factor(self, domain: str) -> int:
        """Return the product of factors for a domain (1 if absent)."""
        factors = self.domain_factors.get(domain)
        if factors is None:
            return 1
        result = 1
        for f in factors:
            result *= f
        return result

    def to_summary(self) -> str:
        """Human-readable format, e.g. ``device=(2, 8)``."""
        if not self.domain_factors:
            return "replicated"
        parts = []
        for label in sorted(self.domain_factors.keys()):
            factors = self.domain_factors[label]
            parts.append(f"{label}={factors}")
        return ", ".join(parts)


def _build_mesh_dim_to_domain(
    topology_metadata: Dict[str, List[int]],
) -> Dict[int, str]:
    """Build reverse map: mesh_dim → domain label from topology_metadata."""
    dim_to_domain: Dict[int, str] = {}
    for key, dims in topology_metadata.items():
        if not key.endswith("_dims"):
            continue
        label = key[: -len("_dims")]
        if label == "mixed":
            continue
        for dim in dims:
            dim_to_domain[int(dim)] = label
    return dim_to_domain


def compute_buffer_logical_shard_factors(
    buffer_shard_specs: "Tuple[object, ...]",
    mesh_shape: Tuple[int, ...],
    topology_metadata: Dict[str, List[int]],
) -> LogicalShardFactors:
    """Compute logical shard factors for a buffer.

    Args:
        buffer_shard_specs: Normalised shard specs — each element is either
            ``("R", ())`` or ``("S", (mesh_dim_0, mesh_dim_1, ...))``.
        mesh_shape: The mesh shape the buffer lives on.
        topology_metadata: Maps ``"<label>_dims"`` to mesh dim indices.

    Returns:
        ``LogicalShardFactors`` with per-domain factor tuples.
    """
    dim_to_domain = _build_mesh_dim_to_domain(topology_metadata)

    # Accumulate per-domain, per-tensor-dim factors
    domain_accum: Dict[str, List[int]] = {}
    for shard_type, mesh_dims in buffer_shard_specs:
        if shard_type == "R" or len(mesh_dims) == 0:
            continue
        for mesh_dim in mesh_dims:
            domain = dim_to_domain.get(int(mesh_dim))
            if domain is None:
                continue
            factor = int(mesh_shape[int(mesh_dim)])
            if domain not in domain_accum:
                domain_accum[domain] = []
            domain_accum[domain].append(factor)

    domain_factors: Dict[str, Tuple[int, ...]] = {}
    for label in sorted(domain_accum.keys()):
        factors = tuple(domain_accum[label])
        if all(f == 1 for f in factors):
            continue
        domain_factors[label] = factors

    return LogicalShardFactors(domain_factors=domain_factors)


def compute_program_logical_shard_factors(
    program: "Program",
    matrix_names: Tuple[str, ...] = ("a", "b", "c"),
) -> Dict[str, "LogicalShardFactors"]:
    """Compute logical shard factors for boundary buffers in a program.

    This is a convenience function that computes ``LogicalShardFactors``
    for all boundary buffers matching ``matrix_names``.

    Args:
        program: A distributed ``Program`` with ``mesh`` and
            ``topology_metadata`` set.
        matrix_names: Tensor names to look for (default: ``("a","b","c")``).

    Returns:
        Dict mapping uppercase matrix name to ``LogicalShardFactors``.
    """
    from mercury.ir.utils import get_io_buffers
    from mercury.search.mapping_constraints import _normalize_exact_spec

    if program.mesh is None:
        return {}
    if not program.topology_metadata:
        return {}

    mesh_shape = tuple(int(d) for d in program.mesh.shape)
    buffers = program.visit(get_io_buffers)
    result: Dict[str, LogicalShardFactors] = {}

    for buf in buffers:
        if buf.tensor not in matrix_names:
            continue
        matrix_name = buf.tensor.upper()
        if matrix_name in result:
            continue
        if buf.shard_spec is None:
            result[matrix_name] = LogicalShardFactors(domain_factors={})
            continue

        normalized_specs = tuple(
            _normalize_exact_spec(spec) for spec in buf.shard_spec.specs
        )
        factors = compute_buffer_logical_shard_factors(
            normalized_specs, mesh_shape, program.topology_metadata
        )
        result[matrix_name] = factors

    return result
