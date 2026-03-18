# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Topology-aware mesh shape policy for domain-aware search enumeration.

Replaces blind world_size factorization in enumerate_mesh_shapes with
controlled per-domain factorization that separates physical topology
from logical sharding.

Topology metadata (inter_node_dims / intra_node_dims) is generated at
shape-construction time rather than post-hoc inferred.
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
    come first, then domains[1], etc.  Default labels for 2-domain case
    are "inter_node" and "intra_node".
    """

    domains: List[DomainSpec] = field(default_factory=list)
    domain_labels: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.domain_labels) == 0 and len(self.domains) > 0:
            if len(self.domains) == 2:
                self.domain_labels = ["inter_node", "intra_node"]
            else:
                self.domain_labels = [f"domain_{i}" for i in range(len(self.domains))]
        if len(self.domain_labels) != len(self.domains):
            raise ValueError(
                f"Number of domain labels ({len(self.domain_labels)}) must match "
                f"number of domains ({len(self.domains)})"
            )


@dataclass
class MeshShapePolicy:
    """Controls how logical mesh shapes are generated from a TopologySpec.

    Main entry-point passed to search().
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

    Returns e.g. {"inter_node_dims": [0, 1], "intra_node_dims": [2], "mixed_dims": []}.
    Generated deterministically from topology — no post-hoc inference.
    """
    dim_offset = 0
    label_to_dims: Dict[str, List[int]] = {}

    for domain, label in zip(topology.domains, topology.domain_labels):
        domain_shapes = enumerate_domain_shapes(domain)

        matched = False
        for factor_tuple in domain_shapes:
            n_dims = len(factor_tuple)
            candidate = mesh_shape[dim_offset : dim_offset + n_dims]
            if candidate == factor_tuple:
                label_to_dims[label] = list(range(dim_offset, dim_offset + n_dims))
                dim_offset += n_dims
                matched = True
                break

        if not matched:
            if domain.size == 1:
                label_to_dims[label] = []
            else:
                raise ValueError(
                    f"Cannot match domain '{label}' (allowed shapes: {domain_shapes}) "
                    f"at offset {dim_offset} in mesh shape {mesh_shape}"
                )

    metadata: Dict[str, List[int]] = {"mixed_dims": []}
    for label, dims in label_to_dims.items():
        metadata[f"{label}_dims"] = dims

    return metadata


def make_gemm_topology_spec(
    inter_node: int,
    intra_node: int,
    inter_factorization: str = "rank_limited",
    inter_max_virtual_dims: int = 2,
    intra_factorization: str = "single_dim",
    intra_max_virtual_dims: int = 2,
) -> TopologySpec:
    """Create a TopologySpec for the standard GEMM search scenario."""
    inter_domain = DomainSpec(
        kind="clique",
        size=inter_node,
        factorization_policy=inter_factorization,
        max_virtual_dims=inter_max_virtual_dims,
    )
    intra_domain = DomainSpec(
        kind="clique",
        size=intra_node,
        factorization_policy=intra_factorization,
        max_virtual_dims=intra_max_virtual_dims,
    )
    return TopologySpec(
        domains=[inter_domain, intra_domain],
        domain_labels=["inter_node", "intra_node"],
    )


def make_gemm_mesh_shape_policy(
    inter_node: int,
    intra_node: int,
    inter_factorization: str = "rank_limited",
    inter_max_virtual_dims: int = 2,
    intra_factorization: str = "single_dim",
    intra_max_virtual_dims: int = 2,
) -> MeshShapePolicy:
    """Create a MeshShapePolicy for the standard GEMM search scenario."""
    topology = make_gemm_topology_spec(
        inter_node=inter_node,
        intra_node=intra_node,
        inter_factorization=inter_factorization,
        inter_max_virtual_dims=inter_max_virtual_dims,
        intra_factorization=intra_factorization,
        intra_max_virtual_dims=intra_max_virtual_dims,
    )
    return MeshShapePolicy(topology=topology)
