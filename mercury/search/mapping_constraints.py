# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Tensor mapping constraints for GEMM and FFN searches."""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from mercury.ir.distributed import ShardType
from mercury.ir.elements import Buffer
from mercury.ir.nodes import Program
from mercury.ir.utils import get_io_buffers


_SUPPORTED_MATRICES = ("A", "B", "C")
_SUPPORTED_MODES = ("flexible", "fixed")
_SUPPORTED_TOPOLOGY_TOKENS = ("inter_node", "intra_node", "mixed")
_SUPPORTED_OPERATORS = ("gate", "up", "down")


@dataclass(frozen=True)
class ExactLayoutSignature:
    """Exact tensor layout signature used by runtime FFN two-step filtering."""

    mesh_shape: Tuple[int, ...]
    local_shape: Tuple[int, ...]
    shard_specs: Tuple[Tuple[str, Tuple[int, ...]], ...]

    def to_summary(self) -> str:
        specs = []
        for shard_type, mesh_dims in self.shard_specs:
            if shard_type == "R":
                specs.append("R")
            else:
                specs.append("S(" + ",".join(str(dim) for dim in mesh_dims) + ")")
        return (
            f"mesh={self.mesh_shape}, local_shape={self.local_shape}, "
            f"specs=[{', '.join(specs)}]"
        )


@dataclass(frozen=True)
class ExactTensorLayoutConstraints:
    """Runtime exact-match layout constraints keyed by matrix name."""

    matrices: Dict[str, ExactLayoutSignature]

    def get(self, matrix_name: str) -> Optional[ExactLayoutSignature]:
        return self.matrices.get(matrix_name)

    def summary_by_matrix(self) -> Dict[str, Optional[str]]:
        return {
            matrix_name: (
                self.matrices[matrix_name].to_summary()
                if matrix_name in self.matrices
                else None
            )
            for matrix_name in _SUPPORTED_MATRICES
        }


@dataclass(frozen=True)
class LogicalBoundaryLayoutSignature:
    """Logical boundary layout independent from execution-local buffer state."""

    mesh_shape: Tuple[int, ...]
    global_shape: Tuple[int, ...]
    shard_specs: Tuple[Tuple[str, Tuple[int, ...]], ...]

    def to_summary(self) -> str:
        specs = []
        for shard_type, mesh_dims in self.shard_specs:
            if shard_type == "R":
                specs.append("R")
            else:
                specs.append("S(" + ",".join(str(dim) for dim in mesh_dims) + ")")
        return (
            f"mesh={self.mesh_shape}, global_shape={self.global_shape}, "
            f"specs=[{', '.join(specs)}]"
        )


@dataclass(frozen=True)
class LogicalTensorLayoutConstraints:
    """Logical boundary constraints keyed by matrix name."""

    matrices: Dict[str, LogicalBoundaryLayoutSignature]

    def get(self, matrix_name: str) -> Optional[LogicalBoundaryLayoutSignature]:
        return self.matrices.get(matrix_name)

    def summary_by_matrix(self) -> Dict[str, Optional[str]]:
        return {
            matrix_name: (
                self.matrices[matrix_name].to_summary()
                if matrix_name in self.matrices
                else None
            )
            for matrix_name in _SUPPORTED_MATRICES
        }


@dataclass(frozen=True)
class MatrixDimMapping:
    """Constraint for one tensor dimension."""

    shard_topology: Optional[Tuple[str, ...]] = None
    shard_factor: Optional[int] = None

    @property
    def is_replicate(self) -> bool:
        return self.shard_topology is None

    def to_summary(self) -> str:
        if self.is_replicate:
            return "R"
        topo = ",".join(self.shard_topology)
        if self.shard_factor is not None:
            return f"S({topo}, factor={self.shard_factor})"
        return f"S({topo})"


@dataclass(frozen=True)
class MatrixMappingConstraint:
    """Constraint for one GEMM matrix."""

    mode: str
    mapping: Optional[Tuple[MatrixDimMapping, ...]] = None

    def to_summary(self) -> str:
        if self.mode == "flexible":
            return "flexible"
        assert self.mapping is not None, "fixed mapping must be present"
        mapping_summary = ", ".join(dim_mapping.to_summary() for dim_mapping in self.mapping)
        return f"fixed [{mapping_summary}]"


@dataclass(frozen=True)
class TensorMappingConstraints:
    """Constraint set for GEMM input/output matrices."""

    matrices: Dict[str, MatrixMappingConstraint]

    def get(self, matrix_name: str) -> MatrixMappingConstraint:
        if matrix_name not in self.matrices:
            return MatrixMappingConstraint(mode="flexible")
        return self.matrices[matrix_name]

    def summary_by_matrix(self) -> Dict[str, str]:
        return {
            matrix_name: self.get(matrix_name).to_summary()
            for matrix_name in _SUPPORTED_MATRICES
        }


@dataclass(frozen=True)
class OperatorTensorMappingConstraints:
    """Operator-scoped tensor mapping constraints for FFN."""

    operators: Dict[str, TensorMappingConstraints]
    operator_names: Tuple[str, ...]

    def get(self, operator_name: str) -> TensorMappingConstraints:
        constraints = self.operators.get(operator_name)
        if constraints is not None:
            return constraints

        return TensorMappingConstraints(
            matrices={
                matrix_name: MatrixMappingConstraint(mode="flexible")
                for matrix_name in _SUPPORTED_MATRICES
            }
        )

    def summary_by_operator(self) -> Dict[str, Dict[str, str]]:
        return {
            operator_name: self.get(operator_name).summary_by_matrix()
            for operator_name in self.operator_names
        }


def _validate_topology_tokens(tokens: List[str], field_name: str) -> Tuple[str, ...]:
    if not isinstance(tokens, list) or len(tokens) == 0:
        raise ValueError(f"{field_name} must be a non-empty list")

    normalized_tokens = []
    for token in tokens:
        if not isinstance(token, str) or token not in _SUPPORTED_TOPOLOGY_TOKENS:
            raise ValueError(f"Unsupported topology token '{token}' in {field_name}")
        normalized_tokens.append(token)

    if len(set(normalized_tokens)) != len(normalized_tokens):
        raise ValueError(f"{field_name} cannot contain duplicate topology tokens")

    return tuple(normalized_tokens)


def _parse_dim_mapping(config: object, field_name: str) -> MatrixDimMapping:
    if config == "R":
        return MatrixDimMapping()

    if not isinstance(config, dict):
        raise ValueError(f"{field_name} must be 'R' or an object with key 'shard'")

    allowed_keys = {"shard", "shard_factor"}
    if not set(config.keys()).issubset(allowed_keys) or "shard" not in config:
        raise ValueError(
            f"{field_name} must contain the 'shard' field and optionally 'shard_factor'"
        )

    tokens = _validate_topology_tokens(config["shard"], f"{field_name}.shard")

    shard_factor = None
    if "shard_factor" in config:
        shard_factor = config["shard_factor"]
        if not isinstance(shard_factor, int) or shard_factor < 2:
            raise ValueError(
                f"{field_name}.shard_factor must be an integer >= 2"
            )

    return MatrixDimMapping(shard_topology=tokens, shard_factor=shard_factor)


def _parse_matrix_constraint(matrix_name: str, config: object) -> MatrixMappingConstraint:
    if not isinstance(config, dict):
        raise ValueError(f"matrices.{matrix_name} must be a JSON object")

    unknown_fields = sorted(set(config.keys()) - {"mode", "mapping"})
    if len(unknown_fields) > 0:
        raise ValueError(
            f"Matrix {matrix_name} has unsupported fields: {', '.join(unknown_fields)}"
        )

    mode = config.get("mode", "flexible")
    if mode not in _SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode '{mode}' for matrix {matrix_name}")

    has_mapping = "mapping" in config
    if mode == "flexible":
        if has_mapping:
            raise ValueError(f"Flexible matrix {matrix_name} cannot define a mapping")
        return MatrixMappingConstraint(mode="flexible")

    if not has_mapping:
        raise ValueError(f"Fixed matrix {matrix_name} must define a mapping")

    mapping = config["mapping"]
    if not isinstance(mapping, list) or len(mapping) != 2:
        raise ValueError(f"Matrix {matrix_name} mapping must be a list of length 2")

    dim_mappings = []
    for dim_id, dim_config in enumerate(mapping):
        dim_mappings.append(
            _parse_dim_mapping(dim_config, f"matrices.{matrix_name}.mapping[{dim_id}]")
        )

    return MatrixMappingConstraint(mode="fixed", mapping=tuple(dim_mappings))


def _parse_matrices_config(matrices_config: object) -> Dict[str, MatrixMappingConstraint]:
    if not isinstance(matrices_config, dict):
        raise ValueError("Tensor mapping config field 'matrices' must be an object")

    unknown_matrices = sorted(set(matrices_config.keys()) - set(_SUPPORTED_MATRICES))
    if len(unknown_matrices) > 0:
        raise ValueError(
            "Unsupported matrices in tensor mapping config: "
            + ", ".join(unknown_matrices)
        )

    matrices: Dict[str, MatrixMappingConstraint] = {}
    for matrix_name in _SUPPORTED_MATRICES:
        if matrix_name in matrices_config:
            matrices[matrix_name] = _parse_matrix_constraint(
                matrix_name,
                matrices_config[matrix_name],
            )
        else:
            matrices[matrix_name] = MatrixMappingConstraint(mode="flexible")
    return matrices


def load_tensor_mapping_constraints(config_path: str) -> TensorMappingConstraints:
    """Load and validate a GEMM tensor mapping constraint config."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError("Tensor mapping config root must be a JSON object")

    version = config.get("version")
    if version != 1:
        raise ValueError("Tensor mapping config version must be 1")

    matrices_config = config.get("matrices", {})
    matrices = _parse_matrices_config(matrices_config)

    return TensorMappingConstraints(matrices=matrices)


def load_operator_tensor_mapping_constraints(
    config_path: str,
    operator_names: List[str],
) -> OperatorTensorMappingConstraints:
    """Load operator-scoped FFN mapping constraints."""
    if len(operator_names) == 0:
        raise ValueError("operator_names must be non-empty")
    if len(set(operator_names)) != len(operator_names):
        raise ValueError("operator_names cannot contain duplicates")

    for operator_name in operator_names:
        if operator_name not in _SUPPORTED_OPERATORS:
            raise ValueError(f"Unsupported FFN operator name '{operator_name}'")

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError("Operator tensor mapping config root must be a JSON object")

    version = config.get("version")
    if version != 1:
        raise ValueError("Operator tensor mapping config version must be 1")

    operators_config = config.get("operators", {})
    if not isinstance(operators_config, dict):
        raise ValueError("Operator tensor mapping config field 'operators' must be an object")

    unknown_operators = sorted(set(operators_config.keys()) - set(_SUPPORTED_OPERATORS))
    if len(unknown_operators) > 0:
        raise ValueError(
            "Unsupported operators in FFN tensor mapping config: "
            + ", ".join(unknown_operators)
        )

    operators: Dict[str, TensorMappingConstraints] = {}
    for operator_name in operator_names:
        operator_config = operators_config.get(operator_name, {})
        if not isinstance(operator_config, dict):
            raise ValueError(f"operators.{operator_name} must be a JSON object")

        unknown_fields = sorted(set(operator_config.keys()) - {"matrices"})
        if len(unknown_fields) > 0:
            raise ValueError(
                f"Operator {operator_name} has unsupported fields: {', '.join(unknown_fields)}"
            )

        matrices_config = operator_config.get("matrices", {})
        constraints = TensorMappingConstraints(matrices=_parse_matrices_config(matrices_config))
        for matrix_name in ("A", "C"):
            if constraints.get(matrix_name).mode != "flexible":
                raise ValueError(
                    f"FFN operator {operator_name} matrix {matrix_name} must be flexible"
                )
        operators[operator_name] = constraints

    return OperatorTensorMappingConstraints(
        operators=operators,
        operator_names=tuple(operator_names),
    )


def _normalize_exact_spec(
    spec: Union[ShardType, Tuple[ShardType, List[int]]]
) -> Tuple[str, Tuple[int, ...]]:
    if isinstance(spec, tuple):
        if spec[0] != ShardType.SHARD:
            raise ValueError(f"Unsupported shard type in exact layout spec: {spec[0]}")
        return ("S", tuple(int(dim) for dim in spec[1]))
    if spec == ShardType.REPLICATE:
        return ("R", tuple())
    raise ValueError(f"Unsupported exact layout spec: {spec}")


def _normalize_logical_spec(
    spec: Union[ShardType, Tuple[ShardType, List[int]]]
) -> Tuple[str, Tuple[int, ...]]:
    if isinstance(spec, tuple):
        if spec[0] != ShardType.SHARD:
            raise ValueError(f"Unsupported shard type in logical layout spec: {spec[0]}")
        return ("S", tuple(int(dim) for dim in spec[1]))
    if spec == ShardType.REPLICATE:
        return ("R", tuple())
    raise ValueError(f"Unsupported logical layout spec: {spec}")


def _mesh_cards(mesh_shape: Tuple[int, ...], mesh_dims: Tuple[int, ...]) -> int:
    cards = 1
    for mesh_dim in mesh_dims:
        cards *= int(mesh_shape[int(mesh_dim)])
    return cards


def derive_logical_local_shape(
    global_shape: Tuple[int, ...],
    mesh_shape: Tuple[int, ...],
    shard_specs: Tuple[Tuple[str, Tuple[int, ...]], ...],
) -> Tuple[int, ...]:
    """Derive local tensor shape implied by a logical global shape and layout."""
    if len(global_shape) != len(shard_specs):
        raise ValueError("global_shape and shard_specs must have identical rank")

    local_shape = []
    for dim_size, shard_spec in zip(global_shape, shard_specs):
        if shard_spec[0] == "R":
            local_shape.append(int(dim_size))
            continue
        if shard_spec[0] != "S":
            raise ValueError(f"Unsupported logical shard spec kind: {shard_spec[0]}")
        shard_cards = _mesh_cards(mesh_shape, shard_spec[1])
        if shard_cards <= 0:
            raise ValueError("shard_cards must be positive")
        if int(dim_size) % shard_cards != 0:
            raise ValueError(
                f"global dim {dim_size} is not divisible by shard cards {shard_cards}"
            )
        local_shape.append(int(dim_size) // shard_cards)
    return tuple(local_shape)


def exact_layout_signature_from_buffer(buffer: Buffer) -> ExactLayoutSignature:
    """Build an exact layout signature from a distributed buffer."""
    if buffer.shard_spec is None:
        raise ValueError(f"Buffer {buffer.tensor} has no shard spec")

    return ExactLayoutSignature(
        mesh_shape=tuple(int(dim) for dim in buffer.shard_spec.mesh.shape),
        local_shape=tuple(int(dim) for dim in buffer.get_shape()),
        shard_specs=tuple(_normalize_exact_spec(spec) for spec in buffer.shard_spec.specs),
    )


def logical_layout_signature_from_buffer(
    buffer: Buffer,
    use_logical_metadata: bool = False,
) -> LogicalBoundaryLayoutSignature:
    """Build a logical boundary layout signature from a distributed buffer."""
    if use_logical_metadata and buffer.logical_shard_spec is not None:
        shard_spec = buffer.logical_shard_spec
    else:
        shard_spec = buffer.shard_spec

    if shard_spec is None:
        raise ValueError(f"Buffer {buffer.tensor} has no shard spec")

    if use_logical_metadata and buffer.global_shape is not None:
        global_shape = tuple(int(dim) for dim in buffer.global_shape)
    else:
        local_shape = tuple(int(dim) for dim in buffer.get_shape())
        shard_specs = tuple(_normalize_logical_spec(spec) for spec in shard_spec.specs)
        inferred = []
        for dim_size, shard in zip(local_shape, shard_specs):
            if shard[0] == "R":
                inferred.append(int(dim_size))
            else:
                inferred.append(int(dim_size) * _mesh_cards(tuple(shard_spec.mesh.shape), shard[1]))
        global_shape = tuple(inferred)

    return LogicalBoundaryLayoutSignature(
        mesh_shape=tuple(int(dim) for dim in shard_spec.mesh.shape),
        global_shape=global_shape,
        shard_specs=tuple(_normalize_logical_spec(spec) for spec in shard_spec.specs),
    )


def exact_layout_signature_equal(
    lhs: ExactLayoutSignature,
    rhs: ExactLayoutSignature,
) -> bool:
    """Return whether two exact layout signatures are identical."""
    return (
        lhs.mesh_shape == rhs.mesh_shape
        and lhs.local_shape == rhs.local_shape
        and lhs.shard_specs == rhs.shard_specs
    )


def logical_layout_signature_equal(
    lhs: LogicalBoundaryLayoutSignature,
    rhs: LogicalBoundaryLayoutSignature,
) -> bool:
    """Return whether two logical boundary layout signatures are identical."""
    return (
        lhs.mesh_shape == rhs.mesh_shape
        and lhs.global_shape == rhs.global_shape
        and lhs.shard_specs == rhs.shard_specs
    )


def _resolve_topology_tokens(program: Program, tokens: Tuple[str, ...]) -> Tuple[int, ...]:
    if program.topology_metadata is None:
        raise ValueError("Program topology metadata must be set before matching tensor mappings")

    return resolve_topology_tokens_from_metadata(program.topology_metadata, tokens)


def resolve_topology_tokens_from_metadata(
    topology_metadata: Dict[str, List[int]],
    tokens: Tuple[str, ...],
) -> Tuple[int, ...]:
    """Resolve topology tokens into concrete mesh dimensions."""
    resolved_dims = []
    for token in tokens:
        dims = topology_metadata.get(f"{token}_dims")
        if dims is None:
            raise ValueError(f"Program topology metadata missing key '{token}_dims'")
        resolved_dims.extend(int(dim) for dim in dims)
    return tuple(sorted(set(resolved_dims)))


def program_satisfies_tensor_mapping_constraints(
    program: Program,
    constraints: Optional[TensorMappingConstraints],
) -> bool:
    """Return whether a program satisfies GEMM tensor mapping constraints.

    For fixed-mode matrices, each tensor dimension's shard spec is checked
    against the constraint's topology tokens and optional shard_factor.

    When ``shard_factor`` is specified, the constraint matches if the actual
    mesh dims are a subset of the resolved topology dims **and** the product
    of the corresponding mesh extents equals the requested factor.  This
    allows the search to produce candidates where only *part* of the
    available topology is used for a given dimension.
    """
    if constraints is None:
        return True

    buffers = program.visit(get_io_buffers)
    matrix_buffers = {}
    for buffer in buffers:
        matrix_name = buffer.tensor.upper()
        if matrix_name in _SUPPORTED_MATRICES and matrix_name not in matrix_buffers:
            matrix_buffers[matrix_name] = buffer

    for matrix_name in _SUPPORTED_MATRICES:
        matrix_constraint = constraints.get(matrix_name)
        if matrix_constraint.mode == "flexible":
            continue

        if matrix_constraint.mapping is None:
            raise ValueError(f"Fixed matrix {matrix_name} is missing a mapping")

        buffer = matrix_buffers.get(matrix_name)
        if buffer is None or buffer.shard_spec is None:
            return False

        if len(buffer.shard_spec.specs) != len(matrix_constraint.mapping):
            return False

        for spec, dim_mapping in zip(buffer.shard_spec.specs, matrix_constraint.mapping):
            if dim_mapping.is_replicate:
                if spec != ShardType.REPLICATE:
                    return False
                continue

            if not isinstance(spec, tuple) or spec[0] != ShardType.SHARD:
                return False

            expected_dims = _resolve_topology_tokens(program, dim_mapping.shard_topology)
            actual_dims = tuple(sorted(int(dim) for dim in spec[1]))

            if dim_mapping.shard_factor is None:
                if actual_dims != expected_dims:
                    return False
            else:
                if not set(actual_dims).issubset(set(expected_dims)):
                    return False
                mesh_shape = buffer.shard_spec.mesh.shape
                product = 1
                for d in actual_dims:
                    product *= int(mesh_shape[d])
                if product != dim_mapping.shard_factor:
                    return False

    return True


def _collect_matrix_buffers(program: Program) -> Dict[str, Buffer]:
    matrix_buffers: Dict[str, Buffer] = {}
    buffers = program.visit(get_io_buffers)
    for buffer in buffers:
        matrix_name = buffer.tensor.upper()
        if matrix_name in _SUPPORTED_MATRICES and matrix_name not in matrix_buffers:
            matrix_buffers[matrix_name] = buffer
    return matrix_buffers


def program_satisfies_logical_layout_constraints(
    program: Program,
    constraints: Optional[LogicalTensorLayoutConstraints],
) -> bool:
    """Return whether a program matches logical A/B/C boundary layouts."""
    if constraints is None:
        return True

    matrix_buffers = _collect_matrix_buffers(program)
    for matrix_name, expected_signature in constraints.matrices.items():
        buffer = matrix_buffers.get(matrix_name)
        if buffer is None or buffer.shard_spec is None:
            return False

        actual_signature = logical_layout_signature_from_buffer(buffer)
        if not logical_layout_signature_equal(actual_signature, expected_signature):
            return False
    return True


def program_satisfies_exact_layout_constraints(
    program: Program,
    constraints: Optional[ExactTensorLayoutConstraints],
) -> bool:
    """Return whether a program matches exact A/B/C layout signatures."""
    if constraints is None:
        return True

    matrix_buffers = _collect_matrix_buffers(program)
    for matrix_name, expected_signature in constraints.matrices.items():
        buffer = matrix_buffers.get(matrix_name)
        if buffer is None or buffer.shard_spec is None:
            return False

        actual_signature = exact_layout_signature_from_buffer(buffer)
        if not exact_layout_signature_equal(actual_signature, expected_signature):
            return False

    return True


# ---------------------------------------------------------------------------
# Logical shard factor helpers
# ---------------------------------------------------------------------------


def logical_shard_factor_for_dim(
    buffer_spec: Tuple[str, Tuple[int, ...]],
    mesh_shape: Tuple[int, ...],
    topology_dims: Tuple[int, ...],
) -> int:
    """Compute the effective logical shard factor for one buffer dimension.

    This function provides an explicit API for computing the shard factor
    that a single buffer dimension contributes within a specific physical
    domain.  It mirrors the logic already used implicitly by
    ``program_satisfies_tensor_mapping_constraints()`` when checking
    ``shard_factor`` constraints.

    Args:
        buffer_spec: Normalised shard spec for one tensor dimension,
            e.g. ``("R", ())`` or ``("S", (0, 1))``.
        mesh_shape: The mesh shape the buffer lives on.
        topology_dims: The mesh dim indices belonging to the target
            physical domain (e.g. the ``inter_node_dims``).

    Returns:
        The product of mesh extents along the intersection of the
        buffer's shard dims and the topology dims.  Returns 1 when the
        dimension is replicated or not sharded on the target domain.
    """
    shard_type, mesh_dims = buffer_spec
    if shard_type == "R" or len(mesh_dims) == 0:
        return 1

    topology_set = set(int(d) for d in topology_dims)
    factor = 1
    for md in mesh_dims:
        md_int = int(md)
        if md_int in topology_set:
            factor *= int(mesh_shape[md_int])
    return factor


def program_satisfies_logical_factor_constraints(
    program: Program,
    required_factors: Dict[str, Dict[str, Tuple[int, ...]]],
) -> bool:
    """Check whether a program's logical shard factors match requirements.

    This is a future-facing constraint checker that matches on
    ``LogicalShardFactors`` directly rather than on raw mesh dim indices.

    Args:
        program: A distributed ``Program`` with ``mesh`` and
            ``topology_metadata`` set.
        required_factors: Constraints keyed by uppercase matrix name,
            then by domain label, mapping to the expected factor tuple.
            Example::

                {
                    "A": {"inter_node": (2, 8)},
                    "B": {"inter_node": (4, 4)},
                }

    Returns:
        ``True`` if every specified matrix/domain pair matches exactly.
    """
    from mercury.search.topology_policy import compute_program_logical_shard_factors

    if program.mesh is None:
        return False

    actual = compute_program_logical_shard_factors(program)

    for matrix_name, domain_constraints in required_factors.items():
        matrix_factors = actual.get(matrix_name)
        if matrix_factors is None:
            return False
        for domain_label, expected_tuple in domain_constraints.items():
            actual_tuple = matrix_factors.domain_factors.get(domain_label)
            if actual_tuple is None:
                if expected_tuple == ():
                    continue
                return False
            if actual_tuple != expected_tuple:
                return False

    return True
