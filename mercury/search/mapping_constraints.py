# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Tensor mapping constraints for GEMM and FFN searches."""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from mercury.ir.distributed import ShardType
from mercury.ir.nodes import Program
from mercury.ir.utils import get_io_buffers


_SUPPORTED_MATRICES = ("A", "B", "C")
_SUPPORTED_MODES = ("flexible", "fixed")
_SUPPORTED_TOPOLOGY_TOKENS = ("inter_node", "intra_node", "mixed")
_SUPPORTED_OPERATORS = ("gate", "up", "down")


@dataclass(frozen=True)
class MatrixDimMapping:
    """Constraint for one tensor dimension."""

    shard_topology: Optional[Tuple[str, ...]] = None

    @property
    def is_replicate(self) -> bool:
        return self.shard_topology is None

    def to_summary(self) -> str:
        if self.is_replicate:
            return "R"
        return "S(" + ",".join(self.shard_topology) + ")"


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

    if set(config.keys()) != {"shard"}:
        raise ValueError(f"{field_name} only supports the 'shard' field")

    tokens = _validate_topology_tokens(config["shard"], f"{field_name}.shard")
    return MatrixDimMapping(shard_topology=tokens)


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


def _resolve_topology_tokens(program: Program, tokens: Tuple[str, ...]) -> Tuple[int, ...]:
    if program.topology_metadata is None:
        raise ValueError("Program topology metadata must be set before matching tensor mappings")

    resolved_dims = []
    for token in tokens:
        dims = program.topology_metadata.get(f"{token}_dims")
        if dims is None:
            raise ValueError(f"Program topology metadata missing key '{token}_dims'")
        resolved_dims.extend(int(dim) for dim in dims)
    return tuple(sorted(set(resolved_dims)))


def program_satisfies_tensor_mapping_constraints(
    program: Program,
    constraints: Optional[TensorMappingConstraints],
) -> bool:
    """Return whether a program satisfies GEMM tensor mapping constraints."""
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
            if actual_dims != expected_dims:
                return False

    return True
