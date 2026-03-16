# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Search utilities and theoretical estimation interfaces."""

from .estimate import EstimateResult, estimate_program, load_hardware_config
from .ffn_graph_search import FFNJointSearchResult, search_ffn
from .ffn_two_step_search import (
    EdgeReshardTransition,
    FFNEdgeLoweringOwnership,
    FFNLayoutPlan,
    FFNOperatorBoundaryClass,
    FFNSegmentSelection,
    FFNStep1LayoutStats,
    FFNTwoStepSearchResult,
    search_ffn_two_step,
)
from .gemm_two_step_search import (
    GEMMLayoutPlan,
    GEMMTwoStepSearchResult,
    search_gemm_two_step,
)
from .mapping_constraints import (
    ExactLayoutSignature,
    ExactTensorLayoutConstraints,
    LogicalBoundaryLayoutSignature,
    LogicalTensorLayoutConstraints,
    OperatorTensorMappingConstraints,
    TensorMappingConstraints,
    derive_logical_local_shape,
    exact_layout_signature_from_buffer,
    logical_layout_signature_from_buffer,
    load_operator_tensor_mapping_constraints,
    load_tensor_mapping_constraints,
    program_satisfies_logical_layout_constraints,
    program_satisfies_exact_layout_constraints,
    program_satisfies_tensor_mapping_constraints,
    resolve_topology_tokens_from_metadata,
)
from .reshard_estimate import estimate_reshard_time
