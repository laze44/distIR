# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""
IR nodes for representing computation in auto-ring compiler.
"""
import ast
from typing import List, Tuple, Union, Optional, Dict, Any, Callable, TypeVar
from dataclasses import dataclass, field
import copy

from mercury.ir.distributed import DeviceMesh
from .elements import Axis, Buffer

T = TypeVar('T')

@dataclass
class IRNode:
    """Base class for all IR nodes."""
    def visit(self, fn: Callable[['IRNode'], T], fn_epiloge: Optional[Callable]=None) -> List[T]:
        """Visit traverses the IR tree and collects results of type T.
        
        Args:
            fn: A visitor function that takes an IRNode and returns a value of type T.
        Returns:
            List of non-None results from applying fn to this node and its children.
        """
        raise NotImplementedError("visit() must be implemented by IRNode subclasses")
    
    def __str__(self, tabs: int = 0) -> str:
        """String representation with proper indentation."""
        return super().__str__()

    def __deepcopy__(self, memo: Dict[int, Any]) -> 'IRNode':
        """Customize deep copy behavior to handle shared nodes.

        This method is called by copy.deepcopy() to create a deep copy of the object.
        """
        return self._deepcopy_impl(memo)

        
    def _deepcopy_impl(self, memo: Dict[int, Any]) -> 'IRNode':
        """Internal implementation of deep copy. Must be implemented by subclasses."""
        raise NotImplementedError("_deepcopy_impl() must be implemented by IRNode subclasses")

    
@dataclass
class RingComm(IRNode):
    """Node for ring communication."""
    axis: Axis
    num_cards: int
    name: str
    shard_dim: int
    write_back: bool = False # whether the data should go back to the original position
    # for reduction ring, it will be true, for other ring, it will be false

    def visit(self, fn: Callable[[IRNode], T], fn_epiloge: Optional[Callable]=None) -> List[T]:
        results = [fn(self)]
        if fn_epiloge:
            fn_epiloge(self)
        return [r for r in results if r is not None]
    
    def __str__(self, tabs: int = 0) -> str:
        """String representation with proper indentation."""
        prefix = "  " * tabs
        return f"{prefix}RingComm(name:{self.name} axis:{self.axis.name}, {self.num_cards})"
        
    def _deepcopy_impl(self, memo: Dict[int, Any]) -> 'RingComm':
        if id(self) in memo:
            return memo[id(self)]
        result = RingComm(
            axis=copy.deepcopy(self.axis, memo),
            num_cards=self.num_cards, 
            name=self.name,
            shard_dim=self.shard_dim,
            write_back=self.write_back,
        )
        memo[id(self)] = result
        return result


@dataclass
class AsyncCollectiveLifecycle:
    """Explicit async collective lifecycle markers for managed reductions."""

    start_op: str = "all_reduce_start"
    wait_on_reuse_op: str = "all_reduce_wait_on_reuse"
    drain_wait_op: str = "all_reduce_wait_drain"


@dataclass
class PendingTileDescriptor:
    """Structured descriptor for in-flight tiles in a managed-reduction pipeline.

    Carries the retire-time tile identity so codegen can retire completed tiles
    without reconstructing identity from ad-hoc arrays like ``pending_j``.

    Args:
        slot_index: The pipeline slot (0 .. stage_count-1) that holds this tile.
        tile_coords: Logical tile coordinates along the overlap axis.
        output_buffer: The output buffer that the tile retires into.
        retire_indices: Index expressions needed to write the retired tile.
        reduce_buffer: The reduction buffer whose collective is in flight.
    """

    slot_index: int = 0
    tile_coords: Optional[List[Union['Axis', int]]] = None
    output_buffer: Optional['Buffer'] = None
    retire_indices: Optional[List[Union[int, 'Axis']]] = None
    reduce_buffer: Optional['Buffer'] = None


@dataclass
class ManagedReductionPipelineRegion(IRNode):
    """Explicit IR form for a legalized async managed-reduction pipeline.

    Represents one managed reduction pipeline over one overlap axis.  Created
    by the legalization pass when an ``async_collective_overlap`` candidate
    satisfies all realizability checks.

    Args:
        reduce_op: The original ``ReduceOp`` that drives the pipeline.
        overlap_axis: The loop axis along which tiles are pipelined.
        stage_count: Number of double-buffer slots (typically 2).
        tile_count: Number of tiles along the overlap axis.
        lifecycle: Collective start/wait lifecycle markers.
        pending_tiles: Structured descriptors for in-flight tile state.
        consumer_store: The ``BufferStore`` that retires a completed tile.
        legalized: Whether the region passed legalization verification.
    """

    reduce_op: 'ReduceOp' = None
    overlap_axis: Optional[Axis] = None
    stage_count: int = 2
    tile_count: int = 1
    lifecycle: Optional[AsyncCollectiveLifecycle] = field(default_factory=AsyncCollectiveLifecycle)
    pending_tiles: List[PendingTileDescriptor] = field(default_factory=list)
    consumer_store: Optional['BufferStore'] = None
    legalized: bool = False

    def visit(self, fn: Callable[['IRNode'], T], fn_epiloge: Optional[Callable] = None) -> List[T]:
        results = [fn(self)]
        if self.reduce_op is not None:
            results.extend(self.reduce_op.visit(fn, fn_epiloge))
        if self.consumer_store is not None:
            results.extend(self.consumer_store.visit(fn, fn_epiloge))
        if fn_epiloge is not None:
            fn_epiloge(self)
        return [r for r in results if r is not None]

    def __str__(self, tabs: int = 0) -> str:
        prefix = "  " * tabs
        overlap_name = self.overlap_axis.name if self.overlap_axis is not None else None
        reduce_buf = self.reduce_op.buffer.tensor if self.reduce_op is not None else None
        return (
            f"{prefix}ManagedReductionPipelineRegion("
            f"reduce_buf={reduce_buf}, overlap_axis={overlap_name}, "
            f"stages={self.stage_count}, tiles={self.tile_count}, "
            f"legalized={self.legalized})"
        )

    def _deepcopy_impl(self, memo: Dict[int, Any]) -> 'ManagedReductionPipelineRegion':
        if id(self) in memo:
            return memo[id(self)]
        result = ManagedReductionPipelineRegion(
            reduce_op=copy.deepcopy(self.reduce_op, memo) if self.reduce_op is not None else None,
            overlap_axis=copy.deepcopy(self.overlap_axis, memo) if self.overlap_axis is not None else None,
            stage_count=self.stage_count,
            tile_count=self.tile_count,
            lifecycle=copy.deepcopy(self.lifecycle, memo),
            pending_tiles=copy.deepcopy(self.pending_tiles, memo),
            consumer_store=copy.deepcopy(self.consumer_store, memo) if self.consumer_store is not None else None,
            legalized=self.legalized,
        )
        memo[id(self)] = result
        return result

@dataclass
class PyNode(IRNode):
    """Node for preserving original Python AST."""
    node: ast.AST
    vars_in_scope: Dict[str, Union[Axis, Buffer]] = field(default_factory=dict)

    def visit(self, fn: Callable[[IRNode], T], fn_epiloge: Optional[Callable]=None) -> List[T]:
        results = [fn(self)]
        if fn_epiloge is not None:
            fn_epiloge(self)
        # Don't traverse into Python AST
        return [r for r in results if r is not None]

    def __str__(self, tabs: int = 0) -> str:
        prefix = "  " * tabs
        return f"{prefix}PyNode({ast.unparse(self.node)})"
        
    def _deepcopy_impl(self, memo: Dict[int, Any]) -> 'PyNode':
        if id(self) in memo:
            return memo[id(self)]
        result = PyNode(
            node=copy.deepcopy(self.node),
            vars_in_scope={k: copy.deepcopy(v, memo) for k, v in self.vars_in_scope.items()})
        memo[id(self)] = result
        return result
    
    def __eq__(self, value):
        if not isinstance(value, PyNode):
            return False
        return ast.unparse(self.node) == ast.unparse(value.node)

@dataclass
class Program(IRNode):
    """Program node representing a complete function."""
    name: str
    inputs: List[Any]
    defaults: List[Any]
    outputs: Any
    body: List[Union[IRNode, PyNode]]
    mesh: Optional[DeviceMesh] = None
    topology_metadata: Dict[str, List[int]] = field(default_factory=dict)

    def visit(self, fn: Callable[[IRNode], T], fn_epiloge: Optional[Callable]=None) -> List[T]:
        results = [fn(self)]
        for node in self.body:
            results.extend(node.visit(fn, fn_epiloge))
        if fn_epiloge is not None:
            fn_epiloge(self)
        return [r for r in results if r is not None]
    
    def __str__(self, tabs: int = 0) -> str:
        """String representation with proper indentation."""
        prefix = "  " * tabs
        res = f"{prefix}Program({self.name})\n"
        res += f"{prefix}Inputs: {self.inputs}\n"
        res += f"{prefix}Defaults: {self.defaults}\n"
        res += f"{prefix}Outputs: {self.outputs}\n"
        if len(self.topology_metadata) > 0:
            res += f"{prefix}Topology: {self.topology_metadata}\n"
        res += f"{prefix}Body:\n"
        for node in self.body:
            res += f"{node.__str__(tabs+1)}\n"
        return res
        
    def _deepcopy_impl(self, memo: Dict[int, Any]) -> 'Program':
        if id(self) in memo:
            return memo[id(self)]
        result = Program(
            name=self.name,
            inputs=copy.deepcopy(self.inputs, memo),
            defaults=copy.deepcopy(self.defaults, memo),
            outputs=copy.deepcopy(self.outputs, memo),
            body=[node._deepcopy_impl(memo) for node in self.body],
            mesh=copy.deepcopy(self.mesh, memo) if self.mesh else None,
            topology_metadata=copy.deepcopy(self.topology_metadata, memo))
        memo[id(self)] = result
        return result

@dataclass
class AxisDef(IRNode):
    """Node for axis definition."""
    axis: Axis

    def visit(self, fn: Callable[[IRNode], T], fn_epiloge: Optional[Callable]=None) -> List[T]:
        res = [fn(self)]
        if fn_epiloge is not None:
            fn_epiloge(self)
        return res
    
    def __str__(self, tabs: int = 0) -> str:
        """String representation with proper indentation."""
        return tabs * "  " + f"{self.axis}"
        
    def _deepcopy_impl(self, memo: Dict[int, Any]) -> 'AxisDef':
        if id(self) in memo:
            return memo[id(self)]
        result = AxisDef(axis=copy.deepcopy(self.axis, memo))
        memo[id(self)] = result
        return result

@dataclass
class BufferMatch(IRNode):
    """Node for buffer matching and axis binding."""
    buffer: Buffer
    tensor_name: Optional[str] # tensor name in the original code, or None if is temporary buffer

    def visit(self, fn: Callable[[IRNode], T], fn_epiloge: Optional[Callable]=None) -> List[T]:
        res = [fn(self)]
        if fn_epiloge is not None:
            fn_epiloge(self)
        return res
    
    def __str__(self, tabs: int = 0) -> str:
        """String representation with proper indentation."""
        return "  " * tabs + f"{self.tensor_name}: {self.buffer}"
        
    def _deepcopy_impl(self, memo: Dict[int, Any]) -> 'BufferMatch':
        if id(self) in memo:
            return memo[id(self)]
        result = BufferMatch(buffer=copy.deepcopy(self.buffer, memo),
                           tensor_name=self.tensor_name)
        memo[id(self)] = result
        return result

@dataclass
class GridLoop(IRNode):
    """Node representing a grid of loops with reduction marking."""
    axes: List[Axis]
    axis_types: str  # "s" for spatial, "r" for reduction
    body: List[Union[IRNode, PyNode]]

    def get_axis_type(self, axis: Axis) -> str:
        """Get the type of an axis in the loop."""
        for i, ax in enumerate(self.axes):
            if ax == axis:
                return self.axis_types[i]
        raise ValueError(f"Axis {axis} not found in loop {self}")

    def visit(self, fn: Callable[[IRNode], T], fn_epiloge: Optional[Callable]=None) -> List[T]:
        results = [fn(self)]
        for node in self.body:
            results.extend(node.visit(fn, fn_epiloge))
        if fn_epiloge is not None:
            fn_epiloge(self)
        return [r for r in results if r is not None]
    
    def __str__(self, tabs: int = 0) -> str:
        """String representation with proper indentation."""
        prefix = "  " * tabs
        axes = [ax.name for ax in self.axes]
        res = f"{prefix}Grid({', '.join(axes)}, types={self.axis_types})\n"
        res += f"{prefix}Body:\n"
        for node in self.body:
            res += f"{node.__str__(tabs+1)}\n"
        return res

    def _deepcopy_impl(self, memo: Dict[int, Any]) -> 'GridLoop':
        if id(self) in memo:
            return memo[id(self)]
        result = GridLoop(
            axes=[copy.deepcopy(axis, memo) for axis in self.axes],
            axis_types=self.axis_types,
            body=[node._deepcopy_impl(memo) for node in self.body])
        memo[id(self)] = result
        return result


# @dataclass
# class InitBlock(IRNode):
#     """Node for reduction initialization block."""
#     init_axis: Union[Axis, int]
#     body: List[Union[IRNode, PyNode]]

#     def visit(self, fn: Callable[[IRNode], T]) -> List[T]:
#         results = [fn(self)]
#         for node in self.body:
#             results.extend(node.visit(fn))
#         return [r for r in results if r is not None]

@dataclass
class BufferStore(IRNode):
    """Node for storing to a buffer."""
    buffer: Buffer
    indices: List[Union[int, Axis, PyNode]]
    value: Union[PyNode, str]
    comm: List[RingComm] = field(default_factory=list)

    def __str__(self, tabs: int = 0) -> str:
        """String representation with proper indentation."""
        prefix = "  " * tabs
        res = f"{prefix}{self.buffer.tensor}[{_format_indices(self.indices)}] = {_format_value(self.value)}"
        res += f" {self.comm}"
        return res
        
    def _deepcopy_impl(self, memo: Dict[int, Any]) -> 'BufferStore':
        if id(self) in memo:
            return memo[id(self)]
        result = BufferStore(
            buffer=copy.deepcopy(self.buffer, memo),
            indices=[copy.deepcopy(idx, memo) if isinstance(idx, (IRNode, Axis)) 
                    else idx for idx in self.indices],
            value=copy.deepcopy(self.value, memo) if isinstance(self.value, IRNode) 
                  else self.value,
            comm=[comm._deepcopy_impl(memo) for comm in self.comm])
        memo[id(self)] = result
        return result

    def visit(self, fn: Callable[[IRNode], T], fn_epiloge: Optional[Callable]=None) -> List[T]:
        results = [fn(self)]
        for idx in self.indices:
            if isinstance(idx, PyNode):
                results.extend(idx.visit(fn))
        for comm in self.comm:
            results.extend(comm.visit(fn))
        if fn_epiloge is not None:
            fn_epiloge(self)
        return [r for r in results if r is not None]

@dataclass
class BufferLoad(IRNode):
    """Node for loading from a buffer."""
    buffer: Buffer
    indices: List[Union[int, Axis, PyNode]]
    target: Union[PyNode, str]
    comm: List[RingComm] = field(default_factory=list)

    def visit(self, fn: Callable[[IRNode], T], fn_epiloge: Optional[Callable]=None) -> List[T]:
        results = [fn(self)]
        for idx in self.indices:
            if isinstance(idx, PyNode):
                results.extend(idx.visit(fn))
        
        for comm in self.comm:
            results.extend(comm.visit(fn))

        if fn_epiloge is not None:
            fn_epiloge(self)

        return [r for r in results if r is not None]
    
    def __str__(self, tabs: int = 0) -> str:
        """String representation with proper indentation."""
        prefix = "  " * tabs
        res = f"{prefix}{self.target} = {self.buffer.tensor}[{_format_indices(self.indices)}]"
        res += f" {self.comm}"
        return res
        
    def _deepcopy_impl(self, memo: Dict[int, Any]) -> 'BufferLoad':
        if id(self) in memo:
            return memo[id(self)]
        result = BufferLoad(
            buffer=copy.deepcopy(self.buffer, memo),
            indices=[copy.deepcopy(idx, memo) if isinstance(idx, (IRNode, Axis)) 
                    else idx for idx in self.indices],
            target=copy.deepcopy(self.target, memo) if isinstance(self.target, IRNode) 
                   else self.target,
            comm=[comm._deepcopy_impl(memo) for comm in self.comm])
        memo[id(self)] = result
        return result
    
# @dataclass
# class BinaryOp(IRNode):
#     """Node for binary operations."""
#     op: str  # "+", "*", etc.
#     left: Union[BufferLoad, 'BinaryOp', float, PyNode]
#     right: Union[BufferLoad, 'BinaryOp', float, PyNode]

#     def visit(self, fn: Callable[[IRNode], T]) -> List[T]:
#         results = [fn(self)]
#         if isinstance(self.left, (BufferLoad, BinaryOp, PyNode)):
#             results.extend(self.left.visit(fn))
#         if isinstance(self.right, (BufferLoad, BinaryOp, PyNode)):
#             results.extend(self.right.visit(fn))
#         return [r for r in results if r is not None]
        
#     def _deepcopy_impl(self, memo: Dict[int, Any]) -> 'BinaryOp':
#         if id(self) in memo:
#             return memo[id(self)]
#         result = BinaryOp(
#             op=self.op,
#             left=copy.deepcopy(self.left, memo) if isinstance(self.left, (IRNode, PyNode)) 
#                  else self.left,
#             right=copy.deepcopy(self.right, memo) if isinstance(self.right, (IRNode, PyNode)) 
#                   else self.right)
#         memo[id(self)] = result
#         return result

@dataclass
class ReduceOp(IRNode):
    """Node for calling reduce operations."""
    op: str
    buffer: Buffer
    src: Union[PyNode, str]
    axes: List[Axis] = field(default_factory=list)
    collective_op: Optional[str] = None
    comm: List[RingComm] = field(default_factory=list)
    shard_dim: List[int] = field(default_factory=list)
    indices: Optional[List[Union[int, Axis, PyNode]]] = None
    managed_collective_strategy: str = "blocking_collective"
    async_collective_overlap_axis: Optional[Axis] = None
    async_collective_tile_count: int = 1
    async_collective_stage_count: int = 1
    async_collective_lifecycle: Optional[AsyncCollectiveLifecycle] = None

    def visit(self, fn: Callable[[IRNode], T], fn_epiloge: Optional[Callable]=None) -> List[T]:
        results = [fn(self)]
        for comm in self.comm:
            results.extend(comm.visit(fn))
        if fn_epiloge is not None:
            fn_epiloge(self)
        return [r for r in results if r is not None]
        
    def _deepcopy_impl(self, memo: Dict[int, Any]) -> 'ReduceOp':
        if id(self) in memo:
            return memo[id(self)]
        new_indices = None
        if self.indices is not None:
            new_indices = [
                copy.deepcopy(idx, memo) if isinstance(idx, (IRNode, Axis)) 
                else idx for idx in self.indices]
        result = ReduceOp(
            op=self.op,
            collective_op=self.collective_op,
            buffer=copy.deepcopy(self.buffer, memo),
            axes=copy.deepcopy(self.axes, memo),
            src=copy.deepcopy(self.src, memo) if isinstance(self.src, IRNode) 
                   else self.src,
            comm=[comm._deepcopy_impl(memo) for comm in self.comm],
            shard_dim=copy.deepcopy(self.shard_dim, memo),
            indices=new_indices,
            managed_collective_strategy=self.managed_collective_strategy,
            async_collective_overlap_axis=copy.deepcopy(self.async_collective_overlap_axis, memo),
            async_collective_tile_count=self.async_collective_tile_count,
            async_collective_stage_count=self.async_collective_stage_count,
            async_collective_lifecycle=copy.deepcopy(self.async_collective_lifecycle, memo),
        )
        memo[id(self)] = result
        return result
    
    def __str__ (self, tabs: int = 0) -> str:
        prefix = "  " * tabs
        res = prefix + f"{self.op}({self.buffer.tensor}[{_format_indices(self.indices) if self.indices is not None else None}], axes={self.axes}, src={self.src}) (collective_op={self.collective_op})"
        res += f" {self.comm} with shard_dim={self.shard_dim}"
        if self.managed_collective_strategy != "blocking_collective":
            overlap_axis = (
                self.async_collective_overlap_axis.name
                if self.async_collective_overlap_axis is not None
                else None
            )
            res += (
                f" strategy={self.managed_collective_strategy}"
                f" overlap_axis={overlap_axis}"
                f" tiles={self.async_collective_tile_count}"
                f" stages={self.async_collective_stage_count}"
            )
        return res

def _format_value(value: Union[BufferLoad, float, PyNode]) -> str:
    """Helper to format values for __str__ methods."""
    if isinstance(value, PyNode):
        return f"py({ast.dump(value.node)})"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)

def _format_indices(indices: List[Union[int, Axis, PyNode]]) -> str:
    """Helper to format indices for __str__ methods."""
    return ", ".join(str(idx) for idx in indices)
