### Module & Class Docstrings
- Modules have triple-quoted docstrings immediately after the copyright header.
- Classes and public methods use Google-style docstrings with `Args:` and `Returns:` sections.

### Imports
Follow this ordering (separated by blank lines):
1. Standard library (`copy`, `typing`, `ast`, `enum`, `dataclasses`)
2. Third-party (`torch`, `numpy`, `pytest`)
3. Local — relative imports within the same package, absolute for cross-package:
   ```python
   from .elements import Axis, Buffer
   from mercury.ir.nodes import IRNode
   ```

### Type Annotations
- Use type hints extensively on function signatures and class fields.
- Import from `typing`: `List`, `Dict`, `Optional`, `Union`, `Tuple`, `Callable`, `TypeVar`.
- Dataclass fields are always annotated.
- Do NOT use `from __future__ import annotations` (project targets Python 3.8+).

### Naming Conventions
| Element | Convention | Examples |
|---------|-----------|----------|
| Classes | PascalCase | `IRNode`, `GridLoop`, `DeviceMesh`, `ShardType` |
| Functions/methods | snake_case | `collect_axis`, `enumerate_mesh_shapes` |
| Visitor methods | `visit_` + class name | `visit_BufferLoad`, `visit_GridLoop` |
| Private methods | Leading underscore | `_deepcopy_impl`, `_device_grid` |
| Constants/enums | UPPER_CASE members | `ShardType.REPLICATE`, `ShardType.SHARD` |
| DSL axis names | Single uppercase letter | `I`, `J`, `K` |
| Test functions | `test_` prefix | `test_matmul_ir_gen`, `test_visitor_traversal` |

### Dataclasses
The IR layer uses `@dataclass` extensively. Every IR node subclass:
- Inherits from `IRNode` (the base dataclass).
- Implements `visit(self, visitor)` dispatching to the appropriate visitor method.
- Implements `_deepcopy_impl(self, memo)` for deep copy support.
- Uses `field(default_factory=...)` for mutable defaults.

```python
@dataclass
class GridLoop(IRNode):
    axes: List[Axis]
    body: List[IRNode] = field(default_factory=list)

    def visit(self, visitor):
        return visitor.visit_GridLoop(self)

    def _deepcopy_impl(self, memo):
        return GridLoop(
            axes=copy.deepcopy(self.axes, memo),
            body=copy.deepcopy(self.body, memo),
        )
```

### Error Handling
- Raise `ValueError` with descriptive messages for invalid inputs or states.
- Do NOT use bare `except:` or empty `except Exception: pass`.
- In tests, assert exceptions with `pytest.raises(ValueError, match="...")`.

### Visitor Pattern
The visitor pattern is central to the IR. When adding a new IR node:
1. Define the dataclass in `mercury/ir/nodes.py`.
2. Add `visit_<ClassName>` method to the `IRVisitor` base class.
3. Implement `visit()` and `_deepcopy_impl()` on the new node.
4. Update all concrete visitors that need to handle the new node.