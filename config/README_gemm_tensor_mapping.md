# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

# GEMM Tensor Mapping Config

这个文档说明 `config/gemm_tensor_mapping*.json` 的配置格式，重点说明矩阵 `fixed mapping` 的写法和含义。

## 用途

GEMM search 会枚举 A、B、C 三个矩阵在多 device 上的 tensor mapping。

- `flexible`：该矩阵不受约束，search 可以自由选择 mapping。
- `fixed`：该矩阵的 mapping 必须严格匹配配置；不匹配的候选会在 `search()` 内部被裁剪。

## 配置文件结构

```json
{
  "version": 1,
  "matrices": {
    "A": {"mode": "flexible"},
    "B": {"mode": "fixed", "mapping": ["R", {"shard": ["intra_node"]}]},
    "C": {"mode": "fixed", "mapping": [{"shard": ["inter_node"]}, {"shard": ["intra_node"]}]}
  }
}
```

字段含义：

- `version`
  - 当前固定为 `1`。
- `matrices`
  - GEMM 三个矩阵的配置集合。
  - 支持的 key 只有 `A`、`B`、`C`。
  - 没写的矩阵默认等价于 `{"mode": "flexible"}`。

## 每个矩阵的配置

每个矩阵对象支持两个字段：

- `mode`
  - 可选值：`"flexible"` 或 `"fixed"`。
- `mapping`
  - 只有 `mode = "fixed"` 时才能出现。
  - GEMM 中 A/B/C 都是二维矩阵，所以 `mapping` 必须是长度为 2 的数组。
  - `mapping[0]` 对应矩阵第 0 维。
  - `mapping[1]` 对应矩阵第 1 维。

非法情况：

- `mode = "fixed"` 但没有 `mapping`
- `mode = "flexible"` 但提供了 `mapping`
- `mapping` 长度不是 2
- 使用了 `A/B/C` 之外的矩阵名

## fixed mapping 的写法

`mapping` 中每一个维度只支持两种形式。

### 1. 复制

```json
"R"
```

含义：

- 该 tensor dim 在这个维度上是 replicated。
- 对应候选程序里的 `ShardingSpec` 必须是 `R`。

### 2. 切分

```json
{"shard": ["intra_node"]}
```

含义：

- 该 tensor dim 必须 shard 到指定的拓扑维度上。
- `shard` 的值是一个非空数组，表示这个 tensor dim 要映射到哪些 mesh 维。

支持的拓扑 token：

- `inter_node`
  - 映射到 `program.topology_metadata["inter_node_dims"]`
- `intra_node`
  - 映射到 `program.topology_metadata["intra_node_dims"]`
- `mixed`
  - 映射到 `program.topology_metadata["mixed_dims"]`

注意：

- `fixed` 是精确匹配，不是模糊匹配。
- 候选程序里实际 shard 的 mesh dim 集合，必须和配置解析出的 mesh dim 集合完全一致。
- `shard` 数组不能为空，也不能重复写同一个 token。

## 示例说明

示例文件：`config/gemm_tensor_mapping_fixed_example.json`

```json
{
  "version": 1,
  "matrices": {
    "A": {"mode": "flexible"},
    "B": {
      "mode": "fixed",
      "mapping": [
        "R",
        {"shard": ["intra_node"]}
      ]
    },
    "C": {
      "mode": "fixed",
      "mapping": [
        {"shard": ["inter_node"]},
        {"shard": ["intra_node"]}
      ]
    }
  }
}
```

含义：

- `A`
  - 不约束，search 自由决定。
- `B`
  - 第 0 维是 `R`
  - 第 1 维必须 shard 到 `intra_node`
- `C`
  - 第 0 维必须 shard 到 `inter_node`
  - 第 1 维必须 shard 到 `intra_node`

## mixed 的含义

当 search 把原始 `(inter_node, intra_node)` mesh reshape 成 `(world_size,)` 这种扁平形态时，某个 mesh dim 可能同时跨越 inter-node 和 intra-node。

这类维度会被标记为 `mixed`。

例如：

- 如果你希望只保留 flatten 后的一维切分候选，可以把某个维度写成：

```json
{"shard": ["mixed"]}
```

## 使用方法

默认配置：

```bash
python example_gemm_ir.py --mapping-config config/gemm_tensor_mapping.json
```

固定 mapping 示例：

```bash
python example_gemm_ir.py --mapping-config config/gemm_tensor_mapping_fixed_example.json
```

运行后，`summary.txt` 中会记录：

- 使用的 mapping config 路径
- A/B/C 的约束摘要
