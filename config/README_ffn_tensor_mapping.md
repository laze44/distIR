# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

# FFN Tensor Mapping Config

这个文档说明 `config/ffn_tensor_mapping.json` 的配置格式。该格式用于
`example_ffn_ir.py` 和 FFN operator-scoped 搜索入口。

## 用途

FFN 搜索包含三个 GEMM operator：

- `gate`
- `up`
- `down`

每个 operator 都有独立的 `A/B/C` 约束。

- `flexible`：不约束该矩阵 mapping。
- `fixed`：必须严格匹配配置 mapping。

## 配置结构

```json
{
  "version": 1,
  "operators": {
    "gate": {
      "matrices": {
        "A": {"mode": "flexible"},
        "B": {"mode": "fixed", "mapping": [{"shard": ["intra_node"]}, "R"]},
        "C": {"mode": "flexible"}
      }
    },
    "up": {
      "matrices": {
        "A": {"mode": "flexible"},
        "B": {"mode": "flexible"},
        "C": {"mode": "flexible"}
      }
    },
    "down": {
      "matrices": {
        "A": {"mode": "flexible"},
        "B": {"mode": "flexible"},
        "C": {"mode": "flexible"}
      }
    }
  }
}
```

字段说明：

- `version`：固定为 `1`
- `operators`：operator 级别配置集合
- operator key 仅支持 `gate` / `up` / `down`
- 每个 operator 的 `matrices` 仅支持 `A` / `B` / `C`

## FFN 特殊约束

FFN 中只允许权重矩阵 `B` 使用 `fixed mapping`：

- `A` 必须是 `flexible`
- `C` 必须是 `flexible`

如果 `A` 或 `C` 被配置为 `fixed`，加载时会报错。

## 默认行为

- 未声明的 operator 默认等价于该 operator 全部 `A/B/C` 为 `flexible`
- 未声明的 matrix 默认等价于该 matrix 为 `flexible`

## 使用方式

```bash
python example_ffn_ir.py \
  --batch 1 \
  --seq-len 64 \
  --d-model 256 \
  --d-ffn 1024 \
  --inter-node 1 \
  --intra-node 2 \
  --mapping-config config/ffn_tensor_mapping.json
```
