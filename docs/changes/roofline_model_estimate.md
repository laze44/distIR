## roofline moddel
单卡上面的单个算子运算时间估算: roofline model

### Roofline Model Formula

The roofline model estimates peak performance as:

$$\text{Performance} = \min(\text{Peak Flops}, \text{Peak Memory Bandwidth} \times \text{Arithmetic Intensity})$$

Where:
- **Peak Flops**: Maximum floating-point operations per second (FP32/FP64)
- **Peak Memory Bandwidth**: Maximum data transfer rate from memory (GB/s)
- **Arithmetic Intensity**: Ratio of operations to data movement (FLOPs/Byte)

## 互联传输时间计算:

### 点对点的互联传输时间可以通过以下公式计算:
connection_time = data_size / bandwidth + connection_latency

### 集合通信互联时间:
Intra-node & Inter-node communication: 采用环形算法估算时间.

all-reduce: connection_time = connection_latency + 2 * (num_nodes - 1) / num_nodes * data_size / bandwidth

all-gather: connection_time = connection_latency + (num_nodes - 1) / num_nodes * data_size / bandwidth

reduce-scatter: connection_time = connection_latency + (num_nodes - 1) / num_nodes * data_size / bandwidth

