# GLM5 FFN模块分步计算
## 1. 基础符号定义
| 符号          | 含义                     | 典型值（GLM5-7B） |
|---------------|--------------------------|-------------------|
| $B$           | 批次大小                 | 自定义            |
| $L$           | 序列长度                 | 4096              |
| $d_{\text{model}}$ | 模型隐藏层维度         | 4096              |
| $d_{\text{ffn}}$   | FFN中间层维度（$4 \times d_{\text{model}}$） | 16384 |

## 2. FFN模块分步计算
### 步骤1：双路投影（升维）
$$
\begin{align}
X_{\text{gate}} &= X \cdot W_{\text{gate}} + b_{\text{gate}} \\
X_{\text{up}} &= X \cdot W_{\text{up}} + b_{\text{up}}
\end{align}
$$
- 输入：$X \in \mathbb{R}^{[B, L, d_{\text{model}}]}$
- 参数：
  - $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{[d_{\text{model}}, d_{\text{ffn}}]}$
  - $b_{\text{gate}}, b_{\text{up}} \in \mathbb{R}^{[d_{\text{ffn}}]}$
- 输出：$X_{\text{gate}}, X_{\text{up}} \in \mathbb{R}^{[B, L, d_{\text{ffn}}]}$

### 步骤2：门控激活与融合
$$
X_{\text{activated}} = \text{SiLU}(X_{\text{gate}}) \odot X_{\text{up}}
$$
- $\text{SiLU}(x) = x \cdot \sigma(x)$（$\sigma$为sigmoid函数）
- $\odot$：逐元素乘法
- 输出：$X_{\text{activated}} \in \mathbb{R}^{[B, L, d_{\text{ffn}}]}$

### 步骤3：降维投影
$$
Y = X_{\text{activated}} \cdot W_{\text{down}} + b_{\text{down}}
$$
- 参数：
  - $W_{\text{down}} \in \mathbb{R}^{[d_{\text{ffn}}, d_{\text{model}}]}$
  - $b_{\text{down}} \in \mathbb{R}^{[d_{\text{model}}]}$
- 输出：$Y \in \mathbb{R}^{[B, L, d_{\text{model}}]}$

### 总结
1. GLM5 FFN核心为「双路升维→门控融合→降维回归」，激活函数为SiLU；
2. 全程保持$[B, L]$维度不变，仅在$d_{\text{model}}$与$4 \times d_{\text{model}}$间切换；
3. 最终输出维度与输入一致（$[B, L, d_{\text{model}}]$），支持残差连接。
