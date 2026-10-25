# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

## Abstract
Transformers are slow and memory-intensive on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. Approximate attention methods have attempted to address this by trading off model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM.

## GPU Memory Hierarchy and IO-Awareness
On modern GPUs (such as NVIDIA A100), compute throughput is significantly faster than memory bandwidth. The GPU memory hierarchy consists of:
1. High Bandwidth Memory (HBM): Large capacity (40GB to 80GB), but relatively low bandwidth (~1.5 TB/s to 2.0 TB/s).
2. On-chip SRAM: Extremely fast memory (~19 TB/s bandwidth), but small capacity (192 KB per Streaming Multiprocessor on A100, totaling around 20MB across the GPU).

Standard attention implementations materialize the intermediate $N \times N$ attention matrix $S = Q K^T$ and $P = \text{softmax}(S)$ in HBM. For sequence length $N = 4096$, this matrix requires gigabytes of memory traffic, causing memory bandwidth bottlenecks.

## Tiling Algorithm
FlashAttention computes exact softmax without storing the full $N \times N$ matrix in HBM by using tiling. It breaks the $Q$, $K$, and $V$ matrices into blocks that fit entirely inside fast SRAM.

### Online Softmax Formulation
Given a vector $x \in \mathbb{R}^B$, standard softmax requires computing:
$m = \max(x)$, $d = \sum_{i} e^{x_i - m}$, and $\text{softmax}(x)_i = \frac{e^{x_i - m}}{d}$.
When streaming over two blocks $x^{(1)}$ and $x^{(2)}$:
$m_{new} = \max(m^{(1)}, m^{(2)})$
$d_{new} = d^{(1)} e^{m^{(1)} - m_{new}} + d^{(2)} e^{m^{(2)} - m_{new}}$
This allows computing the exact attention output $O = P V$ in a single forward pass while keeping all intermediate matrices in SRAM.

## Experimental Results
FlashAttention achieves:
- 2x to 4x wall-clock speedup over standard PyTorch attention for sequence lengths from 128 to 4096.
- Up to 3x speedup on end-to-end BERT and GPT-2 training.
- Enables context lengths up to 64k tokens on single GPUs without memory exhaustion.
- Exact equivalence: FlashAttention produces exact attention weights with zero loss in mathematical precision compared to standard attention.
