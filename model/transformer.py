"""
Problem: Transformer Block
Module: model/transformer.py
Description: Implement a Pre-LayerNorm Transformer block: multi-head causal
             self-attention with a residual connection, then a position-wise
             feed-forward network (model_dim -> 4*model_dim -> model_dim) with
             ReLU and dropout, also wrapped in a residual connection.
Source: https://neetcode.io/problems/transformer-block/question

Công thức:
    x = x + MultiHeadAttention(LayerNorm_1(x))
    x = x + FFN(LayerNorm_2(x))

    Với FFN (VanillaNeuralNetwork):
        FFN(x) = Dropout( W_down · ReLU(W_up · x + b_up) + b_down )
        W_up:   (4 * model_dim, model_dim)
        W_down: (model_dim, 4 * model_dim)

    - Đây là kiến trúc Pre-LN: chuẩn hóa TRƯỚC sub-layer, rồi mới cộng residual.
      Gradient chạy thẳng qua nhánh residual nên mạng sâu train ổn định hơn Post-LN.
    - Residual connection giải quyết vanishing gradient và giữ thông tin gốc.
    - FFN mở rộng 4x rồi thu lại: tăng năng lực biểu diễn phi tuyến cho từng vị trí.

Ý tưởng:
    1. `layer_norm_1(embedded)` -> multi-head causal self-attention -> cộng residual.
    2. `layer_norm_2(x)` -> feed-forward -> cộng residual.
    3. `torch.round(x, decimals=4)` làm tròn kết quả cuối.

Thứ tự khởi tạo (theo đề, vì `torch.manual_seed(0)` làm kết quả phụ thuộc thứ tự):
    1. `attention`       (MultiHeadedSelfAttention)
    2. `feed_forward`    (VanillaNeuralNetwork)
    3. `layer_norm_1`
    4. `layer_norm_2`
    Đừng đổi thứ tự và đừng bỏ `torch.manual_seed(0)`.

Lưu ý khi làm bài:
    - `torch.manual_seed(0)` ở đầu `forward` khiến dropout tất định giữa các lần
      chạy -> output tái lập được (khớp test của đề). Trong code thật KHÔNG seed
      trong forward, vì như vậy dropout luôn dùng cùng một mask và làm giảm tác
      dụng regularization.
    - Các class `MultiHeadedSelfAttention` và `VanillaNeuralNetwork` được lồng bên
      trong `TransformerBlock`; `__init__` của block tham chiếu chúng dạng
      `self.MultiHeadedSelfAttention(...)`.
    - `SingleHeadAttention` (bên trong MultiHeadedSelfAttention) dùng `torch.tril`
      không chỉ định `device`; khi chạy CPU thì ổn, còn GPU cần thêm
      `device=embedded.device`.

Complexity:
    Với B batch, T sequence length, d = model_dim:
    - Attention: O(B * T^2 * d) (chi phối khi T lớn).
    - Feed-forward: O(B * T * d * 4d) = O(B * T * d^2) (chi phối khi d lớn).
    - Bộ nhớ: O(B * T * d) cho activations + O(B * T^2) cho attention scores.

Key Takeaways for mini-GPT:
    - Một GPT = embedding + positional encoding + N block TransformerBlock
      (GPT-2 small: 12 block, d = 768, 12 head) + LayerNorm cuối + LM head.
    - GPT-2 dùng Pre-LN đúng như block này; khác biệt nhỏ: GPT dùng GELU thay ReLU
      và thường có bias trong các Linear (bài này không bias cho projection).
    - Residual + LayerNorm + dropout là bộ ba giúp train ổn định; khi ghép nhiều
      block, residual stream chính là "xương sống" mà mọi sub-layer đọc/ghi vào.
    - Sau khi hoàn thành block, chỉ cần xếp chồng N lần là có `model/gpt.py`.
"""

import torch
import torch.nn as nn
from torchtyping import TensorType


class TransformerBlock(nn.Module):

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        # Thứ tự khởi tạo quan trọng (seed 0) theo yêu cầu của đề:
        # attention -> feed-forward -> layer_norm_1 -> layer_norm_2.
        self.attention = self.MultiHeadedSelfAttention(model_dim, num_heads)
        self.feed_forward = self.VanillaNeuralNetwork(model_dim)
        self.layer_norm_1 = nn.LayerNorm(model_dim)
        self.layer_norm_2 = nn.LayerNorm(model_dim)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Seed lại để dropout tất định -> output tái lập được cho test.
        torch.manual_seed(0)

        # Sub-layer 1: Pre-LN + causal multi-head attention + residual.
        x = embedded + self.attention(self.layer_norm_1(embedded))

        # Sub-layer 2: Pre-LN + feed-forward + residual.
        x = x + self.feed_forward(self.layer_norm_2(x))

        return torch.round(x, decimals=4)

    class MultiHeadedSelfAttention(nn.Module):

        class SingleHeadAttention(nn.Module):

            def __init__(self, model_dim: int, head_size: int):
                super().__init__()
                torch.manual_seed(0)
                self.key_gen = nn.Linear(model_dim, head_size, bias=False)
                self.query_gen = nn.Linear(model_dim, head_size, bias=False)
                self.value_gen = nn.Linear(model_dim, head_size, bias=False)

            def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                # embedded: (B, T, model_dim)
                k = self.key_gen(embedded)      # (B, T, head_size)
                q = self.query_gen(embedded)    # (B, T, head_size)
                v = self.value_gen(embedded)    # (B, T, head_size)

                # scores[b, i, j] = q_i . k_j, scale bởi sqrt(d_head)
                scores = q @ k.transpose(1, 2)
                scores = scores / (k.shape[2] ** 0.5)

                # Causal mask: token i chỉ thấy j <= i.
                context_length = k.shape[1]
                lower_triangular = torch.tril(
                    torch.ones(context_length, context_length)
                )
                scores = scores.masked_fill(
                    lower_triangular == 0, float("-inf")
                )
                scores = nn.functional.softmax(scores, dim=2)

                # Tổng có trọng số của Value
                return scores @ v

        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)
            # Mỗi head có chiều model_dim / num_heads.
            self.att_heads = nn.ModuleList()
            for _ in range(num_heads):
                self.att_heads.append(
                    self.SingleHeadAttention(
                        model_dim, model_dim // num_heads
                    )
                )
            # Trộn thông tin giữa các head.
            self.output_proj = nn.Linear(model_dim, model_dim, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            # Concat các head theo trục embedding -> (B, T, model_dim)
            head_outputs = [
                head(embedded) for head in self.att_heads
            ]
            concatenated = torch.cat(head_outputs, dim=2)
            return self.output_proj(concatenated)

    class VanillaNeuralNetwork(nn.Module):

        def __init__(self, model_dim: int):
            super().__init__()
            torch.manual_seed(0)
            # Mở rộng 4x -> ReLU -> thu về model_dim -> dropout.
            self.up_projection = nn.Linear(model_dim, model_dim * 4)
            self.relu = nn.ReLU()
            self.down_projection = nn.Linear(model_dim * 4, model_dim)
            self.dropout = nn.Dropout(0.2)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            # (B, T, model_dim) -> (B, T, 4 * model_dim)
            x = self.up_projection(embedded)
            x = self.relu(x)
            # (B, T, 4 * model_dim) -> (B, T, model_dim)
            x = self.down_projection(x)
            return self.dropout(x)
