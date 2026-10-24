"""
Problem: Multi-Headed Self-Attention
Module: model/multi_head_attention.py
Description: Implement multi-head causal self-attention: run several single-head
             attention heads in parallel with a smaller head dimension, concatenate
             their outputs along the embedding axis, then apply an output projection.
Source: https://neetcode.io/problems/multi-headed-self-attention/question

Công thức:
    head_i = Attention(X W_Q^i, X W_K^i, X W_V^i)       với d_head = d_model / h
    MultiHead(X) = Concat(head_1, ..., head_h) W_O

    - Mỗi head chiếu input vào không gian con có số chiều `attention_dim // num_heads`.
    - Các head học các kiểu quan hệ khác nhau (cú pháp, đồng tham chiếu, vị trí...).
    - Concat theo trục embedding (dim=2) -> (B, T, h * d_head) = (B, T, attention_dim).
    - Output projection W_O (Linear, không bias) trộn thông tin giữa các head.

Ý tưởng:
    1. `nn.ModuleList` chứa `num_heads` instance `SingleHeadAttention`, mỗi head
       nhận `attention_dim // num_heads`.
    2. Mỗi head tự tính Q, K, V, scaled dot-product + causal mask, trả về
       (B, T, d_head).
    3. `torch.cat(head_outputs, dim=2)` nối các head theo trục embedding.
    4. `output_projection` (attention_dim -> attention_dim, bias=False) trộn các head.
    5. `torch.round(output, decimals=4)` làm tròn kết quả cuối cùng.

Lưu ý quan trọng khi làm bài:
    - `torch.manual_seed(0)` nằm trong `SingleHeadAttention.__init__` nên MỖI head
      đều được khởi tạo lại từ cùng một seed. Hệ quả: các head cùng kích thước có
      trọng số giống hệt nhau -> output các head trùng nhau.
    - Đây là hành vi mà template/đề yêu cầu để khớp kết quả kiểm thử; KHÔNG tự ý
      bỏ seed hay đổi thứ tự khởi tạo K/Q/V. Trong code thực tế ta sẽ seed một lần
      ở đầu chương trình (hoặc để mặc định) để các head học các không gian khác nhau.
    - `SingleHeadAttention` được định nghĩa SAU `MultiHeadAttention` vẫn hợp lệ:
      Python chỉ cần tên này tồn tại tại thời điểm `__init__` được gọi, không phải
      lúc module được import.

Complexity:
    Với B batch, T sequence length, d = attention_dim, h head:
    - Thời gian: O(B * T^2 * d) — tổng chi phí attention của các head cộng lại
      vẫn tuyến tính theo d (mỗi head O(B * T^2 * d/h)).
    - Bộ nhớ:   O(B * T^2 * h) cho attention weights + O(B * T * d) cho output.

Key Takeaways for mini-GPT:
    - Đây là thành phần "Multi-Head Attention" trong Transformer/GPT: nhiều head
      cho phép mô hình chú ý đồng thời vào nhiều vị trí/khía cạnh khác nhau.
    - Trong GPT-2, `d_model = 768`, `h = 12` -> mỗi head `d_head = 64`. Tổng chi phí
      tính toán gần như tương đương 1 head có `d_model = 768`, nhưng biểu diễn tốt hơn.
    - Cài đặt tối ưu thường gộp 3 phép chiếu thành một `Linear(n_embd, 3 * n_embd)`
      rồi `split`, và gộp các head bằng `view/transpose` thay vì vòng lặp Python.
    - Causal mask + nhiều head là nền tảng của decoder-only GPT; phần này sẽ được
      ghép với residual connection và LayerNorm trong `model/transformer.py`.
"""

import torch
import torch.nn as nn
from torchtyping import TensorType


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        # Mỗi head hoạt động trên không gian con có số chiều attention_dim / num_heads.
        # (Giả định attention_dim chia hết cho num_heads.)
        self.heads = nn.ModuleList([
            SingleHeadAttention(embedding_dim, attention_dim // num_heads)
            for _ in range(num_heads)
        ])
        # Trộn thông tin giữa các head: (B, T, attention_dim) -> (B, T, attention_dim)
        self.output_projection = nn.Linear(
            attention_dim, attention_dim, bias=False
        )

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Mỗi head trả về (B, T, attention_dim / num_heads)
        head_outputs = [head(embedded) for head in self.heads]

        # Nối theo trục embedding: (B, T, h * d_head) = (B, T, attention_dim)
        concatenated = torch.cat(head_outputs, dim=2)

        # Chiếu tuyến tính để tổng hợp các head
        output = self.output_projection(concatenated)

        return torch.round(output, decimals=4)


class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        # Seed lại mỗi lần tạo head theo yêu cầu của template/đề bài.
        torch.manual_seed(0)
        self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # embedded: (B, T, embedding_dim)
        k = self.key_gen(embedded)      # (B, T, attention_dim)
        q = self.query_gen(embedded)    # (B, T, attention_dim)
        v = self.value_gen(embedded)    # (B, T, attention_dim)

        # scores[b, i, j] = q_i . k_j  -> (B, T, T)
        scores = q @ k.transpose(1, 2)

        # Scale bởi sqrt(d_k); k.shape[2] chính là d_k.
        scores = scores / (k.shape[2] ** 0.5)

        # Causal mask: token i chỉ được chú ý tới j <= i.
        context_length = k.shape[1]
        lower_triangular = torch.tril(
            torch.ones(
                context_length,
                context_length,
                device=embedded.device
            )
        )
        # Vị trí tương lai -> -inf => softmax cho trọng số 0.
        scores = scores.masked_fill(
            lower_triangular == 0,
            float("-inf")
        )
        scores = nn.functional.softmax(scores, dim=2)

        # Tổng có trọng số của Value: (B, T, T) @ (B, T, d) -> (B, T, d)
        return scores @ v
