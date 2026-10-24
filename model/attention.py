"""
Problem: Self-Attention Head
Module: model/attention.py
Description: Implement a single causal (masked) self-attention head: project the
             input into Query/Key/Value, compute scaled dot-product attention with
             a lower-triangular mask, then return the weighted sum of Values.
Source: https://neetcode.io/problems/self-attention/question

Công thức:
    Q = X W_Q,  K = X W_K,  V = X W_V
    scores = (Q K^T) / sqrt(d_k)
    scores = scores + M      với M[i, j] = 0 nếu j <= i, -inf nếu j > i
    A = softmax(scores, dim=-1)
    out = A V

    - Chia cho sqrt(d_k) giữ phương sai của tích vô hướng ổn định (~1) thay vì
      tăng theo d_k, tránh softmax bị bão hòa (gradient gần 0) khi d_k lớn.
    - Causal mask đảm bảo tính autoregressive: token ở vị trí i chỉ được nhìn
      các token j <= i (không "nhìn thấy tương lai").
    - softmax trên các vị trí bị mask -inf cho trọng số đúng bằng 0.

Ý tưởng:
    1. Ba lớp Linear (không bias) chiếu input từ `embedding_dim` -> `attention_dim`
       để tạo K, Q, V. Khởi tạo theo đúng thứ tự key/query/value và
       `torch.manual_seed(0)` để weight tái lập được (đề bài yêu cầu).
    2. `Q @ K.transpose(-2, -1)` -> scores shape (B, T, T): scores[b, i, j] là độ
       tương đồng giữa query của token i và key của token j.
    3. `torch.tril(torch.ones(T, T))` tạo ma trận tam giác dưới (kể cả đường chéo);
       `masked_fill(mask == 0, -inf)` chặn các vị trí j > i.
    4. `softmax(dim=2)` chuẩn hóa theo trục key (mỗi hàng query cộng lại = 1).
    5. `attention_weights @ V` -> (B, T, attention_dim).
    6. `torch.round(..., decimals=4)` làm tròn output theo yêu cầu.

Complexity:
    Với B batch, T sequence length, d = attention_dim:
    - Thời gian: O(B * T^2 * d) cho QK^T và O(B * T^2 * d) cho A @ V
      (chi phí bậc hai theo độ dài chuỗi là lý do GPT cần KV-cache khi sinh text).
    - Bộ nhớ:   O(B * T^2) cho ma trận attention scores/weights.

Key Takeaways for mini-GPT:
    - Đây chính là "scaled dot-product attention" trong "Attention Is All You Need",
      nhưng chỉ 1 head. Multi-head attention chạy song song nhiều head rồi concat.
    - Mask nhân quả (causal) là khác biệt cốt lõi giữa encoder (bidirectional) và
      decoder/GPT (autoregressive). Khi train GPT, ta tính cả chuỗi một lần nhờ mask
      này thay vì lặp từng token.
    - Trong GPT-2: attention_dim = n_embd / n_head cho mỗi head; Q, K, V thường
      được gộp thành một Linear(n_embd, 3 * n_embd) để tối ưu.
    - Khi inference có KV-cache, chỉ token mới nhất cần tính Q; K và V của các
      token trước được lưu lại, nên không cần mask (mỗi bước chỉ có 1 query).
    - `Q.size(-1)` chính là d_k; nếu Q, K có cùng số chiều thì luôn chia đúng.
"""

import math

import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        # Seed cố định để tái lập weight theo đúng thứ tự khởi tạo của đề bài.
        torch.manual_seed(0)
        # Thứ tự khởi tạo quan trọng: key -> query -> value.
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # embedded: (B, T, embedding_dim)
        K = self.key(embedded)      # (B, T, attention_dim)
        Q = self.query(embedded)    # (B, T, attention_dim)
        V = self.value(embedded)    # (B, T, attention_dim)

        # Scaled dot-product: (B, T, d) @ (B, d, T) -> (B, T, T)
        # Chia sqrt(d_k) để ổn định phương sai trước khi softmax.
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(Q.size(-1))

        # Causal mask: tam giác dưới = 1 (được phép), phần trên = 0 (bị chặn).
        seq_len = embedded.size(1)
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=embedded.device)
        )
        # Vị trí tương lai -> -inf => softmax cho trọng số 0.
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Chuẩn hóa theo trục key (dim=2).
        attention_weights = torch.softmax(scores, dim=2)

        # Tổng có trọng số của các Value: (B, T, T) @ (B, T, d) -> (B, T, d)
        output = attention_weights @ V

        return torch.round(output, decimals=4)
