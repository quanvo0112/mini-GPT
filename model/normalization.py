"""
Problem: Layer Normalization
Module: model/normalization.py
Description: Implementation of Layer Normalization forward pass with learnable scale (gamma) and shift (beta) parameters
Source: https://neetcode.io/practice/machine-learning/problems/layer-normalization

1. Các bước tính Layer Normalization:
    - Bước 1: Tính mean trên toàn bộ feature của vector:
        mu = np.mean(x)
    - Bước 2: Tính population variance:
        sigma^2 = np.var(x)
    - Bước 3: Chuẩn hóa về mean 0, variance 1 (cộng epsilon để tránh chia cho 0):
        x_hat = (x - mu) / np.sqrt(sigma^2 + eps)
    - Bước 4: Affine transform (scale & shift học được):
        out = gamma * x_hat + beta
    - Làm tròn 5 chữ số thập phân: np.round(out, 5)

2. So sánh BatchNorm vs LayerNorm:
    - BatchNorm: Chuẩn hóa theo chiều batch (axis=0) cho từng feature riêng rẽ. Phụ thuộc kích thước batch.
    - LayerNorm: Chuẩn hóa qua toàn bộ features của từng sample độc lập. Hoạt động tốt bất kể batch size = 1
      và cực kỳ phù hợp cho chuỗi có độ dài thay đổi (NLP).

Complexity:
    Với n = len(x) là số features:
    - Time:  O(n)
    - Space: O(n)

Ghi nhớ cho mini-GPT:
    - LayerNorm là thành phần chuẩn mực trong mọi kiến trúc Transformer & GPT:
      * Trong GPT-2/GPT-3, LayerNorm được đặt TRƯỚC Self-Attention và Feed-Forward Network (Pre-LayerNorm):
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
      * Kiến trúc Pre-LN giúp gradient truyền thẳng qua residual stream mà không bị suy giảm,
        cho phép huấn luyện các mạng cực sâu mà không cần warmup quá khắt khe.
"""

import numpy as np
from numpy.typing import NDArray


class Solution:

    def forward(
        self,
        x: NDArray[np.float64],
        gamma: NDArray[np.float64],
        beta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        eps = 1e-5

        # 1. Tính mean và variance
        mean = np.mean(x)
        var = np.var(x)

        # 2. Chuẩn hóa
        x_hat = (x - mean) / np.sqrt(var + eps)

        # 3. Scale và shift
        out = gamma * x_hat + beta

        return np.round(out, 5)
