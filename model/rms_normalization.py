"""
Problem: RMS Normalization (RMSNorm)
Module: model/rms_normalization.py
Description: Implementation of Root Mean Square Normalization (RMSNorm) with learnable scale parameter (gamma)
Source: https://neetcode.io/practice/machine-learning/problems/rms-normalization

1. RMS Normalization là gì?
    - RMSNorm chuẩn hóa độ lớn (magnitude/scale) của vector kích hoạt mà KHÔNG trừ mean (không mean centering).
    - Công thức Root Mean Square (RMS):
        RMS(x) = sqrt( (1 / n) * sum_{i=1}^n x_i^2 + eps ) = sqrt( mean(x^2) + eps )
    - Chuẩn hóa:
        x_hat = x / RMS(x)
    - Scale bằng tham số học được (gamma):
        y = gamma * x_hat
    - Làm tròn 4 chữ số thập phân: np.round(output, 4).tolist()

2. Khác biệt cốt lõi giữa LayerNorm và RMSNorm:
    - LayerNorm:
        * Trừ mean: x - mu
        * Chia standard deviation: sqrt(var + eps)
        * Hai tham số học được: gamma (scale) và beta (shift/bias)
    - RMSNorm:
        * Bỏ qua bước trừ mean (giả định tính bất biến dịch chuyển - shift-invariance không đóng vai trò then chốt).
        * Chia cho RMS(x) = sqrt(mean(x^2) + eps).
        * Chỉ có một tham số học được: gamma (không có beta).

3. Tại sao các LLM hiện đại ưa chuộng RMSNorm?
    - RMSNorm giảm thiểu chi phí tính toán (tiết kiệm khoảng 10% - 50% thời gian tính toán normalization)
      bằng cách loại bỏ phép tính mean và phép trừ.
    - Được áp dụng rộng rãi trong các mô hình ngôn ngữ lớn hàng đầu như LLaMA (Meta), Mistral, Gemma (Google), Qwen.

Complexity:
    Với n = len(x) là số features:
    - Time:  O(n)
    - Space: O(n)

Key Takeaway cho mini-GPT:
    RMSNorm = normalize magnitude without subtracting mean
    RMS = sqrt(mean(x^2) + eps)
    y = gamma * (x / RMS)
"""

from typing import List
import numpy as np


class Solution:

    def rms_norm(
        self,
        x: List[float],
        gamma: List[float],
        eps: float,
    ) -> List[float]:
        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)

        # 1. Tính Root Mean Square (RMS)
        rms = np.sqrt(np.mean(x ** 2) + eps)

        # 2. Chuẩn hóa độ lớn
        x_hat = x / rms

        # 3. Scale bằng gamma (không có beta)
        output = gamma * x_hat

        return np.round(output, 4).tolist()
