"""
Problem: Batch Normalization
Module: model/batch_normalization.py
Description: Implementation of Batch Normalization for training and inference modes with running statistics
Source: https://neetcode.io/practice/machine-learning/problems/batch-normalization

1. Batch Normalization là gì?
    - Chuẩn hóa từng feature column qua toàn bộ các mẫu trong batch (tính thống kê theo axis=0).
    - Giúp giảm hiện tượng Internal Covariate Shift, tăng tốc độ hội tụ và cho phép dùng learning rate lớn hơn.

2. Training vs Inference:
    - Khi training = True:
        * Tính mean và variance trực tiếp từ batch hiện tại:
            batch_mean = np.mean(x, axis=0)
            batch_var  = np.var(x, axis=0)
            x_hat      = (x - batch_mean) / np.sqrt(batch_var + eps)
        * Cập nhật running statistics (Exponential Moving Average):
            running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            running_var  = (1 - momentum) * running_var  + momentum * batch_var
    - Khi training = False (Inference):
        * Không tính statistics của batch hiện tại (để inference độc lập với batch size và các sample xung quanh):
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)
    - Affine transformation (áp dụng cho cả 2 mode):
        y = gamma * x_hat + beta  (gamma: learnable scale, beta: learnable shift)

3. So sánh BatchNorm vs LayerNorm:
    - BatchNorm: Chuẩn hóa từng feature theo chiều batch (axis=0). Phụ thuộc vào batch size.
    - LayerNorm: Chuẩn hóa toàn bộ features của từng sample độc lập (axis=-1). Không phụ thuộc vào batch size.
    - Vì độ dài câu biến thiên và tính độc lập sample, Transformer & GPT chọn dùng LayerNorm / RMSNorm thay vì BatchNorm.

Complexity:
    Với N samples trong batch, D features:
    - Time:  O(N * D)
    - Space: O(N * D) cho các mảng trung gian

Key Takeaways cho mini-GPT:
    - BatchNorm phổ biến trong Computer Vision (CNNs), còn Transformer / LLM chủ yếu dùng LayerNorm / RMSNorm.
    - Lưu ý về convention momentum:
      running = (1 - momentum) * running + momentum * batch (theo chuẩn PyTorch / NeetCode).
"""

from typing import List, Tuple
import numpy as np


class Solution:

    def batch_norm(
        self,
        x: List[List[float]],
        gamma: List[float],
        beta: List[float],
        running_mean: List[float],
        running_var: List[float],
        momentum: float,
        eps: float,
        training: bool,
    ) -> Tuple[List[List[float]], List[float], List[float]]:
        # Chuyển đổi sang NumPy array
        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)
        beta = np.array(beta, dtype=float)
        running_mean = np.array(running_mean, dtype=float)
        running_var = np.array(running_var, dtype=float)

        if training:
            # 1. Tính thống kê batch dọc theo axis=0
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            x_hat = (x - batch_mean) / np.sqrt(batch_var + eps)

            # 2. Cập nhật running statistics (theo chuẩn PyTorch / NeetCode)
            running_mean = (
                (1 - momentum) * running_mean + momentum * batch_mean
            )
            running_var = (
                (1 - momentum) * running_var + momentum * batch_var
            )
        else:
            # 3. Ở chế độ inference, sử dụng running statistics tích lũy
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)

        # 4. Scale và shift (Affine transform)
        y = gamma * x_hat + beta

        return (
            np.round(y, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist(),
        )
