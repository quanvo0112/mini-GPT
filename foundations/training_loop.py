"""
Problem: Training Loop Mechanics
Module: foundations/training_loop.py
Description: Full vectorized training loop for Linear Regression (Forward -> MSE Gradient -> Gradient Descent Update)
Source: https://neetcode.io/practice/machine-learning/problems/training-loop

1. Cấu trúc một Training Loop hoàn chỉnh:
    - Khởi tạo tham số:
        w = np.zeros(n_features)
        b = 0.0
    - Lặp qua từng epoch:
        1. Forward Pass (dự đoán):
            y_hat = X @ w + b
        2. Tính sai số (residual / error vector):
            error = y_hat - y
        3. Tính gradient theo MSE loss (Vectorized hoàn toàn bằng NumPy):
            dw = (2 / n_samples) * (X.T @ error)   # d(MSE)/dw
            db = (2 / n_samples) * np.sum(error)    # d(MSE)/db
        4. Cập nhật tham số theo Gradient Descent:
            w -= lr * dw
            b -= lr * db

2. Tại sao dùng Vectorized Gradients (X.T @ error) thay vì vòng lặp for?
    - Vectorized tận dụng thư viện BLAS / C bên dưới, nhanh hơn rất nhiều so với lặp qua từng feature.
    - Đảm bảo tính toán đồng thời toàn bộ gradient trước khi cập nhật tham số.

Complexity:
    Với n = n_samples, m = n_features, E = epochs:
    - Time:  O(E * n * m)
    - Space: O(n + m) để lưu vector y_hat, error và dw

Ghi nhớ cho mini-GPT:
    Quy trình chuẩn mực của mọi Training Loop trong Deep Learning:
        Input Batch -> Model Forward -> Loss Calculation -> Backward (Gradients) -> Optimizer Step (Weights Update)
    Khi huấn luyện mini-GPT ở các bài sau (trong train.py), quy trình này vẫn giữ nguyên,
    chỉ thay thế Linear Model bằng Transformer, MSE Loss bằng Cross-Entropy Loss,
    và Gradient Descent bằng AdamW Optimizer.
"""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray


class Solution:

    def train(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        epochs: int,
        lr: float,
    ) -> Tuple[NDArray[np.float64], float]:
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0

        for _ in range(epochs):
            # 1. Forward pass
            y_hat = X @ w + b

            # 2. Gradient of MSE
            error = y_hat - y
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            # 3. Gradient descent update
            w -= lr * dw
            b -= lr * db

        return np.round(w, 5), round(b, 5)
