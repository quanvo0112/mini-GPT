"""
Problem: Cross Entropy Loss
Module: foundations/loss.py
Description: Binary & Categorical Cross Entropy Loss
Source: https://neetcode.io/practice/machine-learning/problems/cross-entropy-loss

Mục tiêu:
    Hiện thực Binary Cross-Entropy Loss (BCE) và Categorical Cross-Entropy Loss (CCE)
    sử dụng NumPy vectorization, tránh vòng lặp for.

1. Binary Cross-Entropy (BCE):
    L = - (1/n) * sum_{i=1}^n [ y_i * log(y_hat_i) + (1 - y_i) * log(1 - y_hat_i) ]
    - y_true = 1 -> tính loss theo log(y_pred)
    - y_true = 0 -> tính loss theo log(1 - y_pred)

2. Categorical Cross-Entropy (CCE):
    L = - (1/n) * sum_i sum_j [ y_ij * log(y_hat_ij) ]
    - y_true: shape (n_samples, n_classes), one-hot encoded
    - y_pred: shape (n_samples, n_classes), xác suất sau softmax

Tại sao cần np.clip(y_pred, 1e-7, 1 - 1e-7)?
    Vì log(0) = -inf, dẫn đến lỗi giá trị NaN hoặc Inf khi tính loss.
    np.clip giới hạn giá trị trong khoảng an toàn [1e-7, 1 - 1e-7].

Complexity:
    Binary:
        Time:  O(n)
        Space: O(n)
    Categorical:
        Time:  O(n * c) với c là số class
        Space: O(n * c)

Ghi nhớ cho mini-GPT:
    - Sigmoid -> kết hợp với Binary Cross-Entropy
    - Softmax -> kết hợp với Categorical Cross-Entropy
    Trong GPT và Language Model, Cross-Entropy Loss cực kỳ quan trọng
    vì mô hình dự đoán phân phối xác suất của token tiếp theo trên toàn bộ vocabulary
    và so sánh với token thật (one-hot).
"""

import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> float:
        # Giới hạn xác suất tránh log(0) gây -inf
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return round(loss, 4)

    def categorical_cross_entropy(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> float:
        # Giới hạn xác suất tránh log(0) gây -inf
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # axis=1 tính loss cho từng sample, sau đó np.mean lấy trung bình tất cả samples
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return round(loss, 4)
