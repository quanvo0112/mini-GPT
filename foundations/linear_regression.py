"""
Problem: Linear Regression (Forward Pass)
Module: foundations/linear_regression.py
Description: Implementation of Linear Regression forward prediction and Mean Squared Error (MSE)
Source: https://neetcode.io/practice/machine-learning/problems/linear-regression-forward

1. Model Prediction (Forward Pass):
    Mô hình Linear Regression:
        y_hat = X @ weights
    - X có kích thước (n, m): n samples, m features
    - weights có kích thước (m,): m trọng số
    - Tích ma trận X @ weights cho kết quả vector predictions shape (n,)
    - Làm tròn: np.round(predictions, 5)

2. Mean Squared Error (MSE Loss):
    Hàm mất mát đánh giá độ lệch giữa dự đoán và nhãn thực tế:
        MSE = (1 / n) * sum_{i=1}^n (y_hat_i - y_i)^2
    - Vectorized trong NumPy: np.mean((model_prediction - ground_truth) ** 2)
    - Làm tròn: round(mse, 5)

Complexity:
    Với X shape (n, m):
    - get_model_prediction: Time O(n * m), Space O(n)
    - get_error:            Time O(n), Space O(n)

Ghi nhớ cho mini-GPT:
    Linear Regression là khối xây dựng cơ bản nhất của Deep Learning:
        Input -> Linear Transform (X @ W + b) -> Activation -> Loss
    Trong Transformer / GPT:
    - Linear Layer (Dense Layer / Projection) trong Feed-Forward Network và Attention Heads
      đều sử dụng phép nhân ma trận (X @ W) tương tự như bước prediction này.
"""

import numpy as np
from numpy.typing import NDArray


class Solution:

    def get_model_prediction(
        self, X: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Phép nhân ma trận: (n, m) @ (m,) -> (n,)
        predictions = X @ weights
        return np.round(predictions, 5)

    def get_error(
        self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]
    ) -> float:
        # Mean Squared Error giữa vector dự đoán và nhãn thực tế
        mse = np.mean((model_prediction - ground_truth) ** 2)
        return round(mse, 5)
