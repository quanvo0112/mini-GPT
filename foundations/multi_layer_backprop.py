"""
Problem: Multi-Layer Backpropagation
Module: foundations/multi_layer_backprop.py
Description: Implementation of forward and backward pass for a 2-layer neural network (x -> W1, b1 -> ReLU -> W2, b2 -> y_hat) with MSE loss
Source: https://neetcode.io/practice/machine-learning/problems/multi-layer-backpropagation

Kiến trúc mạng (2-layer MLP):
    x (I,) -> z1 = W1 @ x + b1 (H,) -> a1 = ReLU(z1) (H,) -> predictions = W2 @ a1 + b2 (O,) -> MSE Loss

1. Forward Pass:
    - Hidden Layer:
        z1 = W1 @ x + b1
        a1 = max(0, z1) = np.maximum(0, z1)
    - Output Layer:
        predictions = W2 @ a1 + b2
    - Mean Squared Error (MSE Loss):
        L = (1 / n) * sum_{i=1}^n (predictions_i - y_true_i)^2
        loss = np.mean((predictions - y_true) ** 2)

2. Backward Pass (Chain Rule qua từng layer):
    - Đạo hàm MSE theo predictions:
        dL/dpred = (2 / n) * (predictions - y_true)
    - Gradients của Layer 2 (Output Layer):
        dL/dW2 = outer(dL/dpred, a1)
        dL/db2 = dL/dpred
    - Truyền gradient ngược về hidden activation a1:
        dL/da1 = W2.T @ dL_dpred
    - Truyền gradient qua hàm kích hoạt ReLU:
        dL/dz1 = dL/da1 * ReLU'(z1) = da1 * (z1 > 0)
        (với ReLU'(z) = 1 nếu z > 0, ngược lại = 0)
    - Gradients của Layer 1 (Hidden Layer):
        dL/dW1 = outer(dL/dz1, x)
        dL/db1 = dL/dz1

Complexity:
    Với I features đầu vào, H hidden neurons, O outputs:
    - Time:  O(H*I + O*H) cho cả forward và backward
    - Space: O(H*I + O*H) để lưu ma trận trọng số và gradients

Ghi nhớ cho mini-GPT:
    - Đây là bước chuyển từ Single Neuron sang mạng nhiều tầng (Multi-Layer Perceptron).
    - Toàn bộ quá trình backpropagation thực chất là Chain Rule chạy lùi qua từng layer:
      Loss -> predictions -> (W2, b2) -> a1 -> ReLU -> z1 -> (W1, b1) -> x.
    - Trong Transformer / GPT, nguyên lý lan truyền ngược tương tự được áp dụng
      cho hàng chục layer Feed-Forward và Multi-Head Attention xếp chồng lên nhau.
"""

from typing import List
import numpy as np


class Solution:

    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float],
    ) -> dict:
        # Chuyển đổi đầu vào sang NumPy arrays
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        # 1. Forward pass
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)
        predictions = W2 @ a1 + b2

        # MSE loss
        loss = np.mean((predictions - y_true) ** 2)

        # 2. Backward pass
        # Output layer gradients
        dL_dpred = 2 * (predictions - y_true) / len(y_true)
        dW2 = np.outer(dL_dpred, a1)
        db2 = dL_dpred

        # Backpropagation qua W2 và ReLU
        da1 = W2.T @ dL_dpred
        dz1 = da1 * (z1 > 0)

        # Hidden layer gradients
        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist(),
        }
