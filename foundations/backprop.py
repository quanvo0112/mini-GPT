"""
Problem: Backpropagation
Module: foundations/backprop.py
Description: Implementation of backward pass using Chain Rule for a single neuron with Sigmoid activation
Source: https://neetcode.io/practice/machine-learning/problems/backpropagation

1. Forward Pass:
    z = x · w + b = np.dot(x, w) + b
    y_hat = sigma(z) = 1 / (1 + exp(-z))
    L = 0.5 * (y_hat - y_true)^2  (Squared Error Loss)

2. Backward Pass (Chain Rule):
    Ta cần tính:
        dL/dw = (dL/dy_hat) * (dy_hat/dz) * (dz/dw)
        dL/db = (dL/dy_hat) * (dy_hat/dz) * (dz/db)

    Từng thành phần đạo hàm:
        dL/dy_hat = y_hat - y_true
        dy_hat/dz = y_hat * (1 - y_hat)       (Đạo hàm của hàm Sigmoid)
        dL/dz     = dL/dy_hat * dy_hat/dz

        dz/dw     = x  -->  dL/dw = dL/dz * x
        dz/db     = 1  -->  dL/db = dL/dz

3. Chiều đi của dữ liệu:
    Forward:   x, w, b -> z -> y_hat -> Loss
    Backward:  Loss -> y_hat -> z -> w, b (truyền ngược lỗi)

Complexity:
    Với n = len(x):
    - Time:  O(n)
    - Space: O(n) do gradient vector dL_dw

Ghi nhớ cho mini-GPT:
    - Backpropagation là nền tảng cốt lõi của việc huấn luyện mạng nơ-ron:
      dùng Chain Rule để tính gradient của Loss theo từng tham số (weights, biases).
    - Trong mạng sâu (Multi-Layer Perceptron hay Transformer/GPT), nguyên lý vẫn giữ nguyên:
      gradient lỗi được truyền ngược từ output layer qua từng transformer block,
      multi-head attention, và embedding layer.
"""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray


class Solution:

    def backward(
        self,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        b: float,
        y_true: float,
    ) -> Tuple[NDArray[np.float64], float]:
        # 1. Forward pass
        z = np.dot(x, w) + b
        y_hat = 1 / (1 + np.exp(-z))

        # 2. Chain rule
        dL_dyhat = y_hat - y_true
        dyhat_dz = y_hat * (1 - y_hat)
        dL_dz = dL_dyhat * dyhat_dz

        # 3. Gradients
        dL_dw = dL_dz * x
        dL_db = dL_dz

        return np.round(dL_dw, 5), round(float(dL_db), 5)
