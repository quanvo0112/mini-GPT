"""
Problem: Single Neuron
Module: foundations/neuron.py
Description: Implementation of a single artificial neuron with linear combination and activation function
Source: https://neetcode.io/practice/machine-learning/problems/single-neuron

1. Pre-activation (Linear Combination):
    z = x · w + b = np.dot(x, w) + b
    - x: 1D input array
    - w: 1D weight array (cùng độ dài với x)
    - b: scalar bias

2. Activation Function:
    - Sigmoid:
        sigma(z) = 1 / (1 + exp(-z))
    - ReLU:
        ReLU(z) = max(0.0, float(z))

3. Return Format:
    Làm tròn 5 chữ số thập phân: round(float(output), 5)

Complexity:
    Với n = len(x):
    - Time:  O(n) cho phép tính tích vô hướng np.dot(x, w)
    - Space: O(1)

Ghi nhớ cho mini-GPT:
    Quy trình tính toán của một neuron:
        Input (x) -> Tích vô hướng + Bias (z = x·w + b) -> Phi tuyến hóa (Activation) -> Output
    Nhiều neuron ghép lại thành:
    - Dense Layer / Linear Layer (nhiều neuron cùng nhận input x)
    - Multilayer Perceptron (các layer nối tiếp nhau)
    - Feed-Forward Network (FFN block) trong Transformer / GPT.
"""

import numpy as np
from numpy.typing import NDArray


class Solution:

    def forward(
        self,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        b: float,
        activation: str,
    ) -> float:
        # 1. Tính pre-activation: z = x · w + b
        z = np.dot(x, w) + b

        # 2. Áp dụng hàm kích hoạt tương ứng
        if activation == "sigmoid":
            output = 1 / (1 + np.exp(-z))
        else:
            output = max(0.0, float(z))

        # 3. Làm tròn 5 chữ số thập phân
        return round(float(output), 5)
