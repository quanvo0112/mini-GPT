"""
Problem: Activation Functions (Sigmoid, ReLU)
Module: foundations/activations.py
Description: Implementation of Sigmoid and ReLU activation functions using NumPy vectorization
Source: https://neetcode.io/practice/machine-learning/problems/sigmoid-and-relu

1. Sigmoid:
    Công thức:
        sigma(z) = 1 / (1 + exp(-z))
    - Nhận giá trị trong khoảng (0, 1), thường dùng trong bài toán Binary Classification
    - Vectorized trong NumPy: 1 / (1 + np.exp(-z))
    - Làm tròn: np.round(..., 5)

2. ReLU (Rectified Linear Unit):
    Công thức:
        ReLU(z) = max(0, z)
    - Trả về 0 nếu z <= 0, trả về z nếu z > 0
    - Giúp giảm thiểu triệt để vấn đề vanishing gradient so với Sigmoid trong các lớp ẩn
    - Vectorized trong NumPy: np.maximum(0, z)

Complexity:
    Với n = len(z):
    Time:  O(n)
    Space: O(n) cho output array

Ghi nhớ cho mini-GPT:
    - np.exp(), np.maximum(), np.round(): Các thao tác vectorized chuẩn của NumPy
    - Vectorization xử lý đồng thời trên mảng giúp tối ưu hóa hiệu năng thay vì dùng vòng lặp for
    - Các hàm kích hoạt phi tuyến (non-linear activation functions) là yếu tố sống còn giúp
      mạng neural xấp xỉ được các hàm phức tạp (Universal Approximation Theorem).
"""

import numpy as np
from numpy.typing import NDArray


class Solution:

    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # Vectorized sigmoid: 1 / (1 + e^(-z)) làm tròn 5 chữ số
        return np.round(1 / (1 + np.exp(-z)), 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # Vectorized ReLU: max(0, z) theo từng phần tử
        return np.maximum(0, z)
