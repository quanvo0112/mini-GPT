"""
Problem: Multilayer Perceptron (MLP From Scratch)
Module: foundations/mlp.py
Description: Implementation of forward pass through an arbitrary number of fully-connected layers with ReLU activation for hidden layers
Source: https://neetcode.io/practice/machine-learning/problems/multilayer-perceptron

Kiến trúc & Luồng tính toán:
    x -> [Layer 1: Linear + ReLU] -> [Layer 2: Linear + ReLU] -> ... -> [Output Layer: Linear (No activation)]

Lưu ý quan trọng về Matrix Orientation:
    Đề bài NeetCode biểu diễn trọng số theo convention:
        W có shape: (input_dim, output_dim)
    Do đó phép tính tuyến tính là:
        output = output @ weights[i] + biases[i]
    thay vì weights[i] @ output.

Quy tắc kích hoạt:
    - Các hidden layers (i < len(weights) - 1): áp dụng ReLU qua np.maximum(0, output)
    - Output layer (i == len(weights) - 1): giữ nguyên linear output (không áp dụng activation)

Làm tròn kết quả:
    np.round(output, 5)

Complexity:
    Với layer thứ i có in_i inputs và out_i outputs:
    - Time:  O(sum_i (in_i * out_i))
    - Space: O(max_i (out_i)) để lưu các vector kích hoạt trung gian

Ghi nhớ cho mini-GPT:
    - MLP (Multi-Layer Perceptron) là khối cơ bản xuất hiện xuyên suốt trong Deep Learning.
    - Trong Transformer / GPT, mỗi Transformer Block đều chứa một khối MLP 2 tầng
      (còn gọi là Position-wise Feed-Forward Network - FFN):
          FFN(x) = max(0, x @ W1 + b1) @ W2 + b2
      hoặc biến thể dùng GELU/SwiGLU. Khối này chịu trách nhiệm lưu trữ và xử lý tri thức (factual knowledge).
"""

from typing import List
import numpy as np
from numpy.typing import NDArray


class Solution:

    def forward(
        self,
        x: NDArray[np.float64],
        weights: List[NDArray[np.float64]],
        biases: List[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        output = x
        for i in range(len(weights)):
            # Weight shape: (input_dim, output_dim) -> output @ weights[i] + biases[i]
            output = output @ weights[i] + biases[i]

            # Áp dụng ReLU cho các hidden layer, bỏ qua ở output layer cuối cùng
            if i < len(weights) - 1:
                output = np.maximum(0, output)

        return np.round(output, 5)
