"""
Problem: Softmax
Module: foundations/softmax.py
Description: Softmax activation function
Source: https://neetcode.io/practice/machine-learning/problems/softmax

Mục tiêu: Tính hàm kích hoạt Softmax cho một 1D NumPy array.

Công thức:
    softmax(z_i) = exp(z_i) / sum_j(exp(z_j))

Numerical Stability (Tránh Overflow):
    Trừ max(z) trước khi exp():
    exp(z_i - c) / sum_j(exp(z_j - c)) = (exp(z_i) / exp(c)) / (sum_j(exp(z_j)) / exp(c))
                                      = exp(z_i) / sum_j(exp(z_j))
    Với c = max(z), giá trị lũy thừa lớn nhất chỉ là exp(0) = 1,
    tránh hoàn toàn hiện tượng numerical overflow khi z có giá trị lớn.

Lưu ý đề bài:
    Làm tròn kết quả 4 chữ số thập phân: np.round(..., 4)

Complexity:
    Time:  O(n) với n = len(z)
    Space: O(n)

Ghi nhớ:
    Softmax chuyển đổi vector logits thành phân phối xác suất (tổng = 1).
    Kỹ thuật trừ max(z) là standard practice trong tất cả các framework (PyTorch, TensorFlow)
    để đảm bảo numerical stability khi tính softmax hoặc cross-entropy loss.
"""

import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # Numerical stability trick: trừ max(z) để tránh overflow khi exp()
        exp_z = np.exp(z - np.max(z))
        # Chuẩn hóa về phân phối xác suất và làm tròn 4 chữ số theo yêu cầu NeetCode
        return np.round(exp_z / np.sum(exp_z), 4)
