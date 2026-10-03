"""
Problem: Gradient Descent
Module: foundations/gradient_descent.py
Description: Gradient descent optimization
Source: https://neetcode.io/practice/machine-learning

Mục tiêu: Dùng Gradient Descent để tối thiểu hóa f(x) = x²

Công thức cốt lõi:
    x_new = x - learning_rate * gradient

Với f(x) = x²:
    f'(x) = 2x  →  gradient = 2x

Complexity:
    Time:  O(iterations)
    Space: O(1)

Ghi nhớ:
    parameter = parameter - learning_rate * gradient
    Công thức này giữ nguyên khi mở rộng sang Neural Network,
    chỉ khác gradient trở thành vector/tensor.
    Đây là nền tảng cho Linear Regression (Training),
    Backpropagation và Training loop phía sau.

Example:
    iterations=10, learning_rate=0.01, init=5  →  4.08536
"""


class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        x = init
        for _ in range(iterations):
            gradient = 2 * x              # f'(x) = 2x
            x -= learning_rate * gradient  # x = x - lr * gradient
        return round(x, 5)
