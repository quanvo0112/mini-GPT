"""
Problem: Linear Regression (Training)
Module: foundations/linear_regression_training.py
Description: Implementation of training a Linear Regression model using Gradient Descent
Source: https://neetcode.io/practice/machine-learning/problems/linear-regression-training

Mục tiêu:
    Tự hiện thực quá trình huấn luyện mô hình Linear Regression bằng Gradient Descent.

1. Logic của Training Loop (train_model):
    Trong mỗi iteration:
        - Tính forward prediction: model_prediction = X @ weights
        - Lặp qua từng feature / weight j:
            - Tính gradient của loss theo w_j: d(MSE)/d(w_j)
            - Cập nhật trọng số: w_j -= learning_rate * gradient
        - Sau khi xong num_iterations, làm tròn kết quả 5 chữ số: np.round(weights, 5)

2. Công thức Đạo hàm (get_derivative):
    MSE = (1 / N) * sum_{i=1}^N (y_hat_i - y_i)^2
    Đạo hàm riêng theo w_j:
        d(MSE)/d(w_j) = - (2 / N) * sum_{i=1}^N (y_i - y_hat_i) * X_{ij}
    Tương ứng với:
        -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

3. Tại sao cần weights = initial_weights.copy()?
    Để tránh làm thay đổi mảng gốc initial_weights mà caller truyền vào (side-effect).

Complexity:
    Với N samples, M weights/features, K iterations:
    - Time:  O(K * N * M)
    - Space: O(N) cho vector prediction trung gian

Ghi nhớ cho mini-GPT:
    Bài này là mắt xích kết nối:
        Model Forward Pass + Loss Derivative + Gradient Descent = Complete Training Loop
    Mọi mô hình Deep Learning từ Perceptron đến GPT đều vận hành dựa trên chu trình cơ bản này:
        Forward -> Compute Loss -> Backward (Compute Gradients) -> Optimizer Step (Update Weights)
"""

import numpy as np
from numpy.typing import NDArray


class Solution:

    def get_derivative(
        self,
        model_prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64],
        N: int,
        X: NDArray[np.float64],
        desired_weight: int,
    ) -> float:
        return -2 * np.dot(
            ground_truth - model_prediction, X[:, desired_weight]
        ) / N

    def get_model_prediction(
        self, X: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        weights = initial_weights.copy()
        N = len(X)
        for _ in range(num_iterations):
            model_prediction = self.get_model_prediction(X, weights)
            for j in range(len(weights)):
                gradient = self.get_derivative(
                    model_prediction, Y, N, X, j
                )
                weights[j] -= self.learning_rate * gradient
        return np.round(weights, 5)
