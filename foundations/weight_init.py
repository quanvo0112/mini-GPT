"""
Problem: Weight Initialization
Module: foundations/weight_init.py
Description: Implementation of Xavier, Kaiming (He), and Random weight initialization with activation variance tracking
Source: https://neetcode.io/practice/machine-learning/problems/weight-initialization

1. Overview:
    - Weight Initialization là bước khởi tạo tham số ban đầu cho mạng nơ-ron trước khi huấn luyện.
    - Tránh hai hiện tượng nguy hiểm:
      * Exploding activations / gradients: giá trị tăng mất kiểm soát theo chiều sâu mạng.
      * Vanishing activations / gradients: giá trị triệt tiêu dần về 0.

2. Các phương pháp khởi tạo (Weight shape: (fan_out, fan_in)):
    - Xavier / Glorot Normal:
        std = sqrt(2.0 / (fan_in + fan_out))
        W ~ N(0, std^2)
        * Giữ ổn định variance của activation và gradient khi dùng hàm Sigmoid / Tanh.
    - Kaiming / He Normal:
        std = sqrt(2.0 / fan_in)
        W ~ N(0, std^2)
        * Thiết kế chuyên biệt cho ReLU để bù đắp việc triệt tiêu nửa miền âm (x <= 0).
    - Random Normal (Baseline):
        std = 1.0 (W ~ N(0, 1))
        * Không scale theo fan_in/fan_out, dễ dẫn đến exploding activations ở mạng sâu.

3. Hàm check_activations():
    - Dựng mạng gồm num_layers Linear layers nối tiếp nhau với hàm kích hoạt ReLU.
    - Shape convention: W có shape (fan_out, fan_in), input x có shape (batch_size, fan_in).
      Do đó forward pass là: x @ w.T.
    - Lưu ý về pseudo-random sequence: Cần tạo toàn bộ weights trước, sau đó mới tạo tensor x
      để đảm bảo trạng thái bộ sinh số ngẫu nhiên torch.manual_seed(0) khớp với test case.
    - Theo dõi x.std():
      * std ổn định qua các layer -> mạng hội tụ tốt.
      * std tăng vọt (e.g. 4.06 -> 2878.09) -> Exploding activations.

Ghi nhớ cho mini-GPT:
    - Trong Transformer & GPT, khởi tạo trọng số quyết định sự ổn định khi pre-training:
      * Thường khởi tạo W ~ N(0, 0.02^2) hoặc scaled Normal: std = 0.02 / sqrt(2 * num_layers)
        cho các projection layer ở residual connections (theo chuẩn GPT-2/GPT-3) để ngăn variance bùng nổ.
"""

import math
import torch
import torch.nn as nn


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> list[list[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        weights = torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> list[list[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        weights = torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals=4).tolist()

    def check_activations(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        init_type: str,
    ) -> list[float]:

        torch.manual_seed(0)

        dims = [input_dim] + [hidden_dim] * num_layers

        # Tạo toàn bộ weights trước theo thứ tự layer
        weights = []
        for i in range(num_layers):
            if init_type == "xavier":
                std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            elif init_type == "kaiming":
                std = math.sqrt(2.0 / dims[i])
            else:
                std = 1.0

            w = torch.randn(dims[i + 1], dims[i]) * std
            weights.append(w)

        # Sau đó mới tạo input x
        x = torch.randn(1, input_dim)

        stds = []
        for w in weights:
            x = x @ w.T
            x = torch.relu(x)
            stds.append(round(x.std().item(), 2))

        return stds
