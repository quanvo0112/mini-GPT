"""
Problem: Basics of PyTorch
Module: foundations/pytorch_basics.py
Description: Fundamental PyTorch tensor operations: reshape, column-wise mean, concatenation, and MSE loss
Source: https://neetcode.io/practice/machine-learning/problems/basics-of-pytorch

Các thao tác Tensor cốt lõi trong PyTorch:
1. Reshape:
    - Input shape: (M, N) với tổng M * N phần tử.
    - Yêu cầu shape mới: (M * N // 2, 2).
    - Sử dụng `torch.reshape(to_reshape, (M * N // 2, 2))`.
    - Reshape chỉ thay đổi cấu trúc biểu diễn chiều, giữ nguyên thứ tự phần tử.

2. Average (Column-wise Mean):
    - Tính giá trị trung bình theo từng cột (tức là gộp qua tất cả các hàng dọc theo dim=0).
    - Sử dụng `torch.mean(to_avg, dim=0)`.

3. Concatenate (Ghép side-by-side):
    - Nối hai tensor cạnh nhau theo chiều cột (dim=1).
    - Sử dụng `torch.cat((cat_one, cat_two), dim=1)`.

4. Get Loss (Mean Squared Error):
    - Tính MSE loss giữa tensor prediction và target:
        MSE = (1 / n) * sum_i (prediction_i - target_i)^2
    - Sử dụng PyTorch API: `torch.nn.functional.mse_loss(prediction, target)`.

Complexity:
    Gọi tổng số phần tử của tensor là N:
    - reshape:     O(1) view operation (hoặc O(N) nếu tensor không contiguous cần copy bộ nhớ)
    - average:     O(N)
    - concatenate: O(N)
    - mse_loss:    O(N)

Ghi nhớ cho mini-GPT:
    - PyTorch Tensors là nền tảng tính toán thay thế cho NumPy khi huấn luyện trên GPU / TPU với autograd.
    - Trong mô hình Transformer & GPT:
      * `reshape` / `view`: Dùng liên tục để tách/gộp số attention heads: `(B, T, C) -> (B, T, num_heads, head_dim)`.
      * `torch.cat`: Dùng để ghép key/value trong KV-cache hoặc ghép các attention heads lại với nhau.
      * `F.mse_loss` / `F.cross_entropy`: Được sử dụng trực tiếp trong training loop.
"""

import torch
import torch.nn
from torchtyping import TensorType


# Round all answers to 4 decimal places: torch.round(tensor, decimals=4)
class Solution:

    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # Reshape (M, N) tensor to (M*N/2, 2)
        # Use torch.reshape(tensor, new_shape)
        M, N = to_reshape.shape
        return torch.reshape(to_reshape, (M * N // 2, 2))

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # Compute column-wise mean (average across rows)
        # Use torch.mean(tensor, dim=0)
        return torch.mean(to_avg, dim=0)

    def concatenate(
        self,
        cat_one: TensorType[float],
        cat_two: TensorType[float]
    ) -> TensorType[float]:
        # Join two tensors side-by-side along dim=1
        # Use torch.cat((a, b), dim=1)
        return torch.cat((cat_one, cat_two), dim=1)

    def get_loss(
        self,
        prediction: TensorType[float],
        target: TensorType[float]
    ) -> TensorType[float]:
        # Compute Mean Squared Error between prediction and target
        # Use torch.nn.functional.mse_loss(prediction, target)
        return torch.nn.functional.mse_loss(prediction, target)
