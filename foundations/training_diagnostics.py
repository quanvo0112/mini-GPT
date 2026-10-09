"""
Problem: Training Diagnostics & Learning Rate
Module: foundations/training_diagnostics.py
Description: Monitoring Neural Network training health using activation statistics and gradient statistics
Source: https://neetcode.io/practice/machine-learning/problems/training-diagnostics

1. Mục tiêu:
    Theo dõi và chẩn đoán các bệnh lý thường gặp trong quá trình huấn luyện mạng nơ-ron:
    - Dead Neurons (Neuron chết do ReLU)
    - Exploding Gradients (Gradient bùng nổ)
    - Vanishing Gradients (Gradient triệt tiêu)

2. Thống kê Activations (compute_activation_stats):
    - Chạy forward pass với `torch.no_grad()` qua từng module (chỉ xét `nn.Linear`).
    - Các chỉ số ghi nhận:
      * `mean`: Giá trị trung bình của activation.
      * `std`: Độ lệch chuẩn của activation.
      * `dead_fraction`: Tỷ lệ các neuron có output <= 0 trên toàn bộ batch samples
        tính bằng: `(x <= 0).all(dim=0).float().mean().item()`.

3. Thống kê Gradients (compute_gradient_stats):
    - Tính forward pass, MSE loss và backward pass (`loss.backward()`).
    - Với mỗi `nn.Linear`, trích xuất `module.weight.grad`:
      * `mean`: Giá trị trung bình gradient.
      * `std`: Độ lệch chuẩn gradient.
      * `norm`: L2 Norm (`torch.norm(grad).item()`) để đánh giá độ lớn vector gradient.

4. Quy tắc chẩn đoán có thứ tự ưu tiên (diagnose):
    - Ưu tiên 1 (Dead Neurons): Nếu bất kỳ layer nào có `dead_fraction > 0.5` -> "dead_neurons".
    - Ưu tiên 2 (Exploding Gradients): Nếu bất kỳ layer nào có `norm > 1000` -> "exploding_gradients".
    - Ưu tiên 3 (Vanishing Gradients): Nếu layer cuối cùng có gradient `norm < 1e-5` -> "vanishing_gradients".
    - Ưu tiên 4 (Bất thường về Activation Std):
      * Nếu có layer `std < 0.1` -> "vanishing_gradients".
      * Nếu có layer `std > 10.0` -> "exploding_gradients".
    - Mặc định: "healthy" nếu không vi phạm các điều kiện trên.

Complexity:
    - Activation stats: O(Forward compute qua các layers)
    - Gradient stats: O(Forward + Backward compute)
    - Diagnosis: O(Số layers)

Ghi nhớ cho mini-GPT:
    - Trong huấn luyện GPT và các mô hình ngôn ngữ lớn (LLM):
      * Thường xuyên giám sát Gradient L2 Norm (`grad_norm`) để phát hiện bùng nổ gradient.
      * Áp dụng Gradient Clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
        để bảo vệ quá trình pre-training không bị sụp đổ (training collapse).
      * Quan sát Activation Std giúp kiểm tra hiệu quả của LayerNorm/RMSNorm và residual connections.
"""

from typing import Dict, List
import torch
import torch.nn as nn


class Solution:

    def compute_activation_stats(
        self,
        model: nn.Module,
        x: torch.Tensor,
    ) -> List[Dict[str, float]]:
        stats = []

        with torch.no_grad():
            for module in model.children():
                x = module(x)

                if isinstance(module, nn.Linear):
                    mean_val = round(x.mean().item(), 4)
                    std_val = round(x.std().item(), 4)

                    if x.dim() >= 2:
                        dead_frac = (
                            (x <= 0).all(dim=0).float().mean().item()
                        )
                    else:
                        dead_frac = (x <= 0).float().mean().item()

                    stats.append({
                        "mean": mean_val,
                        "std": std_val,
                        "dead_fraction": round(dead_frac, 4),
                    })

        return stats

    def compute_gradient_stats(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> List[Dict[str, float]]:
        model.zero_grad()

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        stats = []

        for module in model.children():
            if isinstance(module, nn.Linear):
                grad = module.weight.grad

                stats.append({
                    "mean": round(grad.mean().item(), 4),
                    "std": round(grad.std().item(), 4),
                    "norm": round(torch.norm(grad).item(), 4),
                })

        return stats

    def diagnose(
        self,
        activation_stats: List[Dict[str, float]],
        gradient_stats: List[Dict[str, float]],
    ) -> str:
        # Priority 1: Dead neurons
        for stats in activation_stats:
            if stats["dead_fraction"] > 0.5:
                return "dead_neurons"

        # Priority 2: Exploding gradients
        for stats in gradient_stats:
            if stats["norm"] > 1000:
                return "exploding_gradients"

        # Priority 3: Vanishing gradients
        if gradient_stats and gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        # Priority 4: Abnormal activation standard deviation
        for stats in activation_stats:
            if stats["std"] < 0.1:
                return "vanishing_gradients"

            if stats["std"] > 10.0:
                return "exploding_gradients"

        return "healthy"
