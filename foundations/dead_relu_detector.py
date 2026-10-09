"""
Problem: Dead ReLU Detector
Module: foundations/dead_relu_detector.py
Description: Detecting dead neurons after ReLU activations and suggesting remediation actions
Source: https://neetcode.io/practice/machine-learning/problems/dead-relu-detector

1. Dead Neurons Overview:
    - Một neuron được coi là "dead" nếu activation sau ReLU của nó luôn bằng 0
      trên TOÀN BỘ các mẫu trong batch: `(activations == 0).all(dim=0)`.
    - Khi một neuron bị chết, gradient truyền ngược qua ReLU tại điểm đó cũng triệt tiêu về 0,
      khiến cho weights kết nối với neuron đó không thể cập nhật và neuron vĩnh viễn không học được nữa.

2. Hook Mechanism (register_forward_hook):
    - Đăng ký hook lắng nghe forward pass của tất cả `nn.ReLU` modules trong mạng.
    - Trích xuất activation tensor độc lập (`output.detach()`), chuẩn hóa về dạng 2D `(batch_size, num_features)`.
    - Tính tỷ lệ dead neurons per layer: `dead.float().mean().item()`.
    - Sử dụng `try ... finally` để luôn gỡ bỏ hook (`handle.remove()`), tránh memory leak và side-effects.

3. Heuristics đề xuất khắc phục (suggest_fix):
    - Ưu tiên 1: Có bất kỳ layer nào có `dead_fraction > 0.5` -> `"use_leaky_relu"`.
    - Ưu tiên 2: Layer đầu tiên có `dead_fraction > 0.3` -> `"reinitialize"`.
    - Ưu tiên 3: Tỷ lệ dead tăng nghiêm ngặt theo chiều sâu layer và layer cuối `> 0.1` -> `"reduce_learning_rate"`.
    - Mặc định: `"healthy"`.

Key Takeaways for mini-GPT:
    - Dead ReLU là vấn đề kinh điển khiến capacity của mạng nơ-ron bị lãng phí.
    - Trong kiến trúc Transformers / GPT hiện đại:
      * Thay vì standard ReLU, các mô hình như GPT-2/3, BERT sử dụng **GELU** (Gaussian Error Linear Unit)
        hoặc **SwiGLU** (LLaMA).
      * GELU có đạo hàm mượt mà và không hoàn toàn triệt tiêu gradient ở miền âm, triệt để loại bỏ
        hiện tượng dead neurons.
"""

from typing import List
import torch
import torch.nn as nn


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        dead_fractions = []
        handles = []

        def hook(module, inputs, output):
            activations = output.detach()

            if activations.ndim == 1:
                activations = activations.unsqueeze(0)
            else:
                activations = activations.reshape(activations.shape[0], -1)

            dead = (activations == 0).all(dim=0)
            dead_fraction = dead.float().mean().item()

            dead_fractions.append(round(dead_fraction, 4))

        for module in model.modules():
            if isinstance(module, nn.ReLU):
                handles.append(module.register_forward_hook(hook))

        try:
            with torch.no_grad():
                model(x)
        finally:
            for handle in handles:
                handle.remove()

        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Priority 1: Any layer has dead fraction > 0.5
        if any(fraction > 0.5 for fraction in dead_fractions):
            return "use_leaky_relu"

        # Priority 2: First layer has dead fraction > 0.3
        if dead_fractions and dead_fractions[0] > 0.3:
            return "reinitialize"

        # Priority 3: Strictly increasing with depth,
        # and the last layer has dead fraction > 0.1
        if (
            len(dead_fractions) > 1
            and all(
                dead_fractions[i] < dead_fractions[i + 1]
                for i in range(len(dead_fractions) - 1)
            )
            and dead_fractions[-1] > 0.1
        ):
            return "reduce_learning_rate"

        # Priorities 4 and 5 both return healthy
        return "healthy"
