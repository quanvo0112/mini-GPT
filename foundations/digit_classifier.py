"""
Problem: Handwritten Digit Classifier (MNIST)
Module: foundations/digit_classifier.py
Description: Build a neural network in PyTorch to classify handwritten digits (0-9)
             from 784-dimensional flattened MNIST images.
Source: https://neetcode.io/practice/machine-learning/problems/handwritten-digit-classifier

Kiến trúc model (giữ đúng thứ tự các layers theo đề bài):

    Input (784 features)
           ↓
    Linear(784, 512)      # lớp ẩn mở rộng không gian biểu diễn
           ↓
    ReLU                  # activation phi tuyến cho hidden layer
           ↓
    Dropout(p=0.2)        # regularization: tắt ngẫu nhiên 20% activation khi train
           ↓
    Linear(512, 10)       # chiếu về 10 logits cho 10 lớp chữ số 0..9
           ↓
    Sigmoid               # ép từng output về khoảng (0, 1)
           ↓
    Output (10 values)

Các điểm quan trọng:
1. `torch.manual_seed(0)`:
    - Trong `__init__()`: cố định random seed để khởi tạo weights có thể tái lập.
    - Trong `forward()`: giữ đúng yêu cầu của template, đồng thời giúp giá trị random
      của `Dropout` có thể tái lập cho cùng một phép tính.
2. `nn.Dropout(0.2)`:
    - Khi training, Dropout đưa ngẫu nhiên 20% activation về 0 và scale các activation
      còn lại (chia cho 1 - p) để giữ kỳ vọng không đổi.
    - Khi model ở chế độ evaluation (`model.eval()`), Dropout được vô hiệu hóa.
3. `nn.Sigmoid()`:
    Sigmoid đưa từng output về khoảng (0, 1):
        sigma(z) = 1 / (1 + e^(-z))
    Với 10 output, model tạo ra 10 giá trị tương ứng với 10 lớp chữ số 0..9.
    Lưu ý Sigmoid không đảm bảo tổng 10 giá trị bằng 1 (đó là vai trò của Softmax);
    tuy nhiên đề bài yêu cầu kiến trúc này nên giữ nguyên.
4. Làm tròn kết quả:
    `torch.round(output, decimals=4)` giữ output dưới dạng Tensor (không chuyển sang list).

Key Takeaways for mini-GPT:
    - `nn.Sequential`: ghép nhiều layers theo đúng thứ tự luồng dữ liệu.
    - `nn.Linear`: phép biến đổi tuyến tính y = xW^T + b, nền tảng của mọi layer trong GPT
      (Q/K/V projection, feed-forward, lm_head).
    - `nn.ReLU`: activation phi tuyến (dùng trong FFN của Transformer, biến thể GELU).
    - `nn.Dropout`: regularization hạn chế overfitting, cũng có trong Transformer block.
    - `nn.Sigmoid`: ép output về (0, 1) (dùng trong gating / attention weights).
    - `model.train()` vs `model.eval()`: điều khiển hành vi của Dropout và BatchNorm.
"""

import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):

    def __init__(self):
        super().__init__()
        # Cố định random seed để khởi tạo weights có thể tái lập.
        torch.manual_seed(0)
        # Xây dựng mạng theo đúng thứ tự kiến trúc đề bài yêu cầu.
        self.model = nn.Sequential(
            nn.Linear(784, 512),   # Input (784) -> hidden (512)
            nn.ReLU(),             # phi tuyến
            nn.Dropout(0.2),       # regularization (tắt 20% activation khi train)
            nn.Linear(512, 10),    # hidden (512) -> 10 logits
            nn.Sigmoid(),          # ép output về (0, 1)
        )

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        # Giữ đúng yêu cầu của template (giúp Dropout có kết quả tái lập).
        torch.manual_seed(0)
        output = self.model(images)
        # Round đến 4 chữ số thập phân, giữ nguyên dạng Tensor.
        return torch.round(output, decimals=4)
