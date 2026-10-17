"""
Problem: Sentiment Analysis (Binary Classification)
Module: foundations/sentiment.py
Description: Cho trước `vocabulary_size`, xây dựng một module nn.Module phân loại
             cảm xúc nhị phân cho câu. Ý tưởng: embedding từng token -> mean pooling
             để lấy vector đại diện cho cả câu -> linear -> sigmoid -> xác suất.
             Input: token IDs shape (B, T). Output: dự đoán shape (B, 1), làm tròn 4 chữ số.
Source: https://neetcode.io/practice/machine-learning

Ý tưởng:
    - `nn.Embedding(vocabulary_size, 16)` là bảng tra cứu (lookup table) shape
      (vocabulary_size, 16); mỗi token ID được ánh xạ thành vector 16 chiều.
      Với input (B, T) -> output embeddings (B, T, 16).
    - `embeddings.mean(dim=1)` (mean pooling) lấy trung bình embedding của T token
      trong mỗi câu:
          pooled[b] = (1 / T) * Σ_{t=0}^{T-1} embeddings[b, t]
      đưa tensor từ (B, T, 16) về (B, 16) — một vector "đại diện" cho toàn câu.
    - `nn.Linear(16, 1)` chiếu vector (B, 16) -> logit (B, 1).
    - `nn.Sigmoid()` biến logit thành xác suất trong (0, 1):
          σ(z) = 1 / (1 + e^(-z))
      phù hợp cho bài toán binary classification.
    - `torch.round(predictions, decimals=4)` làm tròn kết quả đến 4 chữ số thập phân.
    - `torch.manual_seed(0)` cố định khởi tạo trọng số để kết quả tái lập được.

Lưu ý:
    - `B` là batch size, `T` là số token mỗi câu.
    - Mean pooling ở đây tính trung bình trên TẤT CẢ vị trí token, kể cả padding
      nếu input đã được pad. Trong thực tế cần mask padding khi lấy trung bình.

Key Takeaways for mini-GPT:
    - `nn.Embedding` chính là bước token embedding trong GPT; ở GPT ta còn cộng thêm
      positional embedding: x = token_emb + pos_emb.
    - Mean pooling là cách tổng hợp đơn giản (thô) nhất để gộp chuỗi thành vector câu;
      Transformer/GPT thay nó bằng self-attention để tổng hợp có trọng số theo ngữ cảnh.
    - Sigmoid + binary loss chỉ dùng cho phân loại nhị phân; GPT dùng softmax trên vocab
      để dự đoán token tiếp theo (phân loại đa lớp).
"""

import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):

    def __init__(self, vocabulary_size: int):
        super().__init__()
        # Cố định seed để khởi tạo trọng số tái lập được giữa các lần chạy.
        torch.manual_seed(0)
        # Bảng embedding: token ID -> vector 16 chiều.
        self.embedding = nn.Embedding(vocabulary_size, 16)
        # Chiếu vector câu (16) -> 1 logit.
        self.linear = nn.Linear(16, 1)
        # Đưa logit về xác suất (0, 1).
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        embeddings = self.embedding(x)       # (B, T, 16)
        pooled = embeddings.mean(dim=1)      # (B, 16)
        logits = self.linear(pooled)         # (B, 1)
        predictions = self.sigmoid(logits)   # (B, 1)
        return torch.round(predictions, decimals=4)
