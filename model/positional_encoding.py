"""
Problem: Positional Encoding
Module: model/positional_encoding.py
Description: Given seq_len and d_model, build the sinusoidal positional encoding
             matrix of shape (seq_len, d_model) using the Transformer formula.
Source: https://neetcode.io/problems/positional-encoding/question

Công thức gốc (Vaswani et al., "Attention Is All You Need"):
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    - pos: vị trí trong câu (0 ... seq_len-1)
    - i:   chỉ số của cặp chiều (0 ... d_model/2 - 1)
    - Các cột chẵn dùng sin, các cột lẻ dùng cos; mỗi tần số được dùng cho
      đúng một cặp chiều (sin, cos) nên mỗi vị trí có một "chữ ký" duy nhất.

Ý tưởng:
    - `positions` shape (seq_len, 1) chứa các vị trí 0 ... seq_len-1.
    - `dimensions` = 0, 2, 4, ... là các chỉ số chiều chẵn.
    - `angle_rates = 1 / 10000^(dimensions / d_model)`: tần số giảm dần theo chiều,
      chiều càng lớn thì bước sóng càng dài (biến thiên càng chậm).
      Đây là biến đổi tương đương của 10000^(-2i/d_model).
    - NumPy broadcasting nhân `positions` (seq_len, 1) với `angle_rates`
      (d_model/2,) -> `angles` shape (seq_len, d_model/2); không cần vòng lặp.
    - Gán sin vào các cột chẵn `pe[:, 0::2]` và cos vào các cột lẻ `pe[:, 1::2]`.
      Cắt `angles[:, :d_model // 2]` để số cột cos khớp với slice cột lẻ
      (d_model chẵn); với d_model lẻ cần cắt thêm cả `angles` cho slice chẵn.
    - `np.round(..., 5)` làm tròn đến 5 chữ số thập phân theo yêu cầu.

Complexity:
    Với L = seq_len và d = d_model:
    - Thời gian: O(L * d) để tính toàn bộ ma trận.
    - Bộ nhớ:   O(L * d) cho mảng kết quả.

Key Takeaways for mini-GPT:
    - Self-attention không có khái niệm thứ tự: nếu hoán vị các token, output chỉ
      bị hoán vị theo (permutation equivariance). Positional encoding phá tính đối
      xứng đó bằng cách cộng thêm thông tin vị trí vào token embedding:
          x = token_embedding + positional_encoding
      (shape (B, T, n_embd) + (T, n_embd) -> broadcast theo batch).
    - Công thức sin/cos cho phép mô hình suy ra vị trí tương đối vì PE(pos + k)
      là một phép biến đổi tuyến tính (rotation) của PE(pos) với mọi k.
    - PE này là cố định (không học), nên không tốn tham số; nhiều GPT hiện đại
      thay bằng learned positional embedding hoặc RoPE (Rotary Position Embedding)
      áp dụng trực tiếp lên Q, K trong attention.
    - Giá trị nằm trong [-1, 1] nên không làm lệch scale của embedding.
"""

import numpy as np
from numpy.typing import NDArray


class Solution:

    def get_positional_encoding(
        self,
        seq_len: int,
        d_model: int
    ) -> NDArray[np.float64]:
        # positions: (seq_len, 1) -> mỗi hàng là một vị trí pos
        positions = np.arange(seq_len)[:, np.newaxis]

        # dimensions: các chỉ số chiều chẵn 0, 2, 4, ...
        dimensions = np.arange(0, d_model, 2)

        # 10000^(-2i / d_model) == 1 / 10000^(dimensions / d_model)
        angle_rates = 1.0 / np.power(10000.0, dimensions / d_model)

        # angles[pos, i] = pos * angle_rates[i]  (broadcasting)
        angles = positions * angle_rates

        pe = np.zeros((seq_len, d_model))

        # Cột chẵn: sin; cột lẻ: cos cùng tần số.
        # Cắt :d_model // 2 để số cột khớp với slice cột lẻ.
        pe[:, 0::2] = np.sin(angles)
        pe[:, 1::2] = np.cos(angles[:, :d_model // 2])

        return np.round(pe, 5)
