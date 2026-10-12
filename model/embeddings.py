"""
Problem: Word Embeddings
Module: model/embeddings.py
Description: Given an embedding matrix (vocab_size, embed_dim) and a 1D array of
             integer token IDs, return the embedding vectors for those tokens.
Source: https://neetcode.io/problems/word-embeddings/question

Ý tưởng:
    - `embeddings` là ma trận shape (vocab_size, embed_dim); mỗi hàng là vector
      embedding tương ứng với một token ID (row i ↔ token ID i).
    - `embeddings[token_ids]` dùng NumPy advanced indexing để "tra cứu" nhiều hàng
      cùng lúc theo danh sách `token_ids`.
      * Giữ nguyên thứ tự các ID trong `token_ids`.
      * Nếu một ID xuất hiện nhiều lần, hàng đó được lấy lặp lại tương ứng.
    - Kết quả có shape (len(token_ids), embed_dim).
    - `np.round(..., 5)` làm tròn đến 5 chữ số thập phân theo yêu cầu.

Complexity:
    Với k token cần tra cứu và embedding dimension d:
    - Thời gian: O(k * d) để copy/gather các vector ra ngoài.
    - Bộ nhớ:   O(k * d) cho mảng kết quả.

Key Takeaways for mini-GPT:
    - Embedding lookup thực chất chỉ là một phép index/gather vào một bảng tra cứu
      (lookup table); không có phép nhân ma trận nào diễn ra.
    - Trong GPT: `nn.Embedding(vocab_size, n_embd)` chính là bảng này. Với input
      token IDs (B, T) -> output embeddings (B, T, n_embd).
    - Bảng embedding là tham số học được (trainable): gradient chỉ cập nhật các hàng
      tương ứng với token xuất hiện trong batch (sparse update).
    - Trong bài toán thực tế, ngoài token embedding còn cộng thêm positional
      embedding để đưa thông tin vị trí vào mô hình.
"""

import numpy as np
from numpy.typing import NDArray


class Solution:

    def lookup(
        self,
        embeddings: NDArray[np.float64],
        token_ids: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        # embeddings: (vocab_size, embed_dim) matrix
        # token_ids: 1D array of integer token IDs
        # Advanced indexing: lấy các hàng theo token_ids, giữ nguyên thứ tự/trùng lặp.
        # return np.round(your_answer, 5)
        return np.round(embeddings[token_ids], 5)
