"""
Problem: GPT Dataset Preparation
Module: data/dataset.py
Description: Build a tokenized, padded dataset from raw sentences. Given lists of
             positive and negative sentences, construct a word-level vocabulary,
             encode every sentence into integer token IDs, then pad all sequences
             so they share the same length (forming a rectangular batch tensor).
Source: https://neetcode.io/practice/machine-learning

Notes / Key Insights:
- Vocabulary is built from the combined corpus and sorted alphabetically for a
  deterministic, reproducible ID mapping.
- IDs start at 1; ID 0 is reserved for padding so that `pad_sequence` can fill
  shorter sentences without colliding with real tokens (a common embedding trick:
  index 0 acts as the "ignore"/pad embedding).
- Positive sentences are concatenated before negative sentences, so the output
  order reflects `positive + negative`.
- Token IDs are stored as float32 to match the required return type; in real
  models they would be integer indices fed straight into `nn.Embedding`.
"""

import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List


class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # Combine both label groups into a single corpus (positive first).
        sentences = positive + negative

        # 1. Build vocabulary: unique words sorted alphabetically.
        #    `i + 1` shifts IDs so that 0 stays free for padding.
        vocab = sorted(set(
            word
            for sentence in sentences
            for word in sentence.split()
        ))
        word_to_id = {word: i + 1 for i, word in enumerate(vocab)}

        # 2. Encode each sentence: map every word to its ID, preserving order.
        tensors = [
            torch.tensor(
                [word_to_id[word] for word in sentence.split()],
                dtype=torch.float32
            )
            for sentence in sentences
        ]

        # 3. Pad to the longest sequence in the batch. `batch_first=True` makes the
        #    output shape (batch, max_seq_len) instead of (max_seq_len, batch).
        return nn.utils.rnn.pad_sequence(tensors, batch_first=True)
