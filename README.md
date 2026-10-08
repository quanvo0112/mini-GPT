# mini-GPT

My implementations and notes while learning Machine Learning
through the [NeetCode Machine Learning](https://neetcode.io/practice/machine-learning) track.

The end goal: **build a mini GPT from scratch**.

## Progress (16/36 completed)

| Directory | Files | Solved | Status |
|-----------|-------|--------|--------|
| `foundations/` | 17 | 13 | 🟡 13/17 |
| `model/` | 11 | 3 | 🟡 3/11 |
| `data/` | 6 | 0 | ⬜ 0/6 |
| Root (`train.py`, `generate.py`) | 2 | 0 | ⬜ 0/2 |
| **Total** | **36** | **16** | **🟡 16/36** |

---

## Project Structure

```text
mini-GPT/
├── README.md
├── requirements.txt
├── .gitignore
├── train.py                     # GPT training loop
├── generate.py                  # Text generation
│
├── model/                       # Attention, Transformer, GPT architecture
│   ├── normalization.py         # ✅ Layer normalization
│   ├── batch_normalization.py   # ✅ Batch normalization
│   ├── rms_normalization.py     # ✅ RMS normalization
│   ├── embeddings.py            # Word embeddings
│   ├── positional_encoding.py   # Positional encoding
│   ├── attention.py             # Self-attention head
│   ├── multi_head_attention.py  # Multi-headed self-attention
│   ├── transformer.py           # Transformer block
│   ├── gpt.py                   # GPT model
│   ├── kv_cache.py              # KV-Cache for fast inference
│   └── grouped_query_attention.py # Grouped query attention
│
├── data/                        # Data pipeline
│   ├── nlp_preprocessing.py     # NLP preprocessing
│   ├── tokenizer.py             # BPE tokenizer
│   ├── vocab.py                 # Character-level encode/decode
│   ├── tokenizer_utils.py       # Tokenization edge cases
│   ├── loader.py                # Batched training data
│   └── dataset.py               # GPT dataset preparation
│
└── foundations/                 # Neural network primitives & math foundations
    ├── gradient_descent.py      # ✅ Gradient descent optimization
    ├── linear_regression.py     # ✅ Linear regression (forward)
    ├── linear_regression_training.py # ✅ Linear regression (training)
    ├── activations.py           # ✅ Activation functions (Sigmoid, ReLU)
    ├── softmax.py               # ✅ Softmax activation
    ├── loss.py                  # ✅ Binary & Categorical Cross Entropy Loss
    ├── neuron.py                # ✅ Single neuron
    ├── mlp.py                   # ✅ Multilayer perceptron
    ├── backprop.py              # ✅ Backpropagation
    ├── multi_layer_backprop.py  # ✅ Multi-layer backprop
    ├── weight_init.py           # ✅ Weight initialization
    ├── dead_relu_detector.py    # Dead ReLU detector
    ├── pytorch_basics.py        # ✅ PyTorch basics
    ├── digit_classifier.py      # Handwritten digit classifier (MNIST)
    ├── sentiment.py             # Sentiment analysis
    ├── training_loop.py         # ✅ Training loop mechanics
    └── training_diagnostics.py  # Diagnostics & learning rate
```

---

## Problem Checklist

### Foundations (`foundations/`)
- [x] `gradient_descent.py` — Gradient descent
- [x] `linear_regression.py` — Linear regression (forward)
- [x] `linear_regression_training.py` — Linear regression (training)
- [x] `activations.py` — Sigmoid, ReLU
- [x] `softmax.py` — Softmax activation
- [x] `loss.py` — Cross-entropy loss (BCE & CCE)
- [x] `neuron.py` — Single neuron
- [x] `mlp.py` — Multilayer perceptron
- [x] `backprop.py` — Backpropagation
- [x] `multi_layer_backprop.py` — Multi-layer backprop
- [x] `weight_init.py` — Weight initialization
- [ ] `dead_relu_detector.py` — Dead ReLU detector
- [x] `pytorch_basics.py` — PyTorch basics
- [ ] `digit_classifier.py` — Digit classifier
- [ ] `sentiment.py` — Sentiment analysis
- [x] `training_loop.py` — Training loop mechanics
- [ ] `training_diagnostics.py` — Training diagnostics

### Model (`model/`)
- [x] `normalization.py` — Layer normalization
- [x] `batch_normalization.py` — Batch normalization
- [x] `rms_normalization.py` — RMS normalization
- [ ] `embeddings.py` — Word embeddings
- [ ] `positional_encoding.py` — Positional encoding
- [ ] `attention.py` — Self-attention head
- [ ] `multi_head_attention.py` — Multi-headed self-attention
- [ ] `transformer.py` — Transformer block
- [ ] `gpt.py` — GPT model
- [ ] `kv_cache.py` — KV-Cache for fast inference
- [ ] `grouped_query_attention.py` — Grouped query attention

### Data (`data/`)
- [ ] `nlp_preprocessing.py` — NLP preprocessing
- [ ] `tokenizer.py` — BPE tokenizer
- [ ] `vocab.py` — Character-level encode/decode
- [ ] `tokenizer_utils.py` — Tokenization edge cases
- [ ] `loader.py` — Batched training data
- [ ] `dataset.py` — GPT dataset preparation

### Root Pipeline
- [ ] `train.py` — GPT training loop
- [ ] `generate.py` — Text generation

---

## Study Plan

🎯 **2 problems per day** → Complete in ~18 days

## Disclaimer

This repository is for learning and educational purposes.
Problem structure corresponds to the [NeetCode](https://neetcode.io/) Machine Learning track.
All implementations and notes are personal study records.
