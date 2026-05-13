# Transformers from Scratch: A Mathematical Deep-Dive

> A ground-up implementation of the Transformer architecture using PyTorch —
> built for engineers who already understand linear algebra and want to know
> exactly what is happening inside the model, mathematically and computationally.

---

## Prerequisites

This project assumes you are comfortable with the following:

- Vector embeddings, L2 norm, dot product, cosine similarity
- Matrix multiplication, PCA, K-Means clustering
- Python, NumPy, and basic PyTorch tensor operations
- You have completed a RAG system from scratch (or equivalent experience)

---

## What You Will Build

By the end of this project you will have:

- A complete implementation of Scaled Dot-Product Attention from scratch
- A working Multi-Head Attention module
- A full Transformer encoder block with Layer Norm, FFN, and residual connections
- A small GPT-style language model trained on real text
- A model that can **generate new text**, with visualised attention patterns and a live loss curve

---

## Environment Setup

```bash
pip install torch torchvision matplotlib numpy jupyter ipykernel
```

Verify your install:

```python
import torch
print(torch.__version__)
```

---

## Project Structure

```
transformers-from-scratch/
│
├── README.md                          ← You are here
│
├── requirements.txt                   ← All dependencies pinned
│
├── data/
│   └── shakespeare.txt                ← Tiny corpus, added in Sprint 6
│
├── notebooks/
│   ├── 00_introduction.ipynb
│   ├── 01_input_embeddings_and_positional_encoding.ipynb
│   ├── 02_scaled_dot_product_attention.ipynb
│   ├── 03_multi_head_attention.ipynb
│   ├── 04_full_transformer_block.ipynb
│   ├── 05_building_a_tiny_gpt.ipynb
│   └── 06_training_loop_and_generation.ipynb
│
└── src/
    └── (shared modules added progressively)
```

---

## Sprint Registry

| Notebook | Title | Key Concepts |
|----------|-------|--------------|
| `00` | Introduction | What is a Transformer, project overview, prerequisites |
| `01` | Input Embeddings & Positional Encoding | Token embeddings, sinusoidal PE, position fingerprints |
| `02` | Scaled Dot-Product Attention | Q, K, V matrices, softmax, sqrt(d_k) scaling |
| `03` | Multi-Head Attention | Parallel attention heads, projection, concatenation |
| `04` | Full Transformer Block | Layer Norm, Feed-Forward Network, residual connections |
| `05` | Building a Tiny GPT | Causal masking, stacked blocks, language model head |
| `06` | Training Loop & Generation | Cross-entropy loss, PyTorch training loop, text generation |

---

## Naming Convention

All notebooks follow the `NN_snake_case_title.ipynb` format.

- `00` is always the project introduction
- Each subsequent number maps to one sprint
- Sprints are designed to be completed in a single focused session
- Every notebook begins with a markdown cell stating its purpose, scope, and what the previous notebook covered

---

## Day Plan

| Day | Notebooks | Focus |
|-----|-----------|-------|
| Day 1 | `00`, `01`, `02` | Attention mechanism from first principles |
| Day 2 | `03`, `04` | Full Transformer encoder block |
| Day 3 | `05`, `06` | Build, train, and generate with a tiny GPT |

---

## Design Philosophy

- **No ML libraries for logic.** PyTorch is used for tensor operations and training only. All attention, normalisation, and model logic is written from scratch.
- **Math first, code second.** Every implementation is preceded by the formula in LaTeX with a plain-English breakdown.
- **Visual proof at every step.** Each notebook includes at least one visualisation that confirms the component is working correctly.
- **Depth over breadth.** This is not a survey. Every line of code is explained.