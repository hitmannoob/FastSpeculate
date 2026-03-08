# N-gram Tree Speculative Decoding

Training-free speculative decoding that replaces Medusa's trained heads with n-gram dictionary lookups + tree attention verification. Zero parameters to train, works on any model, up to **2.67x speedup** on repetitive code generation.

**[Blog Post](https://medium.com/@deepshbansal491/n-gram-tree-speculative-decoding-replacing-medusa-heads-with-n-gram-dictionaries-for-faster-code-74eef84bdebe)** | **[Notebook](https://github.com/hitmannoob/FastSpeculate/blob/main/ngram_speculative_decoding_final.ipynb)**

## Key Idea

1. Build n-gram dictionary from prompt context
2. Multiple matching continuations → merge into tree with shared prefixes
3. Verify all branches in one forward pass using Medusa-style tree attention
4. Accept longest valid path

## Results (Qwen2.5-Coder-7B, A100)

| Task | vs Baseline | vs Prompt Lookup |
|------|-------------|------------------|
| Repetitive classes | **2.67x** | **2.21x** |
| API boilerplate | **2.07x** | 0.94x |
| Unit tests | **1.92x** | 1.00x |
| Validation code | **1.64x** | **1.28x** |

## Quick Start

```bash
pip install transformers accelerate torch
```

Open the notebook and run all cells.
