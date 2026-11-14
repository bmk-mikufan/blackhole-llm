Model `Qwen/Qwen3-4B` (bfloat16 compute), 16,384 tokens of `wikitext2_test.txt`, ctx 512. Weights only (KV cache / activations not quantized).

Top-1 agree = share of positions where the quantized model's argmax token matches the unquantized one.

| Config | PPL | Δ vs unquantized | Top-1 agree |
|---|---:|---:|---:|
| unquantized (bfloat16) | 15.418 | — | — |
| `bf16` | 15.418 | +0.00% | 100.00% |
| `all-bfp8` | 15.436 | +0.11% | 97.80% |
| `tt-perf` | 16.195 | +5.04% | 89.42% |
| `mlp-bfp4` | 16.401 | +6.38% | 86.95% |
| `all-bfp4` | 16.492 | +6.97% | 82.32% |
