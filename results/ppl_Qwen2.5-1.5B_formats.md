Model `Qwen/Qwen2.5-1.5B` (float32 compute), 32,768 tokens of `wikitext2_test.txt`, ctx 512. Weights only (KV cache / activations not quantized).

Top-1 agree = share of positions where the quantized model's argmax token matches the unquantized one.

| Config | PPL | Δ vs unquantized | Top-1 agree |
|---|---:|---:|---:|
| unquantized (float32) | 11.790 | — | — |
| `bf16` | 11.790 | +0.00% | 100.00% |
| `all-bfp8` | 11.814 | +0.20% | 98.04% |
| `tt-perf` | 12.382 | +5.02% | 86.08% |
| `mlp-bfp4` | 13.068 | +10.83% | 82.63% |
| `all-bfp4` | 14.472 | +22.74% | 74.21% |
