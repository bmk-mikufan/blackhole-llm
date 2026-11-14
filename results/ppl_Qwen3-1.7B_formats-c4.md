Model `Qwen/Qwen3-1.7B` (float32 compute), 32,768 tokens of `c4_validation.txt`, ctx 512. Weights only (KV cache / activations not quantized).

Top-1 agree = share of positions where the quantized model's argmax token matches the unquantized one.

| Config | PPL | Δ vs unquantized | Top-1 agree |
|---|---:|---:|---:|
| unquantized (float32) | 23.913 | — | — |
| `bf16` | 23.913 | +0.00% | 100.00% |
| `all-bfp8` | 23.949 | +0.15% | 98.21% |
| `tt-perf` | 25.067 | +4.83% | 86.44% |
| `mlp-bfp4` | 25.288 | +5.75% | 84.08% |
| `all-bfp4` | 27.217 | +13.82% | 75.97% |
