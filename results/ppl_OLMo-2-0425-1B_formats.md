Model `allenai/OLMo-2-0425-1B` (float32 compute), 32,768 tokens of `wikitext2_test.txt`, ctx 512. Weights only (KV cache / activations not quantized).

Top-1 agree = share of positions where the quantized model's argmax token matches the unquantized one.

| Config | PPL | Δ vs unquantized | Top-1 agree |
|---|---:|---:|---:|
| unquantized (float32) | 12.155 | — | — |
| `bf16` | 12.153 | -0.01% | 99.64% |
| `all-bfp8` | 12.176 | +0.18% | 97.67% |
| `tt-perf` | 12.792 | +5.24% | 87.21% |
| `mlp-bfp4` | 13.225 | +8.81% | 84.57% |
| `all-bfp4` | 16.188 | +33.18% | 70.02% |
