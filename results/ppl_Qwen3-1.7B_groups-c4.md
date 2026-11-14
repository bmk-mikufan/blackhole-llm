Model `Qwen/Qwen3-1.7B` (float32 compute), 32,768 tokens of `c4_validation.txt`, ctx 512. Weights only (KV cache / activations not quantized).

Top-1 agree = share of positions where the quantized model's argmax token matches the unquantized one.

| Config | PPL | Δ vs unquantized | Top-1 agree |
|---|---:|---:|---:|
| unquantized (float32) | 23.913 | — | — |
| `all-bfp8` | 23.949 | +0.15% | 98.21% |
| `all-bfp8+wqkv=bfp4` | 24.435 | +2.18% | 89.71% |
| `all-bfp8+wo=bfp4` | 24.185 | +1.14% | 92.80% |
| `all-bfp8+ff1_3=bfp4` | 25.067 | +4.83% | 86.44% |
| `all-bfp8+ff2=bfp4` | 24.163 | +1.05% | 90.96% |
| `all-bfp8+lm_head=bfp4` | 25.067 | +4.83% | 86.33% |
