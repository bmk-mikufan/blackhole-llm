Model `Qwen/Qwen2.5-1.5B` (float32 compute), 32,768 tokens of `wikitext2_test.txt`, ctx 512. Weights only (KV cache / activations not quantized).

Top-1 agree = share of positions where the quantized model's argmax token matches the unquantized one.

| Config | PPL | Δ vs unquantized | Top-1 agree |
|---|---:|---:|---:|
| unquantized (float32) | 11.791 | — | — |
| `all-bfp8` | 11.814 | +0.20% | 98.04% |
| `all-bfp8+wqkv=bfp4` | 12.144 | +2.99% | 91.27% |
| `all-bfp8+wo=bfp4` | 11.947 | +1.32% | 93.15% |
| `all-bfp8+ff1_3=bfp4` | 12.383 | +5.02% | 86.08% |
| `all-bfp8+ff2=bfp4` | 12.403 | +5.20% | 89.36% |
| `all-bfp8+lm_head=bfp4` | 12.663 | +7.40% | 83.18% |
