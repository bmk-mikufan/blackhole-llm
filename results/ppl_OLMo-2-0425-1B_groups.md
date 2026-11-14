Model `allenai/OLMo-2-0425-1B` (float32 compute), 32,768 tokens of `wikitext2_test.txt`, ctx 512. Weights only (KV cache / activations not quantized).

Top-1 agree = share of positions where the quantized model's argmax token matches the unquantized one.

| Config | PPL | Δ vs unquantized | Top-1 agree |
|---|---:|---:|---:|
| unquantized (float32) | 12.155 | — | — |
| `all-bfp8` | 12.176 | +0.18% | 97.67% |
| `all-bfp8+wqkv=bfp4` | 12.407 | +2.08% | 91.85% |
| `all-bfp8+wo=bfp4` | 12.337 | +1.50% | 93.73% |
| `all-bfp8+ff1_3=bfp4` | 12.792 | +5.24% | 87.21% |
| `all-bfp8+ff2=bfp4` | 12.563 | +3.36% | 90.22% |
| `all-bfp8+lm_head=bfp4` | 14.137 | +16.31% | 77.27% |
