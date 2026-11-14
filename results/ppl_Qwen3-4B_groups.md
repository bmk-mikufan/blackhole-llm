Model `Qwen/Qwen3-4B` (bfloat16 compute), 16,384 tokens of `wikitext2_test.txt`, ctx 512. Weights only (KV cache / activations not quantized).

Top-1 agree = share of positions where the quantized model's argmax token matches the unquantized one.

| Config | PPL | Δ vs unquantized | Top-1 agree |
|---|---:|---:|---:|
| unquantized (bfloat16) | 15.411 | — | — |
| `all-bfp8` | 15.456 | +0.29% | 97.77% |
| `all-bfp8+wqkv=bfp4` | 15.642 | +1.50% | 91.47% |
| `all-bfp8+wo=bfp4` | 15.221 | -1.23% | 94.45% |
| `all-bfp8+ff1_3=bfp4` | 16.204 | +5.14% | 89.52% |
| `all-bfp8+ff2=bfp4` | 15.550 | +0.90% | 91.66% |
| `all-bfp8+lm_head=bfp4` | 15.654 | +1.58% | 91.91% |
