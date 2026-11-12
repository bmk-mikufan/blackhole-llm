Model `Qwen/Qwen3-1.7B` (float32 compute), 32,768 tokens of `wikitext2_test.txt`, ctx 512. Weights only (KV cache / activations not quantized).

| Config | PPL | Δ vs unquantized |
|---|---:|---:|
| unquantized (float32) | 20.757 | — |
| `all-bfp8` | 20.768 | +0.05% |
| `all-bfp8+wqkv=bfp4` | 21.101 | +1.65% |
| `all-bfp8+wo=bfp4` | 20.798 | +0.20% |
| `all-bfp8+ff1_3=bfp4` | 22.190 | +6.90% |
| `all-bfp8+ff2=bfp4` | 20.510 | -1.19% |
| `all-bfp8+lm_head=bfp4` | 21.897 | +5.49% |
