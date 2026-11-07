Model `Qwen/Qwen3-0.6B` (float32 compute), 32,768 tokens of `wikitext2_test.txt`, ctx 512. Weights only (KV cache / activations not quantized).

| Config | PPL | Δ vs unquantized |
|---|---:|---:|
| unquantized (float32) | 27.688 | — |
| `all-bfp8` | 27.828 | +0.50% |
| `all-bfp8+wqkv=bfp4` | 29.047 | +4.91% |
| `all-bfp8+wo=bfp4` | 28.206 | +1.87% |
| `all-bfp8+ff1_3=bfp4` | 30.661 | +10.74% |
| `all-bfp8+ff2=bfp4` | 29.084 | +5.04% |
| `all-bfp8+lm_head=bfp4` | 29.497 | +6.53% |
