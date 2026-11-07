Model `Qwen/Qwen3-0.6B` (float32 compute), 32,768 tokens of `wikitext2_test.txt`, ctx 512. Weights only (KV cache / activations not quantized).

| Config | PPL | Δ vs unquantized |
|---|---:|---:|
| unquantized (float32) | 27.688 | — |
| `all-bfp8+wo=bfp4+ff2=bfp4` | 29.619 | +6.97% |
| `all-bfp8+wo=bfp4+ff2=bfp4+wqkv=bfp4` | 30.928 | +11.70% |
