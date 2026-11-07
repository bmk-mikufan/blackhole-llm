Model `Qwen/Qwen3-0.6B` (float32 compute), 32,768 tokens of `wikitext2_test.txt`, ctx 512. Weights only (KV cache / activations not quantized).

| Config | PPL | Δ vs unquantized |
|---|---:|---:|
| unquantized (float32) | 27.688 | — |
| `mlp-bfp4+L0.ff1_3=bfp8+L0.ff2=bfp8+L22.ff1_3=bfp8+L22.ff2=bfp8+L27.ff1_3=bfp8+L27.ff2=bfp8` | 30.549 | +10.33% |
| `tt-perf+L0.ff1_3=bfp8+L22.ff1_3=bfp8+L27.ff1_3=bfp8` | 29.911 | +8.03% |
