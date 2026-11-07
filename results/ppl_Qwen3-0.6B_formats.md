Model `Qwen/Qwen3-0.6B` (float32 compute), 32,768 tokens of `wikitext2_test.txt`, ctx 512. Weights only (KV cache / activations not quantized).

| Config | PPL | Δ vs unquantized |
|---|---:|---:|
| unquantized (float32) | 27.688 | — |
| `bf16` | 27.688 | +0.00% |
| `all-bfp8` | 27.827 | +0.50% |
| `tt-perf` | 30.661 | +10.74% |
| `mlp-bfp4` | 32.137 | +16.07% |
| `all-bfp4` | 36.255 | +30.94% |
