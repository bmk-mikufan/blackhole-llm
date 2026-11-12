Model `Qwen/Qwen3-1.7B` (float32 compute), 32,768 tokens of `wikitext2_test.txt`, ctx 512. Weights only (KV cache / activations not quantized).

| Config | PPL | Δ vs unquantized |
|---|---:|---:|
| unquantized (float32) | 20.757 | — |
| `bf16` | 20.757 | +0.00% |
| `all-bfp8` | 20.768 | +0.05% |
| `tt-perf` | 22.190 | +6.90% |
| `mlp-bfp4` | 21.798 | +5.01% |
| `all-bfp4` | 23.397 | +12.72% |
