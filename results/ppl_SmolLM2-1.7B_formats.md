Model `HuggingFaceTB/SmolLM2-1.7B` (float32 compute), 32,768 tokens of `wikitext2_test.txt`, ctx 512. Weights only (KV cache / activations not quantized).

Top-1 agree = share of positions where the quantized model's argmax token matches the unquantized one.

| Config | PPL | Δ vs unquantized | Top-1 agree |
|---|---:|---:|---:|
| unquantized (float32) | 10.645 | — | — |
| `bf16` | 10.645 | +0.00% | 100.00% |
| `all-bfp8` | 10.658 | +0.12% | 97.50% |
| `tt-perf` | 11.387 | +6.97% | 87.08% |
| `mlp-bfp4` | 12.216 | +14.76% | 81.55% |
| `all-bfp4` | 16.746 | +57.31% | 62.84% |
