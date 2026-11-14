Model `HuggingFaceTB/SmolLM2-1.7B` (float32 compute), 32,768 tokens of `wikitext2_test.txt`, ctx 512. Weights only (KV cache / activations not quantized).

Top-1 agree = share of positions where the quantized model's argmax token matches the unquantized one.

| Config | PPL | Δ vs unquantized | Top-1 agree |
|---|---:|---:|---:|
| unquantized (float32) | 10.645 | — | — |
| `all-bfp8` | 10.658 | +0.12% | 97.50% |
| `all-bfp8+wqkv=bfp4` | 11.340 | +6.53% | 86.34% |
| `all-bfp8+wo=bfp4` | 10.774 | +1.21% | 94.00% |
| `all-bfp8+ff1_3=bfp4` | 11.387 | +6.97% | 87.08% |
| `all-bfp8+ff2=bfp4` | 11.625 | +9.21% | 85.54% |
| `all-bfp8+lm_head=bfp4` | 12.887 | +21.06% | 72.75% |
