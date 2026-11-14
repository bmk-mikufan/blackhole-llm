Model `allenai/OLMo-2-0425-1B` (float32 compute), 8,192 tokens of `wikitext2_test.txt`, ctx 512. Weights only (KV cache / activations not quantized).

Top-1 agree = share of positions where the quantized model's argmax token matches the unquantized one.

| Config | PPL | Δ vs unquantized | Top-1 agree |
|---|---:|---:|---:|
| unquantized (float32) | 13.285 | — | — |
| `all-bfp8` | 13.319 | +0.26% | 97.51% |
| `all-bfp8+L0.ff1_3=bfp4+L0.ff2=bfp4` | 13.312 | +0.21% | 96.35% |
| `all-bfp8+L1.ff1_3=bfp4+L1.ff2=bfp4` | 13.335 | +0.38% | 95.67% |
| `all-bfp8+L2.ff1_3=bfp4+L2.ff2=bfp4` | 13.376 | +0.69% | 96.04% |
| `all-bfp8+L3.ff1_3=bfp4+L3.ff2=bfp4` | 13.407 | +0.92% | 95.96% |
| `all-bfp8+L4.ff1_3=bfp4+L4.ff2=bfp4` | 13.370 | +0.65% | 95.54% |
| `all-bfp8+L5.ff1_3=bfp4+L5.ff2=bfp4` | 13.431 | +1.10% | 95.48% |
| `all-bfp8+L6.ff1_3=bfp4+L6.ff2=bfp4` | 13.359 | +0.56% | 95.51% |
| `all-bfp8+L7.ff1_3=bfp4+L7.ff2=bfp4` | 13.346 | +0.46% | 95.69% |
| `all-bfp8+L8.ff1_3=bfp4+L8.ff2=bfp4` | 13.383 | +0.74% | 95.80% |
| `all-bfp8+L9.ff1_3=bfp4+L9.ff2=bfp4` | 13.372 | +0.66% | 95.80% |
| `all-bfp8+L10.ff1_3=bfp4+L10.ff2=bfp4` | 13.372 | +0.66% | 94.93% |
| `all-bfp8+L11.ff1_3=bfp4+L11.ff2=bfp4` | 13.410 | +0.95% | 94.87% |
| `all-bfp8+L12.ff1_3=bfp4+L12.ff2=bfp4` | 13.400 | +0.87% | 95.09% |
| `all-bfp8+L13.ff1_3=bfp4+L13.ff2=bfp4` | 13.401 | +0.87% | 95.08% |
| `all-bfp8+L14.ff1_3=bfp4+L14.ff2=bfp4` | 13.425 | +1.05% | 94.90% |
| `all-bfp8+L15.ff1_3=bfp4+L15.ff2=bfp4` | 13.547 | +1.97% | 92.30% |
