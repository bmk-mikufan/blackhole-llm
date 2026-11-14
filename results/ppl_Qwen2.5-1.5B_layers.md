Model `Qwen/Qwen2.5-1.5B` (float32 compute), 8,192 tokens of `wikitext2_test.txt`, ctx 512. Weights only (KV cache / activations not quantized).

Top-1 agree = share of positions where the quantized model's argmax token matches the unquantized one.

| Config | PPL | Δ vs unquantized | Top-1 agree |
|---|---:|---:|---:|
| unquantized (float32) | 13.848 | — | — |
| `all-bfp8` | 13.871 | +0.16% | 97.75% |
| `all-bfp8+L0.ff1_3=bfp4+L0.ff2=bfp4` | 13.905 | +0.41% | 96.58% |
| `all-bfp8+L1.ff1_3=bfp4+L1.ff2=bfp4` | 14.024 | +1.27% | 95.41% |
| `all-bfp8+L2.ff1_3=bfp4+L2.ff2=bfp4` | 14.055 | +1.49% | 95.53% |
| `all-bfp8+L3.ff1_3=bfp4+L3.ff2=bfp4` | 13.913 | +0.47% | 95.65% |
| `all-bfp8+L4.ff1_3=bfp4+L4.ff2=bfp4` | 13.960 | +0.81% | 96.01% |
| `all-bfp8+L5.ff1_3=bfp4+L5.ff2=bfp4` | 13.916 | +0.49% | 95.86% |
| `all-bfp8+L6.ff1_3=bfp4+L6.ff2=bfp4` | 13.885 | +0.27% | 96.24% |
| `all-bfp8+L7.ff1_3=bfp4+L7.ff2=bfp4` | 13.881 | +0.24% | 96.08% |
| `all-bfp8+L8.ff1_3=bfp4+L8.ff2=bfp4` | 13.943 | +0.68% | 95.97% |
| `all-bfp8+L9.ff1_3=bfp4+L9.ff2=bfp4` | 13.905 | +0.41% | 96.64% |
| `all-bfp8+L10.ff1_3=bfp4+L10.ff2=bfp4` | 13.945 | +0.70% | 96.07% |
| `all-bfp8+L11.ff1_3=bfp4+L11.ff2=bfp4` | 13.872 | +0.17% | 96.20% |
| `all-bfp8+L12.ff1_3=bfp4+L12.ff2=bfp4` | 13.890 | +0.30% | 96.47% |
| `all-bfp8+L13.ff1_3=bfp4+L13.ff2=bfp4` | 13.920 | +0.52% | 96.51% |
| `all-bfp8+L14.ff1_3=bfp4+L14.ff2=bfp4` | 13.888 | +0.29% | 96.47% |
| `all-bfp8+L15.ff1_3=bfp4+L15.ff2=bfp4` | 13.888 | +0.29% | 96.52% |
| `all-bfp8+L16.ff1_3=bfp4+L16.ff2=bfp4` | 13.917 | +0.49% | 96.01% |
| `all-bfp8+L17.ff1_3=bfp4+L17.ff2=bfp4` | 13.917 | +0.50% | 96.31% |
| `all-bfp8+L18.ff1_3=bfp4+L18.ff2=bfp4` | 13.901 | +0.38% | 96.06% |
| `all-bfp8+L19.ff1_3=bfp4+L19.ff2=bfp4` | 13.939 | +0.65% | 96.33% |
| `all-bfp8+L20.ff1_3=bfp4+L20.ff2=bfp4` | 13.929 | +0.58% | 96.35% |
| `all-bfp8+L21.ff1_3=bfp4+L21.ff2=bfp4` | 13.935 | +0.63% | 96.04% |
| `all-bfp8+L22.ff1_3=bfp4+L22.ff2=bfp4` | 13.917 | +0.50% | 96.08% |
| `all-bfp8+L23.ff1_3=bfp4+L23.ff2=bfp4` | 13.918 | +0.51% | 95.97% |
| `all-bfp8+L24.ff1_3=bfp4+L24.ff2=bfp4` | 13.913 | +0.46% | 95.96% |
| `all-bfp8+L25.ff1_3=bfp4+L25.ff2=bfp4` | 13.933 | +0.61% | 96.08% |
| `all-bfp8+L26.ff1_3=bfp4+L26.ff2=bfp4` | 14.204 | +2.56% | 95.43% |
| `all-bfp8+L27.ff1_3=bfp4+L27.ff2=bfp4` | 14.084 | +1.70% | 90.88% |
