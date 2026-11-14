Model `HuggingFaceTB/SmolLM2-1.7B` (float32 compute), 8,192 tokens of `wikitext2_test.txt`, ctx 512. Weights only (KV cache / activations not quantized).

Top-1 agree = share of positions where the quantized model's argmax token matches the unquantized one.

| Config | PPL | Δ vs unquantized | Top-1 agree |
|---|---:|---:|---:|
| unquantized (float32) | 12.696 | — | — |
| `all-bfp8` | 12.723 | +0.21% | 97.31% |
| `all-bfp8+L0.ff1_3=bfp4+L0.ff2=bfp4` | 12.768 | +0.56% | 96.35% |
| `all-bfp8+L1.ff1_3=bfp4+L1.ff2=bfp4` | 13.761 | +8.38% | 88.11% |
| `all-bfp8+L2.ff1_3=bfp4+L2.ff2=bfp4` | 12.760 | +0.50% | 96.45% |
| `all-bfp8+L3.ff1_3=bfp4+L3.ff2=bfp4` | 12.748 | +0.41% | 96.54% |
| `all-bfp8+L4.ff1_3=bfp4+L4.ff2=bfp4` | 12.744 | +0.38% | 96.36% |
| `all-bfp8+L5.ff1_3=bfp4+L5.ff2=bfp4` | 12.743 | +0.37% | 96.56% |
| `all-bfp8+L6.ff1_3=bfp4+L6.ff2=bfp4` | 12.735 | +0.30% | 96.62% |
| `all-bfp8+L7.ff1_3=bfp4+L7.ff2=bfp4` | 12.745 | +0.38% | 95.84% |
| `all-bfp8+L8.ff1_3=bfp4+L8.ff2=bfp4` | 12.729 | +0.26% | 96.45% |
| `all-bfp8+L9.ff1_3=bfp4+L9.ff2=bfp4` | 12.744 | +0.37% | 96.56% |
| `all-bfp8+L10.ff1_3=bfp4+L10.ff2=bfp4` | 12.745 | +0.38% | 96.36% |
| `all-bfp8+L11.ff1_3=bfp4+L11.ff2=bfp4` | 12.748 | +0.40% | 96.50% |
| `all-bfp8+L12.ff1_3=bfp4+L12.ff2=bfp4` | 12.745 | +0.39% | 96.73% |
| `all-bfp8+L13.ff1_3=bfp4+L13.ff2=bfp4` | 12.752 | +0.44% | 96.35% |
| `all-bfp8+L14.ff1_3=bfp4+L14.ff2=bfp4` | 12.746 | +0.39% | 96.53% |
| `all-bfp8+L15.ff1_3=bfp4+L15.ff2=bfp4` | 12.755 | +0.46% | 96.26% |
| `all-bfp8+L16.ff1_3=bfp4+L16.ff2=bfp4` | 12.748 | +0.40% | 96.09% |
| `all-bfp8+L17.ff1_3=bfp4+L17.ff2=bfp4` | 12.781 | +0.67% | 96.13% |
| `all-bfp8+L18.ff1_3=bfp4+L18.ff2=bfp4` | 12.777 | +0.63% | 95.98% |
| `all-bfp8+L19.ff1_3=bfp4+L19.ff2=bfp4` | 12.788 | +0.72% | 95.86% |
| `all-bfp8+L20.ff1_3=bfp4+L20.ff2=bfp4` | 12.774 | +0.61% | 96.07% |
| `all-bfp8+L21.ff1_3=bfp4+L21.ff2=bfp4` | 12.770 | +0.58% | 95.97% |
| `all-bfp8+L22.ff1_3=bfp4+L22.ff2=bfp4` | 12.884 | +1.48% | 95.69% |
| `all-bfp8+L23.ff1_3=bfp4+L23.ff2=bfp4` | 13.305 | +4.79% | 87.49% |
