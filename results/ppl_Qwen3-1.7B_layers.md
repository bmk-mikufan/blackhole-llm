Model `Qwen/Qwen3-1.7B` (float32 compute), 8,192 tokens of `wikitext2_test.txt`, ctx 512. Weights only (KV cache / activations not quantized).

| Config | PPL | Δ vs unquantized |
|---|---:|---:|
| unquantized (float32) | 24.006 | — |
| `all-bfp8` | 23.991 | -0.06% |
| `all-bfp8+L0.ff1_3=bfp4+L0.ff2=bfp4` | 24.115 | +0.46% |
| `all-bfp8+L1.ff1_3=bfp4+L1.ff2=bfp4` | 24.041 | +0.15% |
| `all-bfp8+L2.ff1_3=bfp4+L2.ff2=bfp4` | 24.087 | +0.34% |
| `all-bfp8+L3.ff1_3=bfp4+L3.ff2=bfp4` | 24.073 | +0.28% |
| `all-bfp8+L4.ff1_3=bfp4+L4.ff2=bfp4` | 24.215 | +0.87% |
| `all-bfp8+L5.ff1_3=bfp4+L5.ff2=bfp4` | 24.104 | +0.41% |
| `all-bfp8+L6.ff1_3=bfp4+L6.ff2=bfp4` | 24.158 | +0.63% |
| `all-bfp8+L7.ff1_3=bfp4+L7.ff2=bfp4` | 24.176 | +0.71% |
| `all-bfp8+L8.ff1_3=bfp4+L8.ff2=bfp4` | 23.956 | -0.21% |
| `all-bfp8+L9.ff1_3=bfp4+L9.ff2=bfp4` | 24.004 | -0.01% |
| `all-bfp8+L10.ff1_3=bfp4+L10.ff2=bfp4` | 23.881 | -0.52% |
| `all-bfp8+L11.ff1_3=bfp4+L11.ff2=bfp4` | 24.039 | +0.14% |
| `all-bfp8+L12.ff1_3=bfp4+L12.ff2=bfp4` | 24.056 | +0.21% |
| `all-bfp8+L13.ff1_3=bfp4+L13.ff2=bfp4` | 23.989 | -0.07% |
| `all-bfp8+L14.ff1_3=bfp4+L14.ff2=bfp4` | 24.074 | +0.29% |
| `all-bfp8+L15.ff1_3=bfp4+L15.ff2=bfp4` | 23.977 | -0.12% |
| `all-bfp8+L16.ff1_3=bfp4+L16.ff2=bfp4` | 23.823 | -0.76% |
| `all-bfp8+L17.ff1_3=bfp4+L17.ff2=bfp4` | 24.030 | +0.10% |
| `all-bfp8+L18.ff1_3=bfp4+L18.ff2=bfp4` | 23.927 | -0.33% |
| `all-bfp8+L19.ff1_3=bfp4+L19.ff2=bfp4` | 24.192 | +0.78% |
| `all-bfp8+L20.ff1_3=bfp4+L20.ff2=bfp4` | 24.172 | +0.69% |
| `all-bfp8+L21.ff1_3=bfp4+L21.ff2=bfp4` | 24.041 | +0.15% |
| `all-bfp8+L22.ff1_3=bfp4+L22.ff2=bfp4` | 24.120 | +0.48% |
| `all-bfp8+L23.ff1_3=bfp4+L23.ff2=bfp4` | 24.129 | +0.52% |
| `all-bfp8+L24.ff1_3=bfp4+L24.ff2=bfp4` | 24.082 | +0.32% |
| `all-bfp8+L25.ff1_3=bfp4+L25.ff2=bfp4` | 24.093 | +0.36% |
| `all-bfp8+L26.ff1_3=bfp4+L26.ff2=bfp4` | 24.107 | +0.42% |
| `all-bfp8+L27.ff1_3=bfp4+L27.ff2=bfp4` | 23.574 | -1.80% |
