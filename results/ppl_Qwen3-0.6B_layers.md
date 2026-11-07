Model `Qwen/Qwen3-0.6B` (float32 compute), 8,192 tokens of `wikitext2_test.txt`, ctx 512. Weights only (KV cache / activations not quantized).

| Config | PPL | Δ vs unquantized |
|---|---:|---:|
| unquantized (float32) | 32.386 | — |
| `all-bfp8` | 32.454 | +0.21% |
| `all-bfp8+L0.ff1_3=bfp4+L0.ff2=bfp4` | 32.743 | +1.10% |
| `all-bfp8+L1.ff1_3=bfp4+L1.ff2=bfp4` | 32.439 | +0.16% |
| `all-bfp8+L2.ff1_3=bfp4+L2.ff2=bfp4` | 32.373 | -0.04% |
| `all-bfp8+L3.ff1_3=bfp4+L3.ff2=bfp4` | 32.473 | +0.27% |
| `all-bfp8+L4.ff1_3=bfp4+L4.ff2=bfp4` | 32.634 | +0.77% |
| `all-bfp8+L5.ff1_3=bfp4+L5.ff2=bfp4` | 32.683 | +0.92% |
| `all-bfp8+L6.ff1_3=bfp4+L6.ff2=bfp4` | 32.503 | +0.36% |
| `all-bfp8+L7.ff1_3=bfp4+L7.ff2=bfp4` | 32.682 | +0.91% |
| `all-bfp8+L8.ff1_3=bfp4+L8.ff2=bfp4` | 32.528 | +0.44% |
| `all-bfp8+L9.ff1_3=bfp4+L9.ff2=bfp4` | 32.538 | +0.47% |
| `all-bfp8+L10.ff1_3=bfp4+L10.ff2=bfp4` | 32.583 | +0.61% |
| `all-bfp8+L11.ff1_3=bfp4+L11.ff2=bfp4` | 32.454 | +0.21% |
| `all-bfp8+L12.ff1_3=bfp4+L12.ff2=bfp4` | 32.582 | +0.60% |
| `all-bfp8+L13.ff1_3=bfp4+L13.ff2=bfp4` | 32.533 | +0.45% |
| `all-bfp8+L14.ff1_3=bfp4+L14.ff2=bfp4` | 32.611 | +0.70% |
| `all-bfp8+L15.ff1_3=bfp4+L15.ff2=bfp4` | 32.381 | -0.02% |
| `all-bfp8+L16.ff1_3=bfp4+L16.ff2=bfp4` | 32.498 | +0.34% |
| `all-bfp8+L17.ff1_3=bfp4+L17.ff2=bfp4` | 32.626 | +0.74% |
| `all-bfp8+L18.ff1_3=bfp4+L18.ff2=bfp4` | 32.636 | +0.77% |
| `all-bfp8+L19.ff1_3=bfp4+L19.ff2=bfp4` | 32.714 | +1.01% |
| `all-bfp8+L20.ff1_3=bfp4+L20.ff2=bfp4` | 32.630 | +0.75% |
| `all-bfp8+L21.ff1_3=bfp4+L21.ff2=bfp4` | 32.458 | +0.22% |
| `all-bfp8+L22.ff1_3=bfp4+L22.ff2=bfp4` | 32.977 | +1.82% |
| `all-bfp8+L23.ff1_3=bfp4+L23.ff2=bfp4` | 32.629 | +0.75% |
| `all-bfp8+L24.ff1_3=bfp4+L24.ff2=bfp4` | 32.641 | +0.79% |
| `all-bfp8+L25.ff1_3=bfp4+L25.ff2=bfp4` | 32.535 | +0.46% |
| `all-bfp8+L26.ff1_3=bfp4+L26.ff2=bfp4` | 32.734 | +1.08% |
| `all-bfp8+L27.ff1_3=bfp4+L27.ff2=bfp4` | 33.344 | +2.96% |
