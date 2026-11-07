Published Llama 3.1 8B p150 @ B=32: 33.1 t/s/u -> step 30.2 ms, 6.67 GB moved/step (tt-perf), implied DRAM BW 221 GB/s = 43% of 512 GB/s; ceiling 76.8 t/s/u.
(if weights were all-bfp8: implied 55% of peak)
(if weights were bf16: implied 104% of peak)

Context length per user = 1024 tokens. 'eff' column applies the B=32 calibration to B=1 (optimistic: B=1 usually reaches lower bandwidth efficiency).

| Model | Config | Weights read/step (GB) | B=1 ceiling t/s/u | B=1 @ 43% eff | B=32 ceiling t/s/u | B=32 ceiling t/s |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.2-1B | all-bfp8 | 1.31 | 385 | 166 | 272 | 8699 |
| Llama-3.2-1B | tt-perf | 1.04 | 482 | 208 | 317 | 10145 |
| Llama-3.2-1B | mlp-bfp4 | 0.91 | 552 | 238 | 346 | 11065 |
| Llama-3.2-3B | all-bfp8 | 3.41 | 147 | 63 | 95 | 3029 |
| Llama-3.2-3B | tt-perf | 2.71 | 185 | 80 | 109 | 3482 |
| Llama-3.2-3B | mlp-bfp4 | 2.36 | 212 | 91 | 118 | 3764 |
| Llama-3.1-8B | all-bfp8 | 7.97 | 64 | 27 | 50 | 1598 |
| Llama-3.1-8B | tt-perf | 6.09 | 83 | 36 | 61 | 1956 |
| Llama-3.1-8B | mlp-bfp4 | 5.16 | 98 | 42 | 69 | 2203 |
| Qwen3-8B | all-bfp8 | 8.04 | 63 | 27 | 48 | 1544 |
| Qwen3-8B | tt-perf | 6.23 | 81 | 35 | 58 | 1863 |
| Qwen3-8B | mlp-bfp4 | 5.32 | 95 | 41 | 65 | 2077 |
| Qwen3-14B | all-bfp8 | 14.86 | 34 | 15 | 29 | 925 |
| Qwen3-14B | tt-perf | 11.30 | 45 | 19 | 36 | 1158 |
| Qwen3-14B | mlp-bfp4 | 9.52 | 53 | 23 | 41 | 1325 |
| Mistral-7B-v0.3 | all-bfp8 | 7.56 | 67 | 29 | 52 | 1665 |
| Mistral-7B-v0.3 | tt-perf | 5.68 | 89 | 38 | 64 | 2058 |
| Mistral-7B-v0.3 | mlp-bfp4 | 4.74 | 106 | 46 | 73 | 2333 |
| Gemma-3-12B | all-bfp8 | 12.50 | 40 | 17 | 26 | 847 |
| Gemma-3-12B | tt-perf | 9.67 | 52 | 22 | 31 | 992 |
| Gemma-3-12B | mlp-bfp4 | 8.25 | 60 | 26 | 34 | 1085 |
| Qwen3-32B | all-bfp8 | 33.98 | — | — | does not fit | — |
| Qwen3-32B | tt-perf | 25.59 | 20 | 9 | 17 | 543 |
| Qwen3-32B | mlp-bfp4 | 21.40 | 24 | 10 | 20 | 631 |
