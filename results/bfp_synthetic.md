Weight SQNR / matmul-output SQNR in dB (higher is better), W: 2048x2048, x: 32x2048 ~ N(0,1).

| Weight distribution | bf16 W / y | bfp8 W / y | bfp4 W / y |
|---|---:|---:|---:|
| gaussian | 55.6 / 55.6 | 43.1 / 43.1 | 19.0 / 19.0 |
| student-t (nu=3) | 55.5 / 55.5 | 40.1 / 40.1 | 16.6 / 16.6 |
| gaussian + 1% outlier in-ch x10 | 55.6 / 55.6 | 43.1 / 43.1 | 19.0 / 19.0 |
| gaussian + 1% outlier out-ch x10 | 55.6 / 55.4 | 39.8 / 39.6 | 16.4 / 16.3 |

Block-size sweep (student-t weights, weight SQNR dB; TT hardware uses 16):

| block | bfp8 | bfp4 | bfp4 bytes/elem |
|---:|---:|---:|---:|
| 8 | 42.3 | 18.5 | 0.6250 |
| 16 | 40.1 | 16.6 | 0.5625 |
| 32 | 38.0 | 14.7 | 0.5312 |
| 64 | 36.0 | 12.8 | 0.5156 |

Rounding mode (student-t weights, weight SQNR dB):

| mode | bfp8 | bfp4 |
|---|---:|---:|
| nearest | 40.1 | 16.6 |
| truncate | 34.3 | 12.0 |
