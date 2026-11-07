"""Bit-faithful-ish CPU emulation of Tenstorrent block floating point (bfp8_b / bfp4_b).

Layout mirrors TT tiles: a weight is stored as W^T [in, out]; each 16 consecutive elements of a
tile face row (i.e. 16 neighbouring *output* columns for one input row) share one 8-bit exponent.
Each element keeps a sign and `mant` magnitude bits, aligned to the block's max exponent:
    bfp8_b: 7 magnitude bits, bfp4_b: 3 magnitude bits.

Run:  python bfp.py   -> self-check + synthetic weight-distribution study (results/bfp_synthetic.md)
"""
from pathlib import Path
import numpy as np

MANT = {"bfp8": 7, "bfp4": 3, "bfp2": 1}


def to_bf16(x):
    """Round float32 -> bfloat16 (round-to-nearest-even), returned as float32."""
    b = np.asarray(x, np.float32).view(np.uint32)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.view(np.float32)


def bfp(x, fmt, block=16, rounding="nearest"):
    """Quantize-dequantize `x` (blocks run along the last axis) to a TT block-float format."""
    if fmt == "bf16":
        return to_bf16(x)
    m = MANT[fmt]
    x = np.asarray(x, np.float32)
    s = x.reshape(-1, block)
    amax = np.abs(s).max(1, keepdims=True)
    e = np.floor(np.log2(np.where(amax > 0, amax, 1.0)))
    step = np.exp2(e - (m - 1))                     # value of one mantissa LSB in this block
    mag = np.abs(s) / step
    q = np.floor(mag + 0.5) if rounding == "nearest" else np.floor(mag)
    q = np.minimum(q, 2**m - 1)
    return (np.sign(s) * q * step).reshape(x.shape).astype(np.float32)


def quantize_linear_weight(w, fmt, block=16):
    """torch-style Linear weight [out, in] -> stored as W^T [in, out], blocks along out."""
    return bfp(w.T, fmt, block).T


def sqnr_db(ref, approx):
    return 10 * np.log10(np.sum(ref.astype(np.float64) ** 2) / np.sum((ref - approx).astype(np.float64) ** 2))


def _check():
    # values already representable must survive exactly
    x = np.array([1, -0.5, 0.25, 0.75] + [0] * 12, np.float32)
    assert np.array_equal(bfp(x, "bfp8"), x) and np.array_equal(bfp(x, "bfp4"), x)
    # one outlier crushes its block's small values in bfp4 (3 magnitude bits => step = max/4)
    y = np.array([8.0] + [0.9] * 15, np.float32)
    assert np.all(bfp(y, "bfp4")[1:] == 0) and np.all(bfp(y, "bfp8")[1:] > 0)
    # error ordering on random data
    r = np.random.default_rng(0).standard_normal((64, 256)).astype(np.float32)
    s = [sqnr_db(r, bfp(r, f)) for f in ["bf16", "bfp8", "bfp4"]]
    assert s[0] > s[1] > s[2] > 0, s
    assert abs(to_bf16(np.float32(1 + 2**-9)) - 1.0) == 0      # rounds to even


def weight(kind, n_in, n_out, rng):
    w = rng.standard_normal((n_out, n_in)).astype(np.float32) * 0.02
    if kind == "student-t (nu=3)":
        w = (rng.standard_t(3, (n_out, n_in)) * 0.02).astype(np.float32)
    elif kind == "gaussian + 1% outlier in-ch x10":
        w[:, rng.choice(n_in, n_in // 100, replace=False)] *= 10
    elif kind == "gaussian + 1% outlier out-ch x10":
        w[rng.choice(n_out, n_out // 100, replace=False), :] *= 10
    return w


def study(n=2048, batch=32, seed=0):
    rng = np.random.default_rng(seed)
    kinds = ["gaussian", "student-t (nu=3)", "gaussian + 1% outlier in-ch x10", "gaussian + 1% outlier out-ch x10"]
    fmts = ["bf16", "bfp8", "bfp4"]
    rows = ["Weight SQNR / matmul-output SQNR in dB (higher is better), W: 2048x2048, x: 32x2048 ~ N(0,1).\n",
            "| Weight distribution | " + " | ".join(f"{f} W / y" for f in fmts) + " |",
            "|---|" + "---:|" * len(fmts)]
    for k in kinds:
        w = weight(k, n, n, rng)
        x = rng.standard_normal((batch, n)).astype(np.float32)
        y = x @ w.T
        cells = []
        for f in fmts:
            wq = quantize_linear_weight(w, f)
            cells.append(f"{sqnr_db(w, wq):.1f} / {sqnr_db(y, x @ wq.T):.1f}")
        rows.append(f"| {k} | " + " | ".join(cells) + " |")

    w = weight("student-t (nu=3)", n, n, rng)
    rows += ["\nBlock-size sweep (student-t weights, weight SQNR dB; TT hardware uses 16):\n",
             "| block | bfp8 | bfp4 | bfp4 bytes/elem |", "|---:|---:|---:|---:|"]
    for b in [8, 16, 32, 64]:
        rows.append(f"| {b} | {sqnr_db(w, quantize_linear_weight(w, 'bfp8', b)):.1f} | "
                    f"{sqnr_db(w, quantize_linear_weight(w, 'bfp4', b)):.1f} | {0.5 + 1 / b:.4f} |")

    rows += ["\nRounding mode (student-t weights, weight SQNR dB):\n", "| mode | bfp8 | bfp4 |", "|---|---:|---:|"]
    for mode in ["nearest", "truncate"]:
        rows.append(f"| {mode} | " + " | ".join(f"{sqnr_db(w, bfp(w.T, f, rounding=mode).T):.1f}" for f in ["bfp8", "bfp4"]) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    _check()
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    text = study()
    (out / "bfp_synthetic.md").write_text(text + "\n", encoding="utf-8")
    print(text)
