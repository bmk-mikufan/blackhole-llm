"""Figures from the CPU accuracy-emulation results (results/ppl_*_{groups,layers}.md).

    python plot_ppl.py Qwen3-0.6B
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from roofline import BYTES, MODELS, layer_params

ROOT = Path(__file__).parent


def rows(path):
    """{spec: ppl} from a results table, plus the unquantized baseline under key None."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (`?)(.+?)\1 \| ([\d.]+) \|", line)
        if m:
            out[None if m.group(2).startswith("unquantized") else m.group(2)] = float(m.group(3))
    return out


def main(name):
    m = MODELS[name]
    lp = layer_params(m)
    saved_mb = {g: lp[g] * m["L"] * (BYTES["bfp8"] - BYTES["bfp4"]) / 1e6 for g in lp}
    saved_mb["lm_head"] = m["V"] * m["d"] * (BYTES["bfp8"] - BYTES["bfp4"]) / 1e6

    g = rows(ROOT / "results" / f"ppl_{name}_groups.md")
    ref = g["all-bfp8"]
    groups = ["wqkv", "wo", "ff1_3", "ff2", "lm_head"]
    delta = [100 * (g[f"all-bfp8+{k}=bfp4"] / ref - 1) for k in groups]

    lay = rows(ROOT / "results" / f"ppl_{name}_layers.md")
    lref = lay["all-bfp8"]
    ldelta = [100 * (lay[k] / lref - 1) for k in lay if k and k.startswith("all-bfp8+L")]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 1, 2]})
    ax[0].bar(groups, delta, color="#c05621")
    ax[0].set_title("PPL increase when one group -> BFP4\n(rest BFP8)")
    ax[0].set_ylabel("Δ PPL vs all-BFP8 (%)")
    ax[1].bar(groups, [d / saved_mb[k] * 10 for d, k in zip(delta, groups)], color="#2b6cb0")
    ax[1].set_title("Cost per byte saved\n(Δ PPL % per 10 MB of weights)")
    colors = ["#c05621" if d > 1.0 else "#888" for d in ldelta]
    ax[2].bar(range(len(ldelta)), ldelta, color=colors)
    ax[2].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[2].set_title("Per-layer sensitivity: layer i MLP (FF1/FF3/FF2) -> BFP4, rest BFP8")
    ax[2].set_xlabel("decoder layer")
    ax[2].set_ylabel("Δ PPL vs all-BFP8 (%)")
    for a in ax[:2]:
        a.tick_params(axis="x", rotation=30)
    for a in ax:
        a.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{name}: BFP4 sensitivity (CPU emulation, wikitext-2)", y=1.02)
    fig.tight_layout()
    out = ROOT / "figures" / f"ppl_sensitivity_{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out, dict(zip(groups, [round(d / saved_mb[k] * 10, 2) for d, k in zip(delta, groups)])))
    pareto(name)


def weight_mb(m, cfg):
    lp = layer_params(m)
    return (sum(n * BYTES[cfg.get(f"L{i}.{g}", cfg[g])] for i in range(m["L"]) for g, n in lp.items())
            + m["V"] * m["d"] * BYTES[cfg["lm_head"]]) / 1e6


def pareto(name):
    """Δ PPL vs weight bytes for every config run at the same token budget (all but the layer sweep)."""
    from ppl import parse_spec
    m = MODELS[name]
    pts = {}
    for f in (ROOT / "results").glob(f"ppl_{name}_*.md"):
        if f.stem.endswith("_layers"):
            continue
        r = rows(f)
        pts.update({k: (weight_mb(m, parse_spec(k)), 100 * (v / r[None] - 1)) for k, v in r.items() if k})
    pts.pop("bf16", None)
    pts.pop("all-bfp8+ff1_3=bfp4", None)   # same config as tt-perf
    b8 = pts["all-bfp8"][0]
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, (mb, d) in sorted(pts.items(), key=lambda kv: kv[1][0], reverse=True):
        default = k in ("tt-perf", "mlp-bfp4", "all-bfp4", "all-bfp8")
        ax.scatter(b8 - mb, d, s=60, color="#c05621" if default else "#2b6cb0", zorder=3)
        label = k.replace("all-bfp8+", "").replace("=bfp4", "").replace("+L0.ff1_3=bfp8", " protect L0/22/27") \
            .split("+L22")[0].replace("+L0.ff2=bfp8", "")
        ax.annotate(label, (b8 - mb, d), textcoords="offset points", xytext=(6, -3), fontsize=7)
    ax.set_xlabel("weight bytes saved vs all-BFP8 (MB)")
    ax.set_ylabel("Δ PPL vs fp32 (%)")
    ax.set_title(f"{name}: accuracy cost vs bytes saved (orange = stock configs)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"ppl_pareto_{name}.png", dpi=150)
    for k, (mb, d) in sorted(pts.items(), key=lambda kv: kv[1][0], reverse=True):
        print(f"{b8 - mb:7.1f} MB saved  {d:+6.2f}%  {k}")


def cross():
    """One group at BFP4 (rest BFP8), Δ PPL vs unquantized, for every model/corpus with a groups table."""
    groups = ["wqkv", "wo", "ff1_3", "ff2", "lm_head"]
    files = sorted((ROOT / "results").glob("ppl_*_groups*.md"))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    w = 0.8 / len(files)
    for i, f in enumerate(files):
        r = rows(f)
        label = f.stem[4:].replace("_groups", "").replace("-c4", " (C4)")
        ax.bar([x + (i - len(files) / 2 + 0.5) * w for x in range(len(groups))],
               [100 * (r[f"all-bfp8+{g}=bfp4"] / r[None] - 1) for g in groups], w, label=label)
    ax.set_xticks(range(len(groups)), groups)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Δ PPL vs unquantized (%)")
    ax.set_title("One tensor group -> BFP4 (rest BFP8), across models: WO is always cheapest, FF1/FF3 or LM head always worst")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "ppl_groups_cross_model.png", dpi=150)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-0.6B"
    cross() if arg == "all" else main(arg)
