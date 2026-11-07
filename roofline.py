"""Pre-silicon analytical model for LLM decode on a single Tenstorrent Blackhole p150a.

Answers three questions before we touch the card:
  1. Which open-weight models fit in 32 GB GDDR6, at which weight format, with how much KV cache left?
  2. What is the memory-bandwidth ceiling on decode tokens/s/user, and how far is the
     published Llama 3.1 8B p150 number from it (i.e. how much headroom Phase 2 can chase)?
  3. Can a decoder layer's weights live in the 120 x 1.5 MB Tensix L1 at all?

Run:  python roofline.py        -> writes results/*.md and figures/*.png
"""
from pathlib import Path

# ---- hardware (p150a spec sheet) -------------------------------------------------
DRAM_BYTES = 32 * 2**30          # 32 GB GDDR6
DRAM_BW = 512e9                  # 512 GB/s peak
L1_BYTES = 120 * 1.5 * 2**20     # 120 Tensix cores x 1.5 MB
# ponytail: flat 2 GiB reserve for trace region, program binaries, activations and
# allocator fragmentation; replace with the real number from the allocator once on hardware.
DRAM_RESERVE = 2 * 2**30

# ---- Tenstorrent tile formats: bytes per element ---------------------------------
# A 32x32 tile = 4 faces of 16x16; one shared 8-bit exponent per 16-element face row.
# bfp8_b: 1024 B mantissa+sign + 64 B exponents = 1088 B / 1024 elem.  bfp4_b: 512 + 64 = 576 B.
BYTES = {"bf16": 2.0, "bfp8": 1088 / 1024, "bfp4": 576 / 1024}

# ---- weight-format configurations (tensor groups as in tt_transformers model_config.py) ---
CONFIGS = {
    "bf16":        dict(wqkv="bf16", wo="bf16", ff1_3="bf16", ff2="bf16", lm_head="bf16", kv="bf16"),
    "all-bfp8":    dict(wqkv="bfp8", wo="bfp8", ff1_3="bfp8", ff2="bfp8", lm_head="bfp8", kv="bfp8"),
    # tt_transformers ModelOptimizations.performance: BFP4 FF1/FF3, BFP8 elsewhere.
    # lm_head precision is our assumption (BFP8); verify on hardware.
    "tt-perf":     dict(wqkv="bfp8", wo="bfp8", ff1_3="bfp4", ff2="bfp8", lm_head="bfp8", kv="bfp8"),
    "mlp-bfp4":    dict(wqkv="bfp8", wo="bfp8", ff1_3="bfp4", ff2="bfp4", lm_head="bfp8", kv="bfp8"),
    "all-bfp4":    dict(wqkv="bfp4", wo="bfp4", ff1_3="bfp4", ff2="bfp4", lm_head="bfp4", kv="bfp8"),
}

# ---- model architectures (from each model's HF config.json) ------------------------
# d=hidden, L=layers, H=q heads, KV=kv heads, hd=head dim, I=MLP intermediate, V=vocab
MODELS = {
    "Llama-3.2-1B":      dict(d=2048, L=16, H=32, KV=8,  hd=64,  I=8192,  V=128256),
    "Llama-3.2-3B":      dict(d=3072, L=28, H=24, KV=8,  hd=128, I=8192,  V=128256),
    "Llama-3.1-8B":      dict(d=4096, L=32, H=32, KV=8,  hd=128, I=14336, V=128256),
    "Qwen3-0.6B":        dict(d=1024, L=28, H=16, KV=8,  hd=128, I=3072,  V=151936),
    "Qwen3-1.7B":        dict(d=2048, L=28, H=16, KV=8,  hd=128, I=6144,  V=151936),
    "Qwen3-4B":          dict(d=2560, L=36, H=32, KV=8,  hd=128, I=9728,  V=151936),
    "Qwen3-8B":          dict(d=4096, L=36, H=32, KV=8,  hd=128, I=12288, V=151936),
    "Qwen3-14B":         dict(d=5120, L=40, H=40, KV=8,  hd=128, I=17408, V=151936),
    "Qwen3-32B":         dict(d=5120, L=64, H=64, KV=8,  hd=128, I=25600, V=151936),
    # Gemma 3: text decoder only; KV counted as if every layer were global (upper bound,
    # 5 of every 6 layers are actually sliding-window).
    "Gemma-3-1B":        dict(d=1152, L=26, H=4,  KV=1,  hd=256, I=6912,  V=262144),
    "Gemma-3-4B":        dict(d=2560, L=34, H=8,  KV=4,  hd=256, I=10240, V=262208),
    "Gemma-3-12B":       dict(d=3840, L=48, H=16, KV=8,  hd=256, I=15360, V=262208),
    "Gemma-3-27B":       dict(d=5376, L=62, H=32, KV=16, hd=128, I=21504, V=262208),
    "Mistral-7B-v0.3":   dict(d=4096, L=32, H=32, KV=8,  hd=128, I=14336, V=32768),
    "Mistral-Small-24B": dict(d=5120, L=40, H=32, KV=8,  hd=128, I=32768, V=131072),
    "Llama-3.3-70B":     dict(d=8192, L=80, H=64, KV=8,  hd=128, I=28672, V=128256),
}

# Published reference point (tt-metal models/README.md, v0.64.0-dev20251030, ISL 128):
# Llama 3.1 8B, p150, batch 32 -> 33.1 t/s/u, 1059.2 t/s, TTFT 57 ms.
PUBLISHED = dict(model="Llama-3.1-8B", config="tt-perf", batch=32, tsu=33.1, ctx=128 + 128)
# ponytail: average decode context assumed = 128 prompt + ~128 generated; KV is ~10% of
# traffic at this point so the error is small. Use the demo's real generated length later.


def layer_params(m):
    """Parameter counts of one decoder layer, per tensor group (norms ignored: <0.01%)."""
    return dict(
        wqkv=m["d"] * (m["H"] + 2 * m["KV"]) * m["hd"],
        wo=m["H"] * m["hd"] * m["d"],
        ff1_3=2 * m["d"] * m["I"],
        ff2=m["I"] * m["d"],
    )


def params(m):
    return m["L"] * sum(layer_params(m).values()) + m["V"] * m["d"]  # + lm_head/embedding (one copy)


def layer_bytes(m, cfg):
    return sum(n * BYTES[cfg[g]] for g, n in layer_params(m).items())


def weight_read_bytes(m, cfg):
    """Bytes streamed from DRAM per decode step (all layers + LM head; embedding lookup is one row)."""
    return m["L"] * layer_bytes(m, cfg) + m["V"] * m["d"] * BYTES[cfg["lm_head"]]


def footprint(m, cfg):
    """Resident weight bytes: read set + a bf16 embedding table (tt keeps it separate from lm_head)."""
    return weight_read_bytes(m, cfg) + m["V"] * m["d"] * BYTES["bf16"]


def kv_bytes_per_token(m, cfg):
    return 2 * m["L"] * m["KV"] * m["hd"] * BYTES[cfg["kv"]]


def step_bytes(m, cfg, batch, ctx):
    return weight_read_bytes(m, cfg) + batch * ctx * kv_bytes_per_token(m, cfg)


def decode_tsu(m, cfg, batch, ctx, eff=1.0):
    """Memory-bound decode ceiling in tokens/s/user at a given DRAM-bandwidth efficiency."""
    return eff * DRAM_BW / step_bytes(m, cfg, batch, ctx)


def implied_efficiency():
    p = PUBLISHED
    m, cfg = MODELS[p["model"]], CONFIGS[p["config"]]
    return p["tsu"] / decode_tsu(m, cfg, p["batch"], p["ctx"])


G = 1e9


def fit_table():
    rows = ["| Model | Params (B) | " + " | ".join(f"{c} GB" for c in CONFIGS) + " | KV/token (KB, bfp8) | Max KV tokens @ tt-perf | Users @ 4k ctx |",
            "|---|---:|" + "---:|" * len(CONFIGS) + "---:|---:|---:|"]
    usable = DRAM_BYTES - DRAM_RESERVE
    for name, m in MODELS.items():
        cells = []
        for c in CONFIGS.values():
            fp = footprint(m, c)
            cells.append(f"{fp / G:.1f}" if fp < usable else f"~~{fp / G:.1f}~~")
        perf = CONFIGS["tt-perf"]
        free = usable - footprint(m, perf)
        kvt = kv_bytes_per_token(m, perf)
        tokens = max(0, free // kvt)
        rows.append(f"| {name} | {params(m) / G:.2f} | " + " | ".join(cells)
                    + f" | {kvt / 1024:.0f} | {int(tokens):,} | {int(tokens // 4096)} |")
    return "\n".join(rows)


def ceiling_table(ctx=1024):
    eff = implied_efficiency()
    rows = [f"Context length per user = {ctx} tokens. 'eff' column applies the B=32 calibration to B=1 (optimistic: "
            "B=1 usually reaches lower bandwidth efficiency).\n",
            f"| Model | Config | Weights read/step (GB) | B=1 ceiling t/s/u | B=1 @ {eff:.0%} eff | B=32 ceiling t/s/u | B=32 ceiling t/s |",
            "|---|---|---:|---:|---:|---:|---:|"]
    usable = DRAM_BYTES - DRAM_RESERVE
    for name in ["Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B", "Qwen3-8B", "Qwen3-14B", "Mistral-7B-v0.3", "Gemma-3-12B", "Qwen3-32B"]:
        m = MODELS[name]
        for cname in ["all-bfp8", "tt-perf", "mlp-bfp4"]:
            c = CONFIGS[cname]
            if footprint(m, c) + 32 * ctx * kv_bytes_per_token(m, c) > usable:
                rows.append(f"| {name} | {cname} | {weight_read_bytes(m, c) / G:.2f} | — | — | does not fit | — |")
                continue
            b1, b32 = decode_tsu(m, c, 1, ctx), decode_tsu(m, c, 32, ctx)
            rows.append(f"| {name} | {cname} | {weight_read_bytes(m, c) / G:.2f} | {b1:.0f} | {b1 * eff:.0f} | {b32:.0f} | {b32 * 32:.0f} |")
    return "\n".join(rows)


def l1_table():
    rows = ["| Model | 1 layer bf16 (MB) | 1 layer bfp8 (MB) | 1 layer bfp4 (MB) | Layers in L1 @ bfp4 | KV/user @ 1k ctx bfp8 (MB) |",
            "|---|---:|---:|---:|---:|---:|"]
    MB = 2**20
    for name, m in MODELS.items():
        lb = {f: sum(layer_params(m).values()) * BYTES[f] for f in BYTES}
        rows.append(f"| {name} | {lb['bf16'] / MB:.0f} | {lb['bfp8'] / MB:.0f} | {lb['bfp4'] / MB:.0f} | "
                    f"{L1_BYTES / lb['bfp4']:.2f} | {1024 * kv_bytes_per_token(m, CONFIGS['all-bfp8']) / MB:.1f} |")
    return "\n".join(rows)


def plots(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eff = implied_efficiency()
    m = MODELS["Llama-3.1-8B"]
    batches = [1, 2, 4, 8, 16, 32, 64]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for cname, color in [("bf16", "#888"), ("all-bfp8", "#2b6cb0"), ("tt-perf", "#c05621"), ("mlp-bfp4", "#2f855a")]:
        c = CONFIGS[cname]
        tsu = [decode_tsu(m, c, b, PUBLISHED["ctx"]) for b in batches]
        ax[0].plot(batches, tsu, "o-", color=color, label=f"{cname} ceiling")
        ax[1].plot(batches, [t * b for t, b in zip(tsu, batches)], "o-", color=color, label=cname)
    c = CONFIGS["tt-perf"]
    ax[0].plot(batches, [decode_tsu(m, c, b, PUBLISHED["ctx"], eff) for b in batches], "--", color="#c05621",
               label=f"tt-perf @ {eff:.0%} BW (calibrated)")
    ax[0].plot([32], [PUBLISHED["tsu"]], "k*", ms=14, label="published p150 (33.1)")
    ax[1].plot([32], [PUBLISHED["tsu"] * 32], "k*", ms=14, label="published p150 (1059)")
    for a in ax:
        a.set_xscale("log", base=2)
        a.set_xlabel("batch size (users)")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("tokens/s/user")
    ax[1].set_ylabel("total tokens/s")
    ax[0].set_title("Llama 3.1 8B decode ceiling on p150a (512 GB/s, ctx 256)")
    ax[1].set_title("Total throughput ceiling")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "llama8b_decode_ceiling.png", dpi=150)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    names = list(MODELS)
    width = 0.27
    for i, (cname, color) in enumerate([("bf16", "#888"), ("all-bfp8", "#2b6cb0"), ("tt-perf", "#c05621")]):
        ax.bar([x + (i - 1) * width for x in range(len(names))],
               [footprint(MODELS[n], CONFIGS[cname]) / G for n in names], width, color=color, label=cname)
    ax.axhline(DRAM_BYTES / G, color="k", lw=1, label="32 GiB GDDR6 (34.4 GB)")
    ax.axhline((DRAM_BYTES - DRAM_RESERVE) / G, color="k", ls="--", lw=1, label="usable (2 GiB reserve)")
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("resident weights (GB)")
    ax.set_ylim(0, 80)
    ax.set_title("Which models fit on one p150a?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "model_fit.png", dpi=150)


def main():
    root = Path(__file__).parent
    (root / "results").mkdir(exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)
    eff = implied_efficiency()
    p = PUBLISHED
    m, cfg = MODELS[p["model"]], CONFIGS[p["config"]]
    sb = step_bytes(m, cfg, p["batch"], p["ctx"])
    summary = (f"Published Llama 3.1 8B p150 @ B=32: {p['tsu']} t/s/u -> step {1e3 / p['tsu']:.1f} ms, "
               f"{sb / G:.2f} GB moved/step (tt-perf), implied DRAM BW {sb * p['tsu'] / G:.0f} GB/s "
               f"= {eff:.0%} of 512 GB/s; ceiling {decode_tsu(m, cfg, p['batch'], p['ctx']):.1f} t/s/u.")
    for f in ["all-bfp8", "bf16"]:
        e = p["tsu"] / decode_tsu(m, CONFIGS[f], p["batch"], p["ctx"])
        summary += f"\n(if weights were {f}: implied {e:.0%} of peak)"
    for name, text in [("fit", fit_table()), ("decode_ceiling", summary + "\n\n" + ceiling_table()), ("l1", l1_table())]:
        (root / "results" / f"{name}.md").write_text(text + "\n", encoding="utf-8")
    plots(root / "figures")
    print(summary)


def _check():
    m = MODELS["Llama-3.1-8B"]
    assert abs(params(m) / G - 7.50) < 0.02, params(m)        # 8.03B incl. separate embedding
    assert abs(BYTES["bfp8"] - 1.0625) < 1e-12 and abs(BYTES["bfp4"] - 0.5625) < 1e-12
    c = CONFIGS["tt-perf"]
    assert decode_tsu(m, c, 1, 128) > decode_tsu(m, c, 32, 128) > decode_tsu(m, c, 32, 4096)
    assert 0 < implied_efficiency() < 1


if __name__ == "__main__":
    _check()
    main()
