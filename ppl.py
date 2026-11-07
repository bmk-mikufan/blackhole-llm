"""Perplexity of a HF causal LM with weights fake-quantized to TT formats per tensor group / per layer.

Emulates the *weight* side of tt_transformers precision configs on CPU, so we can rank
BF16 / BFP8 / BFP4 mixes by accuracy before running them on the p150a. Not emulated: KV-cache
format, activation formats, math fidelity (LoFi/HiFi) -- those need the real hardware.

Config specs: a base from roofline.CONFIGS plus '+key=fmt' overrides, where key is a tensor
group (wqkv, wo, ff1_3, ff2, lm_head) or a layer-scoped group 'L<i>.<group>':
    all-bfp8+ff2=bfp4            all-bfp8+L3.ff1_3=bfp4+L3.ff2=bfp4

    python ppl.py --model Qwen/Qwen3-0.6B --text data/wikitext2_test.txt
    python ppl.py --model Qwen/Qwen3-0.6B --text data/wikitext2_test.txt --layer-sweep ff1_3,ff2=bfp4 --tag layers
"""
import argparse
import re
from pathlib import Path

import torch
from bfp import quantize_linear_weight
from roofline import BYTES, CONFIGS

GROUP = {"q_proj": "wqkv", "k_proj": "wqkv", "v_proj": "wqkv", "o_proj": "wo",
         "gate_proj": "ff1_3", "up_proj": "ff1_3", "down_proj": "ff2", "lm_head": "lm_head"}


def parse_spec(spec):
    base, *overrides = spec.split("+")
    cfg = dict(CONFIGS[base])
    for o in overrides:
        key, fmt = o.split("=")
        assert fmt in BYTES and key.split(".")[-1] in cfg, f"bad override {o!r}"
        cfg[key] = fmt
    return cfg


def untie_lm_head(model):
    head, emb = model.get_output_embeddings(), model.get_input_embeddings()
    if head.weight is emb.weight:   # tt stores lm_head separately; don't quantize the embedding with it
        head.weight = torch.nn.Parameter(emb.weight.detach().clone())


def apply_config(model, cfg):
    n = 0
    for name, mod in model.named_modules():
        g = GROUP.get(name.rsplit(".", 1)[-1])
        if g and isinstance(mod, torch.nn.Linear):
            layer = re.search(r"layers\.(\d+)\.", name)
            fmt = cfg.get(f"L{layer.group(1)}.{g}", cfg[g]) if layer else cfg[g]
            w = mod.weight.data.float().numpy()
            # BFP8/BFP4 values have <= 7 significant bits, so storing them back in bf16 is lossless
            mod.weight.data = torch.from_numpy(quantize_linear_weight(w, fmt)).to(mod.weight.dtype)
            n += 1
    return n


@torch.no_grad()
def perplexity(model, ids, ctx):
    nll, count = 0.0, 0
    for i in range(0, len(ids) - 1, ctx):
        chunk = ids[i:i + ctx + 1]
        if len(chunk) < 2:
            break
        x, y = torch.tensor([chunk[:-1]]), torch.tensor([chunk[1:]])
        logits = model(x).logits.float()
        nll += torch.nn.functional.cross_entropy(logits[0], y[0], reduction="sum").item()
        count += y.numel()
    return float(torch.exp(torch.tensor(nll / count)))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True, help="plain-text eval corpus, e.g. wikitext-2 test split")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--configs", default=",".join(CONFIGS), help="comma-separated config specs")
    ap.add_argument("--layer-sweep", help="GROUPS=FMT: for each layer i, base + L<i>.<group>=FMT")
    ap.add_argument("--sweep-base", default="all-bfp8")
    ap.add_argument("--tag", default="formats")
    a = ap.parse_args()
    import sys
    sys.stdout.reconfigure(encoding="utf-8")   # Korean Windows consoles default to cp949

    dtype = getattr(torch, a.dtype)
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(Path(a.text).read_text(encoding="utf-8")).input_ids[:a.max_tokens]

    def load():   # reload instead of keeping a reference copy: halves peak RAM
        m = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).eval()
        untie_lm_head(m)
        return m

    specs = a.configs.split(",")
    if a.layer_sweep:
        groups, fmt = a.layer_sweep.split("=")
        n_layers = load().config.num_hidden_layers
        specs = [a.sweep_base] + [a.sweep_base + "".join(f"+L{i}.{g}={fmt}" for g in groups.split(","))
                                  for i in range(n_layers)]

    rows = [f"Model `{a.model}` ({a.dtype} compute), {len(ids):,} tokens of `{Path(a.text).name}`, ctx {a.ctx}. "
            "Weights only (KV cache / activations not quantized).\n",
            "| Config | PPL | Δ vs unquantized |", "|---|---:|---:|"]
    base = perplexity(load(), ids, a.ctx)
    rows.append(f"| unquantized ({a.dtype}) | {base:.3f} | — |")
    print(rows[-1], flush=True)
    for spec in specs:
        model = load()
        apply_config(model, parse_spec(spec))
        p = perplexity(model, ids, a.ctx)
        rows.append(f"| `{spec}` | {p:.3f} | {100 * (p / base - 1):+.2f}% |")
        print(rows[-1], flush=True)
        del model
    out = Path(__file__).parent / "results" / f"ppl_{a.model.split('/')[-1]}_{a.tag}.md"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
