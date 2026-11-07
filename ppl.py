"""Perplexity of a HF causal LM with weights fake-quantized to TT formats per tensor group.

Emulates the *weight* side of tt_transformers precision configs on CPU, so we can rank
BF16 / BFP8 / BFP4 mixes by accuracy before running them on the p150a. Not emulated: KV-cache
format, activation formats, math fidelity (LoFi/HiFi) -- those need the real hardware.

    python ppl.py --model Qwen/Qwen3-0.6B --text wikitext2_test.txt
"""
import argparse
from pathlib import Path

import torch
from bfp import quantize_linear_weight
from roofline import CONFIGS

GROUP = {"q_proj": "wqkv", "k_proj": "wqkv", "v_proj": "wqkv", "o_proj": "wo",
         "gate_proj": "ff1_3", "up_proj": "ff1_3", "down_proj": "ff2", "lm_head": "lm_head"}


def untie_lm_head(model):
    head, emb = model.get_output_embeddings(), model.get_input_embeddings()
    if head.weight is emb.weight:   # tt stores lm_head separately; don't quantize the embedding with it
        head.weight = torch.nn.Parameter(emb.weight.detach().clone())


def apply_config(model, cfg):
    n = 0
    for name, mod in model.named_modules():
        g = GROUP.get(name.rsplit(".", 1)[-1])
        if g and isinstance(mod, torch.nn.Linear):
            w = mod.weight.data.float().numpy()
            mod.weight.data = torch.from_numpy(quantize_linear_weight(w, cfg[g])).to(mod.weight.dtype)
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
    ap.add_argument("--configs", default=",".join(CONFIGS))
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(Path(a.text).read_text(encoding="utf-8")).input_ids[:a.max_tokens]
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    untie_lm_head(model)
    ref = {k: v.clone() for k, v in model.state_dict().items()}

    rows = [f"Model `{a.model}`, {len(ids):,} tokens of `{Path(a.text).name}`, ctx {a.ctx}. Weights only (KV/activations fp32).\n",
            "| Config | PPL | Δ vs fp32 |", "|---|---:|---:|"]
    base = perplexity(model, ids, a.ctx)
    rows.append(f"| fp32 | {base:.3f} | — |")
    print(rows[-1], flush=True)
    for name in a.configs.split(","):
        model.load_state_dict(ref)
        apply_config(model, CONFIGS[name])
        p = perplexity(model, ids, a.ctx)
        rows.append(f"| {name} | {p:.3f} | {100 * (p / base - 1):+.2f}% |")
        print(rows[-1], flush=True)
    out = Path(__file__).parent / "results" / f"ppl_{a.model.split('/')[-1]}.md"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
