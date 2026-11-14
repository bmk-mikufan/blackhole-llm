"""Smoke test for ppl.py on a tiny random Llama (no download): python tests/test_ppl.py"""
import sys, torch
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import LlamaConfig, LlamaForCausalLM
import ppl
from roofline import CONFIGS
torch.manual_seed(0)
m = LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)).eval()
ppl.untie_lm_head(m)
assert m.lm_head.weight is not m.model.embed_tokens.weight
ids = list(torch.randint(0, 256, (600,)).tolist())
ref = {k: v.clone() for k, v in m.state_dict().items()}
base, ref_top1 = ppl.evaluate(m, ids, 128)
assert len(ref_top1) == len(ids) - 1
out = {}
for name in CONFIGS:
    m.load_state_dict(ref); n = ppl.apply_config(m, CONFIGS[name]); out[name] = ppl.evaluate(m, ids, 128)[0]
    assert n == 2*7 + 1, n
emb_ok = torch.equal(m.model.embed_tokens.weight, ref["model.embed_tokens.weight"])
print(base, out, "embedding untouched:", emb_ok)
assert emb_ok and abs(out["bf16"]-base)/base < 0.01
ppl.CHUNK = 16  # exercise slicing on the tiny model (hidden 64)
c = ppl.parse_spec("all-bfp8+ff2=bfp4+L1.ff1_3=bf16")
assert c["ff2"] == "bfp4" and c["L1.ff1_3"] == "bf16" and c["ff1_3"] == "bfp8"
m.load_state_dict(ref); ppl.apply_config(m, c)
w1 = m.model.layers[1].mlp.gate_proj.weight; w0 = m.model.layers[0].mlp.gate_proj.weight
from bfp import quantize_linear_weight, to_bf16
assert torch.equal(w1, torch.from_numpy(to_bf16(ref["model.layers.1.mlp.gate_proj.weight"].numpy())))
assert torch.equal(w0, torch.from_numpy(quantize_linear_weight(ref["model.layers.0.mlp.gate_proj.weight"].numpy(), "bfp8")))
# bfp values survive a bf16 round trip
import numpy as np
q = quantize_linear_weight(np.random.default_rng(1).standard_normal((64, 64)).astype(np.float32), "bfp8")
assert np.array_equal(to_bf16(q), q)
print("spec/layer override OK")
