#!/usr/bin/env bash
# Phase 1: reproduce the published Llama 3.1 8B p150 number (33.1 t/s/u @ batch 32, ISL 128)
# with the tt-metal TT-Transformers v1 demo. Run inside a tt-metal checkout with the venv active.
# Reference release: v0.64.0-dev20251030 (tt-metal models/README.md).
set -euo pipefail
export HF_MODEL=${HF_MODEL:-meta-llama/Llama-3.1-8B-Instruct}
export MESH_DEVICE=${MESH_DEVICE:-P150}
OUT=${OUT:-$(dirname "$0")/logs/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
tt-smi -s > "$OUT/tt-smi.json" || true          # board info, firmware, power
git -C "${TT_METAL_HOME:?set TT_METAL_HOME}" rev-parse HEAD > "$OUT/tt-metal.commit"

for k in "performance and batch-1" "performance and batch-32" "accuracy and batch-1" "accuracy and batch-32"; do
  pytest "$TT_METAL_HOME/models/ttt_compat/demo/simple_text_demo.py" -k "$k" 2>&1 | tee "$OUT/${k// /_}.log"
done
