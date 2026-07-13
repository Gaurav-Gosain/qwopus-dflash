#!/usr/bin/env bash
# Qwopus3.5-9B-Coder + DFlash block-diffusion speculative decoding on RTX 3070 (8GB)
# Needs llama.cpp master >= d1b34251b (--spec-type draft-dflash) built with CUDA.
# Draft: z-lab/Qwen3.5-9B-DFlash converted against the Qwopus tokenizer (see convert-draft.sh)
#        The draft GGUF has no embeddings/lm_head; it shares the target's at runtime.
#
# Measured on 600-token coding generations, temp 0, target Q3_K_M fully on GPU:
#   no speculation:        38 tok/s
#   DFlash n-max 7:        81 tok/s (acceptance 0.56)
#   DFlash n-max 15:      127 tok/s (acceptance 0.34, mean draft len 6.1)
# Needs ~6.5 GB free VRAM. If other processes hold VRAM the server spills target
# layers to CPU and speculation becomes a net loss; free VRAM first.
set -euo pipefail

LCPP="${LCPP:-$HOME/dev/llama.cpp}"
MODELS="$LCPP/models/qwopus"
PORT="${PORT:-8080}"

# --- Mode: Q3_K_M (default) fits fully in 8GB alongside the draft ---
# --- Set Q4=1 for the higher-quality Q4_K_M target (some layers spill to CPU) ---
if [[ "${Q4:-0}" == "1" ]]; then
  MODEL="$MODELS/Qwopus3.5-9B-coder-Exp-Q4_K_M.gguf"
else
  MODEL="$MODELS/Qwopus3.5-9B-coder-Exp-Q3_K_M.gguf"
fi

DRAFT="$MODELS/Qwopus3.5-9B-Coder-DFlash-Q4_K_M.gguf"
CTX=4096

# No -ngl on purpose: the server fits layer offload to whatever VRAM is free.
# Qwen3.5 is a hybrid linear-attention arch and each context checkpoint stores
# the full recurrent state (~100 MB), so cap checkpoints.
# DFlash block size is 16, so n-max 15 drafts a full block per step.
exec "$LCPP/build/bin/llama-server" \
  -m "$MODEL" \
  -md "$DRAFT" \
  --spec-type draft-dflash --spec-draft-n-max 15 \
  -np 1 -ctxcp 2 \
  -fa on --jinja -c "$CTX" -b 512 -ub 256 -ctk q8_0 -ctv q8_0 \
  --host 127.0.0.1 --port "$PORT" "$@"
