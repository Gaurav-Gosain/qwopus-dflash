#!/usr/bin/env bash
# Convert z-lab/Qwen3.5-9B-DFlash into a DFlash draft GGUF for Qwopus3.5-9B-Coder.
#
# The draft checkpoint has no token embeddings or lm_head; llama.cpp shares the
# target model's at runtime. The conversion only needs the target's tokenizer,
# so the full Qwopus safetensors are never downloaded.
#
# Requires: llama.cpp checkout (master >= d1b34251b), python venv with
# requirements/requirements-convert_hf_to_gguf.txt installed, curl.
set -euo pipefail

LCPP="${LCPP:-$HOME/dev/llama.cpp}"
PY="${PY:-python3}"
WORK="${WORK:-$(mktemp -d)}"
OUT="${OUT:-$LCPP/models/qwopus}"

mkdir -p "$WORK/draft" "$WORK/target" "$OUT"

echo "downloading draft checkpoint (2.6G)"
curl -sL -C - -o "$WORK/draft/model.safetensors" \
  https://huggingface.co/z-lab/Qwen3.5-9B-DFlash/resolve/main/model.safetensors
curl -sL -o "$WORK/draft/config.json" \
  https://huggingface.co/z-lab/Qwen3.5-9B-DFlash/resolve/main/config.json

echo "downloading target tokenizer"
for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja; do
  curl -sL -o "$WORK/target/$f" \
    "https://huggingface.co/Jackrong/Qwopus3.5-9B-Coder/resolve/main/$f"
done

# Qwopus was saved with transformers v5, which writes a tokenizer_class that
# v4 cannot load. The tokenizer.json itself is a standard fast tokenizer.
$PY - "$WORK/target/tokenizer_config.json" <<'EOF'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
d["tokenizer_class"] = "PreTrainedTokenizerFast"
json.dump(d, open(p, "w"), indent=2)
EOF

echo "converting"
$PY "$LCPP/convert_hf_to_gguf.py" "$WORK/draft" \
  --target-model-dir "$WORK/target" \
  --outtype bf16 \
  --outfile "$WORK/Qwopus3.5-9B-Coder-DFlash-bf16.gguf"

echo "quantizing"
"$LCPP/build/bin/llama-quantize" \
  "$WORK/Qwopus3.5-9B-Coder-DFlash-bf16.gguf" \
  "$OUT/Qwopus3.5-9B-Coder-DFlash-Q4_K_M.gguf" Q4_K_M
"$LCPP/build/bin/llama-quantize" \
  "$WORK/Qwopus3.5-9B-Coder-DFlash-bf16.gguf" \
  "$OUT/Qwopus3.5-9B-Coder-DFlash-Q8_0.gguf" Q8_0

echo "done: $OUT"
