#!/usr/bin/env bash
# Qwopus3.5-9B-Coder + DFlash on Apple Silicon via dflash-mlx.
# Measured on an M3 Pro 18GB: code edit 28 -> 55 tok/s (1.95x), codegen 28 -> 42 tok/s (1.49x).
# The 4-bit target decodes at the memory-bandwidth roofline (~29 tok/s on 150 GB/s);
# speculative verification is the only way past it.
#
# Setup (once):
#   uv venv ~/dflash-mlx-env --python 3.12
#   uv pip install --python ~/dflash-mlx-env/bin/python "dflash-mlx[bench]"
#   ~/dflash-mlx-env/bin/python patch-draft-config.py
# The patch script downloads z-lab/Qwen3.5-9B-DFlash and lifts rope_theta and
# block_size out of the nested transformers-v5 config sections that dflash-mlx
# does not read yet.
set -euo pipefail

PORT="${PORT:-8098}"
exec ~/dflash-mlx-env/bin/dflash serve \
  --model snsnc/Qwopus3.5-9B-Coder-MLX-4bit \
  --draft ~/models/qwen35-9b-dflash \
  --draft-quant w4 \
  --block-tokens 8 \
  --port "$PORT"
