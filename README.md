# qwopus-dflash

DFlash speculative decoding for [Qwopus3.5-9B-Coder](https://huggingface.co/Jackrong/Qwopus3.5-9B-Coder) on upstream llama.cpp. 3.3x faster code generation on an RTX 3070 (8 GB).

[DFlash](https://github.com/z-lab/dflash) is block-diffusion drafting: the draft model proposes an entire block of tokens in one forward pass, conditioned on hidden states extracted from the target model. Support landed in llama.cpp master in [#22105](https://github.com/ggml-org/llama.cpp/pull/22105) as `--spec-type draft-dflash`.

## Why this repo

No DFlash draft exists for Qwopus, only for its base model ([z-lab/Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash)). The published GGUF conversions of that draft are not a drop-in fit:

- Qwopus extends the Qwen3.5 tokenizer with 7 extra tokens (ids 248070 to 248076), so drafts converted against the base tokenizer fail the vocab compatibility check or mistokenize.
- Some published conversions target a llama.cpp fork rather than upstream.

The fix is cheap because of how DFlash works in llama.cpp: the draft GGUF carries no token embeddings and no lm_head, it borrows the target model's copies at runtime. Converting a Qwopus-matched draft therefore needs only the Qwopus tokenizer, not its 18 GB of weights, and the resulting draft matches the target embeddings exactly by construction.

## Benchmarks

RTX 3070 8 GB, target Qwopus3.5-9B-Coder Q3_K_M fully on GPU, draft Q4_K_M, 600-token coding generation at temperature 0, llama.cpp master (4193ea697).

| config | speed | acceptance |
| --- | --- | --- |
| no speculation | 38 tok/s | - |
| DFlash, n-max 7 | 81 tok/s | 0.56 |
| DFlash, n-max 15 | 127 tok/s | 0.34 |

Draft quantization barely matters (Q8_0 draft: 128 tok/s at 0.35 acceptance) so the smaller Q4_K_M is the default. Acceptance falls to about 0.15 on freeform prose, still a net win at roughly 1.8x.

Caveat: this needs about 6.5 GB of free VRAM. If other processes hold VRAM, llama-server spills target layers to CPU and speculation becomes a net loss (measured 12 tok/s against a 38 tok/s baseline with 1.7 GB stolen by another process). Do not pin `-ngl`; letting the server fit offload automatically degrades gracefully.

## Usage

Build llama.cpp master with CUDA, then:

```sh
./convert-draft.sh   # downloads draft + Qwopus tokenizer, converts, quantizes
./run.sh             # llama-server on :8080 with draft-dflash
```

`Q4=1 ./run.sh` serves the Q4_K_M target instead of Q3_K_M (better quality, some CPU spill on 8 GB).

## Notes and gotchas

- Qwopus ships a transformers v5 `tokenizer_config.json` (`tokenizer_class: TokenizersBackend`) that transformers v4 refuses to load. The conversion script rewrites it to `PreTrainedTokenizerFast`; the tokenizer.json itself is a standard fast tokenizer.
- Qwen3.5 is a hybrid linear-attention architecture. Every context checkpoint stores the full recurrent state (roughly 100 MB), and llama-server defaults to 32 checkpoints per slot, which alone is 3.2 GB. `run.sh` caps this with `-ctxcp 2`.
- The DFlash mask token id is 248077, which sits just past Qwopus's last added token (248076). No collision, but worth knowing if you retarget another finetune.
- `--spec-draft-n-max` is clamped to block_size - 1 = 15. Long blocks win on code even though per-token acceptance drops, because each verify pass lands about 6 tokens.
