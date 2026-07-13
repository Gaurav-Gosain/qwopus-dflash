# qwopus-dflash

Up to 4.9x faster code generation from [Qwopus3.5-9B-Coder](https://huggingface.co/Jackrong/Qwopus3.5-9B-Coder) on an RTX 3070, same outputs. DFlash block-diffusion speculative decoding on upstream llama.cpp.

![side by side: baseline vs dflash applying the same edit to a Go file](demo.gif)

62 to 304 tok/s renaming a struct field across a Go file. Both panes are real captured token streams: same model, same prompt, temperature 0. The right pane just has a DFlash draft attached.

**Draft GGUFs: [GauravGosain/Qwopus3.5-9B-Coder-DFlash-GGUF](https://huggingface.co/GauravGosain/Qwopus3.5-9B-Coder-DFlash-GGUF)**

## Quick start

```sh
llama-server \
  -m Qwopus3.5-9B-coder-Exp-Q3_K_M.gguf \
  -md Qwopus3.5-9B-Coder-DFlash-Q4_K_M.gguf \
  --spec-type draft-dflash --spec-draft-n-max 15 \
  -fa on --jinja -ctxcp 2 -fitt 256
```

Needs llama.cpp master (DFlash landed in [#22105](https://github.com/ggml-org/llama.cpp/pull/22105)). Or clone this repo: `./convert-draft.sh` builds the draft from scratch, `./run.sh` serves it.

## Why this repo

No DFlash draft exists for Qwopus, only for its base model ([z-lab/Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash)). The published GGUF conversions of that draft are not a drop-in fit:

- Qwopus extends the Qwen3.5 tokenizer with 7 extra tokens (ids 248070 to 248076), so drafts converted against the base tokenizer fail the vocab compatibility check or mistokenize.
- Some published conversions target a llama.cpp fork rather than upstream.

The fix is cheap because of how DFlash works in llama.cpp: the draft GGUF carries no token embeddings and no lm_head, it borrows the target model's copies at runtime. Converting a Qwopus-matched draft therefore needs only the Qwopus tokenizer, not its 18 GB of weights, and the resulting draft matches the target embeddings exactly by construction.

## Benchmarks

RTX 3070 8 GB, target Qwopus3.5-9B-Coder Q3_K_M, draft Q4_K_M, 600-token coding generation at temperature 0, llama.cpp master (4193ea697), measured back to back with both configurations fully on GPU.

| workload | baseline | DFlash | speedup | acceptance |
| --- | --- | --- | --- | --- |
| code editing (rename a field, echo the file) | 62 tok/s | 304 tok/s | 4.9x | 0.84 |
| fresh code generation | 58 tok/s | 145 tok/s | 2.5x | 0.34 |

The speedup tracks how predictable the output is. Editing existing code is DFlash's best case: mean draft length hits 13.7 of 15, so most blocks verify in one pass. Fresh generation still runs 2.5x. Freeform prose drops to about 0.15 acceptance, still a net win. Shorter blocks raise acceptance but lower throughput (n-max 7: 0.56 acceptance, slower overall). Draft quantization barely matters (Q8_0 within noise of Q4_K_M) so the smaller Q4_K_M is the default.

Caveat: this needs about 6.5 GB of free VRAM and a low `--fit-target` margin (run.sh uses `-fitt 256`; the default 1024 reserves too much and spills layers). If other processes hold VRAM, llama-server spills target layers to CPU and speculation goes net-negative: measured 28 tok/s against a 58 tok/s baseline with the GPU shared. Do not pin `-ngl`; the automatic fit degrades gracefully instead of failing to start.

## Modes

`Q4=1 ./run.sh` serves the Q4_K_M target instead of Q3_K_M (better quality, some CPU spill on 8 GB).

## Notes and gotchas

- Qwopus ships a transformers v5 `tokenizer_config.json` (`tokenizer_class: TokenizersBackend`) that transformers v4 refuses to load. The conversion script rewrites it to `PreTrainedTokenizerFast`; the tokenizer.json itself is a standard fast tokenizer.
- Qwen3.5 is a hybrid linear-attention architecture. Every context checkpoint stores the full recurrent state (roughly 100 MB), and llama-server defaults to 32 checkpoints per slot, which alone is 3.2 GB. `run.sh` caps this with `-ctxcp 2`.
- The DFlash mask token id is 248077, which sits just past Qwopus's last added token (248076). No collision, but worth knowing if you retarget another finetune.
- `--spec-draft-n-max` is clamped to block_size - 1 = 15. Long blocks win on code even though per-token acceptance drops, because each verify pass lands about 6 tokens.
