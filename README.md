# GPT From Scratch

A small GPT-style decoder-only transformer, implemented from first principles
in PyTorch — no `nn.Transformer`, no pretrained weights. The goal isn't a
strong model; it's to actually build and understand every piece of the
architecture that sits underneath the LLM-based tools I use daily (agents,
MCP servers, etc.), rather than only ever calling one through an API.

Trained on the classic **tiny-Shakespeare** character-level dataset (the same
corpus the original nanoGPT tutorial uses) — small enough to train a
meaningful model in a few minutes on CPU, since this machine has an AMD GPU
(no CUDA).

## Architecture

Everything in `src/model.py` is built manually:

- **Tokenizer** — character-level by default (`src/data.py`, a fixed
  char-to-int vocabulary built from the training corpus), or a real
  from-scratch byte-level BPE tokenizer (`src/bpe.py`), selectable via
  `TOKENIZER_TYPE`.
- **Token + positional embeddings** — learned embedding tables, not
  sinusoidal.
- **Causal multi-head self-attention** (`CausalSelfAttention`) — scaled
  dot-product attention with a lower-triangular mask so each position can
  only attend to itself and earlier positions (no peeking at the future).
- **Position-wise MLP** — the standard 4x-expansion feedforward block with
  GELU.
- **Transformer block** — pre-norm residual connections around attention and
  MLP (`x = x + attn(ln(x))`, `x = x + mlp(ln(x))`).
- **Autoregressive sampling** (`GPT.generate`) — temperature and top-k
  sampling, one token at a time, feeding each generated token back in as
  context. KV-cached by default (`use_cache=True`) so each new token only
  costs a forward pass over itself, not the whole context so far — see
  `PROJECT_PLAN.md` Chunk 3 for the correctness proof and measured speedup.

## Quick Start

```powershell
# Set up the environment (already done if you're reading this after setup)
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# Train (downloads tiny-Shakespeare automatically on first run, ~3-5 min on CPU)
.\venv\Scripts\python.exe src\train.py

# Generate text from the trained checkpoint (one-shot)
.\venv\Scripts\python.exe src\generate.py --prompt "ROMEO:" --max_new_tokens 300

# Or stay in a loop and type prompts interactively
.\venv\Scripts\python.exe src\generate.py --interactive --checkpoint gpt_stage1c_longer_scheduled.pt
```

There's no web or GUI interface — `--interactive` is a terminal loop: type a prompt,
press Enter, read the completion, repeat. Anything you type that isn't one of the 65
characters the model was trained on (it's a closed character-level vocab, not a full
Unicode-aware tokenizer) gets dropped with a warning rather than crashing.

## Default model size

~800K parameters (4 layers, 4 heads, 128-dim embeddings, 128-token context) —
deliberately small so a full training run finishes in minutes on CPU. Not
tuned for quality; tuned for "runs end-to-end and you can see it learn."

## What this deliberately doesn't do

- No distributed/multi-GPU training or mixed precision — single CPU
  process, kept simple on purpose.
- No relative/rotary position encoding (RoPE, ALiBi) — learned absolute
  position embeddings only, which is why KV-cached generation past
  `block_size` degrades in quality (see `PROJECT_PLAN.md` Chunk 3).
- No pretrained weights loaded from anywhere — every parameter starts from
  random initialization and is learned from this run.

## Status

**Phase 1 (training-mechanics experiments) complete as of 2026-07-22** — see
`PROJECT_PLAN.md` for the full stage-by-stage writeup. Four checkpoints were
trained and compared on both val loss and sampled output quality:

| Checkpoint | Params | Iters | Val loss |
|---|---|---|---|
| `gpt_v1_826k_loss1.75.pt` (original baseline, 2026-07-20) | 826K | 3000 | 1.7458 |
| `gpt_stage1_schedule_only.pt` (LR schedule alone) | 826K | 3000 | 1.8407 |
| `gpt_stage1b_bigger_model.pt` (bigger model alone) | 1.27M | 3000 | 1.6554 |
| **`gpt_stage1c_longer_scheduled.pt` (current best)** | 1.27M | 5000 | **1.6164** |
| `gpt_stage4_tied.pt` (weight tying, Chunk 4, char/same arch as 1c) | 1.268M | 5000 | 1.6465 |

Chunk 4 (weight tying — sharing the token-embedding and output-head matrix, a
real GPT-2 trick) is done and correctness-verified, but **did not become the
new best checkpoint**: close on val loss, noticeably more garbled on sampled
text. Root cause understood, not just observed — see `PROJECT_PLAN.md`'s
Chunk 4 result for the weight-norm comparison that explains it. Also built
BPE tokenization and KV-caching since Phase 1 closed — see below.

**Current best checkpoint: `checkpoints/gpt_stage1c_longer_scheduled.pt`.**
Sample with:

```powershell
.\venv\Scripts\python.exe src\generate.py --prompt "ROMEO:" --checkpoint gpt_stage1c_longer_scheduled.pt
```

Headline finding: model capacity (826K → 1.27M params) was the lever that
actually moved quality at this step budget, not training mechanics alone — a
bare LR schedule applied to the original size (Stage 1a) made things *worse*,
because decaying the LR over a fixed step count lowers the average effective
LR with no offsetting benefit. The schedule only paid off once combined with
the bigger model *and* a longer run (Stage 1c). Earlier checkpoints
(`gpt.pt`, `gpt_v1_826k_loss1.75.pt`) remain on disk as the historical
baseline, superseded rather than deleted. See `PROMPT_INPUTS.md` for the
session history.

**Phase 2, Chunk 2 (real byte-level BPE tokenizer) complete as of
2026-08-13** — see `PROJECT_PLAN.md` for the full writeup, including a real
lost-log bug found and fixed along the way. `checkpoints/gpt_stage2_bpe.pt`
is a from-scratch-BPE-tokenized run, confirmed trained (not just present on
disk) via qualitative sampling — coherent Shakespeare-style dialogue
structure, not random-init noise. BPE and char-level losses aren't on the
same scale (see `PROJECT_PLAN.md`), so this checkpoint is a parallel
demonstration of subword tokenization working end-to-end, not a claimed
replacement for `gpt_stage1c_longer_scheduled.pt` as "the" best checkpoint.
Sample it with `--checkpoint gpt_stage2_bpe.pt`.

**Chunk 3 (KV-caching) and Chunk 4 (weight tying) both complete as of
2026-08-13/14** — full writeups in `PROJECT_PLAN.md`. KV-caching is on by
default (`use_cache=True`) and proven byte-identical to the uncached path
within `block_size`, 1.64x faster on this small model. Weight tying
(`checkpoints/gpt_stage4_tied.pt`, sample with `--checkpoint
gpt_stage4_tied.pt`) is correctness-verified and saves the expected 10,400
params, but comes with a real, root-caused quality tradeoff at this scale —
see `PROJECT_PLAN.md`'s Chunk 4 result for why.

**Chunk 5 (new dataset — Project Gutenberg's *War and Peace*, ~3x
tiny-Shakespeare's size and stylistically different) complete as of
2026-08-16** — `data.py` now supports a `DATASETS` registry instead of one
hardcoded corpus URL; `tiny_shakespeare` stays the unchanged default.
`checkpoints/gpt_stage5_war_and_peace.pt` (same 1.27M-param/5000-iter/cosine
config as Stage 1c, dataset as the only changed variable) finished with a
*numerically lower* val loss (1.4049 vs. Stage 1c's 1.6164) but this is a
comparability trap, not a win: sampled output is consistently more garbled
than Stage 1c's across 4 different prompts, and the val-loss curve shows
training hadn't converged at this step budget (still falling at iter 5000,
no plateau like Stage 1c showed) — real, measured underfitting from ~3x less
per-character training exposure at the same capacity/step budget, not a bug.
See `PROJECT_PLAN.md`'s Chunk 5 result for the full root-cause writeup.
Sample it with `--checkpoint gpt_stage5_war_and_peace.pt --prompt "Natasha"`
(the standardized `"ROMEO:"` prompt doesn't apply to this book).

All five chunks on the original plan are now complete.
