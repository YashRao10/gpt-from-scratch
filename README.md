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

- **Character-level tokenizer** (`src/data.py`) — no BPE, just a fixed
  char-to-int vocabulary built from the training corpus.
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
  context.

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

- No BPE/subword tokenizer — character-level only, to keep the vocab and
  embedding table trivial to reason about.
- No KV-caching during generation — each new token re-runs the full forward
  pass over the context window. Fine at this scale, would matter at real
  scale.
- No distributed/multi-GPU training, mixed precision, or learning-rate
  schedule — single CPU process, constant LR, kept simple on purpose.
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
