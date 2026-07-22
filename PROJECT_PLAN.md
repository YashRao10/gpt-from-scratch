# Project Plan — GPT From Scratch

A chunked roadmap so this can be picked up and paused across sessions instead
of run as one long block. See `PROMPT_INPUTS.md` for the verbatim prompts
that drove each decision below.

| # | Chunk | What it actually does | Teaches you | Est. time | Suggested timing | Status |
|---|-------|------------------------|-------------|-----------|-------------------|--------|
| 0 | **Baseline pipeline** | Tokenizer, model, train loop, sampling — all wired end-to-end | The full GPT architecture skeleton | — | Done | Complete (2026-07-20) |
| 1a | **LR schedule + grad clip** | Same size/steps as v1, add warmup+cosine decay and gradient clipping | Why training *mechanics* (not just size) drive quality | ~35 min | 2026-07-22 | Done — negative result (see below) |
| 1b | **Scale the model** | Bump to 160-dim embeddings, same step count, **constant LR** (fair size-only comparison vs v1) | How much capacity alone buys you | ~50 min | 2026-07-22 | Done — clear win (see below) |
| 1c | **Train longer + LR schedule** | 160-dim (Stage 1b's size), extend to 4-5K steps, **now** add the cosine schedule (it needs a longer horizon to pay off — see 1a finding) | Diminishing returns / when a schedule actually helps | ~65-85 min | Same week | Queued |
| 1d | **Compare & decide** | Sample all checkpoints side by side, pick a winner, update README | How to evaluate a model beyond the loss number | ~15 min | Same week | Queued |
| 2 | **BPE tokenizer** | Replace char-level with a real byte-pair-encoding tokenizer (build merge table, encode/decode) | How GPT-2/3-style subword tokenization actually works | 2-3 hrs, own session | Next session | Planned |
| 3 | **KV-caching** | Rewrite `generate()` to cache past keys/values instead of recomputing the full forward pass each token | How real inference servers get fast | 1-2 hrs | Following session | Planned |
| 4 | **Weight tying** | Share the token-embedding and output-head weight matrix (a real GPT-2 trick) | A classic param-efficiency technique, cheap to implement | ~20 min | Any time, low priority | Stretch |
| 5 | **New/harder dataset** | Swap tiny-Shakespeare for something bigger (e.g. a different corpus) | How dataset size/diversity changes what the model can learn | Varies | Whenever curious | Stretch |

## Stage 1a result (2026-07-22)

Cosine LR decay + grad clipping, tested at v1's exact size/step count (826K
params, 3000 iters), came out **worse** than v1's constant-LR baseline:
train loss 1.6954 vs 1.5736, val loss 1.8407 vs 1.7458 (though the
train/val gap did shrink: 0.145 vs 0.172, i.e. less overfitting). Cause:
decaying from 3e-4 to 3e-5 over a fixed 3000-step budget lowers the
*average* effective LR, so less net learning happens than at constant LR
over the same steps. Conclusion: the schedule isn't a free win in isolation
— it needs a longer step budget to earn back what the decay costs early on.
Moved the schedule test into Stage 1c (train longer) instead. Stage 1b
(bigger model) will use constant LR so it's a clean apples-to-apples size
comparison against v1.

## Stage 1b result (2026-07-22)

Bigger model alone (1.27M params vs v1's 826K, constant LR, same 3000
iters) is a clear, unambiguous win over both v1 and Stage 1a: train loss
1.4520 (vs v1 1.5736, Stage 1a 1.6954), val loss 1.6554 (vs v1 1.7458,
Stage 1a 1.8407). Sample text is also qualitatively better — more real
English words, cleaner dialogue-tag formatting. Checkpoint:
`gpt_stage1b_bigger_model.pt`. Model capacity, not training mechanics, was
the lever that actually mattered at this step budget. Stage 1c carries
forward the 160-dim size and extends training, reintroducing the LR
schedule now that there's a longer horizon for it to pay off.

## Natural stopping points

After 1d you'll have a solid, understood, working small GPT — that alone is
a complete milestone if you want to pause there. Chunks 2-3 are the "go
deeper" phase (real tokenization + real inference tricks); 4-5 are optional
polish, not required for the learning goal.

## Checkpoint naming convention

Each stage in Phase 1 saves its own checkpoint under `checkpoints/` rather
than overwriting `gpt.pt`, so nothing is lost if a later stage doesn't pan
out:

- `gpt_v1_826k_loss1.75.pt` — original baseline (2026-07-20)
- `gpt_stage1_schedule_only.pt` — Stage 1a
- (later stages follow the same `gpt_stage<N>_<what-changed>.pt` pattern)
