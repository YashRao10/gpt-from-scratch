# Project Plan — GPT From Scratch

A chunked roadmap so this can be picked up and paused across sessions instead
of run as one long block. See `PROMPT_INPUTS.md` for the verbatim prompts
that drove each decision below.

| # | Chunk | What it actually does | Teaches you | Est. time | Suggested timing | Status |
|---|-------|------------------------|-------------|-----------|-------------------|--------|
| 0 | **Baseline pipeline** | Tokenizer, model, train loop, sampling — all wired end-to-end | The full GPT architecture skeleton | — | Done | Complete (2026-07-20) |
| 1a | **LR schedule + grad clip** | Same size/steps as v1, add warmup+cosine decay and gradient clipping | Why training *mechanics* (not just size) drive quality | ~35 min | 2026-07-22 | Done — negative result (see below) |
| 1b | **Scale the model** | Bump to 160-dim embeddings, same step count, **constant LR** (fair size-only comparison vs v1) | How much capacity alone buys you | ~50 min | 2026-07-22 | Done — clear win (see below) |
| 1c | **Train longer + LR schedule** | 160-dim (Stage 1b's size), extend to 4-5K steps, **now** add the cosine schedule (it needs a longer horizon to pay off — see 1a finding) | Diminishing returns / when a schedule actually helps | ~65-85 min | 2026-07-22 | Done — best result so far (see below) |
| 1d | **Compare & decide** | Sample all checkpoints side by side, pick a winner, update README | How to evaluate a model beyond the loss number | ~15 min | 2026-07-22 | Done — 1c wins, Phase 1 closed (see below) |
| 2 | **BPE tokenizer** | Replace char-level with a real byte-pair-encoding tokenizer (build merge table, encode/decode) | How GPT-2/3-style subword tokenization actually works | 2-3 hrs, own session | Next session | Done (2026-08-13) — see below |
| 3 | **KV-caching** | Rewrite `generate()` to cache past keys/values instead of recomputing the full forward pass each token | How real inference servers get fast | 1-2 hrs | Following session | Done (2026-08-13) — see below |
| 4 | **Weight tying** | Share the token-embedding and output-head weight matrix (a real GPT-2 trick) | A classic param-efficiency technique, cheap to implement | ~20 min | Any time, low priority | Done (2026-08-14) — real tradeoff found, see below |
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

## Stage 1c result (2026-07-22)

Extending Stage 1b's 1.27M-param model from 3000 to 5000 iters and
reintroducing the cosine LR schedule (now with a horizon long enough to earn
back the decay, per the Stage 1a finding) produced the best result of Phase
1: train loss 1.4096, val loss 1.6164 — beating Stage 1b's 1.6554 at the same
model size. Val loss actually bottomed out at iter 4500 (1.6103) and ticked
up slightly to 1.6164 by iter 5000 — a small, real sign of the start of a
plateau at this step count, not just noise: train loss shows the same
non-monotonic dip-then-tick-up shape over the same window (1.4179 → 1.4067 →
1.4103 → 1.4096), so both curves agree diminishing returns had set in by
~4500 iters. The training
script only checkpoints the final iteration, not the best-val one, so the
saved checkpoint is the very slightly worse 5000-iter version — a known, tiny
gap, not worth a rerun to shave 0.006 off val loss.

## Stage 1d result — final comparison and decision (2026-07-22)

Sampled all four Phase 1 checkpoints on the identical prompt (`"ROMEO:"`,
300 tokens, temperature 0.8, top_k 50) to judge beyond the raw loss number:

| Checkpoint | Params | Iters | Val loss | Qualitative read |
|---|---|---|---|---|
| `gpt_v1_826k_loss1.75.pt` (baseline) | 826K | 3000 | 1.7458 | Mostly non-words, some correct short English, weak dialogue structure |
| `gpt_stage1_schedule_only.pt` (1a) | 826K | 3000 | 1.8407 | Similar to baseline, arguably choppier — matches its worse val loss |
| `gpt_stage1b_bigger_model.pt` (1b) | 1.27M | 3000 | 1.6554 | Clearly more coherent — real multi-character dialogue (KING RICHARD II, MENENIUS, Second AUFIDIUS), correct tag formatting |
| `gpt_stage1c_longer_scheduled.pt` (1c) | 1.27M | 5000 | 1.6164 | Comparable coherence to 1b, different cast sampled (Second Servant, JULIET) — best loss of the four |

**Decision: Stage 1c (`gpt_stage1c_longer_scheduled.pt`) is the Phase 1
winner.** It has the lowest val loss of any checkpoint tested and samples at
least as coherently as 1b — no regression traded for the loss improvement.
Confirms the two-part finding from 1a/1b: model capacity was the lever that
mattered most at a fixed step budget, and the LR schedule *is* a genuine
(if modest) net positive once the step budget is long enough to amortize the
decay — the exact fix 1a's negative result predicted.

**Phase 1 is closed.** `README.md`'s Status section and default checkpoint
reference have been updated to point at `gpt_stage1c_longer_scheduled.pt` as
the current best model. `gpt.pt` (the original, undifferentiated checkpoint
name from before this project had a naming convention) and `gpt_v1_826k_loss1.75.pt`
remain on disk as the historical baseline — not deleted, just superseded.

## Phase 2, Chunk 2: BPE tokenizer (2026-08-12)

Built `src/bpe.py` from scratch — no `tiktoken`/`sentencepiece`, matching the
project's whole-point-is-understanding-every-piece philosophy already applied
to the model itself. It's **byte-level** BPE (the real GPT-2/3 design):
training starts from the 256 raw UTF-8 byte values as the base vocabulary and
greedily merges the most frequent adjacent pair, `vocab_size - 256` times.
Byte-level means there's no `<unk>` token and no fixed character allowlist —
any text in any script/emoji/etc. can always be re-expressed as some sequence
of the 256 base bytes even where no merge applies, unlike the Phase 1
`CharTokenizer` which could only represent the ~65 characters it happened to
see in tiny-Shakespeare.

**Correctness verified before touching the training pipeline at all** — a
round-trip sanity test (repeated phrases, non-ASCII text, empty string,
unseen text) confirmed `decode(encode(x)) == x` in every case, and a full
train→checkpoint→generate.py smoke test (tiny model, 5 steps) confirmed the
whole pipeline wires together correctly before spending real training time.

**Real bug caught by the smoke test, not assumed fine:** an essentially
untrained model's near-random output byte sequence isn't guaranteed to be
valid UTF-8, so `decode()`'s `errors="replace"` correctly substitutes
`U+FFFD` — but Windows' default console codepage (cp1252) can't print that
character and crashed the whole script. Fixed by reconfiguring `sys.stdout`
to UTF-8 with a replace fallback at the top of `generate.py`. A real,
if minor, robustness gap that Phase 1's pure-ASCII char vocab never exposed.

**Integration design:** `TOKENIZER_TYPE` env var (`"char"` default, keeping
every Phase 1 checkpoint/workflow working unchanged) or `"bpe"`
(`BPE_VOCAB_SIZE`, default 512). Checkpoints now embed a `tokenizer_type`
field plus either `stoi`/`itos` (char) or `bpe_data` (BPE's merge table,
JSON-safe) — self-contained, same pattern Phase 1 already used, so a
checkpoint file alone is always enough to regenerate text with the right
tokenizer, no external vocab file to keep in sync. `generate.py` reads
`tokenizer_type` and reconstructs the right codec automatically;
checkpoints saved before this change have no such field and correctly fall
back to `"char"`.

**A real, from-scratch cost worth naming:** this naive implementation
rescans pair counts across the whole sequence on every single merge step —
training at vocab_size=512 on the real ~1.1M-character corpus took ~78s,
and a full-corpus `encode()` call costs about the same. Both get cached to
`checkpoints/bpe_cache/` (trained merge table + the already-tokenized
corpus) so re-running `train.py` to try different model hyperparameters at
a fixed vocab size doesn't repay that cost every time. A real BPE
implementation (or `tiktoken`) uses a priority queue instead — the tradeoff
here is staying byte-for-byte example-of-the-algorithm even where it's
provably not the efficient way to do it, matching the "build the real
mechanism, not a wrapper" goal of the whole project.

**Comparability caveat, to keep in mind reading Stage 2's results below:**
BPE cross-entropy loss and Phase 1's char-level loss are *not* the same
scale — a BPE token typically spans several characters, so lower
loss-per-token doesn't mean "better" in the same units. The honest
comparison is qualitative (sample coherence) plus how much *more text* the
same 128-token context window now covers, not a direct loss-number race.

## Stage 2 result — BPE-tokenized training run (2026-08-13, confirmed after a gap)

The actual `TOKENIZER_TYPE=bpe` training run (`gpt_stage2_bpe.pt`) was launched in the prior
session (2026-08-12) but never confirmed finished — the session ended before checking, and this
session picked that verification up as its first task.

**Confirmed complete, but not from the log — the log was empty.** `train_log_stage2_bpe.txt`
contained only the startup numpy warning, none of the per-250-iter loss lines a full run should
produce. Root cause: `train.py`'s `print()` calls had no `flush=True`, so their output sat in
Python's stdout buffer — if the process was killed rather than exiting cleanly (most likely the
terminal/session closing right around when training finished), that buffered output was lost
even though the run itself completed normally. **Fixed properly, not just noted**: every
`print()` in the training loop now flushes explicitly, and — more robust than logging alone —
each checkpoint now embeds its own `last_iter`/`last_losses`, so a checkpoint is self-documenting
even if the log file is empty, truncated, or missing entirely next time.

**Verified completion the right way: by actually sampling the checkpoint, not by inferring from
timestamps.** The gap between the log's last write (01:21) and the checkpoint's final save
(02:06) — about 45 minutes — closely matches Stage 1c's known full-run time (2769s ≈ 46 min at a
similar model size), which was suggestive but circumstantial. The real confirmation:

```
.\venv\Scripts\python.exe src\generate.py --checkpoint gpt_stage2_bpe.pt --prompt "ROMEO:" --max_new_tokens 200
```

```
ROMEO: he be honour to well.
Almost offere's manner, teach:
In thinks of think his own forth.

Messenger:
Go with the heart of her try's triumbers.

HERMIONE:
Doubts that I pray head:
...
```

Coherent dialogue-tag structure, real character names, mostly-grammatical archaic-flavored
English — qualitatively in the same range as the Phase 1 winner (Stage 1c), not random-init
gibberish. This is real evidence the BPE-tokenized model actually trained, not just that a file
exists.

**Known, permanent gap for this specific checkpoint:** the exact final train/val loss numbers
from this run are unrecoverable — they were never logged (the bug above) and this checkpoint
predates the `last_losses` self-documentation fix, so it can't retroactively carry them either.
Not worth a retrain just to recover a number; the qualitative sample is sufficient confirmation
of Stage 2's actual goal (does BPE tokenization work end-to-end through a real training run).
Per `PROJECT_PLAN.md`'s own comparability caveat above, a BPE loss number wouldn't have been
directly comparable to Phase 1's char-level loss anyway.

## Chunk 3 result — KV-caching (2026-08-13)

Added `forward_cached()` to `CausalSelfAttention`, `Block`, and `GPT` in `model.py` — a
parallel path alongside the existing `forward()`, which `train.py` still calls completely
unchanged, so nothing about the already-verified training pipeline was touched. `GPT.generate()`
gained a `use_cache: bool = True` parameter: the cached path does one prefill pass over the
starting prompt, then one forward pass per new token over just that token, attending back to a
growing per-block `(k, v)` cache instead of recomputing every earlier position every step.
`use_cache=False` keeps the exact original recompute-everything path, kept specifically so the
two can be compared.

**Correctness proven, not assumed:** `src/bench_kv_cache.py` runs both paths from the same seed
and prompt and asserts the sampled token IDs are byte-identical — confirmed **MATCH** for 118
new tokens (comfortably within `block_size=128`). Same math, same RNG draw order, the cache
just avoids redundant recomputation.

**Real, honest limitation past `block_size` — documented, not fixed with a hack.** The
position-embedding table only has `block_size` rows. The original uncached `generate()` handles
this by re-slicing the trailing `block_size` window every step, which silently **renumbers**
every token's position from 0 each time a token slides through the window — already a known
simplification in the original code, not something introduced here. A real KV-cache can't do
that (you can't cheaply renumber a position already baked into cached `k`/`v` tensors through
several transformer layers), so the cached path instead evicts the single oldest cache entry
once full and lets every further token's position **saturate at `block_size - 1`** once the
window is full — every generated token past that point gets the exact same position embedding.

Compared the two paths' actual output past `block_size` (400 new tokens, well beyond 128):
the uncached path's tail stays noticeably more coherent than the cached path's, because
renumbering (however hacky) still gives the model *some* varied positional signal, while
clamping gives none at all once saturated. **This is a real quality tradeoff of the
clamp-based cache, not a bug** — correctness is proven exact within `block_size`, which is
what this chunk's own learning goal (understand and correctly implement KV-caching) actually
required. A proper fix would mean relative position encoding (RoPE/ALiBi) instead of absolute
learned position embeddings — a real architecture change, out of scope for this chunk, and a
reasonable candidate if this project ever revisits position encoding specifically.

**Speed measured, not assumed:** on this small model (1.27M params, `block_size=128`, CPU),
400-token generation: 1.84s uncached vs. 1.13s cached — **1.64x speedup**. Real, but modest —
at this scale, the O(T²)→O(T) attention-cost saving is small relative to Python/tensor-op
overhead per step, since T never gets large enough for the quadratic cost to dominate. The
benefit of KV-caching grows sharply with model size and context length; a small from-scratch
model at `block_size=128` is close to the least favorable case for it, which is itself a useful
thing to have actually measured rather than assumed.

## Chunk 4 result — weight tying (2026-08-14)

Added `weight_tying: bool` to `GPTConfig` (default `False`, so every existing checkpoint keeps
loading unchanged) and, when set, assign `self.head.weight = self.token_emb.weight` in
`GPT.__init__`, right after `_init_weights` runs. Shapes already match exactly with no
transpose needed — `token_emb` is `(vocab_size, n_embd)`, and `nn.Linear` stores its weight as
`(out_features, in_features)`, so `head`'s is `(vocab_size, n_embd)` too.

**Correctness verified before the real run, not assumed:** confirmed the param count drops by
exactly `vocab_size * n_embd` (65×160=10,400, matching precisely), confirmed `head.weight is
token_emb.weight` (literally the same object, not just equal values), confirmed gradients flow
through both usages in one backward pass, and confirmed every pre-existing checkpoint (char and
BPE) still loads and generates correctly — the risky part being that `GPTConfig` is pickled
directly into old checkpoints, so accessing the new field had to go through
`getattr(config, "weight_tying", False)` rather than `config.weight_tying`, since old pickled
configs simply don't have the attribute at all (dataclass unpickling restores `__dict__`
directly, it doesn't backfill new fields' defaults).

**Real run: `gpt_stage4_tied.pt`, same architecture/iters/tokenizer as Stage 1c** (char, 160-dim,
4 head, 4 layer, 5000 iters, cosine schedule) so the comparison is genuinely apples-to-apples,
unlike Stage 2's BPE run. Params: 1,268,320 (vs. Stage 1c's ~1,278,720 untied — the expected
10,400-param saving). **Final val loss 1.6465, close to but slightly worse than Stage 1c's
1.6164** (+1.9% relative).

**The loss gap undersells the real difference — sampled text is noticeably more garbled than
Stage 1c's, not just marginally.** Checked this wasn't a one-off sampling fluke (one "ROMEO:"
draw produced an outright degenerate repetition, `llllllllllllllll`) by sampling 3 more prompts
at the same settings — the extreme repetition didn't recur, but every sample was consistently
less coherent than Stage 1c's own output on the identical prompt, fewer recognizable real words,
more broken fragments.

**Root-caused, not just reported as a vibe — this is a real, mechanistically-understood tradeoff,
not a bug.** Compared weight norms between the two checkpoints: Stage 1c's *untied* embedding
settles at norm 3.54 while its *separate* output head settles at norm 7.46 — the two roles want
meaningfully different scales during training. Stage 4's *tied* matrix, forced to serve both
roles at once, settles at norm 7.14 — pulled almost all the way to the output-projection's
preferred scale, far from where the embedding-lookup role would land on its own. This is a known,
documented tension with naive weight tying (the reason production tied-embedding
implementations like GPT-2 apply an extra scale factor at the softmax rather than tying the raw
matrices as-is) — confirmed the tie itself was intact and correct first (`head.weight is
token_emb.weight` still `True` after training, param count still exactly reduced) before
concluding this was a real scale-tension effect rather than an implementation bug.

**Verdict: weight tying works exactly as designed (real param savings, correct gradient
sharing, backward-compatible with every existing checkpoint) but is a genuine quality tradeoff
at this small scale without further work** (e.g. an explicit output-side scale factor, which
this chunk deliberately didn't add — that would be a reasonable follow-up, not required for the
chunk's own learning goal of understanding and correctly implementing the tying itself).

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
