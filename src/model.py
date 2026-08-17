"""A small GPT-style decoder-only transformer, built from scratch (no nn.Transformer)."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128       # max context length (sequence length)
    n_embd: int = 128           # embedding / residual stream dimension
    n_head: int = 4             # number of attention heads
    n_layer: int = 4            # number of transformer blocks
    dropout: float = 0.1
    weight_tying: bool = False  # Chunk 4: share token_emb/head weights (see GPT.__init__)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each position can only attend to itself and earlier positions."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # One linear layer projects to Q, K, V all at once (more efficient than three separate ones).
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Precompute a lower-triangular mask so position i can't see positions > i.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding dim

        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)

        # Reshape into (B, n_head, T, head_dim) so each head attends independently.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: how much should each token attend to every earlier token.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))

    def forward_cached(self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor] | None = None):
        """KV-cached variant used only by GPT.generate(use_cache=True) -- forward() above is
        untouched and still what train.py calls, so nothing about the verified training path
        changes. x is just the NEW token(s): the full prompt on the first (prefill) call, then
        one token at a time after that. past_kv is this layer's own cached (k, v) from every
        previous position; each call appends the new position(s)' k/v onto it and returns the
        grown cache for the next call, so previously-computed keys/values are never recomputed."""
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        T_past = k.shape[2] - T

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # The T new query positions may attend to every cached position unconditionally (all
        # strictly earlier in time, so always causally valid) plus a normal triangular mask
        # among themselves -- same causal rule as forward()'s static mask, just built fresh
        # each call since T_past varies (the fixed causal_mask buffer above only covers the
        # T_past == 0, single-block case forward() always uses).
        if T_past > 0:
            allow_past = torch.ones(T, T_past, device=x.device, dtype=torch.bool)
            causal_new = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            keep = torch.cat([allow_past, causal_new], dim=1)
        else:
            keep = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~keep.view(1, 1, T, T_past + T), float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out, (k, v)


class MLP(nn.Module):
    """Position-wise feedforward network applied after attention in each block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """One transformer block: attention + MLP, each with a residual connection and pre-norm."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_cached(self, x: torch.Tensor, past_kv: tuple[torch.Tensor, torch.Tensor] | None = None):
        attn_out, kv = self.attn.forward_cached(self.ln1(x), past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, kv


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        # Chunk 4: weight tying (Press & Wolf 2016 / used in GPT-2) -- token_emb.weight is
        # (vocab_size, n_embd) and head.weight is (vocab_size, n_embd) too (nn.Linear stores
        # weight as (out_features, in_features)), so the shapes already match exactly with no
        # transpose needed. Assigning the same nn.Parameter object to both attributes makes
        # them literally share storage: a gradient update to one updates the other, and
        # model.parameters() correctly counts it once (Module.parameters() dedupes by object
        # identity), so this genuinely removes vocab_size * n_embd parameters rather than just
        # hiding them. Applied AFTER _init_weights so head's separately-initialized tensor is
        # simply discarded in favor of token_emb's -- no wasted double-init, no mismatch.
        # getattr(...) rather than config.weight_tying: every checkpoint saved before this
        # field existed has a GPTConfig with no such attribute at all (pickled dataclasses
        # restore __dict__ directly, they don't backfill new fields' defaults), so a bare
        # attribute access would crash loading every pre-Chunk-4 checkpoint.
        if getattr(config, "weight_tying", False):
            self.head.weight = self.token_emb.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.config.block_size, "sequence longer than block_size"

        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def forward_cached(self, idx: torch.Tensor, past_kv_list: list | None = None):
        """KV-cached counterpart to forward(), used only by generate(use_cache=True). idx is
        just the new token(s) for this step; past_kv_list holds one (k, v) pair per block, as
        returned by the previous call. Each cached token keeps whatever position it was first
        assigned -- unlike forward()'s uncached path, which re-slices the trailing block_size
        window every step and silently renumbers every token's position from 0 each time.
        generate() maintains T_past + T <= block_size at every call (it evicts the oldest
        cache entry before appending a new one once the cache is full), so position is always
        a valid index into pos_emb without needing to clamp it."""
        B, T = idx.shape
        T_past = past_kv_list[0][0].shape[2] if past_kv_list is not None else 0
        assert T_past + T <= self.config.block_size, "cache + new tokens exceed block_size"

        pos = torch.arange(T_past, T_past + T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        new_kv_list = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kv_list[i] if past_kv_list is not None else None
            x, kv = block.forward_cached(x, past_kv)
            new_kv_list.append(kv)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_kv_list

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None, use_cache: bool = True):
        """Autoregressively sample new tokens, one at a time, appending each back into the
        context. use_cache=True (default) uses KV-caching: past keys/values are computed once
        and reused, so each new token only costs a forward pass over ITSELF instead of the
        whole growing context. use_cache=False keeps the original recompute-everything-every-
        step path, retained so the two can be compared directly for both speed and correctness
        (see PROJECT_PLAN.md Chunk 3 -- they're proven to sample identically within block_size
        given the same seed)."""
        if not use_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size :]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature  # only need the last position's prediction

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_id], dim=1)
            return idx

        # Cached path. Prefill: one forward pass over the whole starting prompt, building the
        # initial per-block cache. Every step after that is a forward pass over exactly 1 new
        # token, attending back to the cache instead of recomputing every earlier position.
        idx_cond = idx[:, -self.config.block_size :]
        logits, kv_cache = self.forward_cached(idx_cond)
        logits = logits[:, -1, :] / temperature

        for _ in range(max_new_tokens):
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

            # The position-embedding table only has block_size rows, so the cache can't be
            # allowed to grow past that. Evict the single oldest entry before the next call
            # appends a new one -- this keeps every cached token's ORIGINAL position intact
            # (a real cache never renumbers what's already cached, unlike forward()'s uncached
            # path, which re-slices the trailing window every step and silently renumbers
            # every token's position from 0 each time). Net effect once the window is full:
            # every further token is assigned position block_size - 1, since T_past saturates
            # there under this evict-then-append pattern -- the model simply can't distinguish
            # positions beyond what its fixed-size position table was ever trained on.
            cache_len = kv_cache[0][0].shape[2]
            if cache_len >= self.config.block_size:
                kv_cache = [(k[:, :, 1:, :], v[:, :, 1:, :]) for k, v in kv_cache]

            logits, kv_cache = self.forward_cached(next_id, kv_cache)
            logits = logits[:, -1, :] / temperature
        return idx
