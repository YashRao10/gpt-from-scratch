"""Train the from-scratch GPT on tiny-Shakespeare and checkpoint it."""

import math
import os
import time

import torch

from data import get_batch, load_data
from model import GPT, GPTConfig

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
CHECKPOINT_NAME = os.environ.get("CHECKPOINT_NAME", "gpt.pt")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)

# Staged quality-improvement plan (see PROJECT_PLAN.md): each stage changes one
# thing vs. v1 so we can tell what actually helped. Stage 1a (schedule+clip at
# v1's size/iters) came out worse than constant LR -- a fixed-budget cosine
# decay lowers the *average* LR too much to pay for itself in only 3000 steps.
# Stage 1b isolated model size (win: 1.27M params clearly beat both v1 and 1a).
# Stage 1c keeps that size, trains longer, and reintroduces the schedule now
# that there's a longer horizon for the decay to earn out.
BATCH_SIZE = 32
BLOCK_SIZE = 128
N_EMBD = 160
N_HEAD = 4
N_LAYER = 4
MAX_ITERS = 5000
EVAL_INTERVAL = 250
EVAL_ITERS = 50
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WARMUP_ITERS = 200
GRAD_CLIP = 1.0
USE_LR_SCHEDULE = True


def get_lr(it: int) -> float:
    if not USE_LR_SCHEDULE:
        return LEARNING_RATE
    if it < WARMUP_ITERS:
        return LEARNING_RATE * it / WARMUP_ITERS
    if it > MAX_ITERS:
        return MIN_LEARNING_RATE
    decay_ratio = (it - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LEARNING_RATE + coeff * (LEARNING_RATE - MIN_LEARNING_RATE)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, device):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_data, val_data, tokenizer = load_data()
    print(f"Vocab size: {tokenizer.vocab_size} | train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
    )
    model = GPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    start = time.time()

    for it in range(1, MAX_ITERS + 1):
        lr = get_lr(it)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:
            losses = estimate_loss(model, train_data, val_data, device)
            elapsed = time.time() - start
            print(f"iter {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | lr {lr:.2e} | {elapsed:.0f}s elapsed")

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "stoi": tokenizer.stoi,
                    "itos": tokenizer.itos,
                },
                CHECKPOINT_PATH,
            )

    print(f"Done. Checkpoint saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
