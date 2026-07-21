"""Train the from-scratch GPT on tiny-Shakespeare and checkpoint it."""

import os
import time

import torch

from data import get_batch, load_data
from model import GPT, GPTConfig

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "gpt.pt")

# Hyperparameters chosen to train in a few minutes on CPU, not to maximize quality.
BATCH_SIZE = 32
BLOCK_SIZE = 128
MAX_ITERS = 3000
EVAL_INTERVAL = 250
EVAL_ITERS = 50
LEARNING_RATE = 3e-4


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

    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=BLOCK_SIZE)
    model = GPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    start = time.time()

    for it in range(1, MAX_ITERS + 1):
        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:
            losses = estimate_loss(model, train_data, val_data, device)
            elapsed = time.time() - start
            print(f"iter {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | {elapsed:.0f}s elapsed")

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
