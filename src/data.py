"""Tokenizers and data loading for the tiny-Shakespeare corpus.

Supports two tokenizers, selected via `load_data(tokenizer_type=...)`:
- "char" (Phase 1): one token per character, ~65-symbol vocab.
- "bpe" (Phase 2): byte-level byte-pair-encoding, see bpe.py.
"""

import os
import urllib.request

import torch

from bpe import BPETokenizer

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_PATH = os.path.join(DATA_DIR, "input.txt")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "bpe_cache")


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(RAW_PATH):
        print(f"Downloading dataset to {RAW_PATH} ...")
        urllib.request.urlretrieve(DATA_URL, RAW_PATH)
    return RAW_PATH


class CharTokenizer:
    """Maps characters to integer ids and back, built from a corpus."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def _load_or_train_bpe(text: str, vocab_size: int) -> tuple[BPETokenizer, list[int]]:
    """Cache both the trained tokenizer and the encoded corpus -- training and
    (re-)encoding the full ~1.1M-char corpus each cost ~a minute with this
    from-scratch tokenizer's naive O(n) merge scans, so re-paying that on
    every train.py run would make iterating on model hyperparams painfully
    slow for no reason once the tokenizer itself is fixed.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    tok_path = os.path.join(CACHE_DIR, f"bpe_{vocab_size}.json")
    ids_path = os.path.join(CACHE_DIR, f"bpe_{vocab_size}_ids.pt")

    if os.path.exists(tok_path) and os.path.exists(ids_path):
        tokenizer = BPETokenizer.load(tok_path)
        ids = torch.load(ids_path).tolist()
        return tokenizer, ids

    tokenizer = BPETokenizer()
    ids = tokenizer.train(text, vocab_size=vocab_size, verbose=True)
    tokenizer.save(tok_path)
    torch.save(torch.tensor(ids, dtype=torch.long), ids_path)
    return tokenizer, ids


def load_data(val_fraction: float = 0.1, tokenizer_type: str = "char", bpe_vocab_size: int = 512):
    path = download_dataset()
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if tokenizer_type == "char":
        tokenizer = CharTokenizer(text)
        ids = tokenizer.encode(text)
    elif tokenizer_type == "bpe":
        tokenizer, ids = _load_or_train_bpe(text, bpe_vocab_size)
    else:
        raise ValueError(f"unknown tokenizer_type: {tokenizer_type!r}")

    data = torch.tensor(ids, dtype=torch.long)
    n = int(len(data) * (1 - val_fraction))
    train_data, val_data = data[:n], data[n:]
    return train_data, val_data, tokenizer


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """Sample a random batch of (input, target) sequences for next-token prediction."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
