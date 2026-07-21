"""Character-level tokenizer and data loading for the tiny-Shakespeare corpus."""

import os
import urllib.request

import torch

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_PATH = os.path.join(DATA_DIR, "input.txt")


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


def load_data(val_fraction: float = 0.1):
    path = download_dataset()
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = int(len(data) * (1 - val_fraction))
    train_data, val_data = data[:n], data[n:]
    return train_data, val_data, tokenizer


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """Sample a random batch of (input, target) sequences for next-token prediction."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
