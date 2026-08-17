"""Tokenizers and data loading, with a choice of source corpus.

Supports two tokenizers, selected via `load_data(tokenizer_type=...)`:
- "char" (Phase 1): one token per character, ~65-symbol vocab.
- "bpe" (Phase 2): byte-level byte-pair-encoding, see bpe.py.

Chunk 5: supports more than one corpus, selected via `load_data(dataset=...)`.
"tiny_shakespeare" (default) keeps every prior chunk's behavior identical --
same file, same path, same cache -- so nothing before this chunk changes
unless a different dataset is explicitly requested.
"""

import os
import re
import urllib.request

import torch

from bpe import BPETokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "bpe_cache")

# Chunk 5: named corpora. "tiny_shakespeare" is Phase 1-4's original corpus,
# unchanged -- adding entries here must never alter its url/filename, since
# every existing checkpoint's char vocab was built from that exact file.
DATASETS = {
    "tiny_shakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "filename": "input.txt",
    },
    # ~3x tiny_shakespeare's size (3.3MB vs 1.1MB) and stylistically different --
    # 19th-century English prose narrative vs. Early Modern English verse/dialogue --
    # so it exercises both the "size" and "diversity" halves of Chunk 5's goal at once,
    # not just "more of the same corpus." Project Gutenberg #2600 (Maude translation).
    "war_and_peace": {
        "url": "https://www.gutenberg.org/files/2600/2600-0.txt",
        "filename": "war_and_peace.txt",
    },
}

_GUTENBERG_START = re.compile(r"\*\*\* ?START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
_GUTENBERG_END = re.compile(r"\*\*\* ?END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Cut the license/header/footer Project Gutenberg wraps every ebook in, so the
    corpus is just the actual work. No-op (returns text unchanged) if the markers
    aren't present -- e.g. tiny_shakespeare's source is already the bare text.
    """
    start = _GUTENBERG_START.search(text)
    end = _GUTENBERG_END.search(text)
    if not start or not end:
        return text
    return text[start.end():end.start()].strip()


def download_dataset(dataset: str = "tiny_shakespeare"):
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset!r} (known: {list(DATASETS)})")
    spec = DATASETS[dataset]
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_path = os.path.join(DATA_DIR, spec["filename"])
    if not os.path.exists(raw_path):
        print(f"Downloading dataset {dataset!r} to {raw_path} ...")
        urllib.request.urlretrieve(spec["url"], raw_path)
    return raw_path


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


def _load_or_train_bpe(text: str, vocab_size: int, dataset: str) -> tuple[BPETokenizer, list[int]]:
    """Cache both the trained tokenizer and the encoded corpus -- training and
    (re-)encoding the full ~1.1M-char corpus each cost ~a minute with this
    from-scratch tokenizer's naive O(n) merge scans, so re-paying that on
    every train.py run would make iterating on model hyperparams painfully
    slow for no reason once the tokenizer itself is fixed.

    Cache filenames include `dataset` (Chunk 5) so two different corpora at the
    same vocab_size don't collide and silently load the wrong one's cache.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    prefix = "bpe" if dataset == "tiny_shakespeare" else f"bpe_{dataset}"
    tok_path = os.path.join(CACHE_DIR, f"{prefix}_{vocab_size}.json")
    ids_path = os.path.join(CACHE_DIR, f"{prefix}_{vocab_size}_ids.pt")

    if os.path.exists(tok_path) and os.path.exists(ids_path):
        tokenizer = BPETokenizer.load(tok_path)
        ids = torch.load(ids_path).tolist()
        return tokenizer, ids

    tokenizer = BPETokenizer()
    ids = tokenizer.train(text, vocab_size=vocab_size, verbose=True)
    tokenizer.save(tok_path)
    torch.save(torch.tensor(ids, dtype=torch.long), ids_path)
    return tokenizer, ids


def load_data(val_fraction: float = 0.1, tokenizer_type: str = "char", bpe_vocab_size: int = 512,
              dataset: str = "tiny_shakespeare"):
    path = download_dataset(dataset)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = _strip_gutenberg_boilerplate(text)

    if tokenizer_type == "char":
        tokenizer = CharTokenizer(text)
        ids = tokenizer.encode(text)
    elif tokenizer_type == "bpe":
        tokenizer, ids = _load_or_train_bpe(text, bpe_vocab_size, dataset)
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
