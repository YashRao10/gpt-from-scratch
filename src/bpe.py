"""Byte-level BPE tokenizer, built from scratch (same spirit as model.py: no
tiktoken/sentencepiece). Operates on raw UTF-8 bytes rather than characters,
the same design GPT-2/3 use -- it means there's no <unk> token, since any
byte sequence (any text in any language/emoji/etc.) can always be re-expressed
as some sequence of the 256 base byte tokens even if no merge applies.
"""

from __future__ import annotations

import json
import os


def get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count how often each adjacent pair of ids occurs."""
    counts: dict[tuple[int, int], int] = {}
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with the single id `new_id`."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-pair-encoding tokenizer: 256 base byte tokens plus learned merges.

    `merges` records the merge order as {(id_a, id_b): new_id}, built greedily
    by always merging the most frequent adjacent pair first -- the standard
    BPE training algorithm. `vocab` maps every token id (base byte or merged)
    to the raw bytes it expands to, so decode is just a concatenation.
    """

    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (the base byte alphabet)")
        num_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))

        for i in range(num_merges):
            counts = get_pair_counts(ids)
            if not counts:
                break  # corpus fully collapsed to one token, nothing left to merge
            pair = max(counts, key=counts.get)
            new_id = 256 + i
            ids = merge_pair(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (i + 1) % 50 == 0:
                print(f"  merge {i + 1}/{num_merges}: {pair} -> {new_id} "
                      f"({self.vocab[new_id]!r}, count {counts[pair]})")

        return ids  # the fully-encoded training corpus, handy for immediate reuse

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        # Repeatedly apply the *earliest-learned* mergeable pair present in the
        # sequence -- merges must replay in training order, since a later merge
        # can depend on an earlier one having already collapsed its pair.
        while len(ids) >= 2:
            counts = get_pair_counts(ids)
            # Pick whichever present pair has the lowest merge rank (= was
            # learned first). Pairs with no merge get rank +inf and are skipped.
            candidate = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if candidate not in self.merges:
                break  # no known merge applies to anything left in the sequence
            ids = merge_pair(ids, candidate, self.merges[candidate])
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        # JSON/checkpoint keys must be strings, so pairs are serialized as "a,b".
        return {"merges": {f"{a},{b}": new_id for (a, b), new_id in self.merges.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> "BPETokenizer":
        tok = cls()
        for key, new_id in data["merges"].items():
            a, b = key.split(",")
            pair = (int(a), int(b))
            tok.merges[pair] = new_id
            tok.vocab[new_id] = tok.vocab[pair[0]] + tok.vocab[pair[1]]
        return tok

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
