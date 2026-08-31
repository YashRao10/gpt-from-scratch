"""Smoke tests for the from-scratch GPT — shape/finiteness checks and the
KV-cache parity claim the README makes ("proven byte-identical to the uncached
path within block_size"). Tiny config, CPU only, no training, runs in seconds.
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.model import GPT, GPTConfig

VOCAB = 17
CFG = GPTConfig(vocab_size=VOCAB, block_size=16, n_embd=32, n_head=4, n_layer=2, dropout=0.0)


def _model():
    torch.manual_seed(0)
    m = GPT(CFG)
    m.eval()
    return m


def test_forward_shape():
    m = _model()
    idx = torch.randint(0, VOCAB, (2, 8))
    logits, loss = m(idx)
    assert logits.shape == (2, 8, VOCAB)
    assert loss is None


def test_forward_loss_is_finite():
    m = _model()
    idx = torch.randint(0, VOCAB, (2, 8))
    targets = torch.randint(0, VOCAB, (2, 8))
    _, loss = m(idx, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_generate_extends_sequence():
    m = _model()
    idx = torch.randint(0, VOCAB, (1, 4))
    out = m.generate(idx, max_new_tokens=5)
    assert out.shape == (1, 9)
    assert (out[:, :4] == idx).all()  # prompt is preserved
    assert (out < VOCAB).all() and (out >= 0).all()


def test_kv_cache_matches_uncached_within_block_size():
    m = _model()
    idx = torch.randint(0, VOCAB, (1, 4))

    torch.manual_seed(123)
    cached = m.generate(idx, max_new_tokens=10, use_cache=True)
    torch.manual_seed(123)
    uncached = m.generate(idx, max_new_tokens=10, use_cache=False)

    assert torch.equal(cached, uncached)
