"""Verify KV-caching is correct (identical output to the uncached path within block_size)
and measure the actual speedup it buys -- the two things Chunk 3 needs to prove, not just
that the code runs without crashing."""

import sys
import time

import torch

from generate import load_model

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "gpt_stage1c_longer_scheduled.pt"


def timed_sample(model, codec, device, prompt, max_new_tokens, use_cache, seed):
    torch.manual_seed(seed)
    start = time.time()
    ids = codec.encode(prompt) or codec.encode("\n")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50, use_cache=use_cache)
    elapsed = time.time() - start
    return codec.decode(out[0].tolist()), elapsed, out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, codec = load_model(CHECKPOINT, device)
    block_size = model.config.block_size
    print(f"Checkpoint: {CHECKPOINT} | block_size: {block_size} | device: {device}\n")

    # --- Correctness: within block_size, cached and uncached must sample byte-identically
    # given the same seed (same RNG draw order, same math, just less redundant recompute).
    n_correctness = block_size - 10  # comfortably under block_size including the prompt
    _, _, out_uncached = timed_sample(model, codec, device, "ROMEO:", n_correctness, use_cache=False, seed=0)
    _, _, out_cached = timed_sample(model, codec, device, "ROMEO:", n_correctness, use_cache=True, seed=0)
    identical = torch.equal(out_uncached, out_cached)
    print(f"Correctness check (within block_size, {n_correctness} new tokens): "
          f"{'MATCH' if identical else 'MISMATCH'}")
    if not identical:
        print("  uncached:", codec.decode(out_uncached[0].tolist()))
        print("  cached:  ", codec.decode(out_cached[0].tolist()))
        sys.exit(1)

    # --- Speed: the real point of KV-caching. Generate well past block_size so the sliding
    # cache-eviction path gets exercised too, not just the easy short-sequence case.
    n_speed = 400
    text_uncached, t_uncached, _ = timed_sample(model, codec, device, "ROMEO:", n_speed, use_cache=False, seed=1)
    text_cached, t_cached, _ = timed_sample(model, codec, device, "ROMEO:", n_speed, use_cache=True, seed=1)
    print(f"\nSpeed check ({n_speed} new tokens, exceeds block_size so cache eviction runs too):")
    print(f"  use_cache=False: {t_uncached:.2f}s")
    print(f"  use_cache=True:  {t_cached:.2f}s")
    print(f"  speedup: {t_uncached / t_cached:.2f}x")

    print("\n--- sample (cached path, first 200 chars) ---")
    print(text_cached[:200])


if __name__ == "__main__":
    main()
