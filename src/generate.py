"""Load a trained checkpoint and sample text from it — one-shot or interactive."""

import argparse
import os
import sys

import torch

from bpe import BPETokenizer
from model import GPT

# BPE decode() falls back to U+FFFD for any byte sequence that isn't valid
# UTF-8 (expected from an undertrained model, or genuinely from mid-token
# sampling cutting a multi-byte character in half) -- Windows' default
# console codepage (cp1252) can't print that character and would otherwise
# crash the whole script. Reconfigure stdout to a UTF-8 codec that always
# has a fallback instead.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")


class _CharCodec:
    """Wraps a Phase-1-style stoi/itos dict pair in the same encode/decode
    interface BPETokenizer exposes, so sample() doesn't need to branch."""

    def __init__(self, stoi: dict, itos: dict):
        self.stoi, self.itos = stoi, itos

    def encode(self, text: str) -> list[int]:
        # Unlike BPE (byte-level, so it can always represent anything as raw
        # bytes), the char vocab is fixed to whatever chars were seen during
        # training -- drop anything unseen instead of crashing.
        unknown = sorted(set(c for c in text if c not in self.stoi))
        if unknown:
            print(f"(dropping characters not in the training vocab: {''.join(unknown)!r})")
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_model(checkpoint_name: str, device: str):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = GPT(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Checkpoints saved before Phase 2 have no "tokenizer_type" key at all --
    # they're always char-level, so default to that for backward compatibility.
    tokenizer_type = checkpoint.get("tokenizer_type", "char")
    if tokenizer_type == "bpe":
        codec = BPETokenizer.from_dict(checkpoint["bpe_data"])
    else:
        codec = _CharCodec(checkpoint["stoi"], checkpoint["itos"])
    return model, codec


def sample(model, codec, device, prompt, max_new_tokens, temperature, top_k):
    ids = codec.encode(prompt)
    if not ids:
        ids = codec.encode("\n")

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return codec.decode(out[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default="gpt.pt", help="checkpoint filename under checkpoints/")
    parser.add_argument("--interactive", action="store_true", help="keep prompting for input instead of exiting after one generation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, codec = load_model(args.checkpoint, device)

    if not args.interactive:
        print(sample(model, codec, device, args.prompt, args.max_new_tokens, args.temperature, args.top_k))
        return

    print(f"Interactive mode - checkpoint: {args.checkpoint} | device: {device}")
    print("Type a prompt and press Enter to generate. Empty input repeats the last settings. Ctrl+C or 'quit' to exit.\n")
    while True:
        try:
            user_prompt = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if user_prompt.strip().lower() in ("quit", "exit"):
            print("Exiting.")
            break
        if not user_prompt:
            user_prompt = "\n"
        text = sample(model, codec, device, user_prompt, args.max_new_tokens, args.temperature, args.top_k)
        print(text)
        print("-" * 40)


if __name__ == "__main__":
    main()
