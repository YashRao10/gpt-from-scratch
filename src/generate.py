"""Load a trained checkpoint and sample text from it."""

import argparse
import os

import torch

from model import GPT

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "gpt.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    model = GPT(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    stoi, itos = checkpoint["stoi"], checkpoint["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)

    idx = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
