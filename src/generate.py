"""Load a trained checkpoint and sample text from it — one-shot or interactive."""

import argparse
import os

import torch

from model import GPT

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")


def load_model(checkpoint_name: str, device: str):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = GPT(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    stoi, itos = checkpoint["stoi"], checkpoint["itos"]
    return model, stoi, itos


def sample(model, stoi, itos, device, prompt, max_new_tokens, temperature, top_k):
    # The model's vocab is fixed to the 65 characters seen in tiny-Shakespeare —
    # drop anything typed that it was never trained on rather than crashing.
    unknown = sorted(set(c for c in prompt if c not in stoi))
    if unknown:
        print(f"(dropping characters not in the training vocab: {''.join(unknown)!r})")
        prompt = "".join(c for c in prompt if c in stoi)
    if not prompt:
        prompt = "\n"

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)

    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return decode(out[0].tolist())


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
    model, stoi, itos = load_model(args.checkpoint, device)

    if not args.interactive:
        print(sample(model, stoi, itos, device, args.prompt, args.max_new_tokens, args.temperature, args.top_k))
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
        text = sample(model, stoi, itos, device, user_prompt, args.max_new_tokens, args.temperature, args.top_k)
        print(text)
        print("-" * 40)


if __name__ == "__main__":
    main()
