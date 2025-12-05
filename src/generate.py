import torch
import torch.nn.functional as F
import argparse
from tokenizer import ByteLevelTokenizer
from model import SparrowGPT

CHECKPOINT_PATH = "sparrowgpt1.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=50):
    model.eval()
    tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    tokens = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

    for _ in range(max_new_tokens):
        # Only feed the last window of tokens
        idx_cond = tokens[:, -model.max_seq_len:]

        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :]  # final token

            # Temperature scaling
            logits = logits / temperature

            # Top-k filtering
            if top_k is not None:
                v, ix = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            next_id = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_id], dim=1)

            # Stop if EOS token is generated
            if next_id.item() == tokenizer.token_to_id[tokenizer.EOS]:
                break

    return tokenizer.decode(tokens[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Hello Sparrow")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    tokenizer = ByteLevelTokenizer()

    vocab_size = len(tokenizer.vocab)
    model = SparrowGPT(
        vocab_size=vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        hidden_dim=512,
        max_seq_len=128
    ).to(DEVICE)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("\n=== SparrowGPT-1 Output ===\n")
    print(output)
    print("\n===========================\n")


if __name__ == "__main__":
    main()
