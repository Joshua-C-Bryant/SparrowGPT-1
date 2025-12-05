import torch
import torch.nn as nn
from torch.optim import AdamW
from tokenizer import ByteLevelTokenizer
from model import SparrowGPT
from tqdm import tqdm
import os

# ------------------------------
# 1. Training Configuration
# ------------------------------
BATCH_SIZE = 32
SEQ_LEN = 128
LR = 3e-4
EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "data/tiny_shakespeare.txt"
CHECKPOINT_PATH = "sparrowgpt1.pt"


# ------------------------------
# 2. Load Data
# ------------------------------
def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ------------------------------
# 3. Create Batches
# ------------------------------
def get_batch(data_ids):
    # Random starting points
    ix = torch.randint(0, len(data_ids) - SEQ_LEN - 1, (BATCH_SIZE,))
    x = torch.stack([data_ids[i:i + SEQ_LEN] for i in ix])
    y = torch.stack([data_ids[i+1:i + SEQ_LEN + 1] for i in ix])

    return x.to(DEVICE), y.to(DEVICE)


# ------------------------------
# 4. Training Loop
# ------------------------------
def train():
    print(f"Using device: {DEVICE}")

    # Load tokenizer
    tokenizer = ByteLevelTokenizer()

    # Load text
    raw_text = load_text(DATA_PATH)
    print("Loaded dataset, length:", len(raw_text))

    # Encode dataset into integers
    data_ids = torch.tensor(tokenizer.encode(raw_text, add_bos=False, add_eos=False), dtype=torch.long)

    # Define model
    model = SparrowGPT(
        vocab_size=len(tokenizer.vocab),
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        hidden_dim=512,
        max_seq_len=SEQ_LEN
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")

    for epoch in range(EPOCHS):
        pbar = tqdm(range(1000), desc=f"Epoch {epoch+1}/{EPOCHS}")
        for _ in pbar:
            x, y = get_batch(data_ids)

            logits = model(x)                    # forward pass
            logits = logits.view(-1, logits.size(-1))  # flatten batch
            y = y.view(-1)                       # flatten targets

            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})

        # Save checkpoint after each epoch
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"Saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
