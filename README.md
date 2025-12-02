🐦 SparrowGPT-1

A tiny Transformer-based GPT model built from scratch in Python & PyTorch.
The first member of the SparrowGPT flock — lightweight, deployable, educational, and fully open-source.

🚀 Overview

SparrowGPT-1 is a minimal yet functional GPT-style language model, designed to:

Teach the fundamentals of tokenization

Demonstrate how Transformers work under the hood

Provide a clean, well-structured codebase

Run on CPU or low-end GPUs

Serve as the foundation for future models in the SparrowGPT flock

This project is inspired by NanoGPT, MinGPT, and educational ML research code — but written entirely from scratch with readability and learning in mind.

🧠 Features

🔡 Custom byte-level tokenizer (built from scratch)

🧱 Full Transformer decoder architecture

🔥 Causal self-attention

📚 Trainable on TinyShakespeare or any text file

🛠️ Clean modular code (src/ folder)

🧪 Unit tests included (tests/)

🖥️ Runs on CPU or GPU

🌱 Foundation for SparrowGPT-2, SparrowGPT-Chat, and the future “Flock”

📁 Project Structure
SparrowGPT-1/
│
├── src/
│   ├── tokenizer.py      # Byte-level tokenizer (from scratch)
│   ├── model.py          # Transformer model implementation
│   ├── train.py          # Training loop
│   └── generate.py       # Text generation script
│
├── data/
│   └── tiny_shakespeare.txt   # (Added during tutorial)
│
├── notebooks/
│   └── exploration.ipynb
│
├── tests/
│   └── test_model.py
│
├── requirements.txt
├── README.md
└── .gitignore

📦 Installation
git clone https://github.com/Joshua-C-Bryant/SparrowGPT-1.git
cd SparrowGPT-1
pip install -r requirements.txt

🏋️ Training
python src/train.py --data data/tiny_shakespeare.txt

✨ Generate Text
python src/generate.py --prompt "To be or not to be"

🪶 Roadmap (The SparrowGPT “Flock”)

SparrowGPT-1 — tiny LLM from scratch

SparrowGPT-Chat — conversational fine-tuned model

SparrowGPT-Vision — multimodal variant

SparrowGPT-Forge — RAG + tools

SparrowGPT-Flock — multiple cooperating small models (agents)

📜 License

MIT — free to use, modify, and build upon.
