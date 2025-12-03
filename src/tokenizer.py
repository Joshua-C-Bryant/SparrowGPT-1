# src/tokenizer.py

import json
from typing import List

class ByteLevelTokenizer:
    def __init__(self):
        # Special tokens
        self.PAD = "<pad>"
        self.BOS = "<bos>"
        self.EOS = "<eos>"

        # Byte-level vocabulary: 0–255
        self.byte_tokens = [i for i in range(256)]

        # Build vocabulary
        self.vocab = [self.PAD, self.BOS, self.EOS] + [chr(i) for i in range(256)]
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.vocab)}

    def encode(self, text: str, add_bos=True, add_eos=True) -> List[int]:
        """Convert text into a list of token IDs."""
        tokens = []

        if add_bos:
            tokens.append(self.token_to_id[self.BOS])

        for ch in text:
            byte_val = ord(ch)
            if byte_val < 256:
                tokens.append(self.token_to_id[chr(byte_val)])
            else:
                # Handle unicode by breaking into utf-8 bytes
                for b in ch.encode("utf-8"):
                    tokens.append(self.token_to_id[chr(b)])

        if add_eos:
            tokens.append(self.token_to_id[self.EOS])

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back into text."""
        chars = []
        for idx in token_ids:
            tok = self.id_to_token[idx]
            if tok in [self.PAD, self.BOS, self.EOS]:
                continue
            chars.append(tok)
        return "".join(chars)

    def save(self, path: str):
        """Save tokenizer vocabulary to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load tokenizer from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        tok = cls()
        tok.vocab = vocab
        tok.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        tok.id_to_token = {i: tok for i, tok in enumerate(vocab)}
        return tok


# Quick test when run as a script
if __name__ == "__main__":
    tok = ByteLevelTokenizer()
    text = "Hello Sparrow!"
    ids = tok.encode(text)
    print("Encoded:", ids)
    print("Decoded:", tok.decode(ids))
