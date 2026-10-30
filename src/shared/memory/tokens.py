"""count_tokens: one reference tokenizer for budgets, chunk sizes and the `tokens` columns.

The models in the pool have different tokenizers, so every count is an estimate; the budget keeps a
margin for that (docs/rag-planv2.md 4.7). The reference is Qwen2.5's tokenizer (public, multilingual,
loaded with the light `tokenizers` library). The router image downloads it at build time
(TOKENIZER_FILE); where it can't be loaded a conservative character estimate is used instead.
"""
import math
import os
from functools import lru_cache

REFERENCE_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"

# Romanian prose measured ~3.4 characters per Qwen token and code ~4: dividing by 3 overestimates,
# which is the safe direction for a budget
_CHARS_PER_TOKEN_FALLBACK = 3.0

# role markers and separators the chat template adds around every message
MESSAGE_OVERHEAD_TOKENS = 4


@lru_cache(maxsize=1)
def _tokenizer():
    try:
        from tokenizers import Tokenizer

        # images ship the file (TOKENIZER_FILE) and run offline; elsewhere it's fetched once
        path = os.environ.get("TOKENIZER_FILE")
        if path and os.path.exists(path):
            return Tokenizer.from_file(path)
        return Tokenizer.from_pretrained(REFERENCE_TOKENIZER)
    except Exception as e:  # no `tokenizers`, no network and no cache, ...
        print(f"[warn] reference tokenizer unavailable ({type(e).__name__}: {e}); "
              f"estimating {_CHARS_PER_TOKEN_FALLBACK} characters per token")
        return None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    tok = _tokenizer()
    if tok is None:
        return math.ceil(len(text) / _CHARS_PER_TOKEN_FALLBACK)
    return len(tok.encode(text, add_special_tokens=False).ids)


def count_message_tokens(content: str) -> int:
    return count_tokens(content) + MESSAGE_OVERHEAD_TOKENS
