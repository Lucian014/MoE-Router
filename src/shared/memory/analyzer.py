"""The BM25 analyzer (docs/rag-planv2.md 4.5): the same function at indexing (chunks.terms) and at
query time, so a term matches only if both sides produce it the same way.

    - lowercase, diacritics removed (ă -> a, ș/ş -> s), so text typed without diacritics matches
    - LaTeX commands (\\frac, \\sum) dropped; the variables and words in formulas stay
    - identifiers kept whole AND split: create_access_token -> create_access_token, create,
      access, token; getUserName -> getusername, get, user, name. Exact names are where BM25
      beats embeddings
    - words stemmed with the Romanian Snowball stemmer (identifiers are not stemmed)
"""
import re
import unicodedata
from functools import lru_cache

_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
_TOKEN = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
MIN_TERM_LENGTH = 2


@lru_cache(maxsize=1)
def _stemmer():
    import snowballstemmer

    return snowballstemmer.stemmer("romanian")


def strip_diacritics(text: str) -> str:
    # the cedilla forms (ş ţ) and the comma forms (ș ț) both decompose to a base letter + a mark
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def analyze(text: str) -> list[str]:
    """Terms of `text`, in order, with repetitions (BM25 needs the term frequencies)."""
    text = strip_diacritics(_LATEX_COMMAND.sub(" ", text))
    stem = _stemmer().stemWord
    terms: list[str] = []
    for token in _TOKEN.findall(text):
        for name in token.split("."):                     # jwt.decode -> jwt, decode
            if not name:
                continue
            lower = name.lower()
            parts = [p.lower() for p in re.split(r"_+", name) for p in _CAMEL.findall(p)]
            is_identifier = "_" in name or any(c.isdigit() for c in name) or len(parts) > 1
            if is_identifier:
                terms.append(lower)
                terms.extend(p for p in parts if p != lower)
            else:
                terms.append(stem(lower) if lower.isalpha() else lower)
    return [t for t in terms if len(t) >= MIN_TERM_LENGTH]
