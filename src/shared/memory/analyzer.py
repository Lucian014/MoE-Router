"""The BM25 analyzer (docs/rag-planv2.md 4.5): the same function at indexing (chunks.terms) and at
query time, so a term matches only if both sides produce it the same way.

No word lists. Which words carry no information (articles, prepositions, question words, in any
language) is learned from the data, not written down: BM25 takes its IDF from the whole indexed
corpus (retrieval.TermStats), where such words are everywhere and so weigh almost nothing. The
analyzer therefore keeps every token and leaves the weighting to the statistics.

    - lowercase, diacritics removed (ă -> a, ș/ş -> s), so text typed without diacritics matches
    - tokens are runs of letters and digits in any script: numbers of any length (2001, 4),
      Greek letters (λ, σ), Cyrillic, ... -- none is dropped for being short
    - LaTeX commands (\\frac, \\sum) dropped; the variables and words in formulas stay
    - identifiers kept whole AND split: create_access_token -> create_access_token, create,
      access, token; getUserName -> getusername, get, user, name. Exact names are where BM25
      beats embeddings
    - words stemmed with every Snowball stemmer in RAG_BM25_STEMMERS, one term per distinct
      stem: the same word gives the same terms whatever the language of the text around it, so
      the language never has to be detected (identifiers are not stemmed)

Changing any rule here changes ANALYZER_VERSION; the stemmers are part of version() too, so the
memory service recomputes the stored terms when either changes (chunking.index_version).
"""
import hashlib
import re
import unicodedata
from collections.abc import Sequence
from functools import lru_cache

ANALYZER_VERSION = "2"     # 2: any script, every token kept (IDF on the corpus), configurable stemmers

_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
_TOKEN = re.compile(r"\w+(?:\.\w+)*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def configured_stemmers() -> tuple[str, ...]:
    from shared.config import get_retrieval_settings

    return get_retrieval_settings().bm25_stemmer_list


@lru_cache(maxsize=8)
def _stemmers(languages: tuple[str, ...]):
    import snowballstemmer

    return tuple(snowballstemmer.stemmer(lang) for lang in languages)


def version(languages: Sequence[str] | None = None) -> str:
    """ANALYZER_VERSION plus a fingerprint of the stemmers: short enough for chunks.chunker_version."""
    languages = tuple(languages) if languages is not None else configured_stemmers()
    digest = hashlib.sha1(",".join(languages).encode()).hexdigest()[:6]
    return f"a{ANALYZER_VERSION}-{digest}"


def strip_diacritics(text: str) -> str:
    # the cedilla forms (ş ţ) and the comma forms (ș ț) both decompose to a base letter + a mark
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def analyze(text: str, languages: Sequence[str] | None = None) -> list[str]:
    """Terms of `text`, in order, with repetitions (BM25 needs the term frequencies).
    `languages`: the Snowball stemmers to apply; default RAG_BM25_STEMMERS."""
    stemmers = _stemmers(tuple(languages) if languages is not None else configured_stemmers())
    text = strip_diacritics(_LATEX_COMMAND.sub(" ", text))
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
            elif lower.isalpha():
                terms.extend(dict.fromkeys(s.stemWord(lower) for s in stemmers) or (lower,))
            else:
                terms.append(lower)
    return [t for t in terms if t.strip("_")]
