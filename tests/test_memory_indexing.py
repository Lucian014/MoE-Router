"""The indexing module (src/shared/memory/): segmentation, splitters, chunking, header, dense
query, BM25 analyzer, embedding client. Sizes are counted in words here (count=words) so the
tests don't depend on the reference tokenizer; MIN/MAX_TOKENS still apply."""
import asyncio
import json

import httpx
import pytest

from shared.memory import chunking
from shared.memory.analyzer import analyze, strip_diacritics
from shared.memory.chunking import chunk_text, chunk_turn, content_hash, embedding_input
from shared.memory.embedding import OllamaEmbeddingClient
from shared.memory.header import build_dense_query, extract_question, select_sentences, tail_tokens
from shared.memory.segment import Segment, segment
from shared.memory.splitters import SPLITTERS


def words(text):
    return len(text.split())


def kinds(text):
    return [s.kind for s in segment(text)]


FENCE = "```"

ANSWER = f"""Ai dreptate, compararea parolelor în clar nu e sigură. Folosește passlib cu bcrypt.

### Instalare
{FENCE}bash
pip install passlib[bcrypt]
{FENCE}

### Codul
{FENCE}python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)
{FENCE}

La înregistrare salvezi hash_password(parola), iar la login folosești verify_password.
"""

PROMPT_WITH_CODE = f"""am codul ăsta:
{FENCE}python
def login(user, password):
    return user.password == password
{FENCE}
nu e sigur, cum îl fac cu bcrypt?"""

TRACEBACK = """Traceback (most recent call last):
  File "main.py", line 42, in protected_route
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
jose.exceptions.ExpiredSignatureError: Signature has expired
"""


# ------------------------------------------------------------------ segmentation
@pytest.mark.parametrize("text", [ANSWER, PROMPT_WITH_CODE, TRACEBACK, "", "doar proză\n\nși încă un paragraf"])
def test_segments_cover_the_text_exactly(text):
    assert "".join(s.text for s in segment(text)) == text


def test_fenced_blocks_and_languages():
    segs = segment(ANSWER)
    assert [s.kind for s in segs] == ["prose", "code", "prose", "code", "prose"]
    assert [s.lang for s in segs if s.kind == "code"] == ["bash", "python"]


def test_fence_closes_only_with_the_same_character_and_length():
    text = "````md\n```python\nx = 1\n```\n````\nafter\n"
    segs = segment(text)
    assert segs[0].kind == "code" and segs[0].text == "````md\n```python\nx = 1\n```\n````\n"
    assert segs[1] == Segment("prose", "after\n")


def test_unclosed_fence_runs_to_the_end():               # an answer cut at max_tokens
    text = "Iată:\n~~~python\ndef f(\n    x"
    assert kinds(text) == ["prose", "code"]


def test_unfenced_log_and_code_are_found():
    assert kinds(TRACEBACK) == ["log"]
    text = "uite codul:\ndef f(x):\n    y = x + 1\n    return y\n\nce face?"
    assert kinds(text) == ["prose", "code", "prose"]


def test_two_code_looking_lines_stay_prose():
    assert kinds("Folosește f(x) = 2;\nsau g(y) = 3;\nAsta e tot.") == ["prose"]


@pytest.mark.parametrize("text, expected", [
    ("de ce nu merge?\nfunction add(a, b) {\n  const s = a + b;\n  return s;\n}\nmerci", ["prose", "code", "prose"]),
    ("uite:\nfunc main() {\n\tfmt.Println(\"hi\")\n\tos.Exit(1)\n}\n", ["prose", "code"]),
    ("query-ul:\nSELECT name, email\nFROM users\nORDER BY name\n", ["prose", "code"]),
    ("Exception in thread \"main\" java.lang.NullPointerException: x\n\tat a.Main.run(Main.java:42)\n"
     "\tat a.Main.main(Main.java:10)\n", ["log"]),
    ("TypeError: Cannot read properties of undefined\n    at render (/app/src/List.jsx:12:15)\n"
     "    at mount (/app/node_modules/react-dom/index.js:300:7)\n", ["log"]),
    ("2024-05-01 12:03:44 INFO start\n2024-05-01 12:03:45 WARN slow\n2024-05-01 12:03:47 ERROR down\n", ["log"]),
    ("  Warning  FailedScheduling  2m  default-scheduler  0/3 nodes\n  Normal   Pulled  1m  kubelet  pulled\n"
     "  Warning  BackOff  30s  kubelet  restarting\n", ["log"]),
    ("E       assert 422 == 200\nE        +  where 422 = <Response [422]>.status_code\n\n"
     "tests/test_auth.py:9: AssertionError\n", ["log"]),
])
def test_unfenced_code_and_logs_by_shape_in_any_language(text, expected):
    assert kinds(text) == expected


@pytest.mark.parametrize("text", [
    "Am încercat (de trei ori) să rulez scriptul.\nApoi (după restart) a mers.\nDar (din nou) a picat.\n",
    "Acesta e un paragraf lung care continuă\npe mai multe rânduri fără punct\nși tot proză rămâne\n",
    "Pași:\nPrimul: instalezi pachetul\nAl doilea: rulezi scriptul\nAl treilea: verifici\n",
    "API-ul răspunde lent.\nHTTP e protocolul folosit.\nJSON e formatul.\n",
    "Am trei opțiuni: prima e rapidă;\na doua e ieftină;\na treia e sigură;\n",
])
def test_prose_stays_prose(text):
    assert kinds(text) == ["prose"]


def test_bracket_math_and_numbered_list_follow_commonmark():
    assert kinds("Text:\n\\[\nx^2\n\\]\nmai departe") == ["prose", "math", "prose"]
    # a numbered list not starting at 1 doesn't interrupt a paragraph (a chat UI shows it as text)
    assert kinds("Primul paragraf\n3. al treilea pas\n") == ["prose"]
    assert kinds("Primul paragraf\n1. primul pas\n2. al doilea\n") == ["prose", "list"]


def test_math_table_and_list():
    text = ("Avem:\n$$\n\\int_0^3 x^2 dx = 9\n$$\n| a | b |\n|---|---|\n| 1 | 2 |\n"
            "- primul\n- al doilea\n  continuare\n\nfinal\n")
    assert kinds(text) == ["prose", "math", "table", "list", "prose"]


# ------------------------------------------------------------------ splitters
def test_code_is_split_on_top_level_definitions():
    funcs = "\n".join(f"@app.get('/r{i}')\ndef route_{i}(x):\n" + "    y = x\n" * 30 + "    return y\n"
                      for i in range(4))
    seg = Segment("code", f"{FENCE}python\nimport os\n\n{funcs}{FENCE}\n", "python")
    pieces = SPLITTERS["code"].split(seg, 80, words)
    assert "".join(pieces) == seg.text
    assert len(pieces) > 1
    assert all(p.count("def route_") <= 1 for p in pieces)       # never two routes cut apart
    assert all("@app.get" in p for p in pieces if "def route_" in p)  # decorator with its function


def test_any_language_splits_by_indentation_and_a_big_class_by_method():
    methods = "\n".join("  method%d(x) {\n" % i + "    let y = x;\n" * 30 + "    return y;\n  }\n" for i in range(3))
    js = f"{FENCE}js\nimport x from 'y';\n\nclass Repo {{\n{methods}}}\n\nexport default Repo;\n{FENCE}\n"
    pieces = SPLITTERS["code"].split(Segment("code", js, "js"), 110, words)
    assert "".join(pieces) == js
    assert len(pieces) >= 3
    assert all(p.count("method") <= 1 for p in pieces)                # a method is never cut apart


def test_code_without_definitions_is_split_on_blank_lines():
    seg = Segment("code", "def f(:\n" + "a b c\n" * 50 + "\n" + "d e f\n" * 50, "python")
    pieces = SPLITTERS["code"].split(seg, 160, words)
    assert "".join(pieces) == seg.text and len(pieces) == 2


def test_log_keeps_its_end_together():
    lines = "".join(f"  frame {i} in module\n" for i in range(60)) + "ValueError: boom happened here\n"
    pieces = SPLITTERS["log"].split(Segment("log", lines), 50, words)
    assert "".join(pieces) == lines
    assert pieces[-1].endswith("ValueError: boom happened here\n") and words(pieces[-1]) > 40


def test_list_is_cut_between_items():
    text = "".join(f"- item {i} " + "w " * 20 + "\n" for i in range(10))
    pieces = SPLITTERS["list"].split(Segment("list", text), 60, words)
    assert "".join(pieces) == text
    assert all(p.startswith("- item") for p in pieces)


def test_formulas_and_tables_are_atomic():
    seg = Segment("math", "$$\n" + "x + " * 500 + "1\n$$\n")
    assert SPLITTERS["math"].split(seg, 50, words) == [seg.text]


def test_prose_split_by_headings_then_paragraphs():
    text = "# A\n" + "a " * 300 + "\n\n" + "b " * 300 + "\n# B\n" + "c " * 100
    pieces = SPLITTERS["prose"].split(Segment("prose", text), 400, words)
    assert "".join(pieces) == text
    assert all(words(p) <= 400 for p in pieces)


# ------------------------------------------------------------------ chunking
def test_answer_chunks_join_back_to_the_answer():
    long_answer = ANSWER + "\n" + "Explicație lungă. " * 400 + "\n" + ANSWER
    for answer in (ANSWER, long_answer):
        chunks = chunk_turn("q?", answer, count=words)
        assert "".join(c.content for c in chunks if c.source == "response") == answer
        assert [c.ord for c in chunks] == list(range(len(chunks)))


def test_small_pieces_merge_and_take_the_code_kind():
    text = f"Iată funcția:\n{FENCE}python\n" + "x = 1\n" * 60 + f"{FENCE}\n"
    ((kind, lang, content),) = chunk_text(text, words)
    assert (kind, lang) == ("code", "python") and content == text


def test_short_prompt_prose_is_the_header_not_a_chunk():
    chunks = chunk_turn(PROMPT_WITH_CODE, ANSWER, count=words)
    prompt_chunks = [c for c in chunks if c.source == "prompt"]
    assert [c.kind for c in prompt_chunks] == ["code"]
    assert "nu e sigur" not in prompt_chunks[0].content


def test_long_pasted_article_is_chunked():
    article = "Rezumă te rog articolul următor. " + "Propoziție din articol despre economie. " * 80
    chunks = chunk_turn(article, "Pe scurt, articolul spune că...", count=words)
    prompt = [c for c in chunks if c.source == "prompt"]
    assert prompt and "".join(c.content for c in prompt) == article


def test_content_hash_ignores_whitespace_noise():
    assert content_hash("x = 1  \r\n\n\n\ny = 2\n") == content_hash("x = 1\n\ny = 2")
    assert content_hash("x = 1") != content_hash("x = 2")


def test_chunker_constants():
    assert chunking.MIN_TOKENS < chunking.MAX_TOKENS and chunking.CHUNKER_VERSION


def test_embedding_input_puts_the_header_first():
    assert embedding_input("cum verific parola?", "def f(): ...") == "Question: cum verific parola?\ndef f(): ..."
    assert embedding_input("", "x") == "x"


# ------------------------------------------------------------------ header and dense query
def test_header_is_the_whole_short_prose():
    assert extract_question(PROMPT_WITH_CODE, ANSWER, words) == \
        ("am codul ăsta: nu e sigur, cum îl fac cu bcrypt?", False)


def test_header_from_the_answer_when_the_prompt_is_only_a_log():
    header, from_answer = extract_question(TRACEBACK, "Eroarea apare pentru că tokenul a expirat. Soluția...", words)
    assert (header, from_answer) == ("Eroarea apare pentru că tokenul a expirat.", True)


def test_long_prose_keeps_questions_then_the_end():
    filler = " ".join(f"Context {i} despre proiect." for i in range(60))
    prose = f"{filler} Cum adaug refresh token? Vreau să expire în 7 zile."
    header = select_sentences(prose, 20, words)
    assert header.startswith("Cum adaug refresh token?") is False      # original order kept
    assert "Cum adaug refresh token?" in header and header.endswith("Vreau să expire în 7 zile.")
    assert words(header) <= 20


def test_dense_query_is_the_question_next_to_a_traceback():
    assert build_dense_query(TRACEBACK + "\nde ce primesc eroarea asta?", count=words) == "de ce primesc eroarea asta?"


def test_dense_query_on_a_log_only_prompt_keeps_its_end():
    long_tb = "Traceback (most recent call last):\n" + "".join(
        f'  File "lib{i}.py", line {i}, in f{i}\n    call_{i}()\n' for i in range(200)) + TRACEBACK
    assert build_dense_query(TRACEBACK, 512, words) == TRACEBACK.strip()
    query = build_dense_query(long_tb, 40, words)
    assert words(query) <= 40 and query.endswith("ExpiredSignatureError: Signature has expired")


def test_tail_tokens():
    assert tail_tokens("a b c", 5, words) == "a b c"
    assert tail_tokens("line one\nline two three", 3, words) == "line two three"


# ------------------------------------------------------------------ analyzer
def test_analyzer_splits_identifiers_and_stems_words():
    terms = analyze("jwt.decode create_access_token ExpiredSignatureError getUserName")
    for t in ["jwt", "create_access_token", "create", "access", "token",
              "expiredsignatureerror", "expired", "signature", "error", "getusername", "user", "name"]:
        assert t in terms


def test_analyzer_matches_with_and_without_diacritics():
    assert analyze("funcția sortează") == analyze("functia sorteaza")
    assert strip_diacritics("ăâîșțşţ") == "aaistst"            # comma and cedilla forms alike


def test_analyzer_drops_latex_commands():
    terms = analyze(r"\frac{a}{b} + \sum x_i")
    assert "frac" not in terms and "sum" not in terms and "x_i" in terms


def test_analyzer_keeps_term_frequencies():
    assert analyze("token token token").count(analyze("token")[0]) == 3


# ------------------------------------------------------------------ embedding client
def test_ollama_embedding_client_batches_and_checks_the_count():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]] * len(body["input"])})

    client = OllamaEmbeddingClient("http://ollama:11434", transport=httpx.MockTransport(handler))
    assert asyncio.run(client.embed(["a", "b"])) == [[0.1, 0.2], [0.1, 0.2]]
    assert seen == [{"model": "bge-m3", "input": ["a", "b"], "truncate": False}]
    assert asyncio.run(client.embed([])) == []

    broken = OllamaEmbeddingClient("http://o", transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"embeddings": [[0.1]]})))
    with pytest.raises(RuntimeError, match="asked for 2"):
        asyncio.run(broken.embed(["a", "b"]))
