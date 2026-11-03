"""What the classifiers are given (src/router/classifier_input.py, classifier.py): a prompt that fits
as it is, a longer one as its request + the start of the document. A word count stands in for the
classifier's tokenizer."""
from router import classifier, domain_schema
from router.classifier_input import GAP, classifier_text, edges, head, head_tail, tail


def words(text: str) -> int:
    return len(text.split())


DOC = "\n\n".join(" ".join(f"w{p}_{i}" for i in range(80)) for p in range(10))    # 10 paragraphs x 80 words
REQUEST = "Rezumă textul de mai sus în 5 puncte."


def test_head_and_tail_fit_the_budget_and_keep_their_end():
    assert words(head(DOC, 50, words)) <= 50 and head(DOC, 50, words).startswith("w0_0")
    assert words(tail(DOC, 50, words)) <= 50 and tail(DOC, 50, words).endswith("w9_79")
    assert head("short", 50, words) == tail("short", 50, words) == "short"


def test_head_tail_keeps_both_ends():
    text = head_tail(DOC, 60, words)
    assert words(text) <= 60 and GAP in text
    assert text.startswith("w0_0") and text.endswith("w9_79")


def test_edges_puts_the_request_first_wherever_it_was():
    for prompt in (f"{REQUEST}\n\n{DOC}", f"{DOC}\n\n{REQUEST}"):
        text = edges(prompt, 60, words)
        assert text.startswith(REQUEST) and words(text) <= 60
        assert "w0_0" in text                       # then the start of the document


def test_edges_without_a_short_paragraph_falls_back_to_head_tail():
    assert edges(DOC, 60, words) == head_tail(DOC, 60, words)


def test_classifier_text_says_how_the_input_was_made():
    assert classifier_text("Rezumă textul.\n\nUn text scurt.", 60, words) == ("Rezumă textul.\n\nUn text scurt.", "full")
    text, how = classifier_text(f"{DOC}\n\n{REQUEST}", 60, words)
    assert how == "edges" and text.startswith(REQUEST)
    assert classifier_text(f"{DOC}\n\n{REQUEST}", 60, words, strategy="raw") == (f"{DOC}\n\n{REQUEST}", "cut")


class FakeAgent:
    """laya.Agent's surface the classifier uses: cfg, tok, predict."""
    cfg = {"max_len": 200, "head_max_len": 100}                # -> a prompt budget of 92 tokens

    def __init__(self):
        self.seen = []

    def tok(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def predict(self, text, questions):
        self.seen.append(text)
        probs = {label: 0.0 for label in domain_schema.LABELS} | {"summarization": 1.0}
        return {"answers": {domain_schema.QID: {"probabilities": probs}}}


def fake_classifier(strategy: str):
    c = object.__new__(classifier._LayaClassifier)         # no checkpoint to load
    c.schema, c.input_strategy, c.agent = domain_schema, strategy, FakeAgent()
    return c


def test_the_domain_classifier_reads_the_request_of_a_long_prompt(monkeypatch):
    fake = fake_classifier("edges")
    monkeypatch.setattr(classifier, "_DOMAIN", fake)
    assert fake.budget == 92

    short = classifier.predict_domain("Rezumă textul acesta.")
    long = classifier.predict_domain(f"{DOC}\n\n{REQUEST}")
    assert (short.input, long.input) == ("full", "edges")
    assert fake.agent.seen[0] == "Rezumă textul acesta."
    assert fake.agent.seen[1].startswith(REQUEST) and words(fake.agent.seen[1]) <= 92
    assert long.category == "summarization"


def test_raw_gives_the_long_prompt_as_it_is(monkeypatch):
    fake = fake_classifier("raw")
    monkeypatch.setattr(classifier, "_DOMAIN", fake)
    assert classifier.predict_domain(f"{DOC}\n\n{REQUEST}").input == "cut"
    assert fake.agent.seen == [f"{DOC}\n\n{REQUEST}"]
