
QID = "complexity"
KIND = "score"

QUESTION = {
    "type": "score",
    "instructions": "How demanding is this request for an AI assistant to answer well?",
    "criteria": [
        "low: a simple request: one fact, a short answer, or a small well-defined task",
        "high: a demanding request: expert knowledge, multi-step reasoning, or long structured output",
    ],
}

QUESTIONS = {QID: QUESTION}
LABELS = ["low", "high"]