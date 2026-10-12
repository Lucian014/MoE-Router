
QID = "domain"
KIND = "choice"

QUESTION = {
    "type": "choice",
    "instructions": "What kind of task is the user asking the assistant to do?",
    "criteria": {
        "coding": "write, debug, explain or review code, SQL queries or database schemas",
        "text-rewrite": "draft or rewrite an email, message or letter, changing its tone or formality",
        "summarization": "condense a longer text into a summary or its key points",
        "translation": "translate text, say how to express something in another language, or explain a foreign word or idiom",
        "creative-writing": "invent fiction: a story, poem, scene, dialogue, myth or song lyrics",
        "open-qa": "answer a factual or general-knowledge question that has one correct answer",
        "math": "solve a math, statistics or logic problem: calculate, prove, or explain a math concept",
        "brainstorming": "suggest ideas, options, recommendations or advice with many valid answers",
    },
}

QUESTIONS = {QID: QUESTION}
LABELS = list(QUESTION["criteria"].keys())