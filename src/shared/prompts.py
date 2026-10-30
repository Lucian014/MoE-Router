"""Prompts shared by the services that build model input (the worker now, the router's context
assembly later, docs/rag-planv2.md 4.7)."""

# In English: the small models follow English instructions more reliably. Without it, a prompt
# in Romanian can get an answer in English or Chinese. The code-block rule lets the chunker
# find code in the answers (docs/rag-planv2.md 4.3).
SYSTEM_PROMPT = (
    "You are a helpful assistant. Always answer in the same language as the user's last message. "
    "Put any code in fenced Markdown code blocks that name the language, for example ```python."
)


def single_turn_messages(prompt: str) -> list[dict[str, str]]:
    """The messages for a prompt without conversation context."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
