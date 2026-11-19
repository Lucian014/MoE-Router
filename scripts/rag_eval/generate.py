"""
Grow the dataset: write the user's side of a conversation, let a large model write the answers.

    # scenarios/example.yaml: same format as data/sessions/, turns with `user` only
    python scripts/rag_eval/generate.py scripts/rag_eval/scenarios/example.yaml \\
        --base http://localhost:11434 --model qwen2.5:7b
    python scripts/rag_eval/generate.py scripts/rag_eval/data/sessions/s01-fastapi-jwt.yaml --glm

Every turn without an `assistant` gets one, generated with the whole conversation so far as
context (any OpenAI-compatible server: Ollama, vLLM, an API). The result goes to data/sessions/
under the same file name; existing answers are kept, so a half-finished file can be resumed.

Then, by hand (docs/rag-planv2.md 6): read every answer, fix what is wrong, and add the
`questions` -- about OLD turns, with the gold as a quote copied from the text. Check with
`python scripts/rag_eval/dataset.py`. Long conversations (15-25 turns, answers of a few hundred
tokens) are what make the small model need retrieval.

The whole conversation goes to the model on every turn, so it has to fit its context: a turn
whose input + --max-tokens would exceed --max-context stops the run (the turns so far are saved)
instead of failing on the server. --history-tokens N sends only the most recent turns that fit in
N tokens (plus the system prompt and the new message): for very long sessions, where the full
history would make every turn cost the whole conversation again. Tokens are counted with the platform's reference tokenizer
(Qwen2.5), the generator's own if it is a Qwen.

An answer that was cut at --max-tokens, or that degenerated (a chat role written as text, CJK
characters, the same long line of prose over and over, the ending of the two previous answers
repeated), is not
saved: the turn is retried (--retries, warmer, asked to be shorter if it was cut), then the run
stops there. --regenerate-from N drops the answers from turn N on, to redo a session from there.

A scenario may carry keys of its own (`question_plan`, a turn's `domain`): they are kept.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import httpx
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset import DATA_DIR
from shared.memory.tokens import count_tokens

MESSAGE_OVERHEAD = 8          # chat-template tokens per message, roughly

SYSTEM = ("You are a helpful, expert assistant in a chat. Answer in the language of the user's message. "
          "Answer what was asked, as a strong assistant would in a real conversation: usually 100-400 words, "
          "longer only when the user asks for a long text (a chapter, a full letter, the translation of a "
          "long document). When the user asks to change code or text you wrote earlier, show only what "
          "changes, unless they ask for the whole thing again. Put code in fenced blocks with the language "
          "name, and use the current API of the libraries the user names.")
SHORTER = " Your previous attempt at this answer was too long: keep this one well under 300 words."
TEMPERATURES = (0.3, 0.5, 0.7)          # first try, then each retry

_ROLE_LEAK = re.compile(r"(?m)^\s*(user|assistant|system)\s*$|<\|im_(start|end)\|>")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
_FENCED = re.compile(r"```.*?(```|$)|\$\$.*?\$\$|\\\[.*?\\\]", re.S)     # code and display math


class _Literal(str):
    pass


def _literal(dumper, value):
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|")


yaml.add_representer(_Literal, _literal)


def _blocks(obj):
    """Multi-line strings as `|` blocks, so the file stays readable."""
    if isinstance(obj, dict):
        return {k: _blocks(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_blocks(v) for v in obj]
    if isinstance(obj, str) and "\n" in obj:
        return _Literal(obj if obj.endswith("\n") else obj + "\n")
    return obj


def input_tokens(messages: list[dict]) -> int:
    return sum(count_tokens(m["content"]) + MESSAGE_OVERHEAD for m in messages)


def recent_window(messages: list[dict], budget: int) -> list[dict]:
    """The system prompt, the newest user message and the most recent earlier (user, assistant)
    pairs whose tokens fit in `budget`, oldest first; budget <= 0 = everything."""
    if budget <= 0:
        return messages
    system, history, new = messages[0], messages[1:-1], messages[-1]
    kept, used = [], 0
    for i in range(len(history) - 2, -1, -2):              # pairs, newest first
        pair = history[i:i + 2]
        cost = input_tokens(pair)
        if used + cost > budget:
            break
        kept[:0] = pair
        used += cost
    return [system, *kept, new]


_THINK = re.compile(r"<think>.*?</think>", re.S)


def complete(base: str, model: str, messages: list[dict], timeout: float, max_tokens: int,
             temperature: float, api_key: str = "", verify: bool = True) -> tuple[str, str]:
    """`base` with or without the trailing /v1."""
    url = base.rstrip("/").removesuffix("/v1") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = httpx.post(url, timeout=timeout, headers=headers, verify=verify,
                   json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens})
    r.raise_for_status()
    choice = r.json()["choices"][0]
    text = _THINK.sub("", choice["message"]["content"] or "")
    return text.strip() + "\n", choice.get("finish_reason") or ""


def _last_paragraph(text: str) -> str:
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    return paragraphs[-1] if paragraphs else ""


def problem(answer: str, finish_reason: str, previous: list[str]) -> str | None:
    """Why the answer can't be kept, or None. `previous`: the answers before it, oldest first."""
    if finish_reason == "length":
        return "cut at --max-tokens"
    if not answer.strip():
        return "empty"
    if _ROLE_LEAK.search(answer):
        return "a chat role written as text"
    if _CJK.search(answer):
        return "CJK characters"
    prose = _FENCED.sub("", answer)
    long_lines = Counter(line.strip() for line in prose.splitlines() if len(line.strip()) > 40)
    if long_lines and max(long_lines.values()) >= 3:
        return "the same line of prose 3+ times"
    ending = _last_paragraph(answer)
    if len(previous) >= 2 and len(ending) > 60 and all(ending == _last_paragraph(p) for p in previous[-2:]):
        return "the ending of the two previous answers, repeated"
    return None


def generate_turn(args, messages: list[dict], previous: list[str]) -> tuple[str | None, str]:
    """(answer, "") or (None, the last attempt's problem)."""
    why = ""
    for attempt, temperature in enumerate(TEMPERATURES[:args.retries + 1]):
        msgs = messages
        if why == "cut at --max-tokens":
            msgs = [{**messages[0], "content": messages[0]["content"] + SHORTER}] + messages[1:]
        answer, finish = complete(args.base, args.model, msgs, args.timeout, args.max_tokens, temperature,
                                  args.api_key, not args.insecure)
        why = problem(answer, finish, previous)
        if why is None:
            return answer, ""
        print(f"    [!] attempt {attempt + 1}: {why}")
    return None, why


def save(out: Path, raw: dict) -> None:
    out.write_text(yaml.dump(_blocks(raw), allow_unicode=True, sort_keys=False, width=1000), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", type=Path)
    ap.add_argument("--base", default="http://localhost:11434", help="OpenAI-compatible server")
    ap.add_argument("--model")
    ap.add_argument("--api-key", default="", help="sent as a Bearer token")
    ap.add_argument("--insecure", action="store_true", help="skip TLS certificate checks")
    ap.add_argument("--glm", action="store_true",
                    help="the GLM judge's server from scripts/rag_eval/.env (GLM_BASE_URL, GLM_API_KEY, GLM_MODEL, "
                         "GLM_INSECURE) instead of --base/--model/--api-key")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--max-tokens", type=int, default=1500, help="per answer")
    ap.add_argument("--max-context", type=int, default=32768, help="the server's --max-model-len")
    ap.add_argument("--history-tokens", type=int, default=0,
                    help="send only the most recent turns that fit in N tokens (0 = the whole conversation)")
    ap.add_argument("--retries", type=int, default=2, choices=range(len(TEMPERATURES)))
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR, help="where the session goes (data/multihop/ for that set)")
    ap.add_argument("--regenerate-from", type=int, metavar="N", help="drop the answers from turn N on first")
    args = ap.parse_args()
    if args.glm:
        from judge_retrieval import GLM_API_KEY, GLM_BASE_URL, GLM_INSECURE, GLM_MODEL
        args.base, args.model, args.api_key = GLM_BASE_URL, GLM_MODEL, GLM_API_KEY
        args.insecure = args.insecure or GLM_INSECURE
    if not args.model:
        ap.error("--model is required (or --glm)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / args.scenario.name
    raw = yaml.safe_load((out if out.exists() else args.scenario).read_text(encoding="utf-8"))
    if args.regenerate_from:
        for turn in raw["turns"][args.regenerate_from - 1:]:
            turn.pop("assistant", None)
    messages = [{"role": "system", "content": SYSTEM}]
    previous: list[str] = []
    for n, turn in enumerate(raw["turns"], start=1):
        messages.append({"role": "user", "content": turn["user"]})
        if not turn.get("assistant"):
            window = recent_window(messages, args.history_tokens)
            used = input_tokens(window)
            if used + args.max_tokens > args.max_context:
                save(out, raw)
                raise SystemExit(f"[!] {out.name}: turn {n} needs {used} + {args.max_tokens} tokens, over "
                                 f"--max-context {args.max_context}; shorten the scenario (turns 1-{n - 1} saved)")
            print(f"[*] {out.name}: turn {n}/{len(raw['turns'])} ({used} tokens in)")
            answer, why = generate_turn(args, window, previous)
            if answer is None:
                save(out, raw)
                raise SystemExit(f"[!] {out.name}: turn {n} failed {args.retries + 1} times ({why}); "
                                 f"turns 1-{n - 1} saved, rerun to retry")
            turn["assistant"] = answer
            save(out, raw)
        messages.append({"role": "assistant", "content": turn["assistant"]})
        previous.append(turn["assistant"])
    raw.setdefault("questions", [])
    save(out, raw)
    print(f"[OK] {out} -- now review the answers and add the questions")


if __name__ == "__main__":
    main()
