"""
Grow the dataset: write the user's side of a conversation, let a large model write the answers.

    # scenarios/example.yaml: same format as data/sessions/, turns with `user` only
    python scripts/rag_eval/generate.py scripts/rag_eval/scenarios/example.yaml \\
        --base http://localhost:11434 --model qwen2.5:7b

Every turn without an `assistant` gets one, generated with the whole conversation so far as
context (any OpenAI-compatible server: Ollama, vLLM, an API). The result goes to data/sessions/
under the same file name; existing answers are kept, so a half-finished file can be resumed.

Then, by hand (docs/rag-planv2.md 6): read every answer, fix what is wrong, and add the
`questions` -- about OLD turns, with the gold as a quote copied from the text. Check with
`python scripts/rag_eval/dataset.py`. Long conversations (15-25 turns, answers of a few hundred
tokens) are what make the small model need retrieval.
"""
import argparse
import sys
from pathlib import Path

import httpx
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset import DATA_DIR

SYSTEM = ("You are a helpful assistant. Answer in the language of the user's message. Put any code in "
          "fenced blocks with the language name. Answer thoroughly, as a strong assistant would.")


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


def complete(base: str, model: str, messages: list[dict], timeout: float) -> str:
    r = httpx.post(f"{base}/v1/chat/completions", timeout=timeout,
                   json={"model": model, "messages": messages, "temperature": 0.3})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", type=Path)
    ap.add_argument("--base", default="http://localhost:11434", help="OpenAI-compatible server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--timeout", type=float, default=600)
    args = ap.parse_args()

    out = DATA_DIR / args.scenario.name
    raw = yaml.safe_load((out if out.exists() else args.scenario).read_text(encoding="utf-8"))
    messages = [{"role": "system", "content": SYSTEM}]
    for n, turn in enumerate(raw["turns"], start=1):
        messages.append({"role": "user", "content": turn["user"]})
        if not turn.get("assistant"):
            print(f"[*] turn {n}/{len(raw['turns'])}")
            turn["assistant"] = complete(args.base, args.model, messages, args.timeout)
            out.write_text(yaml.dump(_blocks(raw), allow_unicode=True, sort_keys=False, width=1000),
                           encoding="utf-8")
        messages.append({"role": "assistant", "content": turn["assistant"]})
    raw.setdefault("questions", [])
    out.write_text(yaml.dump(_blocks(raw), allow_unicode=True, sort_keys=False, width=1000), encoding="utf-8")
    print(f"[OK] {out} -- now review the answers and add the questions")


if __name__ == "__main__":
    main()
