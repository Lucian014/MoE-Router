#!/usr/bin/env python3
"""
label_categories_llm.py -- eticheteaza fiecare prompt dintr-un JSON de pe
Hugging Face cu una din categoriile din categories.yaml, folosind un LLM
prin tool calling.

LLM-ul primeste promptul si TREBUIE sa apeleze tool-ul `classify_prompt`,
al carui argument `category` e un enum cu numele din categories.yaml.
Descrierile si exemplele categoriilor sunt puse in system prompt.

Merge cu orice endpoint compatibil OpenAI:
  - Ollama:  --base-url http://localhost:11434/v1  --model qwen2.5:7b
  - vLLM:    --base-url http://localhost:8000/v1   --model <model>
  - OpenAI:  --base-url https://api.openai.com/v1  --model gpt-4o-mini
             (cheia din OPENAI_API_KEY sau --api-key)
  - Gemini:  --base-url https://generativelanguage.googleapis.com/v1beta/openai/
             --model gemini-3.1-flash-lite --api-key $GEMINI_API_KEY --tool-choice required

Intrare: JSON array sau JSONL. Iesire: acelasi format, fiecare obiect cu
campurile noi `category` (si `category_reason` daca folosesti --with-reason).

Progresul se salveaza pe parcurs in <output>.partial.jsonl; daca scriptul
se opreste, il rulezi din nou cu aceeasi comanda si continua de unde a
ramas.

Exemple:
    python label_categories_llm.py prompts.json --model qwen2.5:7b
    python label_categories_llm.py prompts.json --model qwen2.5:7b --allow-other --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from openai import OpenAI

OTHER = "other"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_records(path: Path) -> tuple[list[dict], str]:
    """Accepta JSON array, JSONL strict, dar si obiecte JSON concatenate /
    formatate pe mai multe linii, cu sau fara virgula intre ele, inclusiv
    newline-uri brute in interiorul string-urilor (strict=False)."""
    text = path.read_text(encoding="utf-8-sig").strip()
    dec = json.JSONDecoder(strict=False)

    if text.startswith("["):
        return dec.decode(text), "json"

    records, pos, n = [], 0, len(text)
    while pos < n:
        while pos < n and (text[pos].isspace() or text[pos] == ","):
            pos += 1
        if pos >= n:
            break
        try:
            obj, pos = dec.raw_decode(text, pos)
        except json.JSONDecodeError as e:
            line = text.count("\n", 0, e.pos) + 1
            sys.exit(f"[!] JSON invalid in '{path}' la linia {line}, dupa {len(records)} "
                     f"obiecte citite corect: {e.msg}")
        if isinstance(obj, list):
            records.extend(obj)
        else:
            records.append(obj)
    return records, "jsonl"


def save_records(records: list[dict], path: Path, fmt: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        if fmt == "json":
            json.dump(records, f, ensure_ascii=False, indent=2)
        else:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_partial(path: Path) -> dict[int, dict]:
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["idx"]] = row
    return done


# ---------------------------------------------------------------------------
# Prompt + tool
# ---------------------------------------------------------------------------

def build_system_prompt(cats: list[dict], allow_other: bool) -> str:
    parts = [
        "You are a classifier for a prompt-routing system. You will receive a user "
        "prompt (it may be in English, Romanian or other languages). Decide which "
        "category best describes the TASK the user wants done, then call the "
        "`classify_prompt` tool. Do not answer or execute the prompt itself.",
        "",
        "Categories:",
    ]
    for c in cats:
        parts.append(f"\n## {c['name']}\n{' '.join(c['description'].split())}")
        exs = c.get("examples", [])[:6]
        if exs:
            parts.append("Examples:\n" + "\n".join(f"- {e}" for e in exs))
    if allow_other:
        parts.append(
            f"\n## {OTHER}\nUse ONLY if the prompt clearly fits none of the categories above."
        )
    parts.append(
        "\nRules: judge by the actual task, not by surrounding boilerplate such as "
        "'You are an AI assistant...'. Always call the tool exactly once."
    )
    return "\n".join(parts)


def build_tool(names: list[str], with_reason: bool) -> dict:
    props = {"category": {"type": "string", "enum": names,
                          "description": "The single best-fitting category."}}
    required = ["category"]
    if with_reason:
        props["reason"] = {"type": "string", "description": "One short sentence explaining the choice."}
        required.append("reason")
    return {
        "type": "function",
        "function": {
            "name": "classify_prompt",
            "description": "Record the category of the user's prompt.",
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


def parse_args_json(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def extract_category(msg, names: list[str]) -> tuple[str | None, str | None]:
    """Ia categoria din tool call; daca modelul a raspuns in text, incearca si acolo."""
    for tc in (msg.tool_calls or []):
        if tc.function.name == "classify_prompt":
            a = parse_args_json(tc.function.arguments)
            cat = str(a.get("category", "")).strip().lower()
            if cat in names:
                return cat, a.get("reason")
    content = (msg.content or "").lower()
    a = parse_args_json(content)
    if str(a.get("category", "")).lower() in names:
        return str(a["category"]).lower(), a.get("reason")
    hits = [n for n in names if n in content]
    if len(hits) == 1:
        return hits[0], None
    return None, None


# ---------------------------------------------------------------------------

def resolve_tool_choice(mode: str):
    if mode == "named":
        return {"type": "function", "function": {"name": "classify_prompt"}}
    return mode  # "required" / "auto"


def classify_one(client, model, system, tool, names, prompt, temperature, retries,
                 tool_choice="named", reasoning_effort=None):
    last_err = None
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"<prompt>\n{prompt}\n</prompt>"},
                ],
                tools=[tool],
                tool_choice=resolve_tool_choice(tool_choice),
                **extra,
            )
            cat, reason = extract_category(resp.choices[0].message, names)
            if cat:
                return cat, reason, None
            last_err = "no valid category in response"
        except Exception as e:  # noqa: BLE001 -- retry pe orice eroare de retea/API
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(min(2 ** attempt, 20))
    return None, None, last_err


def main() -> None:
    ap = argparse.ArgumentParser(description="Eticheteaza prompturile cu o categorie folosind un LLM (tool calling).")
    ap.add_argument("input", help="Fisier JSON / JSONL.")
    ap.add_argument("-o", "--output", help="Implicit: <input>_categorized.<ext>")
    ap.add_argument("--categories", default="categories.yaml")
    ap.add_argument("--model", required=True, help="Ex: qwen2.5:7b (Ollama), gpt-4o-mini (OpenAI)")
    ap.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"))
    ap.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "ollama"))
    ap.add_argument("--prompt-field", default="prompt")
    ap.add_argument("--max-chars", type=int, default=4000, help="Taie prompturile foarte lungi.")
    ap.add_argument("--workers", type=int, default=4, help="Cereri paralele.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--tool-choice", choices=["named", "required", "auto"], default="named",
                    help="named = forteaza classify_prompt (Ollama/vLLM/OpenAI); "
                         "pentru Gemini foloseste 'required' sau 'auto'.")
    ap.add_argument("--reasoning-effort", default=None,
                    help="Optional, ex. 'low' / 'none' la Gemini/OpenAI ca sa reduci costul si latenta.")
    ap.add_argument("--allow-other", action="store_true",
                    help=f"Permite categoria '{OTHER}' pentru prompturi care nu se potrivesc nicaieri.")
    ap.add_argument("--with-reason", action="store_true", help="Cere si o justificare scurta.")
    ap.add_argument("--limit", type=int, help="Proceseaza doar primele N (pentru test).")
    args = ap.parse_args()

    with open(args.categories, encoding="utf-8") as f:
        cats = yaml.safe_load(f)["categories"]
    names = [c["name"] for c in cats] + ([OTHER] if args.allow_other else [])
    system = build_system_prompt(cats, args.allow_other)
    tool = build_tool(names, args.with_reason)

    in_path = Path(args.input)
    records, fmt = load_records(in_path)
    if args.limit:
        records = records[: args.limit]
    missing = [i for i, r in enumerate(records) if args.prompt_field not in r]
    if missing:
        sys.exit(f"[!] {len(missing)} inregistrari fara campul '{args.prompt_field}' (ex: index {missing[0]}).")

    out_path = Path(args.output) if args.output else in_path.with_name(
        f"{in_path.stem}_categorized{in_path.suffix}")
    partial_path = out_path.with_name(out_path.name + ".partial.jsonl")
    done = load_partial(partial_path)
    todo = [i for i in range(len(records)) if i not in done or done[i].get("category") is None]
    print(f"[*] {len(records)} inregistrari, {len(records) - len(todo)} deja facute, {len(todo)} de procesat.")
    print(f"[*] Model '{args.model}' la {args.base_url}, {args.workers} workeri. Categorii: {names}")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=120)
    t0 = time.time()
    with partial_path.open("a", encoding="utf-8") as pf, ThreadPoolExecutor(args.workers) as ex:
        futs = {
            ex.submit(classify_one, client, args.model, system, tool, names,
                      str(records[i][args.prompt_field])[: args.max_chars],
                      args.temperature, args.retries,
                      args.tool_choice, args.reasoning_effort): i
            for i in todo
        }
        for n, fut in enumerate(as_completed(futs), 1):
            i = futs[fut]
            cat, reason, err = fut.result()
            row = {"idx": i, "category": cat, "reason": reason, "error": err}
            done[i] = row
            pf.write(json.dumps(row, ensure_ascii=False) + "\n")
            pf.flush()
            if err:
                print(f"    [!] #{i}: {err}")
            if n % 50 == 0 or n == len(todo):
                rate = n / (time.time() - t0)
                print(f"    {n}/{len(todo)}  ({rate:.1f}/s, ~{(len(todo) - n) / rate / 60:.1f} min ramase)")

    for i, r in enumerate(records):
        r["category"] = done[i]["category"]
        if args.with_reason:
            r["category_reason"] = done[i].get("reason")
    save_records(records, out_path, fmt)

    failed = sum(r["category"] is None for r in records)
    print(f"\n[OK] Scris in '{out_path}'. Distributie:")
    for name, n in Counter(r["category"] for r in records).most_common():
        print(f"    {str(name):<18} {n:>7}  ({n / len(records):.1%})")
    if failed:
        print(f"[!] {failed} esuate (category = null). Ruleaza din nou aceeasi comanda ca sa le reincerci.")
    else:
        partial_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()