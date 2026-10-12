#!/usr/bin/env python3
"""
Convert every *.json file in a folder (a JSON array of objects) into a
*.jsonl file with the same name: one JSON object per line.

    prompts_brainstorming.json -> prompts_brainstorming.jsonl

Usage (from the repo root):
    python convert_json_to_jsonl.py
    python convert_json_to_jsonl.py --dir scripts/resources/prompts_datasets
    python convert_json_to_jsonl.py --delete-json   # also remove the .json originals
"""
import argparse
import json
import sys
from pathlib import Path

DEFAULT_DIR = "resources/prompts_datasets"


def convert(json_path: Path, delete_json: bool) -> None:
    with json_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        print(f"[!] Skipping {json_path.name}: expected a JSON array, got {type(data).__name__}")
        return

    jsonl_path = json_path.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[OK] {json_path.name} -> {jsonl_path.name} ({len(data)} records)")

    if delete_json:
        json_path.unlink()
        print(f"     deleted {json_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JSON array files to JSONL.")
    parser.add_argument("--dir", default=DEFAULT_DIR, help=f"Folder with .json files (default: {DEFAULT_DIR})")
    parser.add_argument("--delete-json", action="store_true", help="Delete the original .json files after converting")
    args = parser.parse_args()

    folder = Path(args.dir)
    if not folder.is_dir():
        sys.exit(f"[!] Folder not found: {folder.resolve()}")

    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        sys.exit(f"[!] No .json files found in {folder.resolve()}")

    for path in json_files:
        try:
            convert(path, args.delete_json)
        except json.JSONDecodeError as e:
            print(f"[!] Skipping {path.name}: invalid JSON ({e})")


if __name__ == "__main__":
    main()