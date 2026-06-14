"""File I/O utilities (adapted from LeanRAG/tools)."""
from __future__ import annotations
import json
import os


def read_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(data: list, path: str, mode: str = "a"):
    with open(path, mode, encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def create_if_not_exist(path: str):
    os.makedirs(path, exist_ok=True)
