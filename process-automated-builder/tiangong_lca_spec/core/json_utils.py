"""Reusable helpers for parsing JSON-like responses from LLMs or services."""

from __future__ import annotations

import json
import re
from typing import Any

from .exceptions import SpecCodingError

THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
CODE_BLOCK_PATTERN = re.compile(r"```(?:json)?(.*?)```", flags=re.DOTALL | re.IGNORECASE)
LINE_COMMENT_PATTERN = re.compile(r"^\s*//.*?$", flags=re.MULTILINE)


def strip_think(content: str) -> str:
    return THINK_PATTERN.sub("", content)


def extract_json_blob(content: str) -> str:
    stripped = strip_think(content).strip()
    match = CODE_BLOCK_PATTERN.search(stripped)
    if match:
        stripped = match.group(1)
    return stripped.strip()


def parse_json_response(content: str) -> Any:
    cleaned = strip_json_comments(extract_json_blob(content))
    attempts = [cleaned, cleaned.strip('"'), truncate_to_balanced(cleaned)]
    for candidate in attempts:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise SpecCodingError("Unable to parse JSON from response")


def truncate_to_balanced(raw: str) -> str:
    stack = 0
    brackets = 0
    result_chars: list[str] = []
    for char in raw:
        if char == "{":
            stack += 1
        elif char == "}":
            stack -= 1
        elif char == "[":
            brackets += 1
        elif char == "]":
            brackets -= 1
        result_chars.append(char)
        if stack == 0 and brackets == 0 and char in ("}", "]"):
            break
    return "".join(result_chars)


def strip_json_comments(raw: str) -> str:
    return LINE_COMMENT_PATTERN.sub("", raw)
