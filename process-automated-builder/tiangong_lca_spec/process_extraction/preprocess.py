"""Utilities for pre-processing markdown JSON documents."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Iterable

from tiangong_lca_spec.core.exceptions import ProcessExtractionError

INTRO_PATTERN = re.compile(r"^(#+\s*)?(?:\d+\s*[.\-:])?\s*(introduction)\b.*", flags=re.IGNORECASE)
SECTION_PATTERN = re.compile(r"^(#+\s*)(references|appendix|acknowledg(e)?ments?)\b.*", flags=re.IGNORECASE | re.MULTILINE)
RESUME_SECTION_PATTERN = re.compile(r"^(#+\s*)(materials?\s+and\s+methods|methods?)\b.*", flags=re.IGNORECASE | re.MULTILINE)
NUMERIC_TOKEN_PATTERN = re.compile(r"\b([0-9OoIlI]+(?:[.,:/-][0-9OoIlI]+)*)([sS%])?\b")
IMAGE_DESCRIPTION_PREFIX = "image description:"
NON_CONTENT_SECTION_PREFIXES = (
    "acknowledgements",
    "author contributions",
    "funding",
    "competing interests",
    "additional information",
    "correspondence and requests for materials",
    "peer review information",
    "reprints and permissions",
    "publisher's note",
    "publisher’s note",
)
NON_CONTENT_SINGLE_LINES = {
    "open access",
    "check for updates",
}


def preprocess_paper(md_json: str, max_length: int | None = None) -> str:
    """Parse and clean markdown fragments serialized as JSON."""
    try:
        parsed = json.loads(md_json)
    except json.JSONDecodeError as exc:
        raise ProcessExtractionError("Paper JSON is not valid") from exc

    fragments = list(_iter_fragments(parsed))
    if not fragments:
        raise ProcessExtractionError("Paper JSON contains no textual fragments")

    text = "\n\n".join(fragment.strip() for fragment in fragments if fragment and fragment.strip())
    text = _trim_front_matter(text)
    text = _remove_sections(text)
    text = unicodedata.normalize("NFKC", text)
    text = _normalize_numeric_artifacts(text)

    if max_length is not None and len(text) > max_length:
        text = text[:max_length]

    return text


def _iter_fragments(parsed: object) -> Iterable[str]:
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str):
                yield item
    elif isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, str):
                yield value
            elif isinstance(value, list):
                for child in value:
                    if isinstance(child, str):
                        yield child
    elif isinstance(parsed, str):
        yield parsed


def _remove_sections(text: str) -> str:
    lines = text.splitlines()
    filtered: list[str] = []
    skip = False
    skip_non_content_block = False
    for line in lines:
        stripped = line.strip()
        stripped_lower = stripped.lower()

        if skip_non_content_block:
            if not stripped:
                skip_non_content_block = False
            continue

        if SECTION_PATTERN.match(line):
            skip = True
            continue
        if skip and RESUME_SECTION_PATTERN.match(line):
            skip = False
        if skip:
            continue

        if stripped_lower in NON_CONTENT_SINGLE_LINES:
            continue

        if any(stripped_lower.startswith(prefix) for prefix in NON_CONTENT_SECTION_PREFIXES):
            skip_non_content_block = True
            continue

        filtered.append(line)
    return "\n".join(filtered)


def _trim_front_matter(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    non_empty_indices = [idx for idx, line in enumerate(lines) if line.strip()]
    if not non_empty_indices:
        return text

    title_idx = non_empty_indices[0]
    intro_idx = next(
        (idx for idx, line in enumerate(lines) if INTRO_PATTERN.match(line.strip())),
        None,
    )

    if intro_idx is None:
        return "\n".join(lines[title_idx:])
    if intro_idx <= title_idx:
        return "\n".join(lines[intro_idx:])

    return "\n".join(lines[title_idx:])


def _normalize_numeric_artifacts(text: str) -> str:
    lines = text.splitlines()
    normalized: list[str] = []
    inside_table = False
    inside_image_desc = False

    for line in lines:
        stripped_lower = line.strip().lower()

        if stripped_lower.startswith(IMAGE_DESCRIPTION_PREFIX):
            inside_image_desc = True
            normalized.append(line)
            continue

        if inside_image_desc:
            normalized.append(line)
            if not stripped_lower:
                inside_image_desc = False
            continue

        if inside_table:
            normalized.append(line)
            if "</table>" in stripped_lower:
                inside_table = False
            continue

        if "<table" in stripped_lower:
            inside_table = True
            normalized.append(line)
            if "</table>" in stripped_lower:
                inside_table = False
            continue

        normalized.append(_normalize_numeric_line(line))

    return "\n".join(normalized)


def _normalize_numeric_line(line: str) -> str:
    collapsed = _collapse_numeric_spacing(line)
    return NUMERIC_TOKEN_PATTERN.sub(_normalize_numeric_token, collapsed)


def _collapse_numeric_spacing(text: str) -> str:
    text = re.sub(r"(?<=\d)\s+(?=[.,/%])", "", text)
    text = re.sub(r"(?<=[.,/%])\s+(?=\d)", "", text)
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    return text


def _normalize_numeric_token(match: re.Match[str]) -> str:
    core, suffix = match.group(1), match.group(2) or ""
    if not any(char.isdigit() for char in core):
        return match.group(0)

    chars = list(core)
    digit_count = sum(char.isdigit() for char in core)

    for idx, char in enumerate(chars):
        prev_char = chars[idx - 1] if idx > 0 else ""
        next_char = chars[idx + 1] if idx + 1 < len(chars) else ""

        if char in {"O", "o"}:
            prev_is_digit = prev_char.isdigit()
            next_is_digit = next_char.isdigit()
            has_additional_digit = digit_count >= 2
            if prev_is_digit or (next_is_digit and has_additional_digit):
                chars[idx] = "0"
            continue

        if char in {"I", "l"}:
            if prev_char.isdigit() or next_char.isdigit():
                chars[idx] = "1"
            continue

    normalized_core = "".join(chars)
    return f"{normalized_core}{suffix}"
