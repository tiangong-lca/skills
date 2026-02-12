"""Utilities for working with the ILCD/TIDAS location code hierarchy."""

from __future__ import annotations

import importlib.resources as resources
import json
import re
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

# Schema is provided by the installed tidas_tools package
DEFAULT_SCHEMA_PATH = Path(resources.files("tidas_tools.tidas.schemas") / "tidas_locations_category.json")
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


class LocationCatalog:
    """Lightweight helper around the ILCD location schema."""

    def __init__(self, schema_path: Path | None = None) -> None:
        self._schema_path = schema_path or DEFAULT_SCHEMA_PATH
        data = json.loads(self._schema_path.read_text(encoding="utf-8"))
        entries = [(item.get("const"), item.get("description", "")) for item in data.get("oneOf", []) if item.get("const")]
        self._code_to_description: dict[str, str] = {code: desc for code, desc in entries}
        self._parent_map: dict[str, str] = {}
        self._children_map: dict[str, list[str]] = defaultdict(list)
        for code in self._code_to_description:
            parent = "-".join(code.split("-")[:-1])
            self._parent_map[code] = parent
            self._children_map[parent].append(code)
        for child_list in self._children_map.values():
            child_list.sort()
        self._token_index: dict[str, set[str]] = defaultdict(set)
        self._build_token_index(entries)

    def _build_token_index(self, entries: Iterable[tuple[str, str]]) -> None:
        for code, desc in entries:
            for token in self._tokenize(code):
                self._token_index[token].add(code)
            for token in self._tokenize(desc):
                self._token_index[token].add(code)

    @staticmethod
    def _tokenize(value: str | None) -> list[str]:
        if not value:
            return []
        return [match.group(0).upper() for match in _TOKEN_PATTERN.finditer(value)]

    def normalize_code(self, value: str | None) -> str | None:
        if not value:
            return None
        cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().upper())
        cleaned = re.sub(r"-+", "-", cleaned).strip("-")
        if not cleaned:
            return None
        if cleaned in self._code_to_description:
            return cleaned
        return None

    def is_valid(self, code: str | None) -> bool:
        return bool(code and code in self._code_to_description)

    def coerce_code(self, value: str | None) -> str | None:
        normalized = self.normalize_code(value)
        return normalized if normalized else None

    def parent(self, code: str | None) -> str | None:
        if not code:
            return None
        return self._parent_map.get(code)

    def describe(self, code: str | None) -> str:
        if not code:
            return ""
        parts = code.split("-")
        labels: list[str] = []
        for idx in range(len(parts)):
            sub_code = "-".join(parts[: idx + 1])
            label = self._code_to_description.get(sub_code)
            if label:
                labels.append(label)
        return " > ".join(labels) if labels else self._code_to_description.get(code, "")

    def best_guess(self, raw_hint: str | None) -> str | None:
        normalized = self.normalize_code(raw_hint)
        if normalized:
            return normalized
        tokens = self._tokenize(raw_hint)
        if not tokens:
            return None
        candidate_sets = [self._token_index[token] for token in tokens if token in self._token_index]
        if not candidate_sets:
            return None
        if len(candidate_sets) == 1:
            candidates = set(candidate_sets[0])
        else:
            intersection = set.intersection(*candidate_sets)
            candidates = intersection if intersection else set().union(*candidate_sets)
        if not candidates:
            return None
        token_set = set(tokens)

        def score(code: str) -> tuple[int, int, int]:
            code_tokens = set(self._tokenize(code))
            match_count = len(token_set & code_tokens)
            depth = code.count("-")
            length = len(code)
            return (match_count, -depth, -length)

        return max(candidates, key=score, default=None)

    def detect_base_code(self, raw_hint: str | None) -> str | None:
        tokens = self._tokenize(raw_hint)
        for token in tokens:
            if "-" not in token and token in self._code_to_description:
                return token
        guess = self.best_guess(raw_hint)
        if guess:
            return self.parent(guess) or guess
        return None

    def build_candidate_list(self, raw_hint: str | None, depth: int = 2, limit: int = 80) -> list[dict[str, str]]:
        base_code = self.detect_base_code(raw_hint)
        start_key = base_code or ""
        queue: deque[tuple[str, int]] = deque([(start_key, 0)])
        seen: set[str] = set()
        collected: list[str] = []
        while queue and len(collected) < limit:
            parent, level = queue.popleft()
            for child in self._children_map.get(parent, []):
                if child in seen:
                    continue
                seen.add(child)
                collected.append(child)
                if level + 1 < depth:
                    queue.append((child, level + 1))
        payload: list[dict[str, str]] = []
        if base_code and base_code in self._code_to_description:
            payload.append(self._candidate_entry(base_code))
        for code in collected:
            if base_code and code == base_code:
                continue
            payload.append(self._candidate_entry(code))
        return payload

    def _candidate_entry(self, code: str) -> dict[str, str]:
        return {
            "code": code,
            "description": self.describe(code),
        }


def extract_location_response(value: Any) -> tuple[str | None, dict[str, Any]]:
    """Parse a LocationNormalizer JSON response into a (code, payload) tuple."""
    if isinstance(value, dict):
        if "locationOfOperationSupplyOrProduction" in value:
            block = value.get("locationOfOperationSupplyOrProduction") or {}
            if isinstance(block, dict):
                code = block.get("@location") or block.get("code")
                return code, value
        code = value.get("code") or value.get("@location") or value.get("location")
        return code, value
    if isinstance(value, str):
        return None, {"description": value}
    return None, {}


@lru_cache(maxsize=1)
def get_location_catalog() -> LocationCatalog:
    return LocationCatalog()
