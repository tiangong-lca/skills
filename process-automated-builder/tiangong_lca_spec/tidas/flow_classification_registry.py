"""Helpers for validating TIDAS product flow classification paths."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

from .schema_loader import TidasSchemaRepository

_SCHEMA_FILENAME = "tidas_flows_product_category.json"


@dataclass(slots=True)
class FlowClassificationEntry:
    """Normalised representation of a single product flow classification node."""

    level: str
    class_id: str
    text: str

    def as_dict(self) -> dict[str, str]:
        return {
            "@level": self.level,
            "@classId": self.class_id,
            "#text": self.text,
        }


class FlowProductClassificationRegistry:
    """Loads and validates product flow classifications against the TIDAS schema."""

    def __init__(self, *, repository: TidasSchemaRepository | None = None) -> None:
        self._repository = repository or TidasSchemaRepository()
        self._entries, self._entries_by_id = _load_classification_entries(self._repository)

    def normalise_path(
        self,
        entries: Sequence[dict[str, str | None] | None],
    ) -> list[dict[str, str]]:
        """Return a canonicalised classification path or raise ValueError."""

        normalised: list[FlowClassificationEntry] = []
        seen_levels: set[str] = set()
        for raw in entries:
            if not raw:
                continue
            level = (raw.get("@level") or "").strip()
            class_id = (raw.get("@classId") or raw.get("@catId") or raw.get("@code") or "").strip()
            if not level or not class_id:
                raise ValueError("Classification entries must include '@level' and '@classId'.")
            record = self._entries_by_id.get(class_id)
            if record is None:
                raise ValueError(f"Unknown TIDAS product flow classification id '{class_id}'.")
            if record.level != level:
                raise ValueError(f"Classification id '{class_id}' declared at level '{level}' " f"but expected level '{record.level}'.")
            if level in seen_levels:
                raise ValueError(f"Duplicate classification level '{level}' detected.")
            seen_levels.add(level)
            normalised.append(record)

        if not normalised:
            raise ValueError("Product flow classification path is empty.")

        level_order = [entry.level for entry in normalised]
        if level_order != sorted(level_order, key=_level_sort_key):
            raise ValueError("Classification levels must appear in ascending order " f"(received order: {level_order}).")

        return [entry.as_dict() for entry in normalised]

    # ------------------------------------------------------------------ helpers

    def find_entry_by_description(self, text: str) -> FlowClassificationEntry | None:
        """Return the deepest entry whose description appears in text."""
        haystack = text.lower()
        best: FlowClassificationEntry | None = None
        for entry in self._entries:
            needle = entry.text.lower()
            if not needle or needle == "unspecified":
                continue
            if needle in haystack:
                if best is None or int(entry.level) > int(best.level) or (entry.level == best.level and len(entry.class_id) > len(best.class_id)):
                    best = entry
        if best is None:
            # Fallback: match on individual keyword occurrences
            for entry in self._entries:
                needle = entry.text.lower()
                if not needle:
                    continue
                if any(token and token in haystack for token in needle.split()):
                    if best is None or int(entry.level) > int(best.level) or (entry.level == best.level and len(entry.class_id) > len(best.class_id)):
                        best = entry
        return best

    def build_path_for_entry(self, entry: FlowClassificationEntry) -> list[dict[str, str]]:
        """Construct a full ascending path ending with the provided entry."""
        path: list[FlowClassificationEntry] = []
        current: FlowClassificationEntry | None = entry
        while current:
            path.append(current)
            current = self._find_parent(current)
        return [item.as_dict() for item in reversed(path)]

    def infer_path_from_text(self, text: str) -> list[dict[str, str]] | None:
        """Infer a reasonable classification path by searching entry descriptions."""
        if not text:
            return None
        entry = self.find_entry_by_description(text)
        if not entry:
            return None
        return self.build_path_for_entry(entry)

    def _find_parent(self, entry: FlowClassificationEntry) -> FlowClassificationEntry | None:
        target_level = int(entry.level) - 1
        if target_level < 0:
            return None
        best: FlowClassificationEntry | None = None
        for candidate in self._entries:
            if int(candidate.level) != target_level:
                continue
            if entry.class_id.startswith(candidate.class_id):
                if best is None or len(candidate.class_id) > len(best.class_id):
                    best = candidate
        return best


@lru_cache(maxsize=1)
def _load_classification_entries(
    repository: TidasSchemaRepository,
) -> tuple[list[FlowClassificationEntry], dict[str, FlowClassificationEntry]]:

    schema = repository.load(_SCHEMA_FILENAME)
    entries_list: list[FlowClassificationEntry] = []
    entries_map: dict[str, FlowClassificationEntry] = {}
    for candidate in schema.get("oneOf", []):
        props = candidate.get("properties") or {}
        level_const = _extract_const(props.get("@level"))
        class_const = _extract_const(props.get("@classId")) or _extract_const(props.get("@catId")) or _extract_const(props.get("@code"))
        text_const = _extract_const(props.get("#text"))
        if not (level_const and class_const and text_const):
            continue
        entry = FlowClassificationEntry(
            level=level_const,
            class_id=class_const,
            text=text_const,
        )
        entries_list.append(entry)
        entries_map[class_const] = entry
    entries_list.sort(key=lambda item: (int(item.level), len(item.class_id)))
    return entries_list, entries_map


def _extract_const(node: dict | None) -> str | None:
    if not isinstance(node, dict):
        return None
    value = node.get("const")
    if isinstance(value, str):
        return value
    return None


def _level_sort_key(level: str) -> tuple[int, str]:
    try:
        depth = int(level)
    except ValueError:
        depth = 0
    return depth, level


def ensure_valid_product_flow_classification(
    entries: Iterable[dict[str, str | None] | None],
) -> list[dict[str, str]]:
    """Convenience wrapper that validates product flow classifications."""
    registry = FlowProductClassificationRegistry()
    return registry.normalise_path(list(entries))


def infer_flow_classification_from_text(text: str) -> list[dict[str, str]] | None:
    """Attempt to infer a classification path based on descriptive text."""
    registry = FlowProductClassificationRegistry()
    return registry.infer_path_from_text(text.strip())


__all__ = [
    "ensure_valid_product_flow_classification",
    "FlowProductClassificationRegistry",
    "infer_flow_classification_from_text",
]
