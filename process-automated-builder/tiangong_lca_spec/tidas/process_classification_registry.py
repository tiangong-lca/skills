"""Utility helpers for validating TIDAS process classification paths."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

from .schema_loader import TidasSchemaRepository

_SCHEMA_FILENAME = "tidas_processes_category.json"


@dataclass(slots=True)
class ClassificationEntry:
    """Normalised representation of a single classification node."""

    level: str
    class_id: str
    text: str

    def as_dict(self) -> dict[str, str]:
        return {
            "@level": self.level,
            "@classId": self.class_id,
            "#text": self.text,
        }


class ProcessClassificationRegistry:
    """Loads and validates process classifications against the TIDAS schema."""

    def __init__(self, *, repository: TidasSchemaRepository | None = None) -> None:
        self._repository = repository or TidasSchemaRepository()
        self._entries = _load_classification_entries(self._repository)

    def normalise_path(
        self,
        entries: Sequence[dict[str, str | None] | None],
    ) -> list[dict[str, str]]:
        """Return a canonicalised classification path or raise ValueError."""
        normalised: list[ClassificationEntry] = []
        seen_levels: set[str] = set()
        for raw in entries:
            if not raw:
                continue
            level = (raw.get("@level") or "").strip()
            class_id = (raw.get("@classId") or "").strip()
            if not level or not class_id:
                raise ValueError("Classification entries must include '@level' and '@classId'.")
            record = self._entries.get(class_id)
            if record is None:
                raise ValueError(f"Unknown TIDAS process classification id '{class_id}'.")
            if record.level != level:
                raise ValueError(f"Classification id '{class_id}' declared at level '{level}' " f"but expected level '{record.level}'.")
            if level in seen_levels:
                raise ValueError(f"Duplicate classification level '{level}' detected.")
            seen_levels.add(level)
            normalised.append(record)

        if not normalised:
            raise ValueError("Process classification path is empty.")

        # Ensure strictly ascending level order (0 -> 3)
        level_order = [entry.level for entry in normalised]
        if level_order != sorted(level_order, key=_level_sort_key):
            raise ValueError("Classification levels must be provided in ascending order " f"(received order: {level_order}).")

        return [entry.as_dict() for entry in normalised]


@lru_cache(maxsize=1)
def _load_classification_entries(
    repository: TidasSchemaRepository,
) -> dict[str, ClassificationEntry]:
    """Parse the schema into an easy to query dictionary."""
    schema = repository.load(_SCHEMA_FILENAME)
    entries: dict[str, ClassificationEntry] = {}
    for candidate in schema.get("oneOf", []):
        props = candidate.get("properties") or {}
        level_const = _extract_const(props.get("@level"))
        class_const = _extract_const(props.get("@classId"))
        text_const = _extract_const(props.get("#text"))
        if not (level_const and class_const and text_const):
            continue
        entries[class_const] = ClassificationEntry(
            level=level_const,
            class_id=class_const,
            text=text_const,
        )
    return entries


def _extract_const(node: dict | None) -> str | None:
    if not isinstance(node, dict):
        return None
    value = node.get("const")
    if isinstance(value, str):
        return value
    return None


def _level_sort_key(level: str) -> tuple[int, str]:
    # Level is a string but represents integer depth
    try:
        depth = int(level)
    except ValueError:
        depth = 0
    return depth, level


def ensure_valid_classification_path(
    entries: Iterable[dict[str, str | None] | None],
) -> list[dict[str, str]]:
    """Convenience wrapper that uses the shared registry singleton."""
    registry = ProcessClassificationRegistry()
    return registry.normalise_path(list(entries))
