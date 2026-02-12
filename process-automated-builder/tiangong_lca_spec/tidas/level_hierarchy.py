"""Shared helpers for navigating level-based TIDAS classification schemas."""

from __future__ import annotations

import importlib.resources as resources
import json
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

# Keep the Traversable path to bundled schemas (supports / and open()).
SCHEMA_DIR = resources.files("tidas_tools.tidas.schemas")


@dataclass(frozen=True)
class HierarchyEntry:
    """Single entry in a level-based classification schema."""

    level: int
    code: str
    description: str


def _load_entries(schema_path: Path) -> list[HierarchyEntry]:
    if not schema_path.exists():
        raise FileNotFoundError(f"TIDAS schema not found: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)

    entries: list[HierarchyEntry] = []
    for item in document.get("oneOf", []):
        props = item.get("properties", {})
        code = None
        for key in ("@classId", "@catId", "@code"):
            candidate = props.get(key, {}).get("const")
            if candidate:
                code = candidate
                break
        level_str = props.get("@level", {}).get("const")
        description = props.get("#text", {}).get("const", "")
        if not code or not level_str:
            continue
        try:
            level = int(level_str)
        except ValueError:
            continue
        entries.append(HierarchyEntry(level=level, code=str(code), description=str(description)))
    return entries


def _build_child_map(
    entries: Iterable[HierarchyEntry],
) -> tuple[list[HierarchyEntry], dict[str, list[HierarchyEntry]], dict[str, HierarchyEntry | None]]:
    child_map: dict[str, list[HierarchyEntry]] = defaultdict(list)
    last_per_level: dict[int, HierarchyEntry] = {}
    roots: list[HierarchyEntry] = []
    parent_map: dict[str, HierarchyEntry | None] = {}

    for entry in entries:
        if entry.level == 0:
            roots.append(entry)
            child_map.setdefault("", []).append(entry)
            parent_map[entry.code] = None
        else:
            parent = _find_parent(entry, last_per_level)
            if parent:
                child_map.setdefault(parent.code, []).append(entry)
                parent_map[entry.code] = parent
        last_per_level[entry.level] = entry
    return roots, child_map, parent_map


def _find_parent(entry: HierarchyEntry, last_per_level: dict[int, HierarchyEntry]) -> HierarchyEntry | None:
    target_level = entry.level - 1
    while target_level >= 0:
        parent = last_per_level.get(target_level)
        if parent:
            return parent
        target_level -= 1
    return None


class HierarchyNavigator:
    """Provides efficient access to hierarchical children for a schema."""

    def __init__(self, entries: Iterable[HierarchyEntry]) -> None:
        self._entries = list(entries)
        self._roots, self._child_map, self._parent_map = _build_child_map(self._entries)
        self._entry_map = {entry.code: entry for entry in self._entries}
        self._max_level = max((entry.level for entry in self._entries), default=-1)

    @property
    def max_level(self) -> int:
        return self._max_level

    def root_entries(self) -> list[HierarchyEntry]:
        return list(self._roots)

    def children(self, parent_code: str | None) -> list[HierarchyEntry]:
        key = parent_code or ""
        return list(self._child_map.get(key, []))

    def get_entry(self, code: str) -> HierarchyEntry | None:
        return self._entry_map.get(code)

    def path(self, code: str) -> list[HierarchyEntry]:
        """Return the path from root to the given code (empty if not found)."""
        entry = self._entry_map.get(code)
        if not entry:
            return []
        chain = [entry]
        while True:
            parent = self._parent_map.get(entry.code)
            if not parent:
                break
            chain.append(parent)
            entry = parent
        chain.reverse()
        return chain


def load_level_entries(schema_path: Path) -> list[HierarchyEntry]:
    """Load hierarchy entries from the provided schema path."""
    return _load_entries(schema_path)


def build_navigator_from_schema(schema_path: Path) -> HierarchyNavigator:
    """Build a navigator for the given schema."""
    return HierarchyNavigator(load_level_entries(schema_path))


@lru_cache(maxsize=1)
def get_process_category_navigator() -> HierarchyNavigator:
    """Return a cached navigator for process classifications."""
    schema_path = SCHEMA_DIR / "tidas_processes_category.json"
    return build_navigator_from_schema(schema_path)


@lru_cache(maxsize=1)
def get_product_flow_category_navigator() -> HierarchyNavigator:
    """Return a cached navigator for product flow classifications."""
    schema_path = SCHEMA_DIR / "tidas_flows_product_category.json"
    return build_navigator_from_schema(schema_path)


__all__ = [
    "HierarchyEntry",
    "HierarchyNavigator",
    "build_navigator_from_schema",
    "load_level_entries",
    "get_process_category_navigator",
    "get_product_flow_category_navigator",
]
