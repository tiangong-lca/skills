"""Utilities for loading and summarising TIDAS JSON schemas."""

from __future__ import annotations

import copy
import importlib.resources as resources
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep the Traversable path to bundled schemas (supports / and open()).
SCHEMA_DIR = resources.files("tidas_tools.tidas.schemas")


def _json_pointer_get(document: Any, pointer: str) -> Any:
    """Resolve a JSON pointer against a document."""
    if not pointer or pointer == "#":
        return document
    if pointer.startswith("#"):
        pointer = pointer[1:]
    if pointer.startswith("/"):
        pointer = pointer[1:]
    if not pointer:
        return document
    parts = pointer.split("/")
    current = document
    for part in parts:
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    return current


def _ensure_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


@dataclass(slots=True)
class FieldSummary:
    """Lightweight description of a schema field."""

    name: str
    description: str | None
    type: str | None
    required: bool
    reference: str | None = None
    children: list["FieldSummary"] | None = None


class TidasSchemaRepository:
    """Caches and resolves TIDAS schemas with support for local $ref resolution."""

    def __init__(self, schema_dir: Path | None = None) -> None:
        self._schema_dir = schema_dir or SCHEMA_DIR
        self._cache: dict[Path, dict[str, Any]] = {}

    def load(self, schema_name: str) -> dict[str, Any]:
        """Return a deep copy of the raw schema document."""
        path = self._schema_dir / schema_name
        if not path.exists():
            raise FileNotFoundError(f"TIDAS schema '{schema_name}' not found at {path}")
        document = self._cache.get(path)
        if document is None:
            document = json.loads(path.read_text(encoding="utf-8"))
            self._cache[path] = document
        return copy.deepcopy(document)

    def summarize_properties(
        self,
        schema_name: str,
        pointer: str | None = None,
        *,
        include_refs: bool = True,
    ) -> list[FieldSummary]:
        """Return a list of field summaries for the given schema pointer."""
        schema_path = (self._schema_dir / schema_name).resolve()
        document = self.load(schema_name)
        node = _json_pointer_get(document, pointer or "")
        visited: set[str] = set()
        return self._summarize_node(node, schema_path, visited, include_refs=include_refs)

    def resolve_with_references(
        self,
        schema_name: str,
        pointer: str | None = None,
    ) -> dict[str, Any]:
        """Return the schema node with local $ref resolved recursively."""
        schema_path = (self._schema_dir / schema_name).resolve()
        document = self.load(schema_name)
        node = _json_pointer_get(document, pointer or "")
        visited: set[str] = set()
        return self._expand(node, schema_path, visited)

    # ------------------------------------------------------------------ internals

    def _summarize_node(
        self,
        node: Any,
        base_path: Path,
        visited: set[str],
        *,
        include_refs: bool,
    ) -> list[FieldSummary]:
        if not isinstance(node, dict):
            return []
        properties = node.get("properties")
        if not isinstance(properties, dict):
            return []
        required = set(_ensure_list(node.get("required")))
        summaries: list[FieldSummary] = []
        for name, schema in properties.items():
            merged, target_path, ref = self._merged_schema(schema, base_path, visited)
            type_hint = self._infer_type(merged)
            description = merged.get("description") or schema.get("description")
            children: list[FieldSummary] | None = None
            if isinstance(merged.get("properties"), dict):
                children = self._summarize_node(
                    merged,
                    target_path,
                    visited,
                    include_refs=include_refs,
                )
            elif isinstance(merged.get("items"), dict):
                item = merged["items"]
                merged_item, item_path, _ = self._merged_schema(item, target_path, visited)
                children = (
                    self._summarize_node(
                        merged_item,
                        item_path,
                        visited,
                        include_refs=include_refs,
                    )
                    or None
                )
            summaries.append(
                FieldSummary(
                    name=name,
                    description=description,
                    type=type_hint,
                    required=name in required,
                    reference=ref if include_refs else None,
                    children=children or None,
                )
            )
        return summaries

    def _merged_schema(
        self,
        schema: dict[str, Any],
        base_path: Path,
        visited: set[str],
    ) -> tuple[dict[str, Any], Path, str | None]:
        if not isinstance(schema, dict):
            return {"const": schema}, base_path, None
        if "$ref" not in schema:
            return schema, base_path, None

        ref = schema["$ref"]
        cache_key = f"{base_path}::{ref}"
        if cache_key in visited:
            # Avoid infinite recursion by returning the minimal stub.
            merged = {k: v for k, v in schema.items() if k != "$ref"}
            return merged, base_path, ref
        visited.add(cache_key)
        try:
            target_node, target_path = self._resolve_ref(ref, base_path)
            if isinstance(target_node, dict):
                merged = copy.deepcopy(target_node)
                for key, value in schema.items():
                    if key == "$ref":
                        continue
                    merged[key] = value
            else:
                merged = {"const": target_node}
            return merged, target_path, ref
        finally:
            visited.remove(cache_key)

    def _resolve_ref(self, ref: str, base_path: Path) -> tuple[Any, Path]:
        path_part, _, fragment = ref.partition("#")
        if path_part:
            ref_path = (base_path.parent / path_part).resolve()
        else:
            ref_path = base_path
        document = self._load_raw(ref_path)
        if fragment:
            target = _json_pointer_get(document, fragment)
        else:
            target = document
        return copy.deepcopy(target), ref_path

    def _load_raw(self, path: Path) -> dict[str, Any]:
        cached = self._cache.get(path)
        if cached is None:
            cached = json.loads(path.read_text(encoding="utf-8"))
            self._cache[path] = cached
        return cached

    def _expand(self, node: Any, base_path: Path, visited: set[str]) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                merged, target_path, _ = self._merged_schema(node, base_path, visited)
                return self._expand(merged, target_path, visited)
            expanded: dict[str, Any] = {}
            for key, value in node.items():
                expanded[key] = self._expand(value, base_path, visited)
            return expanded
        if isinstance(node, list):
            return [self._expand(item, base_path, visited) for item in node]
        return node

    @staticmethod
    def _infer_type(schema: dict[str, Any]) -> str | None:
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            return " | ".join(schema_type)
        if isinstance(schema_type, str):
            return schema_type
        if "enum" in schema:
            return "enum"
        if "const" in schema:
            return "const"
        if "anyOf" in schema:
            types = [item.get("type") for item in schema["anyOf"] if isinstance(item, dict) and item.get("type")]
            return " | ".join(types) if types else "anyOf"
        return None


_repository: TidasSchemaRepository | None = None


def get_schema_repository() -> TidasSchemaRepository:
    """Return the process-wide schema repository singleton."""
    global _repository  # noqa: PLW0603 - module-level cache
    if _repository is None:
        _repository = TidasSchemaRepository()
    return _repository
