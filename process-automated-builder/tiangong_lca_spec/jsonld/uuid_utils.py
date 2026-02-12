"""Utility helpers to remap UUIDs for JSON-LD derived datasets."""

from __future__ import annotations

from typing import Any, Iterable, Mapping
from uuid import uuid4


class UUIDMapper:
    """Track UUID remapping tables for JSON-LD datasets."""

    def __init__(self, *, preserve_source_uuids: Iterable[str] | None = None) -> None:
        self.flow_map: dict[str, str] = {}
        self.source_map: dict[str, str] = {}
        self.process_map: dict[str, str] = {}
        self._preserve_source_uuids = {value.strip().lower() for value in (preserve_source_uuids or []) if isinstance(value, str) and value.strip()}

    # Flow datasets -----------------------------------------------------------------
    def remap_flow_dataset(self, dataset: dict[str, Any]) -> None:
        flow_root = dataset.get("flowDataSet")
        if not isinstance(flow_root, Mapping):
            return
        info = flow_root.get("flowInformation", {}).get("dataSetInformation", {})
        old_uuid = _ensure_uuid(info)
        new_uuid = _new_uuid()
        self.flow_map[old_uuid] = new_uuid
        info["common:UUID"] = new_uuid
        _replace_uuid_strings(flow_root, old_uuid, new_uuid)

    # Source datasets ----------------------------------------------------------------
    def remap_source_dataset(self, dataset: dict[str, Any]) -> None:
        source_root = dataset.get("sourceDataSet")
        if not isinstance(source_root, Mapping):
            return
        info = source_root.get("sourceInformation", {}).get("dataSetInformation", {})
        old_uuid = _ensure_uuid(info)
        if self._should_preserve_source_uuid(old_uuid):
            self.source_map.setdefault(old_uuid, old_uuid)
            return
        new_uuid = _new_uuid()
        self.source_map[old_uuid] = new_uuid
        info["common:UUID"] = new_uuid
        _replace_uuid_strings(source_root, old_uuid, new_uuid)

    # Process datasets --------------------------------------------------------------
    def remap_process_block(self, block: dict[str, Any]) -> None:
        dataset = block.get("processDataSet")
        if not isinstance(dataset, Mapping):
            return
        info = dataset.get("processInformation", {}).get("dataSetInformation", {})
        old_uuid = _ensure_uuid(info)
        new_uuid = _new_uuid()
        self.process_map[old_uuid] = new_uuid
        info["common:UUID"] = new_uuid
        block["process_id"] = new_uuid
        _replace_uuid_strings(dataset, old_uuid, new_uuid)
        self._rewrite_references(dataset)

    # Reference rewriting -----------------------------------------------------------
    def _rewrite_references(self, node: Any) -> None:
        if isinstance(node, dict):
            ref = node.get("@refObjectId")
            ref_type = node.get("@type")
            if isinstance(ref, str):
                if ref in self.flow_map:
                    node["@refObjectId"] = self.flow_map[ref]
                elif ref in self.source_map:
                    node["@refObjectId"] = self.source_map[ref]
                elif ref in self.process_map:
                    node["@refObjectId"] = self.process_map[ref]
                elif ref_type == "source data set" and not self._should_preserve_source_uuid(ref):
                    new_uuid = self.source_map.setdefault(ref, _new_uuid())
                    node["@refObjectId"] = new_uuid
            uri = node.get("@uri")
            if isinstance(uri, str):
                node["@uri"] = _replace_known_uuid(uri, self.flow_map, self.source_map, self.process_map)
            for key, value in list(node.items()):
                self._rewrite_references(value)
        elif isinstance(node, list):
            for item in node:
                self._rewrite_references(item)

    def _should_preserve_source_uuid(self, uuid_value: str | None) -> bool:
        if not isinstance(uuid_value, str):
            return False
        key = uuid_value.strip().lower()
        return bool(key) and key in self._preserve_source_uuids


# Helpers --------------------------------------------------------------------------


def _replace_uuid_strings(node: Any, old: str, new: str) -> None:
    if old == new:
        return
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if isinstance(value, str):
                if old in value:
                    node[key] = value.replace(old, new)
            else:
                _replace_uuid_strings(value, old, new)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            if isinstance(value, str):
                if old in value:
                    node[index] = value.replace(old, new)
            else:
                _replace_uuid_strings(value, old, new)


def _replace_known_uuid(value: str, *mappings: Mapping[str, str]) -> str:
    result = value
    for mapping in mappings:
        for old, new in mapping.items():
            if old in result:
                result = result.replace(old, new)
    return result


def _ensure_uuid(container: Mapping[str, Any]) -> str:
    candidate = container.get("common:UUID")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    new_uuid = _new_uuid()
    container["common:UUID"] = new_uuid
    return new_uuid


def _new_uuid() -> str:
    return str(uuid4())
