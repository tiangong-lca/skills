"""Helpers for constructing Tiangong portal dataset URLs."""

from __future__ import annotations

from typing import Final

TIDAS_PORTAL_BASE: Final[str] = "https://lcdn.tiangong.earth"

_SUFFIX_MAP: Final[dict[str, str]] = {
    "process": "showProcess.xhtml",
    "flow": "showProductFlow.xhtml",
    "source": "showSource.xhtml",
}

_FOLDER_MAP: Final[dict[str, str]] = {
    "process": "processes",
    "process data set": "processes",
    "flow": "flows",
    "flow data set": "flows",
    "source": "sources",
    "source data set": "sources",
    "contact": "contacts",
    "contact data set": "contacts",
}


def build_portal_uri(dataset_kind: str, uuid_value: str, version: str) -> str:
    """Return the public portal URL for a dataset."""
    if not uuid_value:
        return ""
    suffix = _SUFFIX_MAP.get(dataset_kind, "showDataSet.xhtml")
    version_clean = (version or "").strip() or "01.01.000"
    return f"{TIDAS_PORTAL_BASE}/{suffix}?uuid={uuid_value}&version={version_clean}"


def build_local_dataset_uri(dataset_kind: str, uuid_value: str, version: str) -> str:
    """Return the relative ILCD path used inside dataset references."""
    if not uuid_value:
        return ""
    kind = (dataset_kind or "").strip().lower()
    folder = _FOLDER_MAP.get(kind)
    if folder is None:
        folder = "datasets"
    version_clean = (version or "").strip() or "01.01.000"
    return f"../{folder}/{uuid_value}_{version_clean}.xml"
