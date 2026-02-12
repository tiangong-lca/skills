"""Helpers for rebuilding flow references inside ILCD datasets."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from tiangong_lca_spec.core.uris import build_local_dataset_uri


@dataclass(slots=True)
class FlowReferenceData:
    """Minimal metadata captured from a flow dataset for downstream references."""

    uuid: str
    version: str
    short_description: dict[str, Any]


def reference_data_from_flow_dataset(flow_dataset: dict[str, Any], *, version: str | None = None) -> FlowReferenceData | None:
    """Extract the metadata required to build `referenceToFlowDataSet` nodes."""

    info = flow_dataset.get("flowInformation", {}).get("dataSetInformation", {})
    uuid_value = _normalise_uuid(info.get("common:UUID"))
    if not uuid_value:
        return None

    dataset_version = _normalise_version(version) or _extract_dataset_version(flow_dataset) or "01.01.000"

    short_description = _first_language_entry(info.get("common:shortDescription"))
    if short_description is None:
        name_block = info.get("name", {})
        short_description = _first_language_entry(name_block.get("baseName")) or _first_language_entry(name_block)
    if short_description is None:
        short_description = _language_entry(uuid_value)

    return FlowReferenceData(uuid=uuid_value, version=dataset_version, short_description=short_description)


def build_flow_reference(metadata: FlowReferenceData, *, uri: str | None = None) -> dict[str, Any]:
    """Return a sanitized `referenceToFlowDataSet` block using captured metadata."""

    version = _normalise_version(metadata.version) or "01.01.000"
    reference = {
        "@type": "flow data set",
        "@refObjectId": metadata.uuid,
        "@version": version,
        "@uri": uri or build_local_dataset_uri("flow data set", metadata.uuid, version),
    }
    if metadata.short_description:
        reference["common:shortDescription"] = deepcopy(metadata.short_description)
    return reference


# Internal helpers ----------------------------------------------------------------


def _normalise_uuid(value: Any) -> str | None:
    if isinstance(value, str):
        candidate = value.strip()
        if candidate:
            return candidate
    return None


def _normalise_version(value: Any) -> str | None:
    if isinstance(value, str):
        candidate = value.strip()
        if candidate:
            return candidate
    return None


def _extract_dataset_version(flow_dataset: dict[str, Any]) -> str | None:
    admin = flow_dataset.get("administrativeInformation", {})
    publication = admin.get("publicationAndOwnership", {})
    version_value = publication.get("common:dataSetVersion")
    return _normalise_version(version_value)


def _language_entry(text: str, lang: str = "en") -> dict[str, str]:
    return {"@xml:lang": lang, "#text": text}


def _first_language_entry(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        text = value.get("#text") or value.get("text")
        if isinstance(text, str) and text.strip():
            lang = value.get("@xml:lang") or "en"
            return _language_entry(text.strip(), lang)
        # Some entries are nested dicts keyed by language codes (e.g. {"en": {...}})
        for candidate in value.values():
            result = _first_language_entry(candidate)
            if result:
                return result
        return None
    if isinstance(value, list):
        for item in value:
            result = _first_language_entry(item)
            if result:
                return result
        return None
    if isinstance(value, str) and value.strip():
        return _language_entry(value.strip())
    return None


__all__ = ["FlowReferenceData", "build_flow_reference", "reference_data_from_flow_dataset"]
