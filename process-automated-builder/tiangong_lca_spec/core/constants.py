"""Shared constant values used across the Tiangong LCA workflow."""

from __future__ import annotations

from typing import Any, Final

from tiangong_lca_spec.core.uris import build_local_dataset_uri

ILCD_FORMAT_SOURCE_UUID: Final[str] = "a97a0155-0234-4b87-b4ce-a45da52f2a40"
ILCD_FORMAT_SOURCE_VERSION: Final[str] = "03.00.003"
ILCD_FORMAT_SOURCE_URI: Final[str] = build_local_dataset_uri(
    "source",
    ILCD_FORMAT_SOURCE_UUID,
    ILCD_FORMAT_SOURCE_VERSION,
)
ILCD_FORMAT_SOURCE_SHORT_DESCRIPTION: Final[dict[str, str]] = {
    "@xml:lang": "en",
    "#text": "ILCD format",
}


def build_dataset_format_reference() -> dict[str, Any]:
    """Return the canonical ILCD format reference block."""
    return {
        "@refObjectId": ILCD_FORMAT_SOURCE_UUID,
        "@type": "source data set",
        "@uri": ILCD_FORMAT_SOURCE_URI,
        "@version": ILCD_FORMAT_SOURCE_VERSION,
        "common:shortDescription": dict(ILCD_FORMAT_SOURCE_SHORT_DESCRIPTION),
    }
