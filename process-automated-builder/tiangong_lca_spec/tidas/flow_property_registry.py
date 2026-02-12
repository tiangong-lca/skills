"""Registry utilities for flow property to unit-group mappings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
MAPPING_PATH = SCHEMA_DIR / "flowproperty_unitgroup_mapping.json"

FLOW_PROPERTY_VERSION_OVERRIDES: Mapping[str, str] = {
    # Mass property ships with a published ILCD dataset version.
    "93a60a56-a3c8-11da-a746-0800200b9a66": "03.00.003",
    # mass*distance (kg*km) transport service property.
    "118f2a40-50ec-457c-aa60-9bc6b6af9931": "01.01.000",
}
DEFAULT_FLOW_PROPERTY_VERSION = "01.01.000"


@dataclass(slots=True, frozen=True)
class UnitDescriptor:
    """Single unit entry inside a unit group."""

    name: str
    mean_value: str
    internal_id: str
    general_comment: str | None
    is_reference: bool


@dataclass(slots=True, frozen=True)
class UnitGroupDescriptor:
    """Description of a unit group associated with a flow property."""

    uuid: str
    name: str
    reference_internal_id: str
    units: tuple[UnitDescriptor, ...]

    @property
    def reference_unit(self) -> UnitDescriptor | None:
        for unit in self.units:
            if unit.is_reference:
                return unit
        return None


@dataclass(slots=True, frozen=True)
class FlowPropertyDescriptor:
    """Parsed mapping entry linking a flow property to a unit group."""

    uuid: str
    name: str
    classification: tuple[str, ...]
    reference_unit_description: str
    reference_unit_group_uuid: str
    unit_group: UnitGroupDescriptor


class FlowPropertyRegistry:
    """Loads and indexes flow property mappings for quick lookups."""

    def __init__(self, mapping_path: Path | None = None) -> None:
        self._mapping_path = mapping_path or MAPPING_PATH
        self._properties = self._load()
        self._by_uuid = {descriptor.uuid.lower(): descriptor for descriptor in self._properties}
        self._by_name = {descriptor.name.lower(): descriptor for descriptor in self._properties}

    def _load(self) -> tuple[FlowPropertyDescriptor, ...]:
        if not self._mapping_path.exists():
            raise FileNotFoundError(f"Flow property mapping not found: {self._mapping_path}")
        raw = json.loads(self._mapping_path.read_text(encoding="utf-8"))
        descriptors: list[FlowPropertyDescriptor] = []
        for entry in raw:
            unit_group_raw = entry.get("unit_group") or {}
            units: list[UnitDescriptor] = []
            for unit in unit_group_raw.get("units") or ():
                units.append(
                    UnitDescriptor(
                        name=str(unit.get("name") or ""),
                        mean_value=str(unit.get("mean_value") or ""),
                        internal_id=str(unit.get("internal_id") or ""),
                        general_comment=unit.get("general_comment"),
                        is_reference=bool(unit.get("is_reference")),
                    )
                )
            unit_group = UnitGroupDescriptor(
                uuid=str(unit_group_raw.get("uuid") or ""),
                name=str(unit_group_raw.get("name") or ""),
                reference_internal_id=str(unit_group_raw.get("reference_internal_id") or "0"),
                units=tuple(units),
            )
            descriptors.append(
                FlowPropertyDescriptor(
                    uuid=str(entry.get("flow_property_uuid") or ""),
                    name=str(entry.get("flow_property_name") or ""),
                    classification=tuple(entry.get("flow_property_classification") or ()),
                    reference_unit_description=str(entry.get("flow_property_reference_unit_description") or ""),
                    reference_unit_group_uuid=str(entry.get("reference_unit_group_uuid") or ""),
                    unit_group=unit_group,
                )
            )
        return tuple(descriptors)

    def list(self) -> tuple[FlowPropertyDescriptor, ...]:
        """Return every flow property descriptor."""
        return self._properties

    def get(self, uuid: str) -> FlowPropertyDescriptor:
        """Return descriptor by UUID (case insensitive)."""
        descriptor = self._by_uuid.get(uuid.lower())
        if descriptor is None:
            raise KeyError(f"Unknown flow property UUID: {uuid}")
        return descriptor

    def find(self, name: str) -> FlowPropertyDescriptor | None:
        """Return descriptor by exact name (case insensitive)."""
        return self._by_name.get(name.lower())

    def search_by_unit(self, unit_name: str) -> list[FlowPropertyDescriptor]:
        """Find potential flow properties that contain the given unit."""
        target = unit_name.strip().lower()
        matches: list[FlowPropertyDescriptor] = []
        for descriptor in self._properties:
            for unit in descriptor.unit_group.units:
                if unit.name.lower() == target:
                    matches.append(descriptor)
                    break
        return matches

    def fuzzy_match(self, text: str) -> FlowPropertyDescriptor | None:
        """Attempt to match property name against free text."""
        candidate = text.strip().lower()
        if not candidate:
            return None
        direct = self._by_name.get(candidate)
        if direct:
            return direct
        for descriptor in self._properties:
            if descriptor.name.lower() in candidate:
                return descriptor
            if candidate in descriptor.name.lower():
                return descriptor
        return None

    def build_flow_property_block(
        self,
        flow_property_uuid: str,
        *,
        mean_value: str = "1.0",
        data_set_internal_id: str | None = None,
        version_override: str | None = None,
    ) -> dict[str, object]:
        """Return an ILCD-compatible flowProperties block."""
        descriptor = self.get(flow_property_uuid)
        version = version_override or FLOW_PROPERTY_VERSION_OVERRIDES.get(descriptor.uuid, DEFAULT_FLOW_PROPERTY_VERSION)
        internal_id = data_set_internal_id or descriptor.unit_group.reference_internal_id or "0"
        reference = {
            "@type": "flow property data set",
            "@refObjectId": descriptor.uuid,
            "@uri": f"../flowproperties/{descriptor.uuid}.xml",
            "@version": version,
            "common:shortDescription": {
                "@xml:lang": "en",
                "#text": descriptor.name,
            },
        }
        return {
            "flowProperty": {
                "@dataSetInternalID": internal_id,
                "meanValue": mean_value,
                "referenceToFlowPropertyDataSet": reference,
            }
        }


@lru_cache(maxsize=1)
def get_default_registry() -> FlowPropertyRegistry:
    """Return a cached registry instance."""
    return FlowPropertyRegistry()
