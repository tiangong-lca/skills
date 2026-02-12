"""Converters that map OpenLCA JSON-LD payloads into ILCD-compatible datasets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from tiangong_lca_spec.tidas.flow_classification_registry import ensure_valid_product_flow_classification
from tiangong_lca_spec.tidas.process_classification_registry import ensure_valid_classification_path

ILCD_PROCESS_XMLNS = {
    "@xmlns": "http://lca.jrc.it/ILCD/Process",
    "@xmlns:common": "http://lca.jrc.it/ILCD/Common",
    "@xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
    "@xsi:schemaLocation": "http://lca.jrc.it/ILCD/Process ../../schemas/ILCD_ProcessDataSet.xsd",
    "@version": "1.1",
    "@locations": "../ILCDLocations.xml",
}

ILCD_FLOW_XMLNS = {
    "@xmlns": "http://lca.jrc.it/ILCD/Flow",
    "@xmlns:common": "http://lca.jrc.it/ILCD/Common",
    "@xmlns:ecn": "http://eplca.jrc.ec.europa.eu/ILCD/Extensions/2018/ECNumber",
    "@xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
    "@xsi:schemaLocation": "http://lca.jrc.it/ILCD/Flow ../../schemas/ILCD_FlowDataSet.xsd",
    "@version": "1.1",
    "@locations": "../ILCDLocations.xml",
}

ILCD_SOURCE_XMLNS = {
    "@xmlns": "http://lca.jrc.it/ILCD/Source",
    "@xmlns:common": "http://lca.jrc.it/ILCD/Common",
    "@xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
    "@xsi:schemaLocation": "http://lca.jrc.it/ILCD/Source ../../schemas/ILCD_SourceDataSet.xsd",
    "@version": "1.1",
}

TIANGONG_CONTACT_UUID = "f4b4c314-8c4c-4c83-968f-5b3c7724f6a8"
TIANGONG_CONTACT_URI = "../contacts/f4b4c314-8c4c-4c83-968f-5b3c7724f6a8_01.00.000.xml"
TIANGONG_CONTACT_VERSION = "01.00.000"
TIANGONG_CONTACT_SHORT_DESCRIPTION = [
    {"@xml:lang": "en", "#text": "Tiangong LCA Data Working Group"},
    {"@xml:lang": "zh", "#text": "天工LCA数据团队"},
]
ILCD_FORMAT_SOURCE_UUID = "a97a0155-0234-4b87-b4ce-a45da52f2a40"
ILCD_FORMAT_SOURCE_URI = "../sources/a97a0155-0234-4b87-b4ce-a45da52f2a40_03.00.003.xml"
ILCD_FORMAT_SOURCE_VERSION = "03.00.003"
ILCD_COMPLIANCE_SOURCE_UUID = "d92a1a12-2545-49e2-a585-55c259997756"
ILCD_COMPLIANCE_SOURCE_URI = "../sources/d92a1a12-2545-49e2-a585-55c259997756_20.20.002.xml"
ILCD_COMPLIANCE_SOURCE_VERSION = "20.20.002"
DEFAULT_LICENSE = "Free of charge for all users and uses"
DEFAULT_DATA_SET_VERSION = "01.01.000"
ILCD_FLOW_PROPERTY_XMLNS = {
    "@xmlns": "http://lca.jrc.it/ILCD/FlowProperty",
    "@xmlns:common": "http://lca.jrc.it/ILCD/Common",
    "@xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
    "@xsi:schemaLocation": "http://lca.jrc.it/ILCD/FlowProperty ../../schemas/ILCD_FlowPropertyDataSet.xsd",
    "@version": "1.1",
}
ILCD_UNIT_GROUP_XMLNS = {
    "@xmlns": "http://lca.jrc.it/ILCD/UnitGroup",
    "@xmlns:common": "http://lca.jrc.it/ILCD/Common",
    "@xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
    "@xsi:schemaLocation": "http://lca.jrc.it/ILCD/UnitGroup ../../schemas/ILCD_UnitGroupDataSet.xsd",
    "@version": "1.1",
}

SOURCE_CATEGORY_LOOKUP: dict[str, tuple[str, str]] = {
    "images": ("0", "Images"),
    "data set formats": ("1", "Data set formats"),
    "datasets": ("2", "Databases"),
    "databases": ("2", "Databases"),
    "compliance systems": ("3", "Compliance systems"),
    "statistical classifications": ("4", "Statistical classifications"),
    "publications and communications": ("5", "Publications and communications"),
    "other source types": ("6", "Other source types"),
}

SOURCE_CATEGORY_BY_CODE = {
    code: label
    for code, label in {
        "0": "Images",
        "1": "Data set formats",
        "2": "Databases",
        "3": "Compliance systems",
        "4": "Statistical classifications",
        "5": "Publications and communications",
        "6": "Other source types",
    }.items()
}

NAME_SEGMENT_SPLITTER = re.compile(r"[;|]+")
YEAR_TOKEN_PATTERN = re.compile(r"^\s*(\d{4})(?:\s*[-–/]\s*(\d{2,4}))?\s*$")
COUNTRY_CODE_PATTERN = re.compile(r"^[A-Z]{2}$")
MIX_KEYWORD_FRAGMENTS = (
    "production mix",
    "consumption mix",
    "market mix",
    "market group",
    "market for",
    "supply mix",
    "grid mix",
    "average mix",
)
MIX_KEYWORD_TRAILERS = (" at plant", " to consumer")
LOCATION_PHRASES = {
    "global",
    "world",
    "worldwide",
    "china",
    "mainland china",
    "people's republic of china",
    "asia",
    "east asia",
    "southeast asia",
    "europe",
    "european union",
    "north america",
    "south america",
    "latin america",
    "germany",
    "japan",
    "india",
    "korea",
    "republic of korea",
    "korea, republic of",
    "taiwan",
    "canada",
    "united states",
    "united kingdom",
    "great britain",
    "australia",
}


def _clean_name_token(value: str) -> str:
    return value.strip().strip(",;:()[]{}")


def _looks_like_year_token(value: str) -> bool:
    cleaned = re.sub(r"[^0-9\-–/]", "", value)
    if not cleaned:
        return False
    if YEAR_TOKEN_PATTERN.match(cleaned):
        return True
    if cleaned.isdigit() and len(cleaned) == 4:
        return True
    return False


def _is_location_token(value: str) -> bool:
    stripped = re.sub(r"[^A-Za-z\s]", " ", value).strip().lower()
    if not stripped:
        return False
    normalised = " ".join(stripped.split())
    if normalised in LOCATION_PHRASES:
        return True
    words = normalised.split()
    location_words = {"cn", "china", "global", "world", "asia", "europe", "germany", "japan", "india", "korea", "taiwan", "canada", "united", "states", "kingdom", "america"}
    if any(word in location_words for word in words):
        return True
    if COUNTRY_CODE_PATTERN.match(value.strip()):
        return True
    return False


def _is_mix_descriptor(value: str) -> bool:
    lowered = " ".join(value.strip().lower().split())
    if not lowered:
        return False
    if "generic" in lowered:
        return True
    for fragment in MIX_KEYWORD_FRAGMENTS:
        if fragment in lowered:
            return True
    if lowered.startswith("market ") or lowered.startswith("mix "):
        return True
    for suffix in MIX_KEYWORD_TRAILERS:
        if lowered.endswith(suffix):
            return True
    words = lowered.replace(",", " ").split()
    if any(word in {"mix", "market"} for word in words):
        return True
    return False


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _parse_process_name_fields(raw_name: str) -> tuple[str, str | None, str | None, str | None]:
    text = (raw_name or "").strip()
    if not text:
        return "Unnamed process", None, None, None
    segments = [_clean_name_token(segment) for segment in NAME_SEGMENT_SPLITTER.split(text)]
    segments = [segment for segment in segments if segment]
    base_name = ""
    detail_sources: list[str] = []
    pre_base_details: list[str] = []
    for segment in segments:
        if not base_name:
            if _looks_like_year_token(segment):
                continue
            if _is_mix_descriptor(segment) or _is_location_token(segment):
                pre_base_details.append(segment)
                continue
            base_name = segment
            continue
        detail_sources.append(segment)
    if not base_name:
        base_name = text
    detail_sources = pre_base_details + detail_sources

    detail_tokens: list[str] = []
    for entry in detail_sources:
        parts = [_clean_name_token(part) for part in entry.split(",")]
        for part in parts:
            if not part:
                continue
            if _looks_like_year_token(part):
                continue
            if part.lower() == base_name.lower():
                continue
            detail_tokens.append(part)

    treatment_parts: list[str] = []
    mix_parts: list[str] = []
    for token in detail_tokens:
        if _is_location_token(token) or _is_mix_descriptor(token):
            _append_unique(mix_parts, token)
        else:
            _append_unique(treatment_parts, token)

    treatment_text = ", ".join(treatment_parts) or None
    mix_text = ", ".join(mix_parts) or None
    return base_name, treatment_text, mix_text, None


def collect_jsonld_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(p for p in path.rglob("*.json") if p.is_file())
    if not files:
        raise SystemExit(f"No JSON-LD files found under {path}")
    return files


def _as_language_entry(text: str | None, lang: str = "en") -> dict[str, str]:
    return {"@xml:lang": lang, "#text": (text or "").strip() or "Unnamed"}


def _parse_category_path(category: str | None) -> list[dict[str, str]]:
    if not category:
        return []
    entries: list[dict[str, str]] = []
    segments = [segment.strip() for segment in category.split("/") if segment.strip()]
    for index, segment in enumerate(segments):
        if ":" in segment:
            class_id, label = segment.split(":", 1)
        else:
            class_id, label = segment, segment
        entries.append({"@level": str(index), "@classId": class_id.strip(), "#text": label.strip()})
    return entries


def _process_classification_from_category(category: str | None) -> list[dict[str, str]]:
    entries = _parse_category_path(category)
    if entries:
        try:
            return ensure_valid_classification_path(tuple(entries))
        except ValueError:
            raise ValueError(f"Unsupported process category path: {category!r}")
    raise ValueError("Process category is required to determine Tiangong classification")


def _default_location_code(name: str | None, *, fallback_to_global: bool = True) -> str | None:
    if not name:
        return "GLO" if fallback_to_global else None
    lowered = name.strip().lower()
    if not lowered:
        return "GLO" if fallback_to_global else None
    if lowered in {"glo", "global", "world", "worldwide"}:
        return "GLO"
    if "china" in lowered or lowered in {"cn", "people's republic of china", "prc"}:
        return "CN"
    if "united states" in lowered or lowered in {"usa", "us"}:
        return "US"
    if fallback_to_global:
        return "GLO"
    return None


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _contact_reference() -> dict[str, Any]:
    return {
        "@refObjectId": TIANGONG_CONTACT_UUID,
        "@type": "contact data set",
        "@uri": TIANGONG_CONTACT_URI,
        "@version": TIANGONG_CONTACT_VERSION,
        "common:shortDescription": [dict(entry) for entry in TIANGONG_CONTACT_SHORT_DESCRIPTION],
    }


def _format_reference() -> dict[str, Any]:
    return {
        "@refObjectId": ILCD_FORMAT_SOURCE_UUID,
        "@type": "source data set",
        "@uri": ILCD_FORMAT_SOURCE_URI,
        "@version": ILCD_FORMAT_SOURCE_VERSION,
        "common:shortDescription": _as_language_entry("ILCD format", "en"),
    }


def _ownership_reference() -> dict[str, Any]:
    return _contact_reference()


def _compliance_reference() -> dict[str, Any]:
    return {
        "@refObjectId": ILCD_COMPLIANCE_SOURCE_UUID,
        "@type": "source data set",
        "@uri": ILCD_COMPLIANCE_SOURCE_URI,
        "@version": ILCD_COMPLIANCE_SOURCE_VERSION,
        "common:shortDescription": {
            "@xml:lang": "en",
            "#text": "ILCD Data Network - Entry-level",
        },
    }


def _default_intended_applications() -> list[dict[str, str]]:
    return [_as_language_entry("Life cycle data prepared for Tiangong LCA Spec Coding workflow automation", "en")]


def _flow_classification_from_category(category: str | None) -> list[dict[str, str]]:
    entries = _parse_category_path(category)
    if entries:
        try:
            return ensure_valid_product_flow_classification(tuple(entries))
        except ValueError as exc:
            raise ValueError(f"Unsupported flow category path: {category!r}") from exc
    raise ValueError("Flow category is required to determine Tiangong product classification")


def _source_classification_from_category(category: str | None) -> dict[str, str]:
    if not category:
        raise ValueError("Source category is required to determine Tiangong classification")
    entries = _parse_category_path(category)
    if entries:
        entry = entries[0]
        class_id = (entry.get("@classId") or "").strip()
        label = (entry.get("#text") or "").strip()
        if class_id in SOURCE_CATEGORY_BY_CODE:
            resolved_label = label or SOURCE_CATEGORY_BY_CODE[class_id]
            return {"@level": "0", "@classId": class_id, "#text": resolved_label}
    normalized = category.strip()
    key = normalized.lower()
    if key in SOURCE_CATEGORY_LOOKUP:
        class_id, text = SOURCE_CATEGORY_LOOKUP[key]
        return {"@level": "0", "@classId": class_id, "#text": text}
    if ":" in normalized:
        prefix, label = normalized.split(":", 1)
        code = prefix.strip()
        if code in SOURCE_CATEGORY_BY_CODE:
            resolved_label = label.strip() or SOURCE_CATEGORY_BY_CODE[code]
            return {"@level": "0", "@classId": code, "#text": resolved_label}
    raise ValueError(f"Unsupported source category: {category!r}")


def _derive_short_name(name: str) -> str:
    stripped = (name or "").strip()
    if not stripped:
        return "Source"
    return stripped


def _guess_publication_type(category: str | None) -> str:
    lowered = (category or "").lower()
    if "article" in lowered or "journal" in lowered or "publication" in lowered:
        return "Article in periodical"
    if "manual" in lowered or "monograph" in lowered:
        return "Monograph"
    if "image" in lowered:
        return "Other unpublished and grey literature"
    if "questionnaire" in lowered:
        return "Questionnaire"
    if "software" in lowered or "database" in lowered:
        return "Software or database"
    return "Article in periodical"


def _ensure_exchange_direction(is_input: bool | None) -> str:
    return "Input" if is_input else "Output"


def _reference_to_flow(flow: dict[str, Any]) -> dict[str, Any]:
    ref_uuid = flow.get("@id") or str(uuid4())
    description = flow.get("name") or "Unnamed flow"
    version = DEFAULT_DATA_SET_VERSION
    return {
        "@type": "flow data set",
        "@refObjectId": ref_uuid,
        "@uri": f"../flows/{ref_uuid}_{version}.xml",
        "@version": version,
        "common:shortDescription": _as_language_entry(description, "en"),
    }


def _reference_to_flow_property(flow_property: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(flow_property, dict):
        return None
    ref_uuid = flow_property.get("@id")
    if not ref_uuid:
        return None
    version = flow_property.get("version") or "01.01.000"
    return {
        "@type": "flow property data set",
        "@refObjectId": ref_uuid,
        "@uri": f"../flowproperties/{ref_uuid}_{version}.xml",
        "@version": version,
        "common:shortDescription": _as_language_entry(flow_property.get("name"), "en"),
    }


def _reference_to_unit_group(unit_group: dict[str, Any]) -> dict[str, Any]:
    ref_uuid = unit_group.get("@id") or str(uuid4())
    version = unit_group.get("version") or "01.01.000"
    return {
        "@type": "unit group data set",
        "@refObjectId": ref_uuid,
        "@uri": f"../unitgroups/{ref_uuid}_{version}.xml",
        "@version": version,
        "common:shortDescription": _as_language_entry(unit_group.get("name") or "Unit group", "en"),
    }


@dataclass(slots=True)
class JSONLDProcessConverter:
    jsonld_path: Path

    def load(self) -> dict[str, Any]:
        return json.loads(self.jsonld_path.read_text(encoding="utf-8"))

    def to_process_block(self) -> dict[str, Any]:
        payload = self.load()
        dataset_uuid = payload.get("@id") or str(uuid4())
        name = payload.get("name") or "Unnamed process"
        base_name, treatment_text, mix_text, functional_text = _parse_process_name_fields(name)
        description = payload.get("description")
        general_comment_entry = None
        if isinstance(description, str) and description.strip():
            general_comment_entry = _as_language_entry(description, "en")
        category = payload.get("category")
        try:
            classification = _process_classification_from_category(category)
        except ValueError as exc:
            raise ValueError(f"{self.jsonld_path}: {exc}") from exc
        location_block = payload.get("location") or {}
        location_code = _default_location_code(location_block.get("name"))

        documentation = payload.get("processDocumentation") or {}
        reference_year = None
        valid_from = documentation.get("validFrom")
        if isinstance(valid_from, str) and valid_from[:4].isdigit():
            reference_year = int(valid_from[:4])

        exchanges_payload = payload.get("exchanges") or []
        exchanges: list[dict[str, Any]] = []
        reference_flow_id: str | None = None
        for idx, exchange in enumerate(exchanges_payload, start=1):
            flow = exchange.get("flow") or {}
            unit = exchange.get("unit") or {}
            amount = exchange.get("amount")
            data_entry: dict[str, Any] = {
                "@dataSetInternalID": str(idx),
                "exchangeDirection": _ensure_exchange_direction(exchange.get("isInput")),
                "meanAmount": str(amount) if amount is not None else "0",
                "unit": unit.get("name") or unit.get("@id") or "",
                "exchangeName": flow.get("name") or "Unnamed flow",
            }
            data_entry["referenceToFlowDataSet"] = _reference_to_flow(flow)
            flow_property = exchange.get("flowProperty") or {}
            property_ref = _reference_to_flow_property(flow_property)
            if property_ref:
                data_entry["referenceToFlowPropertyDataSet"] = property_ref
            if exchange.get("isQuantitativeReference") and reference_flow_id is None:
                reference_flow_id = str(idx)
            exchanges.append(data_entry)

        if not reference_flow_id and exchanges:
            reference_flow_id = exchanges[0]["@dataSetInternalID"]

        name_block: dict[str, Any] = {
            "baseName": _as_language_entry(base_name, "en"),
        }
        if treatment_text:
            name_block["treatmentStandardsRoutes"] = _as_language_entry(treatment_text, "en")
        if mix_text:
            name_block["mixAndLocationTypes"] = _as_language_entry(mix_text, "en")
        if functional_text:
            name_block["functionalUnitFlowProperties"] = _as_language_entry(functional_text, "en")

        process_dataset = {
            **ILCD_PROCESS_XMLNS,
            "processInformation": {
                "dataSetInformation": {
                    "common:UUID": dataset_uuid,
                    "name": name_block,
                    **({"common:generalComment": general_comment_entry} if general_comment_entry else {}),
                    "classificationInformation": {"common:classification": {"common:class": classification or [{"@level": "0", "@classId": "Z", "#text": "Unspecified"}]}},
                },
                "quantitativeReference": {
                    "referenceToReferenceFlow": reference_flow_id or "1",
                    "functionalUnitOrOther": _as_language_entry("Reference flow of process", "en"),
                },
                "time": {
                    "common:referenceYear": reference_year or 2020,
                },
                "geography": {
                    "locationOfOperationSupplyOrProduction": {"@location": location_code},
                },
            },
            "modellingAndValidation": {
                "LCIMethodAndAllocation": {
                    "typeOfDataSet": "Unit process, single operation",
                    "LCIMethodPrinciple": "Attributional",
                }
            },
            "administrativeInformation": {
                "dataEntryBy": {
                    "common:referenceToDataSetFormat": _format_reference(),
                },
                "publicationAndOwnership": {
                    "common:dataSetVersion": payload.get("version") or "01.01.000",
                    "common:permanentDataSetURI": f"https://lcdn.tiangong.earth/showProcess.xhtml?uuid={dataset_uuid}&version=01.01.000",
                },
                "common:commissionerAndGoal": {
                    "common:referenceToCommissioner": _contact_reference(),
                    "common:intendedApplications": _default_intended_applications(),
                },
            },
            "exchanges": {
                "exchange": exchanges,
            },
        }

        return {
            "processDataSet": process_dataset,
            "process_id": dataset_uuid,
        }


@dataclass(slots=True)
class JSONLDFlowConverter:
    jsonld_path: Path

    def load(self) -> dict[str, Any]:
        return json.loads(self.jsonld_path.read_text(encoding="utf-8"))

    def to_flow_dataset(self) -> dict[str, Any]:
        payload = self.load()
        flow_uuid = payload.get("@id") or str(uuid4())
        name = payload.get("name") or "Unnamed flow"
        dataset_version = DEFAULT_DATA_SET_VERSION
        try:
            classification = _flow_classification_from_category(payload.get("category"))
        except ValueError as exc:
            raise ValueError(f"{self.jsonld_path}: {exc}") from exc
        description = payload.get("description")
        name_block = {
            "baseName": _as_language_entry(name, "en"),
            "treatmentStandardsRoutes": _as_language_entry("Standard treatment not specified", "en"),
            "mixAndLocationTypes": _as_language_entry("Production mix, at plant", "en"),
            "functionalUnitFlowProperties": _as_language_entry("Declared per reference flow property", "en"),
        }
        flow_properties_payload = payload.get("flowProperties") or []
        flow_properties: list[dict[str, Any]] = []
        for idx, factor in enumerate(flow_properties_payload, start=1):
            reference = _reference_to_flow_property(factor.get("flowProperty") or {})
            if reference is None:
                continue
            flow_properties.append(
                {
                    "@dataSetInternalID": str(idx),
                    "meanValue": str(factor.get("conversionFactor", 1)),
                    "referenceToFlowPropertyDataSet": reference,
                }
            )
        if not flow_properties:
            flow_properties.append(
                {
                    "@dataSetInternalID": "1",
                    "meanValue": "1",
                    "referenceToFlowPropertyDataSet": {
                        "@type": "flow property data set",
                        "@refObjectId": "93a60a56-a3c8-11da-a746-0800200b9a66",
                        "@uri": "../flowproperties/93a60a56-a3c8-11da-a746-0800200b9a66_03.00.003.xml",
                        "@version": "03.00.003",
                        "common:shortDescription": _as_language_entry("Mass", "en"),
                    },
                }
            )
        reference_flow_property_id = flow_properties[0]["@dataSetInternalID"]
        location_code = _default_location_code((payload.get("location") or {}).get("name"), fallback_to_global=False)
        general_comment_entries = None
        if isinstance(description, str) and description.strip():
            general_comment_entries = [_as_language_entry(description, "en")]

        flow_dataset = {
            "flowDataSet": {
                **ILCD_FLOW_XMLNS,
                "flowInformation": {
                    "dataSetInformation": {
                        "common:UUID": flow_uuid,
                        "name": name_block,
                        "common:synonyms": [_as_language_entry(name, "en")],
                        **({"common:generalComment": general_comment_entries} if general_comment_entries else {}),
                        "classificationInformation": {"common:classification": {"common:class": classification}},
                    },
                    "quantitativeReference": {
                        "referenceToReferenceFlowProperty": reference_flow_property_id,
                    },
                    **({"geography": {"locationOfSupply": location_code}} if location_code else {}),
                    "technology": {
                        "technologicalApplicability": [_as_language_entry("Applicable to generic supply mixes.", "en")],
                    },
                },
                "flowProperties": {
                    "flowProperty": flow_properties,
                },
                "modellingAndValidation": {
                    "LCIMethod": {
                        "typeOfDataSet": "Product flow",
                    },
                    "complianceDeclarations": {
                        "compliance": {
                            "common:referenceToComplianceSystem": _compliance_reference(),
                            "common:approvalOfOverallCompliance": "Fully compliant",
                            "common:nomenclatureCompliance": "Fully compliant",
                            "common:methodologicalCompliance": "Not defined",
                            "common:reviewCompliance": "Not defined",
                            "common:documentationCompliance": "Not defined",
                            "common:qualityCompliance": "Not defined",
                        }
                    },
                },
                "administrativeInformation": {
                    "dataEntryBy": {
                        "common:referenceToDataSetFormat": _format_reference(),
                        "common:referenceToPersonOrEntityEnteringTheData": _contact_reference(),
                        "common:timeStamp": _current_timestamp(),
                    },
                    "publicationAndOwnership": {
                        "common:dataSetVersion": dataset_version,
                        "common:permanentDataSetURI": f"https://lcdn.tiangong.earth/showFlow.xhtml?uuid={flow_uuid}&version={dataset_version}",
                        "common:licenseType": DEFAULT_LICENSE,
                        "common:copyright": "false",
                        "common:referenceToOwnershipOfDataSet": _ownership_reference(),
                    },
                },
            }
        }
        return flow_dataset


@dataclass(slots=True)
class JSONLDFlowPropertyConverter:
    jsonld_path: Path

    def load(self) -> dict[str, Any]:
        return json.loads(self.jsonld_path.read_text(encoding="utf-8"))

    def to_flow_property_dataset(self) -> dict[str, Any]:
        payload = self.load()
        property_uuid = payload.get("@id") or str(uuid4())
        name = payload.get("name") or "Flow property"
        category = payload.get("category")
        classification = _parse_category_path(category)
        unit_group = payload.get("unitGroup") or {}
        dataset = {
            "flowPropertyDataSet": {
                **ILCD_FLOW_PROPERTY_XMLNS,
                "flowPropertiesInformation": {
                    "dataSetInformation": {
                        "common:UUID": property_uuid,
                        "common:name": _as_language_entry(name, "en"),
                        "common:synonyms": [_as_language_entry(name, "en")],
                        "classificationInformation": {"common:classification": {"common:class": classification or [{"@level": "0", "@classId": "Z", "#text": "Unspecified"}]}},
                    },
                    "quantitativeReference": {
                        "referenceToReferenceUnitGroup": _reference_to_unit_group(unit_group),
                    },
                },
                "administrativeInformation": {
                    "dataEntryBy": {
                        "common:referenceToDataSetFormat": {
                            "@refObjectId": "a97a0155-0234-4b87-b4ce-a45da52f2a40",
                            "@type": "source data set",
                            "@uri": "../sources/a97a0155-0234-4b87-b4ce-a45da52f2a40_03.00.003.xml",
                            "@version": "03.00.003",
                            "common:shortDescription": _as_language_entry("ILCD format", "en"),
                        }
                    },
                    "publicationAndOwnership": {
                        "common:dataSetVersion": payload.get("version") or "01.01.000",
                    },
                },
            }
        }
        return dataset


@dataclass(slots=True)
class JSONLDUnitGroupConverter:
    jsonld_path: Path

    def load(self) -> dict[str, Any]:
        return json.loads(self.jsonld_path.read_text(encoding="utf-8"))

    def to_unit_group_dataset(self) -> dict[str, Any]:
        payload = self.load()
        group_uuid = payload.get("@id") or str(uuid4())
        name = payload.get("name") or "Unit group"
        category = payload.get("category")
        classification = _parse_category_path(category)
        units_payload = payload.get("units") or []
        units: list[dict[str, Any]] = []
        for idx, unit in enumerate(units_payload):
            units.append(
                {
                    "@dataSetInternalID": str(idx),
                    "name": unit.get("name") or unit.get("@id") or f"unit_{idx}",
                    "meanValue": str(unit.get("conversionFactor", 1)),
                }
            )
        if not units:
            units.append({"@dataSetInternalID": "0", "name": "1", "meanValue": "1"})

        dataset = {
            "unitGroupDataSet": {
                **ILCD_UNIT_GROUP_XMLNS,
                "unitGroupInformation": {
                    "dataSetInformation": {
                        "common:UUID": group_uuid,
                        "common:name": _as_language_entry(name, "en"),
                        "classificationInformation": {"common:classification": {"common:class": classification or [{"@level": "0", "@classId": "Z", "#text": "Unspecified"}]}},
                    },
                    "quantitativeReference": {
                        "referenceToReferenceUnit": "0",
                    },
                },
                "administrativeInformation": {
                    "dataEntryBy": {
                        "common:referenceToDataSetFormat": {
                            "@refObjectId": "a97a0155-0234-4b87-b4ce-a45da52f2a40",
                            "@type": "source data set",
                            "@uri": "../sources/a97a0155-0234-4b87-b4ce-a45da52f2a40_03.00.003.xml",
                            "@version": "03.00.003",
                            "common:shortDescription": _as_language_entry("ILCD format", "en"),
                        }
                    },
                    "publicationAndOwnership": {
                        "common:dataSetVersion": payload.get("version") or "01.01.000",
                    },
                },
                "units": {"unit": units},
            }
        }
        return dataset


@dataclass(slots=True)
class JSONLDSourceConverter:
    jsonld_path: Path

    def load(self) -> dict[str, Any]:
        return json.loads(self.jsonld_path.read_text(encoding="utf-8"))

    def to_source_dataset(self) -> dict[str, Any]:
        payload = self.load()
        source_uuid = payload.get("@id") or str(uuid4())
        name = payload.get("name") or "Source"
        category = payload.get("category")
        try:
            classification = _source_classification_from_category(category)
        except ValueError as exc:
            raise ValueError(f"{self.jsonld_path}: {exc}") from exc
        dataset_version = payload.get("version") or "01.01.000"
        short_name = _derive_short_name(name)
        description = payload.get("description")
        description_entries = None
        if isinstance(description, str) and description.strip():
            description_entries = [_as_language_entry(description, "en")]
        publication_type = _guess_publication_type(category)
        citation = payload.get("textReference") or name or short_name
        dataset = {
            "sourceDataSet": {
                **ILCD_SOURCE_XMLNS,
                "sourceInformation": {
                    "dataSetInformation": {
                        "common:UUID": source_uuid,
                        "common:shortName": _as_language_entry(short_name, "en"),
                        "classificationInformation": {"common:classification": {"common:class": classification}},
                        "sourceCitation": citation,
                        "publicationType": publication_type,
                        **({"sourceDescriptionOrComment": description_entries} if description_entries else {}),
                        "referenceToContact": _contact_reference(),
                    }
                },
                "administrativeInformation": {
                    "dataEntryBy": {
                        "common:timeStamp": _current_timestamp(),
                        "common:referenceToDataSetFormat": _format_reference(),
                    },
                    "publicationAndOwnership": {
                        "common:dataSetVersion": dataset_version,
                        "common:permanentDataSetURI": f"https://lcdn.tiangong.earth/showSource.xhtml?uuid={source_uuid}&version={dataset_version}",
                        "common:referenceToOwnershipOfDataSet": _ownership_reference(),
                    },
                },
            }
        }
        return dataset


def convert_process_directory(path: Path) -> list[dict[str, Any]]:
    files = collect_jsonld_files(path)
    return [JSONLDProcessConverter(file_path).to_process_block() for file_path in files]


def convert_flow_directory(path: Path) -> list[dict[str, Any]]:
    files = collect_jsonld_files(path)
    return [JSONLDFlowConverter(file_path).to_flow_dataset() for file_path in files]


def convert_flow_property_directory(path: Path) -> list[dict[str, Any]]:
    files = collect_jsonld_files(path)
    return [JSONLDFlowPropertyConverter(file_path).to_flow_property_dataset() for file_path in files]


def convert_unit_group_directory(path: Path) -> list[dict[str, Any]]:
    files = collect_jsonld_files(path)
    return [JSONLDUnitGroupConverter(file_path).to_unit_group_dataset() for file_path in files]


def convert_source_directory(path: Path) -> list[dict[str, Any]]:
    files = collect_jsonld_files(path)
    return [JSONLDSourceConverter(file_path).to_source_dataset() for file_path in files]
