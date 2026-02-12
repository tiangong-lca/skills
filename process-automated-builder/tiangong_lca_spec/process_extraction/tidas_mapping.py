"""Utilities for normalising process datasets to the TIDAS ILCD schema."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any
from uuid import uuid4

from tiangong_lca_spec.core.constants import build_dataset_format_reference
from tiangong_lca_spec.core.exceptions import ProcessExtractionError
from tiangong_lca_spec.core.uris import build_local_dataset_uri, build_portal_uri
from tiangong_lca_spec.tidas import get_schema_repository
from tiangong_lca_spec.tidas.process_classification_registry import (
    ensure_valid_classification_path,
)

BASE_METADATA = {
    "@xmlns:common": "http://lca.jrc.it/ILCD/Common",
    "@xmlns": "http://lca.jrc.it/ILCD/Process",
    "@xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
    "@version": "1.1",
    "@locations": "../ILCDLocations.xml",
    "@xsi:schemaLocation": "http://lca.jrc.it/ILCD/Process ../../schemas/ILCD_ProcessDataSet.xsd",
}

DEFAULT_LOCATION = "GLO"
DEFAULT_REFERENCE_TYPE = "Reference flow(s)"
DEFAULT_REFERENCE_ID = "0"
DEFAULT_LANGUAGE = "en"
DEFAULT_DATA_SET_VERSION = "01.01.000"

ILCD_ENTRY_LEVEL_REFERENCE_ID = "d92a1a12-2545-49e2-a585-55c259997756"
ILCD_ENTRY_LEVEL_REFERENCE_VERSION = "20.20.002"

COMPLIANCE_BASE_POINTER = "/properties/modellingAndValidation/properties/complianceDeclarations/properties/compliance"

COMPLIANCE_DEFAULT_PREFERENCES = {
    "common:approvalOfOverallCompliance": "Fully compliant",
    "common:nomenclatureCompliance": "Fully compliant",
    "common:methodologicalCompliance": "Not defined",
    "common:reviewCompliance": "Not defined",
    "common:documentationCompliance": "Not defined",
    "common:qualityCompliance": "Not defined",
}

TYPE_OF_DATA_SET_POINTER = "/properties/modellingAndValidation/properties/LCIMethodAndAllocation/properties/typeOfDataSet"
LCI_METHOD_PRINCIPLE_POINTER = "/properties/modellingAndValidation/properties/LCIMethodAndAllocation/properties/LCIMethodPrinciple"
LCI_METHOD_APPROACH_POINTER = "/properties/modellingAndValidation/properties/LCIMethodAndAllocation/properties/LCIMethodApproaches"


@dataclass(frozen=True)
class ProcessSchemaMetadata:
    enum_fields: dict[str, tuple[str, ...]]
    multilang_fields: frozenset[str]
    compliance_status_fields: tuple[str, ...]
    compliance_field_pointers: dict[str, str]


def build_tidas_process_dataset(process_dataset: dict[str, Any]) -> dict[str, Any]:
    """Return a normalised deep copy of the provided process dataset."""

    dataset = _apply_root_metadata(process_dataset)
    dataset["@locations"] = BASE_METADATA["@locations"]
    process_information = dataset.setdefault("processInformation", {})
    normalised_process_information, name_components = _normalise_process_information(process_information)
    dataset["processInformation"] = normalised_process_information
    dataset_uuid = normalised_process_information.get("dataSetInformation", {}).get("common:UUID")
    modelling = _normalise_modelling_and_validation(dataset.get("modellingAndValidation"))
    if modelling:
        dataset["modellingAndValidation"] = modelling
    else:
        dataset.pop("modellingAndValidation", None)
    administrative = _normalise_administrative_information(
        dataset.get("administrativeInformation"),
        dataset_uuid=dataset_uuid,
        dataset_kind="process",
    )
    if administrative:
        dataset["administrativeInformation"] = administrative
    else:
        dataset.pop("administrativeInformation", None)
    exchanges, reference_id = _normalise_exchanges(dataset.get("exchanges"), name_components)
    dataset["exchanges"] = exchanges
    if reference_id is not None:
        qref = dataset["processInformation"].setdefault("quantitativeReference", {})
        qref["@type"] = DEFAULT_REFERENCE_TYPE
        qref["referenceToReferenceFlow"] = str(reference_id)
        if "functionalUnitOrOther" in qref:
            qref["functionalUnitOrOther"] = _to_multilang(qref["functionalUnitOrOther"])
    lcia = _normalise_lcia_results(dataset.get("LCIAResults"))
    if lcia:
        dataset["LCIAResults"] = lcia
    else:
        dataset.pop("LCIAResults", None)
    _apply_schema_normalisation(dataset)
    _prune_invalid_reference_fields(dataset)
    return _strip_common_other(dataset)


def _apply_root_metadata(process_dataset: dict[str, Any]) -> dict[str, Any]:
    dataset = deepcopy(process_dataset if isinstance(process_dataset, dict) else {})
    for key, value in BASE_METADATA.items():
        dataset[key] = value
    return dataset


def _normalise_process_information(
    process_information: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    info = _ensure_dict(process_information)
    dataset_info, name_components = _normalise_dataset_information(
        info.get("dataSetInformation"),
    )
    info["dataSetInformation"] = dataset_info
    qref = _normalise_quantitative_reference(info.get("quantitativeReference"))
    info["quantitativeReference"] = qref
    name_components["functional_unit"] = _extract_functional_unit_text(qref)
    info["time"] = _normalise_time(info.get("time"))
    geography = _normalise_geography(info.get("geography"))
    info["geography"] = geography
    _finalise_mix_string(name_components, geography)
    dataset_name = info["dataSetInformation"].get("name", {})
    dataset_name["treatmentStandardsRoutes"] = _ensure_multilang(
        _format_name_field_text(name_components["treatment"]),
        separator=", ",
    )
    dataset_name["mixAndLocationTypes"] = _ensure_multilang(
        _format_name_field_text(name_components["mix"]),
        separator=", ",
    )
    info["dataSetInformation"]["name"] = dataset_name
    info["technology"] = _normalise_technology(info.get("technology"))
    if "technology" in info and not info["technology"]:
        info.pop("technology")
    mathematical_relations = _ensure_dict(info.get("mathematicalRelations"))
    if mathematical_relations:
        if "modelDescription" in mathematical_relations:
            entries = _ensure_multilang_list(mathematical_relations["modelDescription"])
            if entries:
                mathematical_relations["modelDescription"] = entries
            else:
                mathematical_relations.pop("modelDescription", None)
        info["mathematicalRelations"] = mathematical_relations
        if not mathematical_relations:
            info.pop("mathematicalRelations", None)
    else:
        info.pop("mathematicalRelations", None)
    return info, name_components


def _normalise_dataset_information(
    data_info: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    info = _ensure_dict(data_info)
    uuid_value = str(uuid4())
    info["common:UUID"] = uuid_value

    identifier = _stringify(info.get("identifierOfSubDataSet")).strip()
    if not identifier:
        identifier = uuid_value
    info["identifierOfSubDataSet"] = identifier

    name_block = _ensure_dict(info.get("name"))
    raw_general_comment = info.get("common:generalComment")
    general_comment_text = _extract_multilang_text(raw_general_comment).strip()
    name_fields = {field: name_block.get(field) for field in ("baseName", "treatmentStandardsRoutes", "mixAndLocationTypes", "functionalUnitFlowProperties")}
    base_name_text = _extract_multilang_text(name_fields.get("baseName"))
    authoritative_treatment_text = _format_name_field_text(_extract_multilang_text(name_fields.get("treatmentStandardsRoutes")))
    authoritative_mix_text = _format_name_field_text(_extract_multilang_text(name_fields.get("mixAndLocationTypes")))
    name_components = _derive_name_components(
        base_name_text,
        name_fields,
        general_comment_text,
    )
    base_text = _format_name_field_text(name_components["base"])
    treatment_text = _format_name_field_text(name_components["treatment"])
    mix_text = _format_name_field_text(name_components["mix"])
    functional_properties_text = _format_name_field_text(name_components.get("functional_unit_properties"))
    refreshed_name_block: dict[str, Any] = {}
    refreshed_name_block["baseName"] = _ensure_multilang(
        base_text,
        fallback="Unnamed process",
        separator=", ",
    )
    treatment_output_text = authoritative_treatment_text or treatment_text
    mix_output_text = authoritative_mix_text or mix_text
    if treatment_output_text:
        name_components["treatment"] = treatment_output_text
    if mix_output_text:
        name_components["mix"] = mix_output_text
    refreshed_name_block["treatmentStandardsRoutes"] = _ensure_multilang(treatment_output_text, separator=", ")
    refreshed_name_block["mixAndLocationTypes"] = _ensure_multilang(mix_output_text, separator=", ")
    if functional_properties_text:
        refreshed_name_block["functionalUnitFlowProperties"] = _ensure_multilang(functional_properties_text, separator=", ")
    info["name"] = refreshed_name_block

    classification_info = _ensure_dict(info.get("classificationInformation"))
    specification_text = (_stringify(classification_info.pop("specification", None)) or _stringify(info.pop("specification", None))).strip()
    classification = classification_info.get("classification")
    common_class = classification_info.get("common:classification")
    if isinstance(classification, list):
        classification_info["common:classification"] = {"common:class": classification}
    elif isinstance(common_class, dict) and "common:class" in common_class:
        pass
    else:
        classification_info.setdefault("common:classification", {"common:class": []})

    current_classes = classification_info.get("common:classification", {}).get("common:class")
    if not current_classes:
        candidate: Any = classification
        if candidate is None:
            candidate = info.get("classification")
        if candidate is None:
            raw_class = info.get("classificationInformation")
            if isinstance(raw_class, dict):
                candidate = raw_class.get("classification") or raw_class.get("common:class")
        if isinstance(candidate, list) and candidate:
            classification_info["common:classification"] = {"common:class": candidate}

    classes = classification_info.get("common:classification", {}).get("common:class")
    if isinstance(classes, list):
        collected_entries: list[dict[str, Any]] = []
        for index, entry in enumerate(classes):
            if index >= 4:
                break
            if not isinstance(entry, dict):
                continue
            class_id = entry.get("@classId") or entry.get("classId") or entry.get("id") or ""
            level = entry.get("@level") or entry.get("level") or entry.get("lvl") or ""
            text_value = entry.get("#text") or entry.get("text") or entry.get("@text") or entry.get("label") or entry.get("name") or class_id
            collected_entries.append(
                {
                    "@level": str(level).strip(),
                    "@classId": str(class_id).strip(),
                    "#text": _strip_class_code_prefix(_stringify(text_value)),
                }
            )
        try:
            normalised_classes = ensure_valid_classification_path(tuple(collected_entries))
        except ValueError as exc:
            raise ProcessExtractionError(f"Invalid process classification path: {exc}") from exc
        classification_info.setdefault("common:classification", {})["common:class"] = normalised_classes
    if specification_text:
        classification_info.setdefault("common:classification", {}).setdefault("common:other", specification_text)
    allowed_keys = {"common:classification", "common:other"}
    classification_info = {key: value for key, value in classification_info.items() if key in allowed_keys}

    info["classificationInformation"] = classification_info

    synonyms_value = info.get("common:synonyms")
    synonyms_entries = _ensure_multilang_list(synonyms_value)
    synonyms_entries = _filter_multilang_entries(synonyms_entries, target_lang=DEFAULT_LANGUAGE, clean_text=True)
    if synonyms_entries:
        info["common:synonyms"] = synonyms_entries
    else:
        info.pop("common:synonyms", None)

    general_comment_entries = _ensure_multilang_list(raw_general_comment)
    if not general_comment_entries and general_comment_text:
        general_comment_entries = _ensure_multilang_list(general_comment_text)
    if general_comment_entries:
        info["common:generalComment"] = general_comment_entries
    else:
        info.pop("common:generalComment", None)

    info.pop("referenceToExternalDocumentation", None)

    if not isinstance(info.get("common:mathematicalRelationsOrReasonsForDataDerivation"), dict):
        info.pop("common:mathematicalRelationsOrReasonsForDataDerivation", None)

    scope = _ensure_dict(info.get("scope"))
    scope.pop("defaultAllocationMethod", None)
    if scope:
        info["scope"] = scope
    else:
        info.pop("scope", None)

    owner = _ensure_dict(info.get("dataSetOwner"))
    if owner:
        info["dataSetOwner"] = owner
    else:
        info.pop("dataSetOwner", None)

    allowed_dataset_info_keys = {
        "common:UUID",
        "name",
        "identifierOfSubDataSet",
        "common:synonyms",
        "complementingProcesses",
        "classificationInformation",
        "common:generalComment",
    }
    info = {key: value for key, value in info.items() if key in allowed_dataset_info_keys and value not in (None, "", {}, [])}
    return info, name_components


def _strip_class_code_prefix(label: str) -> str:
    text = _stringify(label).strip()
    if not text:
        return ""
    match = re.match(r"^[A-Za-z0-9]+(?:[\.\-][A-Za-z0-9]+)*\s*[-–]\s*", text)
    if match:
        text = text[match.end() :].strip()
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    return text or _stringify(label)


FEEDSTOCK_KEYWORDS = [
    "coal",
    "lignite",
    "biomass",
    "wood",
    "natural gas",
    "shale gas",
    "crude oil",
    "petroleum",
    "diesel",
    "gasoline",
    "ethanol",
    "methanol",
    "hydrogen",
    "steam",
    "water",
    "electricity",
    "limestone",
    "iron ore",
    "aluminium",
    "copper",
]


def _derive_name_components(
    base_name: str,
    name_fields: dict[str, Any],
    general_comment: str,
) -> dict[str, Any]:
    base = base_name.strip() or "Unnamed process"
    product, initial_route = _split_product_and_route(base)
    expanded_sources = []
    for value in name_fields.values():
        text_value = _extract_multilang_text(value)
        if text_value:
            expanded_sources.append(text_value)
    if general_comment:
        expanded_sources.append(general_comment)
    route = _resolve_route(product, initial_route, expanded_sources)
    feedstock = _extract_feedstock(expanded_sources, product)
    standards = _shorten_standard_text(_extract_standards(expanded_sources))
    mix_type = _infer_mix_type(expanded_sources)
    location_type = _infer_location_type(expanded_sources)
    treatment_segments = _collect_treatment_segments(product, feedstock, route, standards)
    treatment = _semicolon_join([product] + treatment_segments)
    treatment_short = _semicolon_join(treatment_segments)
    mix = _compose_mix_string(mix_type, location_type, None)
    functional_properties = _extract_multilang_text(name_fields.get("functionalUnitFlowProperties"))
    return {
        "base": base,
        "product": product,
        "feedstock": feedstock,
        "route": route,
        "standards": standards,
        "mix_type": mix_type,
        "location_type": location_type,
        "treatment": treatment,
        "treatment_segments": treatment_segments,
        "treatment_short": treatment_short,
        "mix": mix,
        "functional_unit": None,
        "functional_unit_properties": functional_properties.strip(),
    }


def _normalise_quantitative_reference(section: Any) -> dict[str, Any]:
    ref = _ensure_dict(section)
    ref["@type"] = DEFAULT_REFERENCE_TYPE
    reference_id = ref.get("referenceToReferenceFlow") or DEFAULT_REFERENCE_ID
    try:
        numeric_id = int(str(reference_id))
    except (ValueError, TypeError):
        numeric_id = int(DEFAULT_REFERENCE_ID)
    ref["referenceToReferenceFlow"] = str(numeric_id)
    functional_unit = ref.get("functionalUnitOrOther")
    if functional_unit:
        ref["functionalUnitOrOther"] = _to_multilang(functional_unit)
    other_value = _stringify(ref.get("common:other"))
    if not other_value.strip():
        ref.pop("common:other", None)
    return ref


def _normalise_time(section: Any) -> dict[str, Any]:
    time_info = _ensure_dict(section)
    reference_year = time_info.pop("referenceYear", None)
    if reference_year is None:
        reference_year = time_info.get("common:referenceYear")
    year_value: int | None = None
    if isinstance(reference_year, (int, float)):
        year_value = int(reference_year)
    elif isinstance(reference_year, str) and reference_year.isdigit():
        year_value = int(reference_year)
    if year_value is not None:
        time_info["common:referenceYear"] = year_value
    elif "common:referenceYear" in time_info:
        value = time_info["common:referenceYear"]
        if not isinstance(value, (int, float, str)) or (isinstance(value, str) and not value.isdigit()):
            time_info.pop("common:referenceYear", None)

    valid_until = time_info.get("common:dataSetValidUntil")
    if isinstance(valid_until, (int, float, str)) and str(valid_until).isdigit():
        time_info["common:dataSetValidUntil"] = int(valid_until)
    elif year_value is not None:
        time_info["common:dataSetValidUntil"] = year_value
    else:
        time_info.pop("common:dataSetValidUntil", None)

    description = time_info.get("common:timeRepresentativenessDescription")
    if description:
        entries = _ensure_multilang_list(description)
        if entries:
            time_info["common:timeRepresentativenessDescription"] = entries
        else:
            time_info.pop("common:timeRepresentativenessDescription", None)

    return {k: v for k, v in time_info.items() if v not in (None, "", {})}


def _normalise_geography(section: Any) -> dict[str, Any]:
    geo = _ensure_dict(section)
    block = _ensure_dict(geo.get("locationOfOperationSupplyOrProduction"))
    raw_code = block.pop("@location", None) or block.pop("location", None) or geo.pop("code", None) or geo.pop("@location", None) or DEFAULT_LOCATION
    code, code_description = _extract_location_code(raw_code)
    sub_location = block.pop("subLocation", None) or geo.pop("subLocation", None)

    normalised_block: dict[str, Any] = {"@location": code}
    if sub_location:
        normalised_block.setdefault("common:comment", sub_location)

    return {"locationOfOperationSupplyOrProduction": normalised_block}


def _normalise_technology(section: Any) -> dict[str, Any]:
    technology = _ensure_dict(section)
    if not technology:
        return {}
    if "technologyDescriptionAndIncludedProcesses" in technology:
        technology["technologyDescriptionAndIncludedProcesses"] = _ensure_multilang(technology.get("technologyDescriptionAndIncludedProcesses"), fallback="")
    if technology.get("technologicalApplicability"):
        technology["technologicalApplicability"] = _ensure_multilang(technology.get("technologicalApplicability"), fallback="")
    for key in (
        "referenceToIncludedProcesses",
        "referenceToTechnologyPictogramme",
        "referenceToTechnologyFlowDiagrammOrPicture",
    ):
        if key in technology and not _has_reference(technology[key]):
            technology.pop(key, None)
    technology.pop("common:other", None)
    return technology


def _sanitize_general_comment(comment: Any) -> dict[str, Any] | str | None:
    text = _extract_multilang_text(comment).strip()
    if not text:
        return None
    lang = DEFAULT_LANGUAGE
    if isinstance(comment, dict):
        lang = comment.get("@xml:lang") or lang
    if text.startswith("FlowSearch hints:"):
        # Preserve the structured hint string so downstream consumers can recover the exact metadata.
        return {"@xml:lang": lang, "#text": text}
    return {"@xml:lang": lang, "#text": text}


def _parse_hint_segments(text: str) -> tuple[dict[str, list[str]], list[str]]:
    body = text.strip()
    segments = [segment.strip() for segment in body.split("|") if segment.strip()]
    hints: dict[str, list[str]] = {}
    notes: list[str] = []
    for segment in segments:
        if "=" in segment:
            key, value = segment.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if not value or value.lower() == "na":
                continue
            values = [item.strip() for item in value.split(";") if item.strip()]
            hints[key] = values or [value]
        else:
            notes.append(_clean_note_text(segment))
    return hints, notes


def _compose_comment_from_hints(hints: dict[str, list[str]], notes: list[str]) -> str:
    parts: list[str] = []

    def _join(values: list[str], *, limit: int | None = None, separator: str = "; ") -> str:
        items = [value for value in values if value]
        if limit is not None:
            items = items[:limit]
        return separator.join(items)

    if hints.get("usage_context"):
        parts.append(f"Usage context: {_join(hints['usage_context'])}")
    if hints.get("source_or_pathway"):
        parts.append(f"Source/pathway: {_join(hints['source_or_pathway'])}")
    if hints.get("state_purity"):
        parts.append(f"State/purity: {_join(hints['state_purity'])}")
    if hints.get("abbreviation"):
        parts.append(f"Abbreviation: {_join(hints['abbreviation'])}")
    if hints.get("formula_or_CAS"):
        parts.append(f"Formula/CAS: {_join(hints['formula_or_CAS'])}")
    if hints.get("en_synonyms"):
        parts.append(f"Synonyms (EN): {_join(hints['en_synonyms'], limit=3)}")
    if hints.get("zh_synonyms"):
        parts.append(f"Synonyms (ZH): {_join(hints['zh_synonyms'], limit=3, separator=', ')}")
    for note in notes:
        cleaned = _clean_note_text(note)
        if cleaned:
            parts.append(cleaned)

    sentences = [_ensure_sentence(part) for part in parts if part]
    return " ".join(sentence for sentence in sentences if sentence).strip()


def _clean_note_text(value: str) -> str:
    if not value:
        return ""
    cleaned = value.strip()
    for prefix in ("Notes:", "Note:", "备注:", "说明:", "-"):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
            break
    return cleaned.strip()


def _format_name_field_text(value: Any) -> str:
    text = _stringify(value).strip()
    if not text:
        return ""
    if ";" not in text:
        return text
    segments = [segment.strip() for segment in text.split(";") if segment.strip()]
    if len(segments) <= 1:
        return text
    return ", ".join(segments)


def _ensure_sentence(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped[-1] in ".!?。；;:":
        return stripped
    return f"{stripped}."


def _normalise_exchanges(
    section: Any,
    name_components: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    exchange_container = _ensure_dict(section)
    raw_exchanges = exchange_container.get("exchange")
    if isinstance(raw_exchanges, dict):
        exchanges_iter = [raw_exchanges]
    elif isinstance(raw_exchanges, list):
        exchanges_iter = raw_exchanges
    else:
        exchanges_iter = []

    normalised: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for index, exchange in enumerate(exchanges_iter, start=0):
        item = _ensure_dict(exchange)
        item["@dataSetInternalID"] = str(index)
        item.pop("functionType", None)
        exchange_name = _stringify(item.get("exchangeName"))
        if not exchange_name:
            exchange_name = _stringify(item.get("flowName"))
        if not exchange_name:
            name_candidate = _extract_flow_name(item.get("referenceToFlowDataSet"))
            if not name_candidate:
                name_candidate = _stringify(item.get("name"))
            if not name_candidate:
                name_candidate = _stringify(item.get("generalComment")).split(":")[0]
            if not name_candidate:
                name_candidate = f"exchange_{index:06d}"
            exchange_name = name_candidate
            item["exchangeName"] = exchange_name
        if "name" in item and isinstance(item["name"], dict) and not item["name"].get("#text"):
            item.pop("name")
        direction = _stringify(item.get("exchangeDirection")).lower()
        item["exchangeDirection"] = "Input" if direction != "output" else "Output"
        item.setdefault("meanAmount", "0")
        item.setdefault("resultingAmount", "0")
        item["dataDerivationTypeStatus"] = _normalise_derivation_status(item.get("dataDerivationTypeStatus"))
        short_description = _compose_short_description(exchange_name, item, name_components)
        comment_value = item.get("generalComment")
        if comment_value:
            cleaned_comment = _sanitize_general_comment(comment_value)
            if cleaned_comment:
                item["generalComment"] = _ensure_multilang(cleaned_comment, fallback="")
            else:
                item.pop("generalComment", None)
        else:
            item.pop("generalComment", None)
        # Remove intermediate hint structures to keep the final ILCD exchange schema-compliant.
        item.pop("flowHints", None)
        item.pop("hints", None)
        # Preserve genuine flow references (from alignment) but drop empty placeholders,
        # so Stage 3 can inject authoritative matches when available.
        reference = item.get("referenceToFlowDataSet")
        if not _has_reference(reference):
            item.pop("referenceToFlowDataSet", None)
        if not _stringify(item.get("common:other")).strip():
            item.pop("common:other", None)
        item.pop("flowName", None)
        normalised.append(item)
        metadata.append(
            {
                "id": item["@dataSetInternalID"],
                "name": exchange_name,
                "direction": item.get("exchangeDirection"),
                "mean": item.get("meanAmount"),
                "unit": item.get("unit") or item.get("resultingAmountUnit"),
                "short": short_description,
            }
        )

    reference_id = _select_reference_flow(metadata, name_components)
    return {"exchange": normalised}, reference_id


def _compose_short_description(
    base_name: str,
    exchange: dict[str, Any],
    name_components: dict[str, Any],
) -> str:
    segments: list[str] = []
    base_segment = base_name.strip()
    if base_segment:
        segments.append(base_segment)

    feedstock = name_components.get("feedstock")
    if feedstock:
        segments.append(f"{feedstock} feedstock")

    route = name_components.get("route")
    if route and route.lower() not in {base_segment.lower(), (feedstock or "").lower()}:
        segments.append(route)

    standards = _shorten_standard_text(name_components.get("standards", ""))
    if standards:
        segments.append(standards)

    mix = name_components.get("mix") or _compose_mix_string(
        name_components.get("mix_type", "Production mix"),
        name_components.get("location_type", "at plant"),
        name_components.get("location_code"),
    )
    if mix:
        segments.append(mix)

    flow_properties = _compose_flow_properties(exchange)
    if flow_properties:
        segments.append(flow_properties)

    return "; ".join(_deduplicate_preserve_order(segments))


def _compose_flow_properties(exchange: dict[str, Any]) -> str:
    amount = _stringify(exchange.get("meanAmount") or exchange.get("resultingAmount"))
    unit = _stringify(exchange.get("unit") or exchange.get("resultingAmountUnit"))
    amount_clean = amount.strip()
    unit_clean = unit.strip()
    if amount_clean in {"", "0", "0.0"}:
        amount_clean = ""
    if unit_clean in {"", "0"}:
        unit_clean = ""
    if amount_clean and unit_clean:
        return f"{amount_clean} {unit_clean}".strip()
    if amount_clean:
        return amount_clean
    if unit_clean:
        return unit_clean
    return ""


def _compose_flow_treatment(flow_name: str, name_components: dict[str, Any]) -> str:
    segments: list[str] = []
    feedstock = name_components.get("feedstock")
    if feedstock:
        segments.append(f"{feedstock} feedstock")
    route = name_components.get("route")
    base_lower = flow_name.strip().lower()
    if route and route.lower() not in {base_lower, (feedstock or "").lower()}:
        segments.append(route)
    standards = _shorten_standard_text(name_components.get("standards", ""))
    if standards:
        segments.append(standards)
    return _semicolon_join(segments)


def _normalise_lcia_results(section: Any) -> dict[str, Any]:
    results = _ensure_dict(section)
    lcia_result = _ensure_dict(results.get("LCIAResult"))
    if not lcia_result:
        return {}
    mean_amount = lcia_result.get("meanAmount")
    if mean_amount is not None:
        try:
            mean_value = float(mean_amount)
        except (TypeError, ValueError):
            mean_value = mean_amount
        else:
            lcia_result["meanAmount"] = f"{mean_value}"
    if lcia_result.get("generalComment"):
        lcia_result["generalComment"] = _ensure_multilang(lcia_result.get("generalComment"), fallback="")
    if not _has_reference(lcia_result.get("referenceToLCIAMethodDataSet")):
        lcia_result.pop("referenceToLCIAMethodDataSet", None)
    results = {"LCIAResult": lcia_result}
    return results


def _normalise_modelling_and_validation(section: Any) -> dict[str, Any]:
    mv_raw = _ensure_dict(section)
    if not mv_raw:
        return {}
    mv: dict[str, Any] = {}

    lci = _ensure_dict(mv_raw.get("LCIMethodAndAllocation"))
    type_value = _normalise_dataset_type(lci.get("typeOfDataSet"))
    if not type_value:
        type_value = _default_enum_value(TYPE_OF_DATA_SET_POINTER)
    if type_value:
        lci["typeOfDataSet"] = type_value
    else:
        lci.pop("typeOfDataSet", None)

    principle_options = _get_enum_options(LCI_METHOD_PRINCIPLE_POINTER)
    principle_value = _match_enum_value(lci.get("LCIMethodPrinciple"), principle_options)
    if principle_value:
        lci["LCIMethodPrinciple"] = principle_value
    else:
        lci.pop("LCIMethodPrinciple", None)

    approach_value = _normalise_lci_method_approach(lci.get("LCIMethodApproaches"))
    if approach_value:
        lci["LCIMethodApproaches"] = approach_value
    else:
        lci.pop("LCIMethodApproaches", None)

    lci.pop("common:other", None)
    for key in (
        "deviationsFromLCIMethodApproaches",
        "deviationsFromLCIMethodPrinciple",
        "modellingConstants",
        "deviationsFromModellingConstants",
    ):
        if key in lci:
            entries = _ensure_multilang_list(lci[key])
            if entries:
                lci[key] = entries
            else:
                lci.pop(key, None)
    if lci:
        mv["LCIMethodAndAllocation"] = lci

    dsr = _ensure_dict(mv_raw.get("dataSourcesTreatmentAndRepresentativeness"))
    if dsr:
        for key in (
            "dataCutOffAndCompletenessPrinciples",
            "deviationsFromCutOffAndCompletenessPrinciples",
            "dataSelectionAndCombinationPrinciples",
            "deviationsFromSelectionAndCombinationPrinciples",
            "dataTreatmentAndExtrapolationsPrinciples",
            "deviationsFromTreatmentAndExtrapolationPrinciples",
            "samplingProcedure",
            "uncertaintyAdjustments",
            "useAdviceForDataSet",
        ):
            if key in dsr:
                entries = _ensure_multilang_list(dsr[key])
                if entries:
                    dsr[key] = entries
                else:
                    dsr.pop(key, None)
        for key in ("annualSupplyOrProductionVolume", "dataCollectionPeriod"):
            if key in dsr:
                value = _ensure_multilang(dsr[key], fallback="")
                if value.get("#text"):
                    dsr[key] = value
                else:
                    dsr.pop(key, None)
        dsr.pop("common:other", None)
        coverage_value = _normalise_percentage_coverage(dsr.get("percentageSupplyOrProductionCovered"))
        if coverage_value is not None:
            dsr["percentageSupplyOrProductionCovered"] = coverage_value
        else:
            dsr.pop("percentageSupplyOrProductionCovered", None)
        if dsr:
            mv["dataSourcesTreatmentAndRepresentativeness"] = dsr

    completeness = _ensure_dict(mv_raw.get("completeness"))
    if completeness:
        mv["completeness"] = completeness

    validation = _ensure_dict(mv_raw.get("validation"))
    review = _ensure_dict(validation.get("review"))
    review_type = _stringify(review.get("@type")).strip()
    if review_type not in {
        "Dependent internal review",
        "Independent internal review",
        "Independent external review",
        "Accredited third party review",
        "Independent review panel",
        "Not reviewed",
    }:
        review["@type"] = "Not reviewed"
    validation["review"] = review
    mv["validation"] = validation

    mv["complianceDeclarations"] = _build_compliance_declarations(mv_raw.get("complianceDeclarations"))

    return mv


def _build_compliance_declarations(section: Any) -> dict[str, Any]:
    declarations_container = _ensure_dict(section)
    extras = _collect_additional_compliance_entries(declarations_container)
    baseline = _build_entry_level_compliance()
    if extras:
        compliance_value: Any = [baseline, *extras]
    else:
        compliance_value = baseline
    result: dict[str, Any] = {"compliance": compliance_value}
    other_text = _stringify(declarations_container.get("common:other")).strip()
    if other_text:
        result["common:other"] = other_text
    return result


def _collect_additional_compliance_entries(container: dict[str, Any]) -> list[dict[str, Any]]:
    raw_entries = container.get("compliance")
    if isinstance(raw_entries, list):
        candidates = raw_entries
    elif isinstance(raw_entries, dict):
        candidates = [raw_entries]
    else:
        candidates = []
    extras: list[dict[str, Any]] = []
    for candidate in candidates:
        normalised = _normalise_compliance_entry(candidate)
        if not normalised:
            continue
        if _is_entry_level_reference(normalised):
            continue
        extras.append(normalised)
    return extras


def _normalise_compliance_entry(entry: Any) -> dict[str, Any] | None:
    data = _ensure_dict(entry)
    if not data:
        return None
    reference = _select_reference(data.get("common:referenceToComplianceSystem"))
    if not reference:
        return None
    normalised: dict[str, Any] = {"common:referenceToComplianceSystem": reference}
    for field in _get_compliance_status_fields():
        status = _normalise_compliance_status(field, data.get(field))
        if status is None:
            return None
        normalised[field] = status
    other_value = _stringify(data.get("common:other")).strip()
    if other_value:
        normalised["common:other"] = other_value
    return normalised


def _normalise_compliance_status(field: str, value: Any) -> str | None:
    metadata = _get_process_schema_metadata()
    pointer = metadata.compliance_field_pointers.get(field, "")
    options = list(metadata.enum_fields.get(pointer, ()))
    return _match_enum_value(value, options)


def _select_reference(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict) and "@refObjectId" in value:
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and "@refObjectId" in item:
                return item
    return None


def _is_entry_level_reference(entry: dict[str, Any]) -> bool:
    reference = entry.get("common:referenceToComplianceSystem")
    if isinstance(reference, dict):
        return reference.get("@refObjectId") == ILCD_ENTRY_LEVEL_REFERENCE_ID
    return False


def _build_entry_level_compliance() -> dict[str, Any]:
    metadata = _get_process_schema_metadata()
    entry = {
        "common:referenceToComplianceSystem": _build_compliance_reference(),
    }
    for field in _get_compliance_status_fields():
        pointer = metadata.compliance_field_pointers.get(field, "")
        preferred = COMPLIANCE_DEFAULT_PREFERENCES.get(field)
        entry[field] = _default_enum_value(pointer, preferred)
    return entry


def _normalise_administrative_information(
    section: Any,
    *,
    dataset_uuid: str | None = None,
    dataset_kind: str = "process",
) -> dict[str, Any]:
    admin = _ensure_dict(section)
    admin.pop("dataGenerator", None)

    commissioner = _ensure_dict(admin.get("common:commissionerAndGoal"))
    commissioner["common:referenceToCommissioner"] = _build_commissioner_reference()
    if not commissioner.get("common:intendedApplications"):
        commissioner.pop("common:intendedApplications", None)
    else:
        intended = commissioner.get("common:intendedApplications")
        entries = _ensure_multilang_list(intended)
        if entries:
            commissioner["common:intendedApplications"] = entries
        else:
            commissioner.pop("common:intendedApplications", None)
    admin["common:commissionerAndGoal"] = commissioner

    data_entry = _ensure_dict(admin.get("dataEntryBy"))
    data_entry.pop("common:other", None)
    data_entry["common:referenceToDataSetFormat"] = _build_dataset_format_reference()
    data_entry["common:referenceToPersonOrEntityEnteringTheData"] = _build_commissioner_reference()
    data_entry["common:timeStamp"] = _current_timestamp()
    cleaned_data_entry = {key: value for key, value in data_entry.items() if value not in (None, "", {}, [])}
    admin["dataEntryBy"] = cleaned_data_entry

    publication = _ensure_dict(admin.get("publicationAndOwnership"))
    version_candidate = DEFAULT_DATA_SET_VERSION
    if publication or dataset_uuid:
        publication["common:dataSetVersion"] = version_candidate
        if dataset_uuid:
            publication["common:permanentDataSetURI"] = _build_permanent_dataset_uri(dataset_kind, dataset_uuid, version_candidate)
        if "common:registrationNumber" in publication:
            publication["common:registrationNumber"] = _stringify(publication.get("common:registrationNumber")).strip()
        publication["common:licenseType"] = "Free of charge for all users and uses"
        publication["common:copyright"] = "false"
        publication["common:referenceToOwnershipOfDataSet"] = _build_commissioner_reference()
        publication.pop("common:other", None)
        access_entries = _ensure_multilang_list(publication.get("common:accessRestrictions"))
        filtered_access: list[dict[str, Any]] = []
        for entry in access_entries:
            if not isinstance(entry, dict):
                continue
            lang = _stringify(entry.get("@xml:lang") or DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE
            text = _stringify(entry.get("#text")).strip()
            if not text:
                continue
            if text.lower() in {"none", "na", "n/a"}:
                continue
            filtered_access.append({"@xml:lang": lang, "#text": text})
        if filtered_access:
            publication["common:accessRestrictions"] = filtered_access
        else:
            publication.pop("common:accessRestrictions", None)
        cleaned_publication = {key: value for key, value in publication.items() if value not in (None, "", {}, [])}
        if cleaned_publication:
            admin["publicationAndOwnership"] = cleaned_publication
        else:
            admin.pop("publicationAndOwnership", None)
    else:
        admin.pop("publicationAndOwnership", None)

    admin.pop("common:other", None)
    return {k: v for k, v in admin.items() if v not in (None, "", {}, [])}


def _build_reference(ref_type: str, description: str) -> dict[str, Any]:
    identifier = str(uuid4())
    version = "01.01.000"
    kind_map = {
        "flow data set": "flow",
        "process data set": "process",
        "source data set": "source",
    }
    dataset_kind = kind_map.get(ref_type.lower())
    uri = build_portal_uri(dataset_kind, identifier, version) if dataset_kind else ""
    return {
        "@type": ref_type,
        "@refObjectId": identifier,
        "@uri": uri,
        "@version": version,
        "common:shortDescription": _ensure_multilang(description),
    }


def _build_commissioner_reference() -> dict[str, Any]:
    ref_object_id = "f4b4c314-8c4c-4c83-968f-5b3c7724f6a8"
    version = "01.00.000"
    return {
        "@refObjectId": ref_object_id,
        "@type": "contact data set",
        "@uri": build_local_dataset_uri("contact data set", ref_object_id, version),
        "@version": version,
        "common:shortDescription": [
            {"@xml:lang": "en", "#text": "Tiangong LCA Data Working Group"},
            {"@xml:lang": "zh", "#text": "天工LCA数据团队"},
        ],
    }


def _build_dataset_format_reference() -> dict[str, Any]:
    return build_dataset_format_reference()


def _current_timestamp() -> str:
    """Return an ISO 8601 timestamp ending with 'Z' (UTC)."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_compliance_reference() -> dict[str, Any] | None:
    """Return the ILCD entry-level compliance reference using ILCD-relative paths."""

    if not ILCD_ENTRY_LEVEL_REFERENCE_ID:
        return None

    return {
        "@refObjectId": ILCD_ENTRY_LEVEL_REFERENCE_ID,
        "@type": "source data set",
        "@uri": build_local_dataset_uri(
            "source data set",
            ILCD_ENTRY_LEVEL_REFERENCE_ID,
            ILCD_ENTRY_LEVEL_REFERENCE_VERSION,
        ),
        "@version": ILCD_ENTRY_LEVEL_REFERENCE_VERSION,
        "common:shortDescription": {"@xml:lang": "en", "#text": "ILCD Data Network - Entry-level"},
    }


def _ensure_global_reference(
    value: Any,
    *,
    ref_type: str,
    description: str,
) -> dict[str, Any]:
    if isinstance(value, dict) and "@refObjectId" in value:
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and "@refObjectId" in item:
                return item
    return _build_reference(ref_type, description)


def _ensure_reference_field(
    container: dict[str, Any],
    key: str,
    *,
    ref_type: str,
    description: str,
) -> None:
    value = container.get(key)
    if _has_reference(value):
        return
    container[key] = _ensure_global_reference(value, ref_type=ref_type, description=description)


_OPTIONAL_REFERENCE_PATHS: tuple[tuple[str, ...], ...] = (
    ("modellingAndValidation", "dataSourcesTreatmentAndRepresentativeness", "referenceToDataHandlingPrinciples"),
    ("modellingAndValidation", "LCIMethodAndAllocation", "referenceToLCAMethodDetails"),
    ("modellingAndValidation", "completeness", "referenceToSupportedImpactAssessmentMethods"),
    ("administrativeInformation", "publicationAndOwnership", "common:referenceToPrecedingDataSetVersion"),
    ("administrativeInformation", "publicationAndOwnership", "common:referenceToUnchangedRepublication"),
    ("administrativeInformation", "publicationAndOwnership", "common:referenceToRegistrationAuthority"),
    ("administrativeInformation", "publicationAndOwnership", "common:referenceToEntitiesWithExclusiveAccess"),
    ("administrativeInformation", "common:commissionerAndGoal", "common:project"),
)


def _prune_invalid_reference_fields(dataset: dict[str, Any]) -> None:
    for path in _OPTIONAL_REFERENCE_PATHS:
        container: Any = dataset
        for key in path[:-1]:
            if not isinstance(container, dict):
                break
            container = container.get(key)
        else:
            if not isinstance(container, dict):
                continue
            field = path[-1]
            value = container.get(field)
            if value in (None, "", [], {}):
                container.pop(field, None)
                continue
            if isinstance(value, str) and not value.strip():
                container.pop(field, None)
                continue
            if not _contains_structured_reference(value):
                container.pop(field, None)


def _contains_structured_reference(value: Any) -> bool:
    if value in (None, "", [], {}):
        return False
    if _has_reference(value):
        return True
    if isinstance(value, dict):
        for child in value.values():
            if isinstance(child, str) and _is_valid_uuid(child.strip()):
                return True
            if isinstance(child, (dict, list)) and _contains_structured_reference(child):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_structured_reference(item) for item in value)
    return False


def _normalise_derivation_status(value: Any) -> str:
    mapping = {
        "measured": "Measured",
        "measured/calculated": "Calculated",
        "calculated": "Calculated",
        "estimated": "Estimated",
        "unknown": "Unknown derivation",
        "unknown derivation": "Unknown derivation",
        "missing important": "Missing important",
        "missing unimportant": "Missing unimportant",
    }
    text = _stringify(value).strip().lower()
    return mapping.get(text, "Unknown derivation")


def _normalise_license(value: Any) -> str:
    allowed = {
        "free of charge for all users and uses": "Free of charge for all users and uses",
        "free of charge for some user types or use types": ("Free of charge for some user types or use types"),
        "free of charge for members only": "Free of charge for members only",
        "license fee": "License fee",
        "other": "Other",
    }
    text = _stringify(value).strip().lower()
    return allowed.get(text, "Other")


def _normalise_boolean(value: Any) -> str:
    text = _stringify(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return "true"
    return "false"


def _select_from_enum(value: Any, allowed: list[str], default: str) -> str:
    text = _stringify(value).strip().lower()
    for candidate in allowed:
        if text == candidate.lower():
            return candidate
    return default


def _is_valid_uuid(value: str) -> bool:
    return bool(
        re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            value,
        )
    )


def _build_permanent_dataset_uri(dataset_kind: str, uuid_value: str, version: str) -> str:
    if not uuid_value:
        return ""
    return build_portal_uri(dataset_kind, uuid_value, version)


def _ensure_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _to_multilang(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        if "@xml:lang" not in value:
            value.setdefault("@xml:lang", DEFAULT_LANGUAGE)
        return value
    if isinstance(value, list):
        return {
            "@xml:lang": DEFAULT_LANGUAGE,
            "#text": "; ".join(_stringify(item) for item in value),
        }
    return {"@xml:lang": DEFAULT_LANGUAGE, "#text": _stringify(value)}


def _normalize_multilang_dict(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    lang = _stringify(value.get("@xml:lang") or DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE
    for key in ("#text", "%23text", "_text", "text", "@text"):
        if key in value and value[key] not in (None, ""):
            text = _stringify(value[key]).strip()
            return {"@xml:lang": lang, "#text": text}
    return None


def _ensure_multilang_list(value: Any) -> list[dict[str, Any]]:
    if value in (None, "", [], {}):
        return []
    items = value if isinstance(value, list) else [value]
    entries: list[dict[str, Any]] = []
    for item in items:
        if item in (None, ""):
            continue
        normalized = _ensure_multilang(item)
        text = _stringify(normalized.get("#text")).strip()
        if not text:
            continue
        lang = _stringify(normalized.get("@xml:lang") or DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE
        entries.append({"@xml:lang": lang, "#text": text})
    return entries


def _filter_multilang_entries(
    entries: list[dict[str, Any]],
    *,
    target_lang: str | None,
    clean_text: bool = False,
) -> list[dict[str, Any]]:
    if not entries:
        return []
    filtered: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    target = target_lang.lower() if isinstance(target_lang, str) else None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        lang_candidate = _stringify(entry.get("@xml:lang") or DEFAULT_LANGUAGE).strip().lower() or DEFAULT_LANGUAGE
        if target is not None and lang_candidate != target:
            continue
        text_value = _stringify(entry.get("#text")).strip()
        if clean_text and text_value:
            text_value = _clean_english_text(text_value)
        if not text_value:
            continue
        canonical_lang = target if target is not None else lang_candidate or DEFAULT_LANGUAGE
        key = (canonical_lang, text_value)
        if key in seen:
            continue
        filtered.append({"@xml:lang": canonical_lang, "#text": text_value})
        seen.add(key)
    return filtered


def _clean_english_text(text: str) -> str:
    if not text:
        return ""
    segments = [segment.strip() for segment in re.split(r"[;,]", text) if segment and segment.strip()]
    english_segments = [segment for segment in segments if segment.isascii()]
    if english_segments:
        return "; ".join(english_segments)
    return text if text.isascii() else ""


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _extract_multilang_text(value: Any) -> str:
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = _extract_multilang_text(item)
            if text:
                parts.append(text)
        return "; ".join(parts)
    if isinstance(value, dict):
        normalized = _normalize_multilang_dict(value)
        if normalized:
            return normalized["#text"]
        return ""
    return _stringify(value)


def _ensure_multilang(value: Any, *, fallback: str | None = None, separator: str = "; ") -> dict[str, Any]:
    if isinstance(value, dict):
        normalized = _normalize_multilang_dict(value)
        if normalized:
            return normalized
    if isinstance(value, list):
        segments = [segment for segment in (_extract_multilang_text(item).strip() for item in value) if segment]
        text = separator.join(segments)
    else:
        text = _stringify(value)
    if (not text) and fallback is not None:
        text = fallback
    if text is None:
        text = ""
    if isinstance(text, str):
        text = text.strip()
    return {"@xml:lang": DEFAULT_LANGUAGE, "#text": text}


def _normalize_enum_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _schema_pointer_get(schema: dict[str, Any], pointer: str) -> Any:
    if not pointer or pointer == "/":
        return schema
    parts = pointer.lstrip("/").split("/")
    node: Any = schema
    for part in parts:
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            try:
                index = int(key)
            except ValueError:
                return {}
            if index < 0 or index >= len(node):
                return {}
            node = node[index]
        elif isinstance(node, dict):
            if key not in node:
                return {}
            node = node[key]
        else:
            return {}
    return node


def _collect_enum_options(node: Any) -> list[str]:
    if not isinstance(node, dict):
        return []
    options: list[str] = []
    enum_values = node.get("enum")
    if isinstance(enum_values, list):
        options.extend(str(item) for item in enum_values)
    for key in ("allOf", "anyOf", "oneOf"):
        variants = node.get(key)
        if isinstance(variants, list):
            for variant in variants:
                options.extend(_collect_enum_options(variant))
    seen: set[str] = set()
    unique: list[str] = []
    for option in options:
        if option not in seen:
            seen.add(option)
            unique.append(option)
    return unique


def _pointer_from_path(path: tuple[str, ...]) -> str:
    if not path:
        return ""
    return "/" + "/".join(path)


def _build_process_schema_metadata(schema: dict[str, Any]) -> ProcessSchemaMetadata:
    enum_fields: dict[str, tuple[str, ...]] = {}
    multilang_fields: set[str] = set()
    compliance_fields: list[str] = []
    compliance_pointers: dict[str, str] = {}
    visited: set[int] = set()

    def walk(node: Any, path: tuple[str, ...]) -> None:
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        pointer = _pointer_from_path(path)
        last_segment = path[-1] if path else None
        second_last_segment = path[-2] if len(path) >= 2 else None
        is_property_field = second_last_segment == "properties"
        is_items_field = last_segment == "items"
        is_additional_field = last_segment == "additionalProperties"
        is_field_node = is_property_field or is_items_field or is_additional_field

        if isinstance(node, dict):
            if is_field_node and _schema_is_multilang(node):
                multilang_fields.add(pointer)

            enum_values = node.get("enum")
            if is_field_node and isinstance(enum_values, list) and enum_values:
                enum_fields[pointer] = tuple(_collect_enum_options(node))

            if is_property_field and pointer.startswith(COMPLIANCE_BASE_POINTER) and pointer != COMPLIANCE_BASE_POINTER and "/properties/common:referenceToComplianceSystem/" not in pointer:
                field_name = path[-1]
                if field_name and field_name not in compliance_pointers:
                    compliance_pointers[field_name] = pointer
                    if field_name not in {"compliance", "common:referenceToComplianceSystem", "common:other"}:
                        compliance_fields.append(field_name)

            for key in ("allOf", "anyOf", "oneOf"):
                variants = node.get(key)
                if isinstance(variants, list):
                    for idx, variant in enumerate(variants):
                        walk(variant, path + (key, str(idx)))

            properties = node.get("properties")
            if isinstance(properties, dict):
                for name, child in properties.items():
                    walk(child, path + ("properties", name))

            items = node.get("items")
            if isinstance(items, dict):
                walk(items, path + ("items",))

            additional = node.get("additionalProperties")
            if isinstance(additional, dict):
                walk(additional, path + ("additionalProperties",))

    walk(schema, ())

    compliance_status_fields = tuple(compliance_fields)
    return ProcessSchemaMetadata(
        enum_fields=enum_fields,
        multilang_fields=frozenset(multilang_fields),
        compliance_status_fields=compliance_status_fields,
        compliance_field_pointers=compliance_pointers,
    )


@lru_cache(maxsize=None)
def _get_enum_options(pointer: str) -> list[str]:
    metadata = _get_process_schema_metadata()
    options = metadata.enum_fields.get(pointer)
    return list(options) if options else []


@lru_cache(maxsize=1)
def _get_process_schema_metadata() -> ProcessSchemaMetadata:
    schema = _get_process_dataset_schema()
    return _build_process_schema_metadata(schema)


def _get_compliance_status_fields() -> tuple[str, ...]:
    return _get_process_schema_metadata().compliance_status_fields


def _match_enum_value(value: Any, options: list[str]) -> str | None:
    if not options:
        return None
    text = _stringify(value).strip()
    if not text:
        return None
    segments = [segment.strip() for segment in re.split(r"[;,/]| and ", text) if segment and segment.strip()]
    for segment in segments:
        match = _match_enum_token(segment, options)
        if match is not None:
            return match
    return _match_enum_token(text, options)


def _match_enum_token(value: Any, options: list[str]) -> str | None:
    text = _stringify(value).strip()
    if not text:
        return None
    lower = text.lower()
    normalized = _normalize_enum_text(text)
    best_match: str | None = None
    best_score = 0
    for option in options:
        candidate = option.strip()
        candidate_lower = candidate.lower()
        candidate_normalized = _normalize_enum_text(candidate)
        if lower == candidate_lower:
            return candidate
        if normalized and candidate_normalized == normalized:
            return candidate
        score = 0
        if candidate_lower in lower or lower in candidate_lower:
            score = max(score, len(candidate_lower))
        overlap = set(normalized.split()) & set(candidate_normalized.split())
        if overlap:
            score = max(score, len(overlap))
        if score > best_score:
            best_score = score
            best_match = candidate
    return best_match if best_score > 0 else None


def _find_enum_option_by_keywords(options: list[str], *keywords: str) -> str | None:
    if not keywords:
        return None
    lowered_keywords = [keyword.lower() for keyword in keywords]
    for option in options:
        candidate_lower = option.lower()
        if all(keyword in candidate_lower for keyword in lowered_keywords):
            return option
    return None


def _default_enum_value(pointer: str, preferred: str | None = None) -> str | None:
    options = _get_enum_options(pointer)
    if preferred:
        match = _match_enum_value(preferred, options)
        if match is not None:
            return match
    return options[0] if options else preferred


@lru_cache(maxsize=1)
def _get_process_dataset_schema() -> dict[str, Any]:
    repo = get_schema_repository()
    return repo.resolve_with_references("tidas_processes.json", "/properties/processDataSet")


def _apply_schema_normalisation(dataset: dict[str, Any]) -> None:
    schema = _get_process_dataset_schema()
    metadata = _get_process_schema_metadata()
    _coerce_schema_fields(dataset, schema, (), metadata)


def _coerce_schema_fields(
    value: Any,
    schema: Any,
    path: tuple[str, ...],
    metadata: ProcessSchemaMetadata,
) -> Any:
    if not isinstance(schema, dict):
        return value

    pointer = _pointer_from_path(path)

    if pointer in metadata.multilang_fields:
        coerced_multilang = _coerce_value_to_multilang(value)
        return coerced_multilang

    enum_values = metadata.enum_fields.get(pointer)
    if enum_values:
        return _coerce_enum_value(value, list(enum_values))

    # Apply composite schema options before descending to properties/items.
    for composite_key in ("allOf",):
        options = schema.get(composite_key)
        if isinstance(options, list):
            for idx, option in enumerate(options):
                value = _coerce_schema_fields(value, option, path + (composite_key, str(idx)), metadata)

    for composite_key in ("anyOf", "oneOf"):
        options = schema.get(composite_key)
        if isinstance(options, list):
            for idx, option in enumerate(options):
                value = _coerce_schema_fields(value, option, path + (composite_key, str(idx)), metadata)
            return value

    if isinstance(value, dict):
        properties = schema.get("properties")
        known_keys = set(properties or {})
        if isinstance(properties, dict):
            for key, child_schema in properties.items():
                if key in value:
                    coerced_child = _coerce_schema_fields(
                        value[key],
                        child_schema,
                        path + ("properties", key),
                        metadata,
                    )
                    if coerced_child is None:
                        value.pop(key, None)
                        continue
                    if isinstance(coerced_child, str) and not coerced_child.strip():
                        value.pop(key, None)
                        continue
                    if isinstance(coerced_child, list) and not coerced_child:
                        value.pop(key, None)
                        continue
                    if isinstance(coerced_child, dict) and not coerced_child:
                        value.pop(key, None)
                        continue
                    value[key] = coerced_child
        additional = schema.get("additionalProperties")
        extra_keys = [key for key in list(value.keys()) if key not in known_keys]
        if isinstance(additional, dict):
            for key in extra_keys:
                coerced_child = _coerce_schema_fields(
                    value[key],
                    additional,
                    path + ("additionalProperties",),
                    metadata,
                )
                if coerced_child in (None, "", [], {}):
                    value.pop(key, None)
                else:
                    value[key] = coerced_child
        elif additional is True:
            pass
        else:
            if isinstance(properties, dict):
                for key in extra_keys:
                    value.pop(key, None)
        return value

    if isinstance(value, list):
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for index, item in enumerate(value):
                value[index] = _coerce_schema_fields(item, items_schema, path + ("items",), metadata)
        return value

    return value


def _schema_is_multilang(schema: dict[str, Any]) -> bool:
    if not isinstance(schema, dict):
        return False
    if _schema_has_multilang_signature(schema):
        return True
    for composite_key in ("allOf", "anyOf", "oneOf"):
        options = schema.get(composite_key)
        if isinstance(options, list) and any(_schema_is_multilang(option) for option in options if isinstance(option, dict)):
            return True
    return False


def _schema_has_multilang_signature(schema: dict[str, Any]) -> bool:
    if not isinstance(schema, dict):
        return False
    if isinstance(schema.get("properties"), dict):
        properties = schema["properties"]
        if "@xml:lang" in properties and "#text" in properties:
            return True
    ref = schema.get("$ref")
    if isinstance(ref, str) and "MultiLang" in ref:
        return True
    type_value = schema.get("type")
    type_candidates: tuple[str, ...]
    if isinstance(type_value, list):
        type_candidates = tuple(str(item) for item in type_value)
    elif isinstance(type_value, str):
        type_candidates = (type_value,)
    else:
        type_candidates = ()
    if "array" in type_candidates:
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            return _schema_has_multilang_signature(item_schema)
    return False


def _coerce_value_to_multilang(value: Any) -> Any:
    entries: list[dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            if item in (None, "", [], {}):
                continue
            normalized = _ensure_multilang(item)
            text = normalized.get("#text", "")
            if isinstance(text, str):
                text = text.strip()
            if text:
                normalized["#text"] = text
                entries.append(normalized)
    else:
        normalized = _ensure_multilang(value)
        text = normalized.get("#text", "")
        if isinstance(text, str):
            text = text.strip()
        if text:
            normalized["#text"] = text
            entries.append(normalized)
    if not entries:
        return None
    if len(entries) == 1:
        return entries[0]
    return entries


def _coerce_enum_value(value: Any, options: list[Any]) -> Any:
    if not options:
        return value
    lookup: dict[str, Any] = {}
    ordered_keys: list[str] = []
    for option in options:
        key = str(option)
        if key not in lookup:
            lookup[key] = option
            ordered_keys.append(key)
    if isinstance(value, list):
        coerced: list[Any] = []
        for item in value:
            match_key = _match_enum_value(item, ordered_keys)
            if match_key is not None:
                coerced.append(lookup.get(match_key, item))
            else:
                coerced.append(item)
        return coerced
    match_key = _match_enum_value(value, ordered_keys)
    if match_key is not None:
        return lookup.get(match_key, value)
    return value


def _extract_flow_name(reference: Any) -> str | None:
    if isinstance(reference, dict):
        if isinstance(reference.get("common:shortDescription"), dict):
            text = reference["common:shortDescription"].get("#text")
            if text:
                return text
        name = reference.get("name") or reference.get("baseName")
        if isinstance(name, dict):
            return name.get("#text") or name.get("text")
        if isinstance(name, str):
            return name
        for value in reference.values():
            if isinstance(value, str):
                return value
    return None


def _has_reference(value: Any) -> bool:
    if isinstance(value, dict):
        return "@refObjectId" in value
    if isinstance(value, list):
        return any(isinstance(item, dict) and "@refObjectId" in item for item in value)
    return False


def _normalise_percentage_coverage(value: Any) -> str | None:
    text = _stringify(value).strip()
    if not text or text.upper() == "NA":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number < 0:
        number = 0.0
    if number > 100:
        number = 100.0
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _extract_location_code(value: Any) -> tuple[str, str | None]:
    if isinstance(value, str):
        text = value.strip()
        code = _parse_location_code(text)
        if code:
            return code, None
        return (text or DEFAULT_LOCATION, None)
    if isinstance(value, dict):
        fallback_description = _stringify(value.get("description") or value.get("name") or value.get("common:other")) or None
        for key in ("code", "@location", "location", "country", "region"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                parsed = _parse_location_code(candidate)
                return (parsed or candidate.strip(), fallback_description)
        return DEFAULT_LOCATION, fallback_description
    if isinstance(value, (list, tuple)):
        for item in value:
            code, desc = _extract_location_code(item)
            if code:
                return code, desc
    return DEFAULT_LOCATION, None


def _parse_location_code(text: str) -> str | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    upper = cleaned.upper()
    iso_match = re.search(r"ISO[:\s\-]*([A-Z]{2,3})", upper)
    if iso_match:
        return iso_match.group(1)
    paren_match = re.search(r"\(([A-Z]{2,3})\)", upper)
    if paren_match:
        return paren_match.group(1)
    start_match = re.match(r"^([A-Z]{2,3})(?:\b|[^A-Z0-9])", upper)
    if start_match:
        return start_match.group(1)
    tokens = [token for token in re.split(r"[\s,;/()\-]+", upper) if token]
    for token in tokens:
        if token == "ISO":
            continue
        if len(token) in {2, 3} and token.isalpha():
            return token
    return None


def _normalise_dataset_type(value: Any) -> str | None:
    options = _get_enum_options(TYPE_OF_DATA_SET_POINTER)
    match = _match_enum_value(value, options)
    if match:
        return match
    text_raw = _stringify(value)
    text = text_raw.strip().lower()
    if not text:
        return None
    single_operation = _find_enum_option_by_keywords(options, "single", "operation")
    if (("non" in text and "aggreg" in text) or ("single" in text and "operation" in text)) and single_operation:
        return single_operation
    black_box = _find_enum_option_by_keywords(options, "black", "box")
    if "black" in text and "box" in text and black_box:
        return black_box
    lci_result = _find_enum_option_by_keywords(options, "lci")
    if ("lci" in text or "inventory" in text) and lci_result:
        return lci_result
    partly_terminated = _find_enum_option_by_keywords(options, "partly")
    if ("partly" in text or "terminated" in text) and partly_terminated:
        return partly_terminated
    avoided_system = _find_enum_option_by_keywords(options, "avoid")
    if "avoid" in text and avoided_system:
        return avoided_system
    return None


def _normalise_lci_method_approach(value: Any) -> str | None:
    options = _get_enum_options(LCI_METHOD_APPROACH_POINTER)
    match = _match_enum_value(value, options)
    if match:
        return match
    text_raw = _stringify(value)
    text = text_raw.strip()
    if not text:
        return None
    lowered = text.lower()
    keyword_map: dict[tuple[str, ...], str | None] = {
        ("market", "value"): _find_enum_option_by_keywords(options, "market", "value"),
        ("gross", "calorific"): _find_enum_option_by_keywords(options, "gross", "calorific"),
        ("net", "calorific"): _find_enum_option_by_keywords(options, "net", "calorific"),
        ("exergetic",): _find_enum_option_by_keywords(options, "exergetic"),
        ("element", "content"): _find_enum_option_by_keywords(options, "element", "content"),
        ("mass",): _find_enum_option_by_keywords(options, "mass"),
        ("volume",): _find_enum_option_by_keywords(options, "volume"),
        ("ability", "bear"): _find_enum_option_by_keywords(options, "ability", "bear"),
        ("marginal", "causality"): _find_enum_option_by_keywords(options, "marginal", "causality"),
        ("physical", "causality"): _find_enum_option_by_keywords(options, "physical", "causality"),
        ("100",): _find_enum_option_by_keywords(options, "100"),
        ("other", "explicit"): _find_enum_option_by_keywords(options, "other", "explicit"),
        ("equal", "distribution"): _find_enum_option_by_keywords(options, "equal", "distribution"),
        ("recycled", "content"): _find_enum_option_by_keywords(options, "recycled", "content"),
        ("bat",): _find_enum_option_by_keywords(options, "bat"),
        ("market", "price"): _find_enum_option_by_keywords(options, "market", "price"),
        ("technical", "properties"): _find_enum_option_by_keywords(options, "technical", "properties"),
        ("recycling", "potential"): _find_enum_option_by_keywords(options, "recycling", "potential"),
        ("no", "correction"): _find_enum_option_by_keywords(options, "no", "correction"),
        ("specific",): _find_enum_option_by_keywords(options, "specific"),
        ("consequential",): _find_enum_option_by_keywords(options, "consequential"),
    }
    for keywords, option in keyword_map.items():
        if option and all(keyword in lowered for keyword in keywords):
            return option
    return None


def _split_product_and_route(base_name: str) -> tuple[str, str]:
    cleaned = base_name.strip()
    lower = cleaned.lower()
    for token in [" for ", " to ", " via ", " -> ", " - ", " — ", ":"]:
        idx = lower.find(token)
        if idx != -1:
            product = cleaned[:idx].strip()
            route = cleaned[idx + len(token) :].strip()
            if not product:
                product = cleaned
            if not route:
                route = cleaned
            return product, route
    return cleaned, cleaned


def _extract_feedstock(sources: list[str], product: str) -> str:
    for text in sources:
        match = re.search(
            r"([A-Za-z0-9\s\-/]+?)\s+as\s+feedstock",
            text or "",
            re.IGNORECASE,
        )
        if match:
            candidate = match.group(1).strip(" ,.;:")
            if candidate:
                return _title_case_phrase(_clean_feedstock_phrase(candidate))
    for text in sources:
        match = re.search(r"feedstock\s*(?:is|are|:)?\s*([A-Za-z0-9\s\-/]+)", text or "", re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" ,.;:")
            if candidate:
                return _title_case_phrase(_clean_feedstock_phrase(candidate))
    for text in sources:
        match = re.search(r"([A-Za-z0-9\s\-/]+?)\s+feedstock", text or "", re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" ,.;:")
            if candidate:
                return _title_case_phrase(_clean_feedstock_phrase(candidate))
    for keyword in FEEDSTOCK_KEYWORDS:
        for text in sources:
            if keyword in (text or "").lower():
                return _title_case_phrase(keyword)
    for keyword in FEEDSTOCK_KEYWORDS:
        if keyword in product.lower():
            return _title_case_phrase(keyword)
    return product


def _clean_feedstock_phrase(text: str) -> str:
    cleaned = re.sub(r"^(of|the|a|an)\s+", "", text, flags=re.IGNORECASE).strip()
    cleaned = re.split(r"\b(used|consumed|for)\b", cleaned, maxsplit=1)[0].strip()
    return cleaned or text


def _extract_standards(sources: list[str]) -> str:
    for text in sources:
        if not text:
            continue
        match = re.search(r"([^.]*standard[^.]*)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip(" ;,.")
        match = re.search(r"(ISO\s?\d+(?:[:/]\d+)?)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _infer_mix_type(sources: list[str]) -> str:
    for text in sources:
        if not text:
            continue
        lower = text.lower()
        if "consumption mix" in lower:
            return "Consumption mix"
    return "Production mix"


LOCATION_TYPE_KEYWORDS = {
    "consumer": "to consumer",
    "wholesale": "at wholesale",
    "retail": "at sales point",
    "plant": "at plant",
    "factory": "at plant",
    "gate": "at plant",
}


def _infer_location_type(sources: list[str]) -> str:
    for text in sources:
        if not text:
            continue
        lower = text.lower()
        for keyword, location in LOCATION_TYPE_KEYWORDS.items():
            if keyword in lower:
                return location
    return "at plant"


def _compose_treatment_string(
    product: str,
    feedstock: str,
    route: str,
    standards: str,
) -> str:
    treatment_segments = _collect_treatment_segments(product, feedstock, route, standards)
    return _semicolon_join([product] + treatment_segments)


def _compose_mix_string(mix_type: str, location_type: str, code: str | None) -> str:
    components = [mix_type, location_type]
    mix = ", ".join(filter(None, components))
    if code and code.upper() != "GLO":
        mix = f"{mix}, {code}"
    return mix


def _resolve_route(product: str, route_candidate: str, sources: list[str]) -> str:
    candidate = route_candidate.strip()
    if candidate and candidate.lower() not in {product.lower(), "unnamed process"}:
        return candidate
    for text in sources:
        if not text:
            continue
        match = re.search(r"(?:technical|technology)\s+route[:：]\s*([^;\n,]+)", text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate
        match = re.search(r"route(?:\s+is|:)?\s*([^;\n,]+)", text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate
    for text in sources:
        if text and "CGTM" in text.upper():
            return "Coal Gasification to Methanol (CGTM)"
    return route_candidate.strip()


def _collect_treatment_segments(
    product: str,
    feedstock: str,
    route: str,
    standards: str,
) -> list[str]:
    segments: list[str] = []
    feedstock_clean = _clean_feedstock_phrase(feedstock)
    if feedstock_clean:
        segment = f"{feedstock_clean} feedstock"
        if segment.lower() != product.lower():
            segments.append(segment)
    route_clean = route.strip()
    if route_clean and route_clean.lower() not in {product.lower(), feedstock_clean.lower()}:
        segments.append(route_clean)
    standards_clean = standards.strip()
    if standards_clean and standards_clean.lower() not in {
        product.lower(),
        feedstock_clean.lower(),
        route_clean.lower(),
    }:
        segments.append(_shorten_standard_text(standards_clean))
    return segments


def _shorten_standard_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    for delimiter in [";", ","]:
        if delimiter in cleaned:
            cleaned = cleaned.split(delimiter)[0].strip()
    lower = cleaned.lower()
    if "cleaner production" in lower:
        subject = ""
        match = re.search(r"for\s+the\s+(.+)", cleaned, re.IGNORECASE)
        if match:
            subject = match.group(1).strip(" ,.()")
        match = re.search(r"for\s+(.+)", cleaned, re.IGNORECASE)
        if not subject and match:
            subject = match.group(1).strip(" ,.()")
        subject = re.sub(r"industry$", "", subject, flags=re.IGNORECASE).strip()
        cleaned = "Cleaner production standard"
        if subject:
            cleaned = f"{cleaned} ({subject})"
    return cleaned


def _semicolon_join(parts: list[str]) -> str:
    return "; ".join(part.strip() for part in parts if part and part.strip())


def _deduplicate_preserve_order(parts: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        key = part.strip().lower()
        if not part or key in seen:
            continue
        seen.add(key)
        result.append(part.strip())
    return result


def _strip_common_other(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_common_other(val) for key, val in value.items() if key != "common:other"}
    if isinstance(value, list):
        return [_strip_common_other(item) for item in value]
    return value


def _finalise_mix_string(name_components: dict[str, Any], geography: dict[str, Any]) -> None:
    mix_type = name_components.get("mix_type") or "Production mix"
    location_type = name_components.get("location_type") or "at plant"
    location_block = _ensure_dict(geography.get("locationOfOperationSupplyOrProduction"))
    code = location_block.get("@location") or geography.get("code")
    name_components["location_code"] = code
    existing_mix = _format_name_field_text(name_components.get("mix"))
    if existing_mix:
        name_components["mix"] = existing_mix
        return
    name_components["mix"] = _compose_mix_string(mix_type, location_type, code)


def _title_case_phrase(text: str) -> str:
    if not text:
        return text
    tokens = [token.strip() for token in text.split() if token.strip()]
    return " ".join(token.capitalize() if len(token) > 1 else token.upper() for token in tokens)


def _extract_functional_unit_text(qref: dict[str, Any]) -> str:
    if not isinstance(qref, dict):
        return ""
    fu = qref.get("functionalUnitOrOther")
    if isinstance(fu, dict):
        return _stringify(fu.get("#text") or fu.get("text"))
    return _stringify(fu)


def _select_reference_flow(
    candidates: list[dict[str, Any]],
    name_components: dict[str, Any],
) -> str | None:
    if not candidates:
        return None
    functional_unit = name_components.get("functional_unit") or ""
    fu_amount, fu_unit = _parse_amount_unit(functional_unit)
    output_candidates = [candidate for candidate in candidates if candidate.get("direction", "").lower() == "output"]
    search_pool = output_candidates or candidates

    for candidate in search_pool:
        name = (candidate.get("name") or "").lower()
        if "reference" in name or "functional" in name:
            return candidate["id"]

    if fu_amount is not None and fu_unit:
        match = _match_candidate_by_amount(search_pool, fu_amount, fu_unit)
        if match:
            return match

    product = (name_components.get("product") or "").lower()
    route = (name_components.get("route") or "").lower()
    for candidate in search_pool:
        base = _extract_exchange_base_name(candidate)
        if product and product in base:
            return candidate["id"]
        if route and route in base:
            return candidate["id"]

    if search_pool:
        return search_pool[0]["id"]
    return candidates[0]["id"]


def _parse_amount_unit(text: str) -> tuple[float | None, str | None]:
    if not text:
        return None, None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Zµμ%/]+)", text)
    if not match:
        return None, None
    try:
        amount = float(match.group(1))
    except (TypeError, ValueError):
        amount = None
    unit = match.group(2)
    return amount, unit


def _match_candidate_by_amount(
    candidates: list[dict[str, Any]],
    fu_amount: float,
    fu_unit: str,
) -> str | None:
    for candidate in candidates:
        amount = candidate.get("mean")
        unit = candidate.get("unit") or ""
        try:
            amount_value = float(amount)
        except (TypeError, ValueError):
            continue
        if abs(amount_value - fu_amount) <= 1e-6 and unit and unit.lower() == fu_unit.lower():
            return candidate["id"]
    return None


def _extract_exchange_base_name(candidate: dict[str, Any]) -> str:
    short_desc = candidate.get("short") or ""
    base = short_desc.split(";")[0].strip().lower()
    name = (candidate.get("name") or "").strip().lower()
    if base:
        return base
    return name
