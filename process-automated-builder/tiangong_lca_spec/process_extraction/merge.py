"""Merger utilities for combining extraction results with flow matches."""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Iterable
from uuid import uuid4

from tiangong_lca_spec.core.models import FlowCandidate, ProcessDataset
from tiangong_lca_spec.core.uris import build_portal_uri

_CJK_CHAR_PATTERN = re.compile(r"[\u2e80-\u2eff\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\ua000-\ua4cf\uac00-\ud7af\uff00-\uffef]+")


def _resolve_base_name(name_block: Any) -> str | None:
    if isinstance(name_block, dict):
        base = name_block.get("baseName")
        if isinstance(base, dict):
            text = base.get("#text") or base.get("text")
            if text:
                return text
            for value in base.values():
                if isinstance(value, str):
                    return value
        elif base:
            return str(base)
        text = name_block.get("#text") or name_block.get("text")
        if text:
            return str(text)
        for value in name_block.values():
            if isinstance(value, str):
                return value
    elif isinstance(name_block, list) and name_block:
        return _resolve_base_name(name_block[0])
    elif isinstance(name_block, str):
        return name_block
    return None


def merge_results(
    process_blocks: list[dict[str, Any]],
    matched_lookup: dict[str, list[FlowCandidate]],
    origin_exchanges: dict[str, list[dict[str, Any]]],
) -> list[ProcessDataset]:
    datasets: list[ProcessDataset] = []
    for block in process_blocks:
        process_dataset = block.get("processDataSet")
        if not isinstance(process_dataset, dict):
            raise ValueError("Expected `processDataSet` in process block")
        process_information = process_dataset.get("processInformation", {})
        modelling = process_dataset.get("modellingAndValidation", {})
        administrative = process_dataset.get("administrativeInformation", {})
        base_exchanges = process_dataset.get("exchanges", {}).get("exchange") or []

        process_name = _extract_process_name_from_dataset(process_dataset, block)
        exchanges = origin_exchanges.get(process_name) or base_exchanges
        exchanges_list = _ensure_list(exchanges)
        merged_exchanges = _merge_exchange_candidates(
            exchanges_list,
            matched_lookup.get(process_name, []),
        )

        dataset = ProcessDataset(
            process_information=process_information,
            modelling_and_validation=modelling,
            administrative_information=administrative,
            exchanges=merged_exchanges,
            process_data_set=process_dataset,
        )
        datasets.append(dataset)
    return datasets


def _extract_process_name_from_dataset(
    process_dataset: dict[str, Any],
    block: dict[str, Any],
) -> str:
    process_info = process_dataset.get("processInformation", {})
    dataset_info = process_info.get("dataSetInformation", {})
    name_block = dataset_info.get("name")
    resolved = _resolve_base_name(name_block)
    if resolved:
        return resolved
    return block.get("process_name", "unknown_process")


def _merge_exchange_candidates(
    exchanges: list[dict[str, Any]],
    candidates: list[FlowCandidate],
) -> list[dict[str, Any]]:
    candidate_map_by_uuid = {candidate.uuid.lower(): candidate for candidate in candidates if getattr(candidate, "uuid", None) and isinstance(candidate.uuid, str)}
    candidate_map_by_name = {candidate.base_name.lower(): candidate for candidate in candidates if getattr(candidate, "base_name", None)}
    merged: list[dict[str, Any]] = []
    for exchange in exchanges:
        enriched = dict(exchange)
        candidate: FlowCandidate | None = None
        ref_uuid = _extract_reference_uuid(enriched.get("referenceToFlowDataSet"))
        if ref_uuid:
            candidate = candidate_map_by_uuid.get(ref_uuid.lower())
        if candidate is None:
            base_name = _exchange_base_name(enriched).lower()
            candidate = candidate_map_by_name.get(base_name)
        if candidate is None:
            candidate = _candidate_from_matching_detail(enriched)
        if candidate and candidate.uuid:
            enriched["referenceToFlowDataSet"] = _reference_from_candidate(candidate, enriched)
        elif not _has_reference(enriched.get("referenceToFlowDataSet")):
            enriched["referenceToFlowDataSet"] = _placeholder_reference(_exchange_base_name(enriched) or "Unspecified flow")
        if candidate:
            enriched.setdefault("matchingDetail", asdict(candidate))
        merged.append(enriched)
    return merged


def _ensure_list(exchanges: Iterable[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(exchanges, list):
        return exchanges
    if isinstance(exchanges, dict):
        return [exchanges]
    return list(exchanges or [])


def _exchange_base_name(exchange: dict[str, Any]) -> str:
    name = exchange.get("exchangeName") or exchange.get("name") or exchange.get("flowName")
    if name:
        return str(name)
    reference = exchange.get("referenceToFlowDataSet")
    if isinstance(reference, dict):
        text = _short_description_text(reference.get("common:shortDescription"))
        if text:
            parts = [part.strip() for part in text.split(";")]
            return parts[0] if parts else text.strip()
    return ""


def _short_description_text(value: Any) -> str:
    if isinstance(value, dict):
        text = value.get("#text") or value.get("text")
        if text:
            return str(text)
        for candidate in value.values():
            candidate_text = _short_description_text(candidate)
            if candidate_text:
                return candidate_text
        return ""
    if isinstance(value, list):
        for item in value:
            candidate = _short_description_text(item)
            if candidate:
                return candidate
        return ""
    if value is None:
        return ""
    return str(value)


def determine_functional_unit(exchanges: list[dict[str, Any]]) -> str | None:
    for exchange in exchanges:
        name = _exchange_base_name(exchange).lower()
        if not name:
            continue
        if _is_waste(name):
            continue
        amount = exchange.get("resultingAmount") or exchange.get("amount")
        unit = exchange.get("unit") or exchange.get("resultingAmountUnit")
        if amount and unit:
            return f"{amount} {unit} {_exchange_base_name(exchange)}"
    return None


def _is_waste(name: str) -> bool:
    waste_keywords = ["waste", "slag", "flue gas", "residue"]
    return any(keyword in name for keyword in waste_keywords)


def _sanitize_component_text(value: str | None) -> str:
    if not value:
        return ""
    sanitized = value.replace("；", ";").replace("，", ",")
    sanitized = _CJK_CHAR_PATTERN.sub("", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized)
    return sanitized.strip(" ,;")


def _normalise_candidate_component(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.replace("；", ";").replace("，", ",")
    segments = [segment.strip(" ,;") for segment in re.split(r"[;,]", cleaned) if segment and segment.strip(" ,;")]
    if not segments:
        return None
    unique: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        normalised = re.sub(r"\s+", " ", segment)
        lower = normalised.lower()
        if lower not in seen:
            seen.add(lower)
            english = _sanitize_component_text(normalised)
            if english:
                unique.append(english)
    return ", ".join(unique)


def _join_short_description_parts(parts: list[str]) -> str:
    cleaned: list[str] = []
    for part in parts:
        candidate = part.replace("；", ";").replace("，", ",").strip()
        candidate = re.sub(r"\s*,\s*", ", ", candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip(" ,;")
        candidate = _sanitize_component_text(candidate)
        if candidate:
            cleaned.append(candidate)
    return "; ".join(cleaned)


def _candidate_reference_parts(candidate: FlowCandidate, exchange: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    fallback = _exchange_base_name(exchange)
    field_map = (
        ("baseName", candidate.base_name or fallback),
        ("treatmentStandardsRoutes", candidate.treatment_standards_routes),
        ("mixAndLocationTypes", candidate.mix_and_location_types),
        ("flowProperties", candidate.flow_properties),
    )
    parts: list[str] = []
    field_values: dict[str, str] = {}
    for key, raw_value in field_map:
        formatted = _normalise_candidate_component(raw_value)
        if formatted:
            field_values[key] = formatted
            parts.append(formatted)
    if not parts and fallback:
        formatted = _normalise_candidate_component(fallback) or _sanitize_component_text(fallback)
        if formatted:
            parts.append(formatted)
    return parts, field_values


def _reference_from_candidate(candidate: FlowCandidate, exchange: dict[str, Any]) -> dict[str, Any]:
    version = candidate.version or "01.01.000"
    uuid_value = candidate.uuid or str(uuid4())
    uri = build_portal_uri("flow", uuid_value, version)
    parts, _ = _candidate_reference_parts(candidate, exchange)
    short_description = _join_short_description_parts(parts or ["Matched flow"])
    reference: dict[str, Any] = {
        "@type": "flow data set",
        "@refObjectId": uuid_value,
        "@version": version,
        "@uri": uri,
        "common:shortDescription": _multilang(short_description or "Matched flow"),
    }
    return reference


def _placeholder_reference(name: str) -> dict[str, Any]:
    identifier = str(uuid4())
    version = "00.00.000"
    uri = build_portal_uri("flow", identifier, version)
    short_description = _join_short_description_parts([name or "Unspecified flow"])
    return {
        "@type": "flow data set",
        "@refObjectId": identifier,
        "@version": version,
        "@uri": uri,
        "common:shortDescription": _multilang(short_description or "Unspecified flow"),
    }


def _has_reference(value: Any) -> bool:
    if isinstance(value, dict):
        return "@refObjectId" in value
    if isinstance(value, list):
        return all(isinstance(item, dict) and "@refObjectId" in item for item in value)
    return False


def _extract_reference_uuid(value: Any) -> str | None:
    if isinstance(value, dict):
        ref = value.get("@refObjectId") or value.get("@id")
        if isinstance(ref, str) and ref.strip():
            return ref.strip()
    return None


def _candidate_from_matching_detail(exchange: dict[str, Any]) -> FlowCandidate | None:
    detail = exchange.get("matchingDetail")
    if not isinstance(detail, dict):
        return None
    selected = detail.get("selectedCandidate")
    if not isinstance(selected, dict):
        return None
    base_name = selected.get("base_name")
    if not base_name:
        return None
    return FlowCandidate(
        uuid=selected.get("uuid"),
        base_name=base_name,
        treatment_standards_routes=selected.get("treatment_standards_routes"),
        mix_and_location_types=selected.get("mix_and_location_types"),
        flow_properties=selected.get("flow_properties"),
        version=selected.get("version"),
        general_comment=selected.get("general_comment"),
        geography=selected.get("geography"),
        classification=selected.get("classification"),
        reasoning=selected.get("reasoning", ""),
    )


def _multilang(text: str) -> dict[str, Any]:
    label = text or "Unnamed flow"
    return {"@xml:lang": "en", "#text": label}
