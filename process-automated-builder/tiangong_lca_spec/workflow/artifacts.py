"""Utilities to build ILCD artifacts directly from Stage 2/3 outputs."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from tiangong_lca_spec.core.constants import (
    ILCD_FORMAT_SOURCE_UUID,
    ILCD_FORMAT_SOURCE_VERSION,
    build_dataset_format_reference,
)
from tiangong_lca_spec.core.logging import get_logger
from tiangong_lca_spec.core.models import FlowCandidate, ProcessDataset
from tiangong_lca_spec.core.uris import build_local_dataset_uri, build_portal_uri
from tiangong_lca_spec.flow_alignment.selector import LanguageModelProtocol
from tiangong_lca_spec.process_extraction.merge import merge_results
from tiangong_lca_spec.process_extraction.tidas_mapping import ILCD_ENTRY_LEVEL_REFERENCE_ID
from tiangong_lca_spec.process_extraction.validators import is_placeholder_value
from tiangong_lca_spec.product_flow_creation import ProductFlowCreateRequest, ProductFlowCreationService
from tiangong_lca_spec.tidas_validation import TidasValidationService

DEFAULT_FORMAT_SOURCE_UUID = ILCD_FORMAT_SOURCE_UUID
SOURCE_CLASSIFICATIONS: dict[str, tuple[str, str]] = {
    "images": ("0", "Images"),
    "data set formats": ("1", "Data set formats"),
    "databases": ("2", "Databases"),
    "compliance systems": ("3", "Compliance systems"),
    "statistical classifications": ("4", "Statistical classifications"),
    "publications and communications": ("5", "Publications and communications"),
    "other source types": ("6", "Other source types"),
}

REQUIRED_FLOW_HINT_FIELDS: tuple[str, ...] = (
    "basename",
    "treatment",
    "mix_location",
    "flow_properties",
    "en_synonyms",
    "zh_synonyms",
    "abbreviation",
    "state_purity",
    "source_or_pathway",
    "usage_context",
)

OPTIONAL_FLOW_HINT_FIELDS: tuple[str, ...] = ("formula_or_CAS",)

FLOW_HINT_FIELDS: tuple[str, ...] = REQUIRED_FLOW_HINT_FIELDS + OPTIONAL_FLOW_HINT_FIELDS

FLOW_COMMENT_SYSTEM_PROMPT = (
    "You are producing an FTMultiLang entry for ILCD `common:generalComment`.\n"
    "Input JSON contains only baseName, treatmentStandardsRoutes, mixAndLocationTypes, "
    "flowProperties, synonyms_en, and synonyms_zh.\n"
    "Return a single English paragraph following this structure: (1) flow definition and identity, "
    "(2) typical application context in LCA/LCI, (3) differentiation from similar flows, "
    "(4) key properties or classification notes, (5) usage considerations or warnings.\n"
    "Write in third person, present tense, objective and factual tone. No bullet lists, no field=value recitations, no Markdown.\n"
    "Do not copy the JSON literally or restate the FlowSearch hints template. Mention Chinese aliases only if needed, placing them in parentheses after the English term.\n"
    "If information is missing, acknowledge it briefly instead of inventing data.\n"
    "Return only the paragraph text."
)

DEFAULT_DATA_SET_VERSION = "01.01.000"


def resolve_dataset_version(ilcd_dataset: Mapping[str, Any] | None) -> str:
    """Extract the dataset version from an ILCD node, falling back to the default."""
    if isinstance(ilcd_dataset, Mapping):
        admin = ilcd_dataset.get("administrativeInformation")
        if isinstance(admin, Mapping):
            publication = admin.get("publicationAndOwnership")
            if isinstance(publication, Mapping):
                version = publication.get("common:dataSetVersion")
                if isinstance(version, str):
                    version = version.strip()
                    if version:
                        return version
    return DEFAULT_DATA_SET_VERSION


def build_export_filename(uuid_value: str, dataset_version: str) -> str:
    """Return the canonical export filename <uuid>_<version>.json."""
    safe_uuid = (uuid_value or "").strip()
    if not safe_uuid:
        raise ValueError("UUID required to build export filename.")
    version = (dataset_version or "").strip() or DEFAULT_DATA_SET_VERSION
    safe_version = re.sub(r"[^0-9A-Za-z._-]", "_", version)
    if not safe_version:
        safe_version = DEFAULT_DATA_SET_VERSION
    return f"{safe_uuid}_{safe_version}.json"


CJK_CHAR_PATTERN = re.compile(r"[\u2e80-\u2eff\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\ua000-\ua4cf\uac00-\ud7af\uff00-\uffef]+")
CHINESE_PUNCT_REPLACEMENTS: dict[str, str] = {
    "，": ", ",
    "。": ". ",
    "；": "; ",
    "：": ": ",
    "、": ", ",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "《": "",
    "》": "",
}

PRODUCT_FALLBACK_CLASSIFICATION = [
    {
        "@level": "0",
        "@classId": "1",
        "#text": "Ores and minerals; electricity, gas and water",
    },
    {
        "@level": "1",
        "@classId": "17",
        "#text": "Electricity, town gas, steam and hot water",
    },
    {"@level": "2", "@classId": "171", "#text": "Electrical energy"},
    {"@level": "3", "@classId": "1710", "#text": "Electrical energy"},
    {"@level": "4", "@classId": "17100", "#text": "Electrical energy"},
]

WASTE_FALLBACK_CLASSIFICATION = [
    {
        "@level": "0",
        "@classId": "3",
        "#text": "Other transportable goods, except metal products, machinery and equipment",
    },
    {"@level": "1", "@classId": "39", "#text": "Wastes or scraps"},
    {"@level": "2", "@classId": "399", "#text": "Other wastes and scraps"},
    {"@level": "3", "@classId": "3999", "#text": "Other wastes n.e.c."},
    {"@level": "4", "@classId": "39990", "#text": "Other wastes n.e.c."},
]

ELEMENTARY_CATEGORY_AIR = [
    {"@level": "0", "@catId": "1", "#text": "Emissions"},
    {"@level": "1", "@catId": "1.3", "#text": "Emissions to air"},
    {"@level": "2", "@catId": "1.3.4", "#text": "Emissions to air, unspecified"},
]
ELEMENTARY_CATEGORY_WATER = [
    {"@level": "0", "@catId": "1", "#text": "Emissions"},
    {"@level": "1", "@catId": "1.1", "#text": "Emissions to water"},
    {"@level": "2", "@catId": "1.1.3", "#text": "Emissions to water, unspecified"},
]
ELEMENTARY_CATEGORY_SOIL = [
    {"@level": "0", "@catId": "1", "#text": "Emissions"},
    {"@level": "1", "@catId": "1.2", "#text": "Emissions to soil"},
    {"@level": "2", "@catId": "1.2.3", "#text": "Emissions to soil, unspecified"},
]
ELEMENTARY_CATEGORY_OTHER = [
    {"@level": "0", "@catId": "4", "#text": "Other elementary flows"},
]
ELEMENTARY_CATEGORY_RESOURCES = [
    {"@level": "0", "@catId": "2", "#text": "Resources"},
]


@dataclass(slots=True)
class ArtifactBuildSummary:
    """Lightweight summary returned after generating artifacts."""

    process_count: int
    flow_count: int
    source_count: int
    validation_report: list[dict[str, Any]]


def generate_artifacts(
    process_blocks: list[dict[str, Any]],
    alignment_entries: list[dict[str, Any]],
    *,
    artifact_root: Path,
    merged_output: Path,
    validation_output: Path,
    workflow_output: Path | None = None,
    format_source_uuid: str = DEFAULT_FORMAT_SOURCE_UUID,
    run_validation: bool = True,
    primary_source_title: str | None = None,
    comment_llm: LanguageModelProtocol | None = None,
) -> ArtifactBuildSummary:
    """Merge aligned results and materialise ILCD artifacts required by downstream tools."""

    matched_lookup, origin_exchanges = _build_alignment_indexes(alignment_entries)
    datasets = merge_results(process_blocks, matched_lookup, origin_exchanges)
    merged_serialised: list[dict[str, Any]] = []
    for dataset in datasets:
        serialised = _serialise_dataset(dataset)
        _sanitize_process_dataset(serialised)
        merged_serialised.append(serialised)

    merged_payload = {"process_datasets": merged_serialised}
    _dump_json(merged_payload, merged_output)

    timestamp = _utc_timestamp()
    _ensure_directories(artifact_root)

    primary_source_uuid: str | None = None
    if primary_source_title:
        primary_source_uuid = str(uuid4())

    source_references: dict[str, dict[str, Any]] = {}
    sanitized_ilcd_datasets: list[dict[str, Any]] = []
    for dataset in datasets:
        ilcd_dataset = dataset.as_dict()
        _sanitize_process_dataset(ilcd_dataset)
        if primary_source_uuid and primary_source_title:
            _attach_primary_source(ilcd_dataset, primary_source_uuid, primary_source_title)
        sanitized_ilcd_datasets.append(deepcopy(ilcd_dataset))
        uuid_value = ilcd_dataset.get("processInformation", {}).get("dataSetInformation", {}).get("common:UUID")
        if not uuid_value:
            raise ValueError("Process dataset missing common:UUID.")
        dataset_version = resolve_dataset_version(ilcd_dataset)
        process_filename = build_export_filename(uuid_value, dataset_version)
        process_path = artifact_root / "processes" / process_filename
        _dump_json({"processDataSet": ilcd_dataset}, process_path)

        source_references |= _collect_source_references(ilcd_dataset)

    unmatched_entries = _collect_unmatched_exchanges(alignment_entries)
    flow_count = 0
    for process_name, exchange in unmatched_entries:
        flow_dataset = _build_flow_dataset(
            exchange,
            process_name,
            timestamp,
            format_source_uuid,
            comment_llm,
        )
        if not flow_dataset:
            continue
        uuid_value, dataset = flow_dataset
        flow_ilcd = dataset.get("flowDataSet", {})
        dataset_version = resolve_dataset_version(flow_ilcd)
        flow_filename = build_export_filename(uuid_value, dataset_version)
        flow_path = artifact_root / "flows" / flow_filename
        _dump_json(dataset, flow_path)
        flow_count += 1

    written_sources = 0
    format_uuid_lower = (format_source_uuid or "").strip().lower()
    compliance_uuid_lower = (ILCD_ENTRY_LEVEL_REFERENCE_ID or "").strip().lower()
    for uuid_value, reference in source_references.items():
        candidate_uuid = (uuid_value or "").strip()
        if not candidate_uuid:
            continue
        candidate_lower = candidate_uuid.lower()
        if candidate_lower == format_uuid_lower:
            continue
        if compliance_uuid_lower and candidate_lower == compliance_uuid_lower:
            continue
        include_format = not (primary_source_uuid and candidate_uuid == primary_source_uuid)
        stub = _build_source_stub(
            candidate_uuid,
            reference,
            timestamp,
            format_source_uuid,
            include_format_reference=include_format,
        )
        source_ilcd = stub.get("sourceDataSet", {})
        dataset_version = resolve_dataset_version(source_ilcd)
        source_filename = build_export_filename(candidate_uuid, dataset_version)
        source_path = artifact_root / "sources" / source_filename
        _dump_json(stub, source_path)
        written_sources += 1

    if run_validation:
        validation_report = _run_validation(artifact_root)
    else:
        validation_report = []

    _dump_json({"validation_report": validation_report}, validation_output)

    if workflow_output is not None:
        payload = {
            "process_datasets": sanitized_ilcd_datasets,
            "alignment": [_sanitize_alignment_entry(entry) for entry in alignment_entries],
            "validation_report": validation_report,
        }
        _dump_json(payload, workflow_output)

    return ArtifactBuildSummary(
        process_count=len(datasets),
        flow_count=flow_count,
        source_count=written_sources,
        validation_report=validation_report,
    )


def _build_alignment_indexes(
    alignment_entries: list[dict[str, Any]],
) -> tuple[dict[str, list[FlowCandidate]], dict[str, list[dict[str, Any]]]]:
    matched_lookup: dict[str, list[FlowCandidate]] = {}
    origin_exchanges: dict[str, list[dict[str, Any]]] = {}
    for entry in alignment_entries:
        process_name = entry.get("process_name") or "unknown_process"
        matched_lookup[process_name] = _hydrate_flow_candidates(entry)
        origin: list[dict[str, Any]] = []
        origin_exchanges_block = entry.get("origin_exchanges") or {}
        if isinstance(origin_exchanges_block, dict):
            for exchanges in origin_exchanges_block.values():
                if isinstance(exchanges, list):
                    origin.extend(exchanges)
                elif isinstance(exchanges, dict):
                    origin.append(exchanges)
        origin_exchanges[process_name] = origin
    return matched_lookup, origin_exchanges


def _hydrate_flow_candidates(entry: dict[str, Any]) -> list[FlowCandidate]:
    candidates_raw = entry.get("matched_flows") or []
    hydrated: list[FlowCandidate] = []
    for item in candidates_raw:
        if isinstance(item, dict):
            hydrated.append(FlowCandidate(**item))
    return hydrated


def _serialise_dataset(dataset: ProcessDataset) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "process_information": deepcopy(dataset.process_information),
        "modelling_and_validation": deepcopy(dataset.modelling_and_validation),
        "administrative_information": deepcopy(dataset.administrative_information),
        "exchanges": [deepcopy(exchange) for exchange in dataset.exchanges],
    }
    if dataset.process_data_set is not None:
        payload["process_data_set"] = deepcopy(dataset.process_data_set)
    return payload


def _language_entry(text: str, lang: str = "en") -> dict[str, str]:
    return {"@xml:lang": lang, "#text": text}


def _dataset_format_reference() -> dict[str, Any]:
    return build_dataset_format_reference()


def _format_reference_block(format_source_uuid: str) -> dict[str, Any]:
    canonical_uuid = (format_source_uuid or "").strip()
    if canonical_uuid and canonical_uuid != ILCD_FORMAT_SOURCE_UUID:
        return {
            "@type": "source data set",
            "@refObjectId": canonical_uuid,
            "@uri": build_local_dataset_uri("source", canonical_uuid, ILCD_FORMAT_SOURCE_VERSION),
            "@version": ILCD_FORMAT_SOURCE_VERSION,
            "common:shortDescription": _language_entry("ILCD format"),
        }
    return build_dataset_format_reference()


def _unique_join(entries: Iterable[str]) -> str:
    seen: list[str] = []
    for entry in entries:
        candidate = entry.strip()
        if candidate and candidate not in seen:
            seen.append(candidate)
    return "; ".join(seen)


def _normalise_language(value: Any, default_lang: str = "en") -> list[dict[str, str]]:
    if value is None:
        return []
    if isinstance(value, list):
        normalised: list[dict[str, str]] = []
        for item in value:
            if isinstance(item, dict) and "#text" in item:
                lang = item.get("@xml:lang") or default_lang
                normalised.append(_language_entry(str(item["#text"]), lang))
            else:
                normalised.append(_language_entry(str(item), default_lang))
        return normalised
    if isinstance(value, dict) and "#text" in value:
        lang = value.get("@xml:lang") or default_lang
        return [_language_entry(str(value["#text"]), lang)]
    return [_language_entry(str(value), default_lang)]


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        text = value.get("#text")
        if isinstance(text, str):
            return text.strip()
    return str(value).strip()


def _extract_lang_text(value: Any, lang: str = "en") -> str:
    entries = _normalise_language(value, default_lang=lang)
    target = (lang or "en").strip().lower()
    fallback = ""
    for entry in entries:
        text = _extract_text(entry.get("#text"))
        if not text:
            continue
        entry_lang = _extract_text(entry.get("@xml:lang")).lower()
        if entry_lang == target:
            return text
        if not fallback:
            fallback = text
    return fallback


def _split_synonyms(value: str) -> list[str]:
    text = _extract_text(value).replace("；", ";")
    if not text:
        return []
    return [chunk.strip() for chunk in text.split(";") if chunk.strip()]


def _classification_entries(classification: Mapping[str, Any]) -> list[dict[str, str]]:
    payload = classification.get("common:classification")
    if not isinstance(payload, Mapping):
        return []
    classes = payload.get("common:class")
    if isinstance(classes, Mapping):
        classes = [classes]
    if not isinstance(classes, list):
        return []
    entries: list[dict[str, str]] = []
    for index, item in enumerate(classes):
        if not isinstance(item, Mapping):
            continue
        text = _extract_text(item.get("#text"))
        if not text:
            continue
        level = _extract_text(item.get("@level")) or str(index)
        class_id = _extract_text(item.get("@classId"))
        entry: dict[str, str] = {"@level": level, "#text": text}
        if class_id:
            entry["@classId"] = class_id
        entries.append(entry)
    return entries


ALLOWED_CHINESE_VALUES = {"天工LCA数据团队"}
PLACEHOLDER_COMMENT_VALUES = {
    "none",
    "no specific explanation",
    "no specific explanations",
    "no specific explanation provided",
    "no specific detail",
    "no specific details",
    "no specific comment",
    "no specific comments",
    "no additional information",
    "no further information",
    "not provided",
    "not specified",
    "no data",
    "no information",
}


def _sanitize_to_english(text: str) -> str:
    if not text:
        return ""
    sanitized = text
    for src, dst in CHINESE_PUNCT_REPLACEMENTS.items():
        sanitized = sanitized.replace(src, dst)
    sanitized = CJK_CHAR_PATTERN.sub("", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized)
    return sanitized.strip()


def _generate_flow_comment_entries(
    *,
    base_name: str | None,
    treatment: str | None,
    mix: str | None,
    flow_properties: str | list[str] | None,
    en_synonyms: str | list[str] | None,
    zh_synonyms: str | list[str] | None,
    fallback_comment: Any,
    process_name: str,
    llm: LanguageModelProtocol | None,
) -> list[dict[str, str]]:
    payload = {
        "baseName": _sanitize_to_english(base_name or ""),
        "treatmentStandardsRoutes": _sanitize_to_english(treatment or ""),
        "mixAndLocationTypes": _sanitize_to_english(mix or ""),
        "flowProperties": _semicolon_join(flow_properties),
        "synonyms_en": _synonym_list(en_synonyms),
        "synonyms_zh": _synonym_list(zh_synonyms),
    }

    if llm:
        generated = _invoke_flow_comment_llm(llm, payload)
        if generated:
            return [{"@xml:lang": "en", "#text": generated}]

    return _build_fallback_comment_entries(fallback_comment, process_name)


def _invoke_flow_comment_llm(llm: LanguageModelProtocol, payload: Mapping[str, Any]) -> str | None:
    try:
        response = llm.invoke({"prompt": FLOW_COMMENT_SYSTEM_PROMPT, "context": payload})
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("artifact_builder.flow_comment_llm_failed", error=str(exc))
        return None
    text = _coerce_llm_text(response)
    normalized = _normalise_ft_text(text)
    if not normalized:
        return None
    return normalized


def _coerce_llm_text(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, Mapping):
        for key in ("text", "output", "message", "content"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if "choices" in response and isinstance(response["choices"], list):
            for choice in response["choices"]:
                if isinstance(choice, Mapping):
                    message = choice.get("message")
                    if isinstance(message, Mapping):
                        content = message.get("content")
                        if isinstance(content, str) and content.strip():
                            return content.strip()
    return str(response or "").strip()


def _normalise_ft_text(text: str | None) -> str:
    if not text:
        return ""
    collapsed = re.sub(r"\s+", " ", text).strip()
    if not collapsed:
        return ""
    if len(collapsed) > 600:
        collapsed = collapsed[:600].rstrip()
    return collapsed


def _build_fallback_comment_entries(raw_comment: Any, process_name: str) -> list[dict[str, str]]:
    comment_entries = _normalise_language(raw_comment or f"Generated for {process_name}")
    comment_entries = [entry for entry in comment_entries if isinstance(entry, dict) and (entry.get("@xml:lang") or "en").lower() == "en" and _extract_text(entry.get("#text"))]
    sanitized_comments: list[dict[str, str]] = []
    for entry in comment_entries:
        text = _sanitize_comment_text(entry.get("#text", ""))
        if text:
            sanitized_comments.append(_language_entry(text, entry.get("@xml:lang", "en")))
    if sanitized_comments:
        return sanitized_comments
    fallback_comment = _extract_text(raw_comment)
    if not fallback_comment or not fallback_comment.isascii():
        sanitized_name = "".join(ch for ch in process_name if ch.isascii()).strip()
        fallback_comment = f"Generated for {sanitized_name}" if sanitized_name else "Generated placeholder comment"
    return [_language_entry(fallback_comment, "en")]


def _semicolon_join(value: str | list[str] | None) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        tokens = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return "; ".join(tokens)
    return str(value).strip()


def _synonym_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [token.strip() for token in value if isinstance(token, str) and token.strip()]
    return [token.strip() for token in value.split(";") if token.strip()]


def _normalize_flowsearch_hints(text: str) -> str:
    prefix = "FlowSearch hints:"
    body = text[len(prefix) :].strip()
    segments = []
    seen: set[str] = set()
    canonical_map = {field.lower(): field for field in FLOW_HINT_FIELDS}
    fields_to_include = [field for field in FLOW_HINT_FIELDS if field != "zh_synonyms"]
    for raw_segment in body.split("|"):
        segment = raw_segment.strip()
        if not segment:
            continue
        key, _, value = segment.partition("=")
        key = key.strip()
        key_lower = key.lower()
        canonical_key = canonical_map.get(key_lower, key)
        if not key_lower or canonical_key == "zh_synonyms":
            continue
        value = value.strip()
        if not key_lower:
            continue
        if key_lower in seen:
            continue
        seen.add(key_lower)
        value = value or "NA"
        segments.append(f"{canonical_key}={value}")
    for field in fields_to_include:
        if field.lower() not in seen:
            segments.append(f"{field}=NA")
    return f"{prefix} " + " | ".join(segments)


def _sanitize_comment_text(text: str) -> str:
    if not text:
        return ""
    sanitized = _sanitize_to_english(text)
    if not sanitized:
        return ""
    normalized = sanitized.strip()
    normalized_lower = normalized.lower().strip(".").strip()
    normalized_lstrip = normalized_lower.lstrip()
    if is_placeholder_value(normalized):
        return ""
    if normalized_lower in PLACEHOLDER_COMMENT_VALUES:
        return ""
    if normalized_lstrip.startswith("{'@xml:lang") or normalized_lstrip.startswith('{"@xml:lang'):
        return ""
    if sanitized.startswith("FlowSearch hints:"):
        sanitized = _normalize_flowsearch_hints(sanitized)
        sanitized = re.sub(r"(?:\|\s*)?zh_synonyms=[^|]*", "", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"\|\s*\|", "|", sanitized)
    sanitized = re.sub(r"Synonyms\s*\(ZH\)\s*:[^.;|]*", "", sanitized, flags=re.IGNORECASE)
    for prefix in (
        "Usage context: ",
        "Notes: ",
        "Source/pathway: ",
        "State/purity: ",
        "Synonyms (EN): ",
        "Abbreviation: ",
        "Formula/CAS: ",
    ):
        sanitized = sanitized.replace(prefix, "")
    sanitized = CJK_CHAR_PATTERN.sub("", sanitized)
    sanitized = re.sub(r"\s{2,}", " ", sanitized)
    sanitized = re.sub(r"\|\s*\|", "|", sanitized)
    sanitized = re.sub(r"\s*\|\s*$", "", sanitized)
    sanitized = re.sub(r"\s*;\s*;", ";", sanitized)
    return sanitized.strip(" ;|.")


def _normalize_candidate_component_text(value: Any) -> str:
    if value is None:
        return ""
    sanitized = _sanitize_to_english(_extract_text(value))
    if not sanitized:
        return ""
    tokens = [token.strip(" ,;") for token in re.split(r"[;,]", sanitized) if token.strip(" ,;")]
    if not tokens:
        return ""
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(token)
    return ", ".join(deduped)


def _build_candidate_short_description(candidate: dict[str, Any], fallback: str | None) -> str:
    if not candidate:
        return _normalize_candidate_component_text(fallback)
    parts: list[str] = []
    for field in ("base_name", "treatment_standards_routes", "mix_and_location_types", "flow_properties"):
        formatted = _normalize_candidate_component_text(candidate.get(field))
        if formatted:
            parts.append(formatted)
    if parts:
        return "; ".join(parts)
    return _normalize_candidate_component_text(fallback)


def _sanitize_language_entry(entry: Any) -> dict[str, Any] | None:
    if isinstance(entry, dict):
        text = _extract_text(entry)
        if text in ALLOWED_CHINESE_VALUES:
            return entry
        sanitized_text = _sanitize_comment_text(text)
        if not sanitized_text:
            return None
        return {"@xml:lang": "en", "#text": sanitized_text}
    if isinstance(entry, str):
        sanitized_text = _sanitize_comment_text(entry)
        if not sanitized_text:
            return None
        return {"@xml:lang": "en", "#text": sanitized_text}
    return None


def _truncate_language_entry(entry: dict[str, Any], max_length: int) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return entry
    text = entry.get("#text")
    if not isinstance(text, str):
        return entry
    if len(text) <= max_length:
        return entry
    truncated = text[:max_length].rstrip()
    fallback = max(truncated.rfind(". "), truncated.rfind("; "), truncated.rfind(", "), truncated.rfind(" "))
    if fallback > max_length * 0.6:
        truncated = truncated[:fallback].rstrip()
    if not truncated:
        truncated = text[:max_length].rstrip()
    entry["#text"] = truncated
    return entry


def _sanitize_matching_detail(detail: dict[str, Any]) -> None:
    for key, value in list(detail.items()):
        if isinstance(value, str):
            detail[key] = _sanitize_comment_text(value) if "comment" in key.lower() else _sanitize_to_english(value)
    selected = detail.get("selectedCandidate")
    if isinstance(selected, dict):
        for field in (
            "base_name",
            "treatment_standards_routes",
            "mix_and_location_types",
            "flow_properties",
            "version",
            "general_comment",
            "reasoning",
            "evaluation_reason",
            "combined_name",
        ):
            value = selected.get(field)
            if isinstance(value, str):
                if field == "general_comment":
                    selected[field] = _sanitize_comment_text(value)
                else:
                    selected[field] = _sanitize_to_english(value)


def _normalize_short_description_text(text: str) -> str:
    if not text:
        return ""
    normalized = text.replace("；", ";").replace("，", ",")
    normalized = re.sub(r"\s*;\s*", "; ", normalized)
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" ;,")


def _localize_reference_uri(node: dict[str, Any]) -> None:
    if not isinstance(node, dict):
        return
    uuid_value = str(node.get("@refObjectId") or "").strip()
    if not uuid_value:
        return
    version = str(node.get("@version") or "").strip() or DEFAULT_DATA_SET_VERSION
    ref_type = str(node.get("@type") or "").strip().lower()
    uri_text = str(node.get("@uri") or "")

    dataset_kind: str | None = None
    if ref_type in {"flow data set", "flow"}:
        dataset_kind = "flow"
    elif ref_type in {"source data set", "source"}:
        dataset_kind = "source"
    elif ref_type in {"process data set", "process"}:
        dataset_kind = "process"
    elif "showproductflow" in uri_text.lower():
        dataset_kind = "flow"
    elif "showsource" in uri_text.lower():
        dataset_kind = "source"
    elif "showprocess" in uri_text.lower():
        dataset_kind = "process"

    if dataset_kind:
        node["@uri"] = build_local_dataset_uri(dataset_kind, uuid_value, version)


def _sanitize_reference_node(node: dict[str, Any]) -> None:
    if not isinstance(node, dict):
        return
    _localize_reference_uri(node)
    short_desc = node.get("common:shortDescription")
    if isinstance(short_desc, dict):
        text = _extract_text(short_desc)
        lang = short_desc.get("@xml:lang") or "en"
        normalized = _normalize_short_description_text(text)
        node["common:shortDescription"] = {"@xml:lang": lang, "#text": normalized or "Unnamed flow"}
    elif isinstance(short_desc, list):
        for entry in short_desc:
            candidate = _sanitize_language_entry(entry)
            if candidate:
                normalized_text = _normalize_short_description_text(candidate.get("#text"))
                if normalized_text:
                    candidate["#text"] = normalized_text
                    node["common:shortDescription"] = candidate
                else:
                    node["common:shortDescription"] = {"@xml:lang": candidate.get("@xml:lang") or "en", "#text": "Unnamed flow"}
                break
        else:
            node["common:shortDescription"] = {"@xml:lang": "en", "#text": "Unnamed flow"}
    elif isinstance(short_desc, str):
        normalized = _normalize_short_description_text(short_desc)
        node["common:shortDescription"] = {"@xml:lang": "en", "#text": normalized or "Unnamed flow"}


def _sanitize_language_field(container: dict[str, Any], key: str) -> None:
    if not isinstance(container, dict) or key not in container:
        return
    value = container[key]
    if isinstance(value, list):
        sanitized_entries = [_sanitize_language_entry(entry) for entry in value]
        sanitized_entries = [entry for entry in sanitized_entries if entry]
        if sanitized_entries:
            container[key] = sanitized_entries
        else:
            container.pop(key, None)
    elif isinstance(value, dict):
        sanitized = _sanitize_language_entry(value)
        if sanitized:
            container[key] = sanitized
        else:
            container.pop(key, None)
    elif isinstance(value, str):
        sanitized = _sanitize_comment_text(value)
        if sanitized:
            container[key] = sanitized
        else:
            container.pop(key, None)


def _sanitize_name_block(name_block: Any) -> None:
    if not isinstance(name_block, dict):
        return
    for key, value in list(name_block.items()):
        if isinstance(value, dict):
            sanitized = _sanitize_language_entry(value)
            if sanitized:
                name_block[key] = sanitized
            else:
                name_block.pop(key, None)
        elif isinstance(value, list):
            sanitized_entries = [_sanitize_language_entry(entry) for entry in value]
            sanitized_entries = [entry for entry in sanitized_entries if entry]
            if sanitized_entries:
                name_block[key] = sanitized_entries
            else:
                name_block.pop(key, None)
        elif isinstance(value, str):
            sanitized = _sanitize_to_english(value)
            if sanitized:
                name_block[key] = sanitized
            else:
                name_block.pop(key, None)


def _sanitize_exchange_language(exchange: dict[str, Any]) -> dict[str, Any]:
    sanitized = deepcopy(exchange)
    name = _sanitize_to_english(_extract_text(sanitized.get("exchangeName")))
    if name:
        sanitized["exchangeName"] = name
    comment = sanitized.get("generalComment")
    if isinstance(comment, list):
        sanitized_comment = None
        for entry in comment:
            sanitized_comment = _sanitize_language_entry(entry)
            if sanitized_comment:
                break
    else:
        sanitized_comment = _sanitize_language_entry(comment)
    if sanitized_comment:
        sanitized["generalComment"] = _truncate_language_entry(sanitized_comment, 500)
    else:
        sanitized.pop("generalComment", None)
    reference = sanitized.get("referenceToFlowDataSet")
    if isinstance(reference, dict):
        _sanitize_reference_node(reference)
        candidate = _extract_candidate(sanitized)
        if candidate:
            fallback = _extract_text(reference.get("common:shortDescription")) or name
            short_desc = _build_candidate_short_description(candidate, fallback)
            if short_desc:
                reference["common:shortDescription"] = _language_entry(short_desc, "en")
    matching_detail = sanitized.get("matchingDetail")
    if isinstance(matching_detail, dict):
        _sanitize_matching_detail(matching_detail)
    return sanitized


def _merge_intended_applications(container: dict[str, Any]) -> None:
    key = "common:intendedApplications"
    if not isinstance(container, dict) or key not in container:
        return
    value = container[key]
    entries = value if isinstance(value, list) else [value]
    merged: dict[str, list[str]] = {}
    order: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            lang = (entry.get("@xml:lang") or "en").lower()
            text = _extract_text(entry)
        else:
            lang = "en"
            text = _extract_text(entry)
        text = re.sub(r"\s+", " ", text).strip(" ;,")
        if not text:
            continue
        if lang not in merged:
            merged[lang] = []
            order.append(lang)
        if text not in merged[lang]:
            merged[lang].append(text)
    if not merged:
        container.pop(key, None)
        return
    preferred_order = [lang for lang in order if lang == "en"]
    if preferred_order:
        order = preferred_order
    container[key] = [{"@xml:lang": lang, "#text": "; ".join(merged[lang])} for lang in order if merged.get(lang)]


def _sanitize_process_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    if "processDataSet" in dataset and isinstance(dataset["processDataSet"], dict):
        dataset["processDataSet"] = _sanitize_process_dataset(dataset["processDataSet"])
        return dataset
    if "process_data_set" in dataset and isinstance(dataset["process_data_set"], dict):
        dataset["process_data_set"] = _sanitize_process_dataset(dataset["process_data_set"])

    info = dataset.get("processInformation") or dataset.get("process_information")
    if isinstance(info, dict):
        data_info = info.get("dataSetInformation") or info.get("data_set_information")
        if isinstance(data_info, dict):
            _sanitize_language_field(data_info, "common:generalComment")
            name_node = data_info.get("name")
            _sanitize_name_block(name_node)
            _sanitize_language_field(data_info, "common:synonyms")
        _sanitize_language_field(info, "generalComment")

    modelling = dataset.get("modellingAndValidation") or dataset.get("modelling_and_validation")
    if isinstance(modelling, dict):
        _sanitize_language_field(modelling, "common:generalComment")
        validation = modelling.get("validation")
        if isinstance(validation, dict):
            review = validation.get("review")
            if isinstance(review, list):
                review = review[0] if review else {}
            if isinstance(review, dict):
                review["@type"] = "Not reviewed"
                for key in list(review.keys()):
                    if key != "@type":
                        review.pop(key, None)
                validation["review"] = review
            else:
                validation["review"] = {"@type": "Not reviewed"}
        else:
            modelling["validation"] = {"review": {"@type": "Not reviewed"}}
        _localize_compliance_references(modelling.get("complianceDeclarations"))

    exchanges_container = dataset.get("exchanges")
    if isinstance(exchanges_container, dict):
        exchanges = exchanges_container.get("exchange")
        if isinstance(exchanges, list):
            exchanges_container["exchange"] = [_sanitize_exchange_language(item) for item in exchanges if isinstance(item, dict)]
        elif isinstance(exchanges, dict):
            exchanges_container["exchange"] = [_sanitize_exchange_language(exchanges)]
    elif isinstance(exchanges_container, list):
        dataset["exchanges"] = [_sanitize_exchange_language(item) for item in exchanges_container if isinstance(item, dict)]

    admin = dataset.get("administrativeInformation") or dataset.get("administrative_information")
    if isinstance(admin, dict):
        _sanitize_language_field(admin, "common:generalComment")
        commissioner = admin.get("common:commissionerAndGoal")
        if isinstance(commissioner, dict):
            _merge_intended_applications(commissioner)
        data_entry = admin.get("dataEntryBy")
        if isinstance(data_entry, dict):
            data_entry["common:referenceToDataSetFormat"] = _dataset_format_reference()
            data_entry.pop("common:referenceToDataSetUseApproval", None)
            data_entry.pop("common:referenceToConvertedOriginalDataSetFrom", None)

    return dataset


def _localize_compliance_references(container: Any) -> None:
    if not isinstance(container, dict):
        return
    reference = container.get("common:referenceToComplianceSystem")
    if isinstance(reference, dict):
        _localize_reference_uri(reference)
    compliance_entries = container.get("compliance")
    if isinstance(compliance_entries, list):
        for entry in compliance_entries:
            if isinstance(entry, dict):
                ref = entry.get("common:referenceToComplianceSystem")
                if isinstance(ref, dict):
                    _localize_reference_uri(ref)
    elif isinstance(compliance_entries, dict):
        ref = compliance_entries.get("common:referenceToComplianceSystem")
        if isinstance(ref, dict):
            _localize_reference_uri(ref)


def _sanitize_alignment_entry(entry: dict[str, Any]) -> dict[str, Any]:
    sanitized = deepcopy(entry)
    process_name = sanitized.get("process_name")
    if isinstance(process_name, str):
        sanitized["process_name"] = _sanitize_to_english(process_name)

    for key in ("matched_flows", "unmatched_flows"):
        value = sanitized.get(key)
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                for field in ("base_name", "general_comment", "process_name"):
                    if field in item and isinstance(item[field], str):
                        sanitizer = _sanitize_comment_text if "comment" in field else _sanitize_to_english
                        item[field] = sanitizer(item[field])

    origin = sanitized.get("origin_exchanges")
    if isinstance(origin, dict):
        sanitized_origin: dict[str, list[dict[str, Any]]] = {}
        for name, exchanges in origin.items():
            sanitized_name = _sanitize_to_english(name) if isinstance(name, str) else name
            if isinstance(exchanges, list):
                sanitized_origin[sanitized_name] = [_sanitize_exchange_language(exchange) for exchange in exchanges if isinstance(exchange, dict)]
        sanitized["origin_exchanges"] = sanitized_origin
    return sanitized


def _parse_flowsearch_hints(comment: Any) -> dict[str, list[str]]:
    text = _extract_text(comment)
    if not text:
        return {}
    prefix = "FlowSearch hints:"
    if text.startswith(prefix):
        text = text[len(prefix) :].strip()
    segments = [segment.strip() for segment in text.split("|") if segment.strip()]
    hints: dict[str, list[str]] = {}
    for segment in segments:
        key, _, value = segment.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if not value or value == "NA":
            hints[key] = []
            continue
        entries = [item.strip() for item in value.split(";") if item.strip()]
        hints[key] = entries or [value]
    return hints


def _infer_flow_type(exchange: dict[str, Any], hints: dict[str, list[str]]) -> str:
    direction = _extract_text(exchange.get("exchangeDirection")).lower()
    name = _extract_text(exchange.get("exchangeName")).lower()
    combined = " ".join(
        [
            name,
            _extract_text(exchange.get("generalComment")).lower(),
            " ".join(hints.get("usage_context", [])).lower(),
            " ".join(hints.get("state_purity", [])).lower(),
        ]
    )
    if any(keyword in combined for keyword in ("emission", "to air", "to water", "wastewater")):
        return "Elementary flow"
    if "waste" in combined or "slag" in combined:
        return "Waste flow"
    if direction == "input" and ("air" in name or "water" in name):
        return "Elementary flow"
    return "Product flow"


def _clone_entries(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in entries]


def _extract_candidate(exchange: dict[str, Any]) -> dict[str, Any]:
    matching = exchange.get("matchingDetail")
    if isinstance(matching, dict):
        candidate = matching.get("selectedCandidate")
        if isinstance(candidate, dict):
            return candidate
    return {}


def _normalise_product_classes(classes: Any) -> list[dict[str, str]]:
    normalised: list[dict[str, str]] = []
    if not isinstance(classes, list):
        return normalised
    for entry in classes:
        if not isinstance(entry, dict):
            continue
        class_id = entry.get("@classId") or entry.get("classId")
        if not class_id:
            continue
        text_value = entry.get("#text") or entry.get("text") or ""
        if isinstance(text_value, dict):
            text_value = text_value.get("#text", "")
        level_value = entry.get("@level")
        if level_value is None:
            level_value = len(normalised)
        normalised.append(
            {
                "@level": str(level_value),
                "@classId": str(class_id),
                "#text": str(text_value),
            }
        )
    return normalised


def _build_product_classification(candidate: dict[str, Any]) -> dict[str, Any]:
    classes = _normalise_product_classes(candidate.get("classification"))
    if not classes:
        classes = _clone_entries(PRODUCT_FALLBACK_CLASSIFICATION)
    return {"common:classification": {"common:class": classes}}


def _build_waste_classification(candidate: dict[str, Any]) -> dict[str, Any]:
    classes = _normalise_product_classes(candidate.get("classification"))
    if not classes:
        classes = _clone_entries(WASTE_FALLBACK_CLASSIFICATION)
    return {"common:classification": {"common:class": classes}}


def _infer_elementary_categories(exchange: dict[str, Any], hints: dict[str, list[str]]) -> list[dict[str, Any]]:
    parts = [
        _extract_text(exchange.get("location")).lower(),
        " ".join(hints.get("usage_context") or []).lower(),
        _extract_text(exchange.get("generalComment")).lower(),
        _extract_text(exchange.get("exchangeName")).lower(),
    ]
    combined = " ".join(filter(None, parts))
    if any(token in combined for token in ("resource", "extraction", "raw material")):
        return _clone_entries(ELEMENTARY_CATEGORY_RESOURCES)
    if "water" in combined or "wastewater" in combined:
        return _clone_entries(ELEMENTARY_CATEGORY_WATER)
    if "soil" in combined or "ground" in combined or "land" in combined:
        return _clone_entries(ELEMENTARY_CATEGORY_SOIL)
    if "air" in combined or "atmosphere" in combined:
        return _clone_entries(ELEMENTARY_CATEGORY_AIR)
    return _clone_entries(ELEMENTARY_CATEGORY_OTHER)


def _build_elementary_classification(exchange: dict[str, Any], hints: dict[str, list[str]]) -> dict[str, Any]:
    categories = _infer_elementary_categories(exchange, hints)
    return {"common:elementaryFlowCategorization": {"common:category": categories}}


def _source_classification_entry(class_id: str, label: str) -> dict[str, Any]:
    return {
        "common:classification": {
            "common:class": {
                "@level": "0",
                "@classId": class_id,
                "#text": label,
            }
        }
    }


def _build_source_classification(reference_node: dict[str, Any], uuid_value: str, format_source_uuid: str) -> dict[str, Any]:
    existing = reference_node.get("classificationInformation")
    if isinstance(existing, dict) and existing.get("common:classification"):
        return existing

    ref_uuid = str(reference_node.get("@refObjectId") or "").lower()
    short_desc = _extract_text(reference_node.get("common:shortDescription")).lower()
    uri = str(reference_node.get("@uri") or "").lower()

    def _match_any(haystack: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in haystack for keyword in keywords if keyword)

    class_id, label = SOURCE_CLASSIFICATIONS["other source types"]
    if uuid_value.lower() == format_source_uuid.lower() or _match_any(short_desc, ("format", "schema")):
        class_id, label = SOURCE_CLASSIFICATIONS["data set formats"]
    elif ref_uuid == DEFAULT_FORMAT_SOURCE_UUID:
        class_id, label = SOURCE_CLASSIFICATIONS["data set formats"]
    elif _match_any(short_desc, ("ilcd data network", "compliance", "conformity", "certification")) or _match_any(uri, ("compliance", "conformity")):
        class_id, label = SOURCE_CLASSIFICATIONS["compliance systems"]
    elif _match_any(short_desc, ("database", "data bank", "dataset")) or _match_any(uri, ("database",)):
        class_id, label = SOURCE_CLASSIFICATIONS["databases"]
    elif _match_any(short_desc, ("nace", "isic", "cpc", "statistical", "classification")) or _match_any(uri, ("classification",)):
        class_id, label = SOURCE_CLASSIFICATIONS["statistical classifications"]
    elif _match_any(short_desc, ("image", "photo", "figure", "diagram")) or uri.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".bmp")):
        class_id, label = SOURCE_CLASSIFICATIONS["images"]
    elif _match_any(
        short_desc,
        (
            "publication",
            "report",
            "article",
            "paper",
            "journal",
            "communication",
            "study",
            "thesis",
            "book",
        ),
    ):
        class_id, label = SOURCE_CLASSIFICATIONS["publications and communications"]

    return _source_classification_entry(class_id, label)


def flow_compliance_declarations() -> dict[str, Any]:
    """Return the default compliance declaration block for generated datasets.

    The compliance system reference reuses the shared Tiangong ILCD Entry-level UUID.
    Stage 3 does not export a local source stub for this reference; downstream systems
    resolve it during publication using the stored UUID.
    """

    compliance_uuid = ILCD_ENTRY_LEVEL_REFERENCE_ID
    compliance_version = "20.20.002"
    return {
        "compliance": {
            "common:referenceToComplianceSystem": {
                "@refObjectId": compliance_uuid,
                "@type": "source data set",
                "@uri": build_local_dataset_uri("source", compliance_uuid, compliance_version),
                "@version": compliance_version,
                "common:shortDescription": _language_entry("ILCD Data Network - Entry-level"),
            },
            "common:approvalOfOverallCompliance": "Fully compliant",
        }
    }


def _data_entry_reference() -> dict[str, Any]:
    return _ownership_reference()


def _ownership_reference() -> dict[str, Any]:
    ref_object_id = "f4b4c314-8c4c-4c83-968f-5b3c7724f6a8"
    version = "01.00.000"
    return {
        "@refObjectId": ref_object_id,
        "@type": "contact data set",
        "@uri": build_local_dataset_uri("contact data set", ref_object_id, version),
        "@version": version,
        "common:shortDescription": [
            _language_entry("Tiangong LCA Data Working Group", "en"),
            _language_entry("天工LCA数据团队", "zh"),
        ],
    }


def _permanent_dataset_uri(dataset_kind: str, uuid_value: str, version: str) -> str:
    return build_portal_uri(dataset_kind, uuid_value, version)


def _build_flow_dataset(
    exchange: dict[str, Any],
    process_name: str,
    timestamp: str,
    format_source_uuid: str,
    comment_llm: LanguageModelProtocol | None,
) -> tuple[str, dict[str, Any]] | None:
    exchange = _sanitize_exchange_language(exchange)
    ref = exchange.get("referenceToFlowDataSet") or {}
    uuid_value = ref.get("@refObjectId") or str(uuid4())
    name = _extract_text(exchange.get("exchangeName")) or _extract_text(ref.get("common:shortDescription"))
    name = _sanitize_to_english(name)
    if not name:
        name = "Unnamed flow"
    hints = _parse_flowsearch_hints(exchange.get("generalComment"))
    flow_type = _infer_flow_type(exchange, hints)
    candidate = _extract_candidate(exchange)
    if flow_type == "Elementary flow":
        LOGGER.info(
            "artifact_builder.skip_elementary_flow",
            process=process_name,
            exchange=name,
            reason="Placeholder flows are only emitted for product flows.",
        )
        return None
    if flow_type == "Waste flow":
        classification = _build_waste_classification(candidate)
    else:
        classification = _build_product_classification(candidate)
    class_entries = _classification_entries(classification)
    if not class_entries:
        class_entries = _classification_entries(_build_product_classification(candidate))

    treatment_candidates = hints.get("state_purity") or hints.get("source_or_pathway") or hints.get("abbreviation") or [name]
    treatment_text = _unique_join(treatment_candidates)
    treatment_text = _sanitize_to_english(treatment_text)

    mix_candidates = hints.get("mix_location") or hints.get("usage_context") or hints.get("source_or_pathway") or []
    location_hint = _extract_text(exchange.get("location"))
    if location_hint:
        mix_candidates = list(mix_candidates) + [location_hint]
    if not mix_candidates:
        mix_candidates = ["Unspecified mix"]
    mix_text = _unique_join(mix_candidates)
    mix_text = _sanitize_to_english(mix_text)

    comment_entries = _generate_flow_comment_entries(
        base_name=name,
        treatment=treatment_text,
        mix=mix_text,
        flow_properties=hints.get("flow_properties"),
        en_synonyms=hints.get("en_synonyms"),
        zh_synonyms=hints.get("zh_synonyms"),
        fallback_comment=exchange.get("generalComment"),
        process_name=process_name,
        llm=comment_llm,
    )
    zh_candidates = hints.get("zh_synonyms") or []
    base_name_zh = _extract_text(zh_candidates[0]) if zh_candidates else name
    service = ProductFlowCreationService()
    request = ProductFlowCreateRequest(
        class_id=str(class_entries[-1].get("@classId") or ""),
        classification=class_entries,
        base_name_en=name,
        base_name_zh=base_name_zh,
        treatment_en=treatment_text or name,
        mix_en=mix_text,
        comment_en=_extract_lang_text(comment_entries, "en") or _extract_text(exchange.get("generalComment")) or f"Auto-generated for {name}",
        comment_zh=_extract_lang_text(comment_entries, "zh") or None,
        synonyms_en=[value for value in (hints.get("en_synonyms") or []) if _extract_text(value)],
        synonyms_zh=[value for value in (hints.get("zh_synonyms") or []) if _extract_text(value)],
        flow_type=flow_type,
        flow_uuid=uuid_value,
        version=DEFAULT_DATA_SET_VERSION,
        timestamp=timestamp,
        mean_value="1",
    )

    try:
        built = service.build(request, allow_validation_fallback=True)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "artifact_builder.flow_build_failed",
            process=process_name,
            exchange=name,
            error=str(exc),
        )
        return None

    flow_dataset = dict(built.dataset)
    publication = flow_dataset.get("administrativeInformation", {}).get("publicationAndOwnership")
    if isinstance(publication, dict):
        publication["common:permanentDataSetURI"] = _permanent_dataset_uri("flow", built.flow_uuid, built.version)
    return built.flow_uuid, {"flowDataSet": flow_dataset}


def _collect_unmatched_exchanges(
    alignment: Iterable[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    collected: dict[str, tuple[str, dict[str, Any]]] = {}
    for entry in alignment:
        process_name = entry.get("process_name") or "Unnamed process"
        origin = entry.get("origin_exchanges") or {}
        if not isinstance(origin, dict):
            continue
        for exchanges in origin.values():
            if isinstance(exchanges, dict):
                exchanges_iter = [exchanges]
            else:
                exchanges_iter = list(exchanges or [])
            for exchange in exchanges_iter:
                if not isinstance(exchange, dict):
                    continue
                ref = exchange.get("referenceToFlowDataSet")
                if not isinstance(ref, dict):
                    continue
                if not ref.get("unmatched:placeholder"):
                    continue
                uuid_value = ref.get("@refObjectId")
                if uuid_value and uuid_value not in collected:
                    collected[uuid_value] = (process_name, exchange)
    return list(collected.values())


def _collect_source_references(process_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            node_type = node.get("@type")
            ref_id = node.get("@refObjectId")
            if node_type == "source data set" and ref_id:
                uri = str(node.get("@uri") or "")
                if uri.startswith("../"):
                    references.setdefault(ref_id, node)
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(process_payload)
    return references


def _build_source_stub(
    uuid_value: str,
    reference_node: dict[str, Any],
    timestamp: str,
    format_source_uuid: str,
    *,
    include_format_reference: bool = True,
) -> dict[str, Any]:
    short_desc = reference_node.get("common:shortDescription")
    description_entries = _normalise_language(short_desc or "Source reference")
    citation_text = description_entries[0]["#text"] if description_entries else "Source reference"
    classification = _build_source_classification(reference_node, uuid_value, format_source_uuid)
    dataset_version = DEFAULT_DATA_SET_VERSION
    dataset = {
        "sourceDataSet": {
            "@xmlns": "http://lca.jrc.it/ILCD/Source",
            "@xmlns:common": "http://lca.jrc.it/ILCD/Common",
            "@xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "@version": "1.1",
            "@xsi:schemaLocation": "http://lca.jrc.it/ILCD/Source ../../schemas/ILCD_SourceDataSet.xsd",
            "sourceInformation": {
                "dataSetInformation": {
                    "common:UUID": uuid_value,
                    "common:shortName": description_entries,
                    "classificationInformation": classification,
                    "sourceCitation": citation_text,
                    "publicationType": "Other unpublished and grey literature",
                    "sourceDescriptionOrComment": description_entries,
                    "referenceToContact": _ownership_reference(),
                }
            },
            "administrativeInformation": {
                "dataEntryBy": {
                    "common:timeStamp": timestamp,
                },
                "publicationAndOwnership": {
                    "common:dataSetVersion": dataset_version,
                    "common:permanentDataSetURI": _permanent_dataset_uri("source", uuid_value, dataset_version),
                    "common:referenceToOwnershipOfDataSet": _ownership_reference(),
                },
            },
        }
    }
    # Always attach the dataset-format reference so downstream validators receive
    # a complete administrative block, regardless of whether the stub was auto-created.
    dataset["sourceDataSet"]["administrativeInformation"]["dataEntryBy"]["common:referenceToDataSetFormat"] = _format_reference_block(format_source_uuid)
    dataset["sourceDataSet"]["administrativeInformation"]["dataEntryBy"]["common:referenceToPersonOrEntityEnteringTheData"] = _data_entry_reference()
    return dataset


def _ensure_directories(root: Path) -> None:
    for name in ("processes", "flows", "sources"):
        (root / name).mkdir(parents=True, exist_ok=True)


def _build_source_reference(uuid_value: str, title: str) -> dict[str, Any]:
    dataset_version = DEFAULT_DATA_SET_VERSION
    return {
        "@type": "source data set",
        "@refObjectId": uuid_value,
        "@uri": build_local_dataset_uri("source", uuid_value, dataset_version),
        "@version": dataset_version,
        "common:shortDescription": [_language_entry(title)],
    }


def _attach_primary_source(ilcd_dataset: dict[str, Any], source_uuid: str, source_title: str) -> None:
    admin = ilcd_dataset.setdefault("administrativeInformation", {})
    data_entry = admin.get("dataEntryBy")
    if not isinstance(data_entry, dict):
        data_entry = {}
        admin["dataEntryBy"] = data_entry
    data_entry.pop("common:referenceToDataSetFormat", None)

    modelling = ilcd_dataset.setdefault("modellingAndValidation", {})
    data_sources = modelling.setdefault("dataSourcesTreatmentAndRepresentativeness", {})
    reference_entry = _build_source_reference(source_uuid, source_title)
    data_sources["referenceToDataSource"] = [reference_entry]
    data_entry["common:referenceToDataSetFormat"] = _dataset_format_reference()


def _run_validation(artifact_root: Path) -> list[dict[str, Any]]:
    service = TidasValidationService()
    try:
        findings = service.validate_directory(artifact_root)
    finally:
        service.close()

    for finding in findings:
        if finding.severity != "info":
            print(finding.message)
    return [asdict(finding) for finding in findings]


def _dump_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


LOGGER = get_logger(__name__)
