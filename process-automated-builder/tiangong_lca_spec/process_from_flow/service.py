"""High-level facade and LangGraph assembly for building processes from a reference flow."""

from __future__ import annotations

import copy
import csv
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, TypedDict
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5
from zipfile import ZipFile

from langgraph.graph import END, StateGraph
from tidas_sdk import create_process, create_source
from tidas_sdk.core.multilang import MultiLangList
from tidas_sdk.entities.utils import default_timestamp
from tidas_sdk.generated.tidas_data_types import GlobalReferenceTypeVariant0, GlobalReferenceTypeVariant1Item
from tidas_sdk.generated.tidas_processes import (
    CommonClassItemOption0,
    ComplianceDeclarationsComplianceOption0,
    DataSetInformationClassificationInformationCommonClassification,
    ExchangeItemReferencesToDataSource,
    ExchangesExchangeItem,
    ModellingAndValidationValidationReview,
    ProcessDataSetAdministrativeInformationCommonCommissionerAndGoal,
    ProcessDataSetAdministrativeInformationDataEntryBy,
    ProcessDataSetAdministrativeInformationPublicationAndOwnership,
    ProcessDataSetModellingAndValidationComplianceDeclarations,
    ProcessDataSetModellingAndValidationDataSourcesTreatmentAndRepresentativeness,
    ProcessDataSetModellingAndValidationLCIMethodAndAllocation,
    ProcessDataSetModellingAndValidationValidation,
    ProcessDataSetProcessInformationDataSetInformation,
    ProcessDataSetProcessInformationGeography,
    ProcessDataSetProcessInformationQuantitativeReference,
    ProcessDataSetProcessInformationTechnology,
    ProcessDataSetProcessInformationTime,
    Processes,
    ProcessesProcessDataSet,
    ProcessesProcessDataSetAdministrativeInformation,
    ProcessesProcessDataSetExchanges,
    ProcessesProcessDataSetModellingAndValidation,
    ProcessesProcessDataSetProcessInformation,
    ProcessInformationDataSetInformationClassificationInformation,
    ProcessInformationDataSetInformationName,
    ProcessInformationGeographyLocationOfOperationSupplyOrProduction,
)
from tidas_sdk.generated.tidas_sources import (
    ClassificationInformationCommonClassificationCommonClass as SourceClassificationCommonClass,
)
from tidas_sdk.generated.tidas_sources import (
    DataSetInformationClassificationInformationCommonClassification as SourceClassificationInformationCommonClassification,
)
from tidas_sdk.generated.tidas_sources import (
    SourceDataSetAdministrativeInformationDataEntryBy,
    SourceDataSetAdministrativeInformationPublicationAndOwnership,
    SourceDataSetSourceInformationDataSetInformation,
    SourceInformationDataSetInformationClassificationInformation,
    SourceInformationDataSetInformationReferenceToDigitalFile,
    Sources,
    SourcesSourceDataSet,
    SourcesSourceDataSetAdministrativeInformation,
    SourcesSourceDataSetSourceInformation,
)

from tiangong_lca_spec.core.config import Settings, get_settings
from tiangong_lca_spec.core.constants import (
    ILCD_FORMAT_SOURCE_SHORT_DESCRIPTION,
    ILCD_FORMAT_SOURCE_URI,
    ILCD_FORMAT_SOURCE_UUID,
    ILCD_FORMAT_SOURCE_VERSION,
)
from tiangong_lca_spec.core.json_utils import parse_json_response
from tiangong_lca_spec.core.logging import get_logger
from tiangong_lca_spec.core.mcp_client import MCPToolClient
from tiangong_lca_spec.core.models import FlowCandidate, FlowQuery
from tiangong_lca_spec.core.uris import build_local_dataset_uri, build_portal_uri
from tiangong_lca_spec.flow_alignment.selector import (
    CandidateSelector,
    LanguageModelProtocol,
    LLMCandidateSelector,
    NoFallbackCandidateSelector,
    SimilarityCandidateSelector,
)
from tiangong_lca_spec.flow_search import search_flows
from tiangong_lca_spec.flow_search.client import FlowSearchClient
from tiangong_lca_spec.flow_search.validators import hydrate_candidate
from tiangong_lca_spec.location import extract_location_response, get_location_catalog
from tiangong_lca_spec.process_extraction.extractors import LocationNormalizer, ProcessClassifier
from tiangong_lca_spec.process_extraction.tidas_mapping import (
    COMPLIANCE_DEFAULT_PREFERENCES,
    ILCD_ENTRY_LEVEL_REFERENCE_ID,
    ILCD_ENTRY_LEVEL_REFERENCE_VERSION,
)
from tiangong_lca_spec.publishing.crud import DatabaseCrudClient
from tiangong_lca_spec.utils.translate import Translator

from .prompts import (
    DATA_CUTOFF_COMPLETENESS_PROMPT,
    DENSITY_ESTIMATE_PROMPT,
    EXCHANGE_VALUE_PROMPT,
    EXCHANGES_PROMPT,
    INDUSTRY_AVERAGE_PROMPT,
    INTENDED_APPLICATIONS_PROMPT,
    PLACEHOLDER_QUERY_BUILDER_PROMPT,
    PLACEHOLDER_UUID_SELECTOR_PROMPT,
    PROCESS_SPLIT_PROMPT,
    REFERENCE_CLUSTER_PROMPT,
    TECH_DESCRIPTION_PROMPT,
)

LOGGER = get_logger(__name__)
SCIENTIFIC_REFERENCE_TOP_K = 10
SCIENTIFIC_REFERENCE_FULLTEXT_TOP_K = 1
SCIENTIFIC_REFERENCE_FULLTEXT_EXT_K = 200
REFERENCE_CLUSTER_MAX_CHARS = 1200
REFERENCE_CLUSTER_MAX_RECORDS = 2
INDUSTRY_AVERAGE_TOP_K = 5
REFERENCE_COUNTRY_PREFERENCE = "China"
REFERENCE_COUNTRY_ALIASES = ("China", "Chinese", "中国")
_LOCATION_DATA_DIR = Path(__file__).resolve().parents[3] / "input_data" / "location"
_LOCATION_EN_PATH = _LOCATION_DATA_DIR / "tidas_locations.json"
_LOCATION_ZH_PATH = _LOCATION_DATA_DIR / "tidas_locations_zh.json"
_LOCATION_SPLIT_PATTERN = re.compile(r"[;,/|>]+")

REFERENCE_SEARCH_KEY = "step_1a_reference_search"
REFERENCE_FULLTEXT_KEY = "step_1b_reference_fulltext"
REFERENCE_CLUSTERS_KEY = "step_1c_reference_clusters"
INDUSTRY_AVERAGE_KEY = "industry_average"

STOP_RULE_PROCESS_COVERAGE = 0.5
STOP_RULE_EXCHANGE_COVERAGE = 0.6
STOP_RULE_MIN_DELTA = 0.1

SI_SNIPPET_MAX_CHARS = 2000
SI_SNIPPET_MAX_FILES = 3
SI_SNIPPET_MAX_BLOCKS = 40
SI_TABLE_MAX_ROWS = 80
SI_TABLE_MAX_COLS = 12
SI_DOCX_MAX_TABLES = 6
SI_DOCX_MAX_PARAGRAPHS = 40
SI_XLSX_MAX_SHEETS = 3

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FLOWPROPERTY_DIR = _REPO_ROOT / "input_data" / "flowproperties"
_UNIT_GROUP_DIR = _REPO_ROOT / "input_data" / "units"
_PFF_RUNTIME_STATE_PATH_ENV = "TIANGONG_PFF_STATE_PATH"
_PFF_RUNTIME_RUN_ID_ENV = "TIANGONG_PFF_RUN_ID"
_PFF_RUNTIME_ARTIFACTS_ROOT = Path("artifacts/process_from_flow")


@dataclass(frozen=True, slots=True)
class FlowPropertyInfo:
    flow_property_id: str
    name: str
    unit_group_id: str | None


@dataclass(frozen=True, slots=True)
class UnitGroupInfo:
    unit_group_id: str
    name: str
    reference_unit: str
    units: dict[str, float]


@dataclass(frozen=True, slots=True)
class FlowReferenceInfo:
    flow_property_id: str | None
    unit_group: UnitGroupInfo | None


_FLOW_PROPERTY_CACHE: dict[str, FlowPropertyInfo] = {}
_UNIT_GROUP_CACHE: dict[str, UnitGroupInfo] = {}


FLOW_QUERY_REWRITE_PROMPT = (
    "You rewrite LCA exchange names for flow search.\n"
    "Given one exchange name and constraints, output concise alternative product/exchange names that may exist in a flow database.\n"
    "Rules:\n"
    "1. Keep the same semantic object; do NOT broaden scope.\n"
    "2. Preserve constraints (flow_type, direction, unit, compartment) conceptually.\n"
    "3. Prefer canonical commodity names and common aliases.\n"
    "4. Remove end-use qualifiers like 'for pigs' when they are not intrinsic to the commodity.\n"
    "Return strict JSON object with key `query_variants` as an array of strings (max 5), no extra text."
)


def _search_scientific_references(
    query: str,
    *,
    mcp_client: MCPToolClient | None = None,
    top_k: int = SCIENTIFIC_REFERENCE_TOP_K,
    filters: dict[str, Any] | None = None,
    ext_k: int | None = None,
    limit: int | None = None,
    keep_all: bool = False,
    country_preference: str | None = None,
    country_aliases: tuple[str, ...] | None = None,
    fallback_to_global: bool = True,
) -> list[dict[str, Any]]:
    """Search scientific literature using tiangong_kb_remote search_Sci_Tool.

    Args:
        query: Search query string describing the technical context
        mcp_client: Optional MCP client instance; creates new one if None
        top_k: Maximum number of references to return
        filters: Optional filter payload passed to the search tool
        ext_k: Optional extK parameter for retrieving extended content
        limit: Optional max results to keep (defaults to top_k)
        keep_all: Whether to keep all returned records without trimming
        country_preference: Optional country hint for a first-pass query
        country_aliases: Strings that count as a country hit in results
        fallback_to_global: Whether to retry without country hint if no hits

    Returns:
        List of reference dictionaries with keys like 'content', 'metadata', 'score'
    """
    if not query or not query.strip():
        return []

    def _has_country_hit(records: list[dict[str, Any]], aliases: tuple[str, ...]) -> bool:
        if not records:
            return False
        lowered_aliases = [alias.casefold() for alias in aliases if alias]
        for record in records:
            if not isinstance(record, dict):
                continue
            for key in ("content", "source", "title", "snippet", "abstract"):
                value = record.get(key)
                if not isinstance(value, str):
                    continue
                text = value.casefold()
                if any(alias in text for alias in lowered_aliases):
                    return True
        return False

    should_close_client = False
    client = mcp_client
    if client is None:
        client = MCPToolClient()
        should_close_client = True

    def _run_query(search_query: str) -> list[dict[str, Any]]:
        try:
            arguments: dict[str, Any] = {
                "query": search_query.strip(),
                "topK": top_k,
            }
            if filters:
                arguments["filter"] = filters
            if ext_k is not None:
                arguments["extK"] = ext_k

            result = client.invoke_json_tool(
                server_name="TianGong_KB_Remote",
                tool_name="Search_Sci_Tool",
                arguments=arguments,
            )

            if not result:
                return []

            references: list[dict[str, Any]] = []
            if isinstance(result, dict):
                records = result.get("records") or result.get("results") or result.get("data") or []
                if isinstance(records, list):
                    references = [item for item in records if isinstance(item, dict)]
            elif isinstance(result, list):
                references = [item for item in result if isinstance(item, dict)]

            LOGGER.info(
                "process_from_flow.search_references",
                query_preview=search_query[:100],
                count=len(references),
            )
            trimmed = references
            max_keep = top_k if limit is None else limit
            if not keep_all and max_keep is not None:
                trimmed = references[:max_keep]
            return [
                {
                    **item,
                    "no": idx,
                }
                for idx, item in enumerate(trimmed, start=1)
            ]
        except Exception as exc:
            LOGGER.warning(
                "process_from_flow.search_references_failed",
                query_preview=search_query[:100],
                error=str(exc),
            )
            return []

    try:
        if country_preference and not (filters and "doi" in filters):
            aliases = country_aliases or (country_preference,)
            preferred_queries: list[str] = []
            for token in (country_preference, *aliases):
                token_text = str(token).strip()
                if not token_text:
                    continue
                preferred = f"{query} {token_text}"
                if preferred not in preferred_queries:
                    preferred_queries.append(preferred)
            fallback_records: list[dict[str, Any]] = []
            for preferred in preferred_queries:
                records = _run_query(preferred)
                if records:
                    if not fallback_records:
                        fallback_records = records
                    if _has_country_hit(records, aliases):
                        return records
            if fallback_to_global:
                global_records = _run_query(query)
                if global_records:
                    return global_records
            return fallback_records
        return _run_query(query)
    finally:
        if should_close_client and client:
            client.close()


def _format_references_for_prompt(references: list[dict[str, Any]]) -> str:
    """Format scientific references into a readable string for LLM prompts.

    Args:
        references: List of reference dictionaries from search_Sci_Tool

    Returns:
        Formatted string with numbered references
    """
    if not references:
        return ""

    lines = ["Scientific References:"]
    for idx, ref in enumerate(references, start=1):
        # Extract content and metadata
        content = ref.get("content") or ref.get("text") or ref.get("segment", {}).get("content") or ""
        metadata = ref.get("metadata") or {}
        source = ref.get("source") or (metadata.get("source") if isinstance(metadata, dict) else None)
        doi = _extract_reference_doi(ref)

        # Build reference entry
        entry_parts = [f"[{idx}]"]
        if isinstance(metadata, dict):
            meta_str = metadata.get("meta") or ""
            if meta_str:
                entry_parts.append(f"Source: {meta_str}")
        if isinstance(source, str) and source.strip():
            entry_parts.append(f"Source: {_compact_text(source.strip(), limit=280)}")
        if isinstance(doi, str) and doi.strip():
            entry_parts.append(f"DOI: {doi.strip()}")

        if content:
            content_text = content if isinstance(content, str) else str(content)
            content_text = content_text.strip()
            if content_text:
                entry_parts.append(f"Content: {content_text}")

        lines.append(" ".join(entry_parts))
        lines.append("")  # Empty line between references

    return "\n".join(lines)


def _reference_identity(reference: dict[str, Any]) -> str:
    doi = _extract_reference_doi(reference)
    if doi:
        return f"doi:{doi.lower()}"
    source = str(reference.get("source") or "").strip()
    if source:
        return f"source:{source.lower()}"
    content = str(reference.get("content") or reference.get("text") or "").strip()
    if content:
        return f"content:{_compact_text(content, limit=80).lower()}"
    return f"ref:{hash(str(reference))}"


def _merge_reference_records(existing: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = list(existing)
    seen = {_reference_identity(item) for item in existing if isinstance(item, dict)}
    for item in new:
        if not isinstance(item, dict):
            continue
        ident = _reference_identity(item)
        if ident in seen:
            continue
        merged.append(item)
        seen.add(ident)
    return merged


def _extract_reference_evidence(references: list[dict[str, Any]]) -> list[str]:
    evidence: list[str] = []
    for ref in references:
        if not isinstance(ref, dict):
            continue
        doi = _extract_reference_doi(ref)
        if doi:
            evidence.append(f"DOI {doi}")
            continue
        source_text = str(ref.get("source") or ref.get("content") or ref.get("text") or "").strip()
        url = _extract_reference_url(source_text)
        if url:
            evidence.append(url)
            continue
        if source_text:
            evidence.append(_compact_text(source_text, limit=160))
    return evidence


def _first_nonempty(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_location_key(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum())


def _clean_location_hint(value: str | None) -> str:
    if not value:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if any(token in lowered for token in ("unspecified", "unknown", "not specified", "n/a", "mix/location")):
        return ""
    if any(token in text for token in ("未指定", "未知")):
        return ""
    return text


@lru_cache(maxsize=1)
def _load_location_maps() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    code_to_en: dict[str, str] = {}
    code_to_zh: dict[str, str] = {}
    name_to_code: dict[str, str] = {}
    for path, target in ((_LOCATION_EN_PATH, code_to_en), (_LOCATION_ZH_PATH, code_to_zh)):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("process_from_flow.location_mapping_load_failed", path=str(path), error=str(exc))
            continue
        locations = data.get("ILCDLocations", {}).get("location", [])
        if isinstance(locations, dict):
            locations = [locations]
        if not isinstance(locations, list):
            continue
        for item in locations:
            if not isinstance(item, dict):
                continue
            code = str(item.get("@value") or "").strip()
            name = str(item.get("#text") or "").strip()
            if not code:
                continue
            target[code] = name
            code_key = _normalize_location_key(code)
            if code_key and code_key not in name_to_code:
                name_to_code[code_key] = code
            if name:
                name_key = _normalize_location_key(name)
                if name_key and name_key not in name_to_code:
                    name_to_code[name_key] = code
    for alias, code in {"global": "GLO", "world": "GLO", "china": "CN", "cn": "CN"}.items():
        alias_key = _normalize_location_key(alias)
        if alias_key and alias_key not in name_to_code:
            name_to_code[alias_key] = code
    return code_to_en, code_to_zh, name_to_code


def _lookup_location_code(value: str | None, name_to_code: dict[str, str]) -> str | None:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    key = _normalize_location_key(raw)
    if key:
        match = name_to_code.get(key)
        if match:
            return match
    for part in _LOCATION_SPLIT_PATTERN.split(raw):
        chunk = part.strip()
        if not chunk:
            continue
        key = _normalize_location_key(chunk)
        if not key:
            continue
        match = name_to_code.get(key)
        if match:
            return match
    return None


def _normalize_location_code(value: str | None, valid_codes: set[str]) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9-]+", "-", str(value).strip().upper()).strip("-")
    if not cleaned:
        return None
    if valid_codes:
        return cleaned if cleaned in valid_codes else None
    return cleaned


def _resolve_location_code_rule_based(candidates: list[str], mix_location: str | None) -> str | None:
    code_to_en, code_to_zh, name_to_code = _load_location_maps()
    valid_codes = set(code_to_en) | set(code_to_zh)
    for candidate in candidates:
        code = _normalize_location_code(candidate, valid_codes)
        if code:
            return code
    for candidate in candidates:
        code = _lookup_location_code(candidate, name_to_code)
        if code:
            return code
    mix_hint = _clean_location_hint(mix_location)
    if mix_hint:
        code = _lookup_location_code(mix_hint, name_to_code)
        if code:
            return code
    return None


def _build_location_candidates(raw_hint: str) -> list[dict[str, str]]:
    if not raw_hint:
        return []
    try:
        catalog = get_location_catalog()
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("process_from_flow.location_catalog_load_failed", error=str(exc))
        return []
    try:
        return catalog.build_candidate_list(raw_hint, depth=2, limit=80)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("process_from_flow.location_candidate_build_failed", hint=raw_hint, error=str(exc))
        return []


def _resolve_location_code_with_llm(
    *,
    llm: LanguageModelProtocol,
    process_info: dict[str, Any],
    raw_hint: str,
) -> str | None:
    if not raw_hint:
        return None
    candidates = _build_location_candidates(raw_hint)
    try:
        normalizer = LocationNormalizer(llm)
        response = normalizer.run(
            process_info,
            hint=raw_hint,
            candidates=candidates or None,
        )
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("process_from_flow.location_llm_failed", error=str(exc))
        return None
    code, payload = extract_location_response(response)
    code_to_en, code_to_zh, name_to_code = _load_location_maps()
    valid_codes = set(code_to_en) | set(code_to_zh)
    normalized = _normalize_location_code(code, valid_codes)
    if normalized:
        return normalized
    if isinstance(payload, dict):
        hint_text = _first_nonempty(payload.get("description"), payload.get("subLocation"))
        if hint_text:
            resolved = _lookup_location_code(hint_text, name_to_code)
            if resolved:
                return resolved
    return None


def _resolve_location_code(
    *,
    candidates: list[str],
    mix_location: str | None,
    llm: LanguageModelProtocol | None,
    process_info: dict[str, Any],
) -> str:
    rule_based = _resolve_location_code_rule_based(candidates, mix_location)
    if rule_based:
        return rule_based
    raw_hint = _clean_location_hint(_first_nonempty(*candidates, mix_location))
    if llm is not None and raw_hint:
        llm_code = _resolve_location_code_with_llm(llm=llm, process_info=process_info, raw_hint=raw_hint)
        if llm_code:
            return llm_code
    return "GLO"


def _compact_text(value: str, *, limit: int = 160) -> str:
    text = str(value).strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    trimmed = text[:limit].rsplit(" ", 1)[0]
    return trimmed or text[:limit]


def _extract_rewrite_variant_strings(raw: Any) -> list[str]:
    values: list[str] = []

    def _append(value: Any) -> None:
        if isinstance(value, str):
            text = value.strip()
            if text:
                values.append(text)
            return
        if isinstance(value, dict):
            for key in ("query", "exchange_name", "name", "text", "value"):
                item = value.get(key)
                if isinstance(item, str) and item.strip():
                    values.append(item.strip())
            return

    if isinstance(raw, list):
        for item in raw:
            _append(item)
        return values
    _append(raw)
    return values


def _generate_flow_query_rewrites_with_llm(
    *,
    llm: LanguageModelProtocol | None,
    exchange_name: str,
    comment: str | None,
    flow_type: str | None,
    direction: str | None,
    unit: str | None,
    expected_compartment: str | None,
    search_hints: list[str] | None,
    max_variants: int = 5,
) -> list[str]:
    if llm is None:
        return []
    if not exchange_name.strip():
        return []

    payload = {
        "prompt": FLOW_QUERY_REWRITE_PROMPT,
        "context": {
            "exchange_name": exchange_name,
            "general_comment": comment,
            "flow_type": flow_type,
            "direction": direction,
            "unit": unit,
            "expected_compartment": expected_compartment,
            "search_hints": search_hints or [],
        },
        "response_format": {"type": "json_object"},
    }
    try:
        raw = llm.invoke(payload)
        data = _ensure_dict(raw)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("process_from_flow.query_rewrite_llm_failed", exchange=exchange_name, error=str(exc))
        return []

    candidates: list[str] = []
    for key in ("query_variants", "variants", "rewrites", "search_queries", "queries"):
        candidates.extend(_extract_rewrite_variant_strings(data.get(key)))
    if not candidates and isinstance(data.get("primary_query"), str):
        candidates.append(str(data.get("primary_query")).strip())

    deduped: list[str] = []
    seen: set[str] = set()
    original_norm = _normalize_exchange_label(exchange_name)
    for item in candidates:
        text = str(item).strip()
        if not text:
            continue
        normalized = _normalize_exchange_label(text)
        if not normalized:
            continue
        if normalized == original_norm:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(text)
        if len(deduped) >= max(1, max_variants):
            break
    return deduped


def _extract_doi_from_text(value: str) -> str | None:
    if not value:
        return None
    match = _DOI_PATTERN.search(value)
    if not match:
        return None
    doi = match.group(0).strip()
    doi = doi.rstrip(").,;")
    return doi or None


def _extract_reference_doi(reference: dict[str, Any]) -> str | None:
    if not isinstance(reference, dict):
        return None
    for key in ("doi", "DOI"):
        value = reference.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("source", "content", "text"):
        value = reference.get(key)
        if isinstance(value, str):
            doi = _extract_doi_from_text(value)
            if doi:
                return doi
    metadata = reference.get("metadata")
    if isinstance(metadata, dict):
        for key in ("doi", "DOI"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        meta_text = metadata.get("meta")
        if isinstance(meta_text, str):
            doi = _extract_doi_from_text(meta_text)
            if doi:
                return doi
    segment = reference.get("segment")
    if isinstance(segment, dict):
        doi = _extract_reference_doi(segment)
        if doi:
            return doi
    return None


def _group_references_by_doi(
    references: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    missing: list[int] = []
    for idx, ref in enumerate(references, start=1):
        if not isinstance(ref, dict):
            continue
        doi = _extract_reference_doi(ref)
        if not doi:
            ref_no = ref.get("no")
            missing.append(ref_no if isinstance(ref_no, int) else idx)
            continue
        grouped.setdefault(doi, []).append(ref)
    return grouped, missing


def _build_fulltext_query(doi: str, references: list[dict[str, Any]], fallback: str) -> str:
    chunks: list[str] = []
    for ref in references:
        for key in ("content", "text"):
            value = ref.get(key)
            if isinstance(value, str) and value.strip():
                chunks.append(value.strip())
                break
    merged = " ".join(chunks)
    merged = re.sub(r"\s+", " ", merged).strip()
    query = _compact_text(merged, limit=200)
    if not query:
        query = _compact_text(fallback or "", limit=200)
    if not query:
        query = f"doi {doi}"
    return query


def _fetch_fulltext_references(
    references: list[dict[str, Any]],
    *,
    mcp_client: MCPToolClient | None = None,
    fallback_query: str = "",
    top_k: int = SCIENTIFIC_REFERENCE_FULLTEXT_TOP_K,
    ext_k: int = SCIENTIFIC_REFERENCE_FULLTEXT_EXT_K,
) -> tuple[list[dict[str, Any]], list[int]]:
    doi_groups, missing = _group_references_by_doi(references)
    entries: list[dict[str, Any]] = []
    for doi, refs in doi_groups.items():
        query = _build_fulltext_query(doi, refs, fallback_query)
        records = _search_scientific_references(
            query,
            mcp_client=mcp_client,
            top_k=top_k,
            ext_k=ext_k,
            filters={"doi": [doi]},
            keep_all=True,
        )
        source_refs = [ref.get("no") for ref in refs if isinstance(ref.get("no"), int)]
        entries.append(
            {
                "doi": doi,
                "query": query,
                "source_refs": source_refs,
                "records": records,
            }
        )
    return entries, missing


def _extract_record_text(record: dict[str, Any]) -> str:
    for key in ("content", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    segment = record.get("segment")
    if isinstance(segment, dict):
        for key in ("content", "text"):
            value = segment.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _collect_article_text(
    records: list[dict[str, Any]],
    *,
    max_chars: int = REFERENCE_CLUSTER_MAX_CHARS,
    max_records: int = REFERENCE_CLUSTER_MAX_RECORDS,
) -> str:
    chunks: list[str] = []
    for record in records:
        if max_records and len(chunks) >= max_records:
            break
        if not isinstance(record, dict):
            continue
        text = _extract_record_text(record)
        if text:
            chunks.append(text)
    combined = "\n\n".join(chunks).strip()
    if max_chars and len(combined) > max_chars:
        combined = combined[:max_chars]
    return combined


def _reference_usability_map(state: ProcessFromFlowState) -> dict[str, dict[str, Any]]:
    usability = state.get("scientific_references", {}).get("usability")
    if not isinstance(usability, dict):
        return {}
    results = usability.get("results")
    if not isinstance(results, list):
        return {}
    mapping: dict[str, dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        doi = str(item.get("doi") or "").strip()
        if not doi:
            continue
        mapping[doi] = item
    return mapping


def _reference_clusters(scientific_references: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(scientific_references, dict):
        return None
    value = scientific_references.get(REFERENCE_CLUSTERS_KEY)
    return value if isinstance(value, dict) else None


def _primary_cluster_dois(scientific_references: dict[str, Any] | None) -> list[str]:
    clusters = _reference_clusters(scientific_references)
    if not isinstance(clusters, dict):
        return []
    primary_id = str(clusters.get("primary_cluster_id") or "").strip()
    clusters_list = clusters.get("clusters")
    if not isinstance(clusters_list, list) or not primary_id:
        return []
    for cluster in clusters_list:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        if cluster_id != primary_id:
            continue
        dois = [str(item).strip() for item in (cluster.get("dois") or []) if str(item).strip()]
        return dois
    return []


def _has_reference_entries(scientific_references: dict[str, Any], key: str) -> bool:
    block = scientific_references.get(key)
    if not isinstance(block, dict):
        return False
    references = block.get("references")
    return isinstance(references, list) and len(references) > 0


def _references_usable(scientific_references: dict[str, Any] | None) -> bool:
    if not isinstance(scientific_references, dict):
        return False
    if not _has_reference_entries(scientific_references, REFERENCE_SEARCH_KEY):
        return False
    if not _has_reference_entries(scientific_references, REFERENCE_FULLTEXT_KEY):
        return False
    clusters = _reference_clusters(scientific_references)
    if not isinstance(clusters, dict):
        return False
    cluster_list = clusters.get("clusters")
    if not isinstance(cluster_list, list) or not cluster_list:
        return False
    usability = scientific_references.get("usability")
    if isinstance(usability, dict):
        results = usability.get("results")
        if isinstance(results, list):
            usable_found = any(isinstance(item, dict) and str(item.get("decision") or "").strip().lower() == "usable" for item in results)
            if not usable_found:
                return False
    return True


def _parse_reference_source(value: str) -> tuple[str, str]:
    text = value.strip()
    if not text:
        return "", ""
    match = _MARKDOWN_LINK_PATTERN.search(text)
    if match:
        label = match.group(1).strip()
        link = match.group(2).strip()
        return label, link
    link = _extract_reference_url(text)
    return text, link


def _extract_reference_url(value: str) -> str:
    if not value:
        return ""
    match = _URL_PATTERN.search(value)
    return match.group(0) if match else ""


def _reference_key(
    doi: str | None,
    url: str | None,
    citation: str | None,
    title: str | None,
    description: str | None,
) -> str:
    if doi:
        return f"doi:{doi.lower()}"
    if url:
        return f"url:{url.lower()}"
    if citation:
        return f"citation:{citation.lower()}"
    if title:
        return f"title:{title.lower()}"
    if description:
        return f"desc:{description.lower()}"
    return ""


def _reference_title_from_citation(citation: str | None) -> str:
    if not citation:
        return ""
    parts = re.split(r"\\.\\s+", citation.strip(), maxsplit=1)
    candidate = parts[0].strip() if parts else citation.strip()
    if len(candidate) < 8:
        candidate = citation.strip()
    return _compact_text(candidate, limit=160)


def _reference_info_from_record(
    record: dict[str, Any],
    *,
    origin_step: str | None = None,
    doi_override: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    doi = str(doi_override or _extract_reference_doi(record) or "").strip() or None
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    source_text = _first_nonempty(record.get("source"), metadata.get("meta"), metadata.get("citation"), metadata.get("source"))
    citation, link = _parse_reference_source(source_text) if source_text else ("", "")
    if not citation:
        citation = _first_nonempty(metadata.get("meta"), metadata.get("citation"), metadata.get("title"), source_text)
    url = _first_nonempty(link, metadata.get("url"), metadata.get("uri"), metadata.get("link"))
    if not url:
        url = _extract_reference_url(source_text or "")
    if not url and doi:
        url = f"https://doi.org/{doi}"
    title = _first_nonempty(metadata.get("title"), metadata.get("paper_title"), metadata.get("name"), metadata.get("document_title"))
    if not title:
        title = _reference_title_from_citation(citation)
    content = _extract_record_text(record)
    description = _compact_text(content, limit=360)
    if not citation and title:
        citation = title
    if not citation and doi:
        citation = f"DOI {doi}"
    if not any([doi, citation, title, description]):
        return None
    short_name = _compact_text(title or citation or (doi or "Reference source"), limit=180)
    key = _reference_key(doi, url, citation, title, description)
    if not key and short_name:
        key = f"short:{short_name.lower()}"
    return {
        "key": key,
        "doi": doi,
        "url": url,
        "citation": citation,
        "title": title,
        "short_name": short_name,
        "description": description,
        "origin_steps": [origin_step] if origin_step else [],
    }


def _merge_reference_info(target: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    if not target:
        return incoming
    for field in ("doi", "url"):
        if not target.get(field) and incoming.get(field):
            target[field] = incoming.get(field)
    for field in ("citation", "title", "short_name", "description"):
        incoming_value = incoming.get(field)
        if not incoming_value:
            continue
        current = target.get(field)
        if not current or len(str(incoming_value)) > len(str(current)):
            target[field] = incoming_value
    steps = set(target.get("origin_steps") or [])
    steps.update(incoming.get("origin_steps") or [])
    target["origin_steps"] = sorted(step for step in steps if step)
    return target


def _collect_reference_infos(scientific_references: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(scientific_references, dict):
        return []
    collected: dict[str, dict[str, Any]] = {}

    def register(info: dict[str, Any] | None) -> None:
        if not info:
            return
        key = str(info.get("key") or "").strip()
        if not key:
            key = f"ref:{len(collected) + 1}"
            info["key"] = key
        existing = collected.get(key)
        if existing is None:
            collected[key] = info
        else:
            collected[key] = _merge_reference_info(existing, info)

    fulltext_entries = scientific_references.get(REFERENCE_FULLTEXT_KEY, {}).get("references", [])
    if isinstance(fulltext_entries, list):
        for entry in fulltext_entries:
            if not isinstance(entry, dict):
                continue
            doi = str(entry.get("doi") or "").strip() or None
            records = entry.get("records") or []
            record = next((item for item in records if isinstance(item, dict)), None)
            if record is None and doi:
                record = {"doi": doi, "source": f"https://doi.org/{doi}"}
            info = _reference_info_from_record(record or {}, origin_step=REFERENCE_FULLTEXT_KEY, doi_override=doi)
            register(info)

    for step_key in (REFERENCE_SEARCH_KEY, "step2", "step3", INDUSTRY_AVERAGE_KEY):
        block = scientific_references.get(step_key)
        if not isinstance(block, dict):
            continue
        references = block.get("references")
        if not isinstance(references, list):
            continue
        for ref in references:
            if not isinstance(ref, dict):
                continue
            info = _reference_info_from_record(ref, origin_step=step_key)
            register(info)

    return list(collected.values())


def _build_reference_cluster_summaries(
    fulltext_entries: list[dict[str, Any]],
    *,
    usability_map: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    usability_map = usability_map or {}
    for entry in fulltext_entries:
        if not isinstance(entry, dict):
            continue
        doi = str(entry.get("doi") or "").strip()
        records = entry.get("records") or []
        records_list = [item for item in records if isinstance(item, dict)]
        snippet = _collect_article_text(records_list)
        usability = usability_map.get(doi) if doi else None
        supported_steps = _normalize_steps(usability.get("supported_steps") if isinstance(usability, dict) else None)
        decision = usability.get("decision") if isinstance(usability, dict) else None
        usage_tags = _usage_tags_from_supported_steps(supported_steps, decision)
        summaries.append(
            {
                "doi": doi,
                "supported_steps": supported_steps,
                "decision": decision,
                "reason": usability.get("reason") if isinstance(usability, dict) else None,
                "evidence": usability.get("evidence") if isinstance(usability, dict) else None,
                "si_hint": usability.get("si_hint") if isinstance(usability, dict) else None,
                "si_reason": usability.get("si_reason") if isinstance(usability, dict) else None,
                "usage_tags": usage_tags,
                "snippet": snippet,
            }
        )
    return summaries


def _source_uuid_for_info(info: dict[str, Any]) -> str:
    key = str(info.get("key") or "").strip()
    if not key:
        return str(uuid4())
    return str(uuid5(NAMESPACE_URL, key))


def _build_source_reference_payload(
    *,
    uuid_value: str,
    version: str,
    short_description: Any,
) -> dict[str, Any]:
    reference = _global_reference(
        ref_type="source data set",
        ref_object_id=uuid_value,
        version=version,
        uri=build_local_dataset_uri("source data set", uuid_value, version),
        short_description=short_description,
    )
    return reference.model_dump(mode="json", by_alias=True, exclude_none=True)


def _build_source_dataset(
    info: dict[str, Any],
    *,
    translator: Translator | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not info:
        return None
    uuid_value = _source_uuid_for_info(info)
    version = "01.01.000"
    short_name = str(info.get("short_name") or info.get("title") or info.get("citation") or info.get("doi") or "Source reference").strip()
    citation = str(info.get("citation") or short_name or "Source reference").strip()
    description = str(info.get("description") or "").strip()
    origin_steps = [str(step).strip() for step in (info.get("origin_steps") or []) if str(step).strip()]
    if origin_steps:
        description = f"{description} Search steps: {', '.join(origin_steps)}.".strip() if description else f"Search steps: {', '.join(origin_steps)}."
    if info.get("doi"):
        description = f"{description} DOI: {info['doi']}.".strip() if description else f"DOI: {info['doi']}."
    url = str(info.get("url") or "").strip()

    short_entries = _build_multilang_entries(short_name, translator=translator)
    description_entries = _build_multilang_entries(description, translator=translator) if description else []
    classification_class = SourceClassificationCommonClass(level="0", class_id="C", text="Literature")
    classification = SourceClassificationInformationCommonClassification(common_class=classification_class)
    classification_info = SourceInformationDataSetInformationClassificationInformation(common_classification=classification)
    reference_to_file = SourceInformationDataSetInformationReferenceToDigitalFile(uri=url) if url else None
    publication_type = "Article in periodical" if info.get("doi") else "Other unpublished and grey literature"

    data_info = SourceDataSetSourceInformationDataSetInformation(
        common_uuid=uuid_value,
        common_short_name=_as_multilang_list(short_entries or short_name),
        classification_information=classification_info,
        source_citation=citation,
        publication_type=publication_type,
        source_description_or_comment=_as_multilang_list(description_entries or description),
        reference_to_digital_file=reference_to_file,
        reference_to_contact=_contact_reference(),
    )
    source_information = SourcesSourceDataSetSourceInformation(data_set_information=data_info)
    data_entry = SourceDataSetAdministrativeInformationDataEntryBy(
        common_time_stamp=default_timestamp(),
        common_reference_to_data_set_format=_dataset_format_reference(),
    )
    publication = SourceDataSetAdministrativeInformationPublicationAndOwnership(
        common_data_set_version=version,
        common_permanent_data_set_uri=build_portal_uri("source", uuid_value, version),
        common_reference_to_ownership_of_data_set=_contact_reference(),
    )
    administrative_information = SourcesSourceDataSetAdministrativeInformation(
        data_entry_by=data_entry,
        publication_and_ownership=publication,
    )
    source_dataset = SourcesSourceDataSet(
        xmlns="http://lca.jrc.it/ILCD/Source",
        xmlns_common="http://lca.jrc.it/ILCD/Common",
        xmlns_xsi="http://www.w3.org/2001/XMLSchema-instance",
        version="1.1",
        xsi_schema_location="http://lca.jrc.it/ILCD/Source ../../schemas/ILCD_SourceDataSet.xsd",
        source_information=source_information,
        administrative_information=administrative_information,
    )
    source_model = Sources(source_data_set=source_dataset)

    validated_on_init = False
    try:
        entity = create_source(source_model, validate=True)
        validated_on_init = True
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("process_from_flow.source_validation_failed", source_key=info.get("key"), error=str(exc))
        entity = create_source(source_model, validate=False)

    if validated_on_init:
        errors = entity.last_validation_error()
        if errors:
            LOGGER.warning("process_from_flow.source_not_valid", source_key=info.get("key"), error=str(errors))
    else:
        valid = entity.validate(mode="pydantic")
        if not valid:
            errors = entity.last_validation_error()
            LOGGER.warning("process_from_flow.source_not_valid", source_key=info.get("key"), error=str(errors))

    payload = entity.model.model_dump(mode="json", by_alias=True, exclude_none=True)
    reference_payload = _build_source_reference_payload(
        uuid_value=uuid_value,
        version=version,
        short_description=short_entries or short_name,
    )
    return payload, reference_payload


def _build_source_datasets_from_references(
    scientific_references: dict[str, Any] | None,
    *,
    translator: Translator | None = None,
    reference_infos: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if reference_infos is not None:
        infos = reference_infos
    else:
        if not isinstance(scientific_references, dict):
            return [], []
        infos = _collect_reference_infos(scientific_references)
    source_datasets: list[dict[str, Any]] = []
    source_references: list[dict[str, Any]] = []
    for info in infos:
        result = _build_source_dataset(info, translator=translator)
        if not result:
            continue
        payload, reference = result
        source_datasets.append(payload)
        source_references.append(reference)
    return source_datasets, source_references


def _build_source_reference_entries(source_datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for dataset in source_datasets:
        if not isinstance(dataset, dict):
            continue
        source_data_set = dataset.get("sourceDataSet")
        if not isinstance(source_data_set, dict):
            continue
        info = source_data_set.get("sourceInformation")
        if not isinstance(info, dict):
            continue
        data_info = info.get("dataSetInformation")
        if not isinstance(data_info, dict):
            continue
        uuid_value = str(data_info.get("common:UUID") or "").strip()
        if not uuid_value:
            continue
        short_desc = data_info.get("common:shortName")
        admin = source_data_set.get("administrativeInformation")
        version = None
        if isinstance(admin, dict):
            publication = admin.get("publicationAndOwnership")
            if isinstance(publication, dict):
                version = publication.get("common:dataSetVersion")
        version_text = str(version or "01.01.000").strip()
        if not short_desc:
            short_desc = [_language_entry("Source reference", "en")]
        references.append(
            {
                "@type": "source data set",
                "@refObjectId": uuid_value,
                "@uri": build_local_dataset_uri("source data set", uuid_value, version_text),
                "@version": version_text,
                "common:shortDescription": short_desc,
            }
        )
    return references


def _coerce_global_reference_items(references: list[Any]) -> list[GlobalReferenceTypeVariant1Item]:
    items: list[GlobalReferenceTypeVariant1Item] = []
    for ref in references:
        if isinstance(ref, GlobalReferenceTypeVariant1Item):
            items.append(ref)
            continue
        if isinstance(ref, GlobalReferenceTypeVariant0):
            items.append(
                GlobalReferenceTypeVariant1Item(
                    type=ref.type,
                    ref_object_id=ref.ref_object_id,
                    version=ref.version,
                    uri=ref.uri,
                    common_short_description=ref.common_short_description,
                )
            )
            continue
        if not isinstance(ref, dict):
            continue
        ref_type = ref.get("@type") or ref.get("type")
        ref_id = ref.get("@refObjectId") or ref.get("refObjectId")
        version = ref.get("@version") or ref.get("version") or "01.01.000"
        uri = ref.get("@uri") or ref.get("uri") or ""
        short_desc = ref.get("common:shortDescription") or ref.get("common_short_description") or ref.get("commonShortDescription")
        if not ref_type or not ref_id:
            continue
        items.append(
            GlobalReferenceTypeVariant1Item(
                type=str(ref_type),
                ref_object_id=str(ref_id),
                version=str(version),
                uri=str(uri),
                common_short_description=_as_multilang_list(short_desc or "Source reference"),
            )
        )
    return items


def _reference_item_id(item: Any) -> str | None:
    if isinstance(item, dict):
        return str(item.get("@refObjectId") or item.get("refObjectId") or "").strip() or None
    ref_object_id = getattr(item, "ref_object_id", None)
    if ref_object_id:
        return str(ref_object_id).strip() or None
    ref_object_id = getattr(item, "refObjectId", None)
    if ref_object_id:
        return str(ref_object_id).strip() or None
    return None


def _normalize_reference_text(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip().lower()
    return text


def _normalize_reference_url(value: str) -> str:
    text = value.strip().rstrip(").,;").lower()
    return text


def _collect_lang_values(value: Any) -> list[str]:
    lang_map = _extract_lang_texts(value)
    values: list[str] = []
    for items in lang_map.values():
        for item in items:
            text = str(item).strip()
            if text:
                values.append(text)
    return values


def _build_source_reference_index(
    source_datasets: list[dict[str, Any]],
    source_references: list[dict[str, Any]],
) -> dict[str, Any]:
    by_uuid: dict[str, dict[str, Any]] = {}
    for ref in source_references:
        if not isinstance(ref, dict):
            continue
        uuid_value = str(ref.get("@refObjectId") or "").strip()
        if uuid_value:
            by_uuid[uuid_value] = ref

    doi_map: dict[str, dict[str, Any]] = {}
    url_map: dict[str, dict[str, Any]] = {}
    text_entries: list[tuple[str, dict[str, Any]]] = []

    for dataset in source_datasets:
        if not isinstance(dataset, dict):
            continue
        source_data_set = dataset.get("sourceDataSet")
        if not isinstance(source_data_set, dict):
            continue
        info_block = source_data_set.get("sourceInformation")
        if not isinstance(info_block, dict):
            continue
        data_info = info_block.get("dataSetInformation")
        if not isinstance(data_info, dict):
            continue
        uuid_value = str(data_info.get("common:UUID") or "").strip()
        if not uuid_value:
            continue
        ref = by_uuid.get(uuid_value)
        if ref is None:
            ref_candidates = _build_source_reference_entries([dataset])
            ref = ref_candidates[0] if ref_candidates else None
        if not isinstance(ref, dict):
            continue

        citation = data_info.get("sourceCitation")
        short_name = data_info.get("common:shortName")
        description = data_info.get("sourceDescriptionOrComment")
        digital_file = data_info.get("referenceToDigitalFile")
        url = None
        if isinstance(digital_file, dict):
            url = digital_file.get("@uri")

        for text in [citation, short_name, description]:
            for value in _collect_lang_values(text):
                doi = _extract_doi_from_text(value)
                if doi:
                    doi_map[doi.lower()] = ref
                text_key = _normalize_reference_text(value)
                if text_key:
                    text_entries.append((text_key, ref))

        if isinstance(url, str) and url.strip():
            url_key = _normalize_reference_url(url)
            if url_key:
                url_map[url_key] = ref
            doi = _extract_doi_from_text(url)
            if doi:
                doi_map[doi.lower()] = ref

    return {
        "doi": doi_map,
        "url": url_map,
        "text_entries": text_entries,
    }


def _collect_exchange_citations(exchange: dict[str, Any]) -> list[str]:
    data_source = exchange.get("data_source") or exchange.get("dataSource")
    if not isinstance(data_source, dict):
        return []
    citations = _clean_evidence_list(data_source.get("citations"))
    return citations


def _match_source_references(
    evidence: list[str],
    reference_index: dict[str, Any],
) -> list[dict[str, Any]]:
    if not evidence:
        return []
    doi_map = reference_index.get("doi", {})
    url_map = reference_index.get("url", {})
    text_entries = reference_index.get("text_entries", [])

    matched: dict[str, dict[str, Any]] = {}
    for item in evidence:
        text = str(item).strip()
        if not text:
            continue
        doi = _extract_doi_from_text(text)
        if doi:
            ref = doi_map.get(doi.lower())
            if isinstance(ref, dict):
                ref_id = ref.get("@refObjectId")
                if ref_id:
                    matched[str(ref_id)] = ref
                continue
        url = _extract_reference_url(text)
        if url:
            url_key = _normalize_reference_url(url)
            ref = url_map.get(url_key)
            if ref is None and url_key.rstrip("/") != url_key:
                ref = url_map.get(url_key.rstrip("/"))
            if isinstance(ref, dict):
                ref_id = ref.get("@refObjectId")
                if ref_id:
                    matched[str(ref_id)] = ref
                continue
            doi = _extract_doi_from_text(url_key)
            if doi:
                ref = doi_map.get(doi.lower())
                if isinstance(ref, dict):
                    ref_id = ref.get("@refObjectId")
                    if ref_id:
                        matched[str(ref_id)] = ref
                    continue
        norm = _normalize_reference_text(text)
        if not norm:
            continue
        for key, ref in text_entries:
            if len(key) < 12:
                continue
            if key in norm or norm in key:
                ref_id = ref.get("@refObjectId") if isinstance(ref, dict) else None
                if ref_id:
                    matched[str(ref_id)] = ref
    return list(matched.values())


def _extract_mineru_text_blocks(payload: Any, *, max_blocks: int) -> list[str]:
    texts: list[str] = []

    def add_text(value: Any) -> None:
        if not value:
            return
        text = str(value).strip()
        if text:
            texts.append(text)

    def handle_item(item: Any) -> None:
        if not isinstance(item, dict):
            return
        text = item.get("text") or item.get("content")
        if text:
            add_text(text)
            return
        blocks = item.get("blocks")
        if isinstance(blocks, list):
            for block in blocks:
                if len(texts) >= max_blocks:
                    return
                if isinstance(block, dict):
                    add_text(block.get("text") or block.get("content"))

    if isinstance(payload, dict):
        if "result" in payload:
            payload = payload.get("result")
        elif "pages" in payload:
            payload = payload.get("pages")

    if isinstance(payload, list):
        for item in payload:
            if len(texts) >= max_blocks:
                break
            handle_item(item)

    return texts


def _collect_snippet(texts: list[str], *, max_chars: int) -> str:
    if not texts:
        return ""
    chunks: list[str] = []
    total = 0
    for text in texts:
        if not text:
            continue
        if total + len(text) + 1 > max_chars:
            break
        chunks.append(text)
        total += len(text) + 1
    return "\n".join(chunks).strip()


def _normalize_doi(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().lower().replace("_", "/")


def _normalize_si_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _should_keep_si_paragraph(text: str) -> bool:
    if re.search(r"\d", text):
        return True
    lowered = text.lower()
    return any(token in lowered for token in ("table", "figure", "supplement", "appendix", "inventory", "input", "output"))


def _format_table_rows(rows: list[list[str]], *, max_rows: int, max_cols: int) -> list[str]:
    lines: list[str] = []
    for idx, row in enumerate(rows):
        if len(lines) >= max_rows:
            break
        cleaned = [_normalize_si_text(cell or "") for cell in row[:max_cols]]
        while cleaned and not cleaned[-1]:
            cleaned.pop()
        if not cleaned:
            continue
        row_text = " | ".join(cleaned)
        if idx == 0 or re.search(r"\d", row_text):
            lines.append(row_text)
    return lines


def _extract_docx_table_rows(table: ET.Element) -> list[list[str]]:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    rows: list[list[str]] = []
    for row in table.findall(".//w:tr", ns):
        cells: list[str] = []
        for cell in row.findall(".//w:tc", ns):
            cell_texts = [text.text or "" for text in cell.findall(".//w:t", ns)]
            cell_text = _normalize_si_text("".join(cell_texts))
            cells.append(cell_text)
        while cells and not cells[-1]:
            cells.pop()
        if cells:
            rows.append(cells)
    return rows


def _extract_docx_text(path: Path) -> str:
    try:
        with ZipFile(path) as zip_file:
            xml_payload = zip_file.read("word/document.xml")
    except (OSError, KeyError):
        return ""
    try:
        root = ET.fromstring(xml_payload)
    except ET.ParseError:
        return ""
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    body = root.find("w:body", ns)
    if body is None:
        return ""
    table_lines: list[str] = []
    para_lines: list[str] = []
    table_count = 0
    para_count = 0
    for child in body:
        tag = child.tag.split("}")[-1]
        if tag == "tbl" and table_count < SI_DOCX_MAX_TABLES:
            rows = _extract_docx_table_rows(child)
            if rows:
                table_count += 1
                table_lines.append(f"Table {table_count}:")
                table_lines.extend(_format_table_rows(rows, max_rows=SI_TABLE_MAX_ROWS, max_cols=SI_TABLE_MAX_COLS))
        elif tag == "p" and para_count < SI_DOCX_MAX_PARAGRAPHS:
            texts = [text.text or "" for text in child.findall(".//w:t", ns)]
            paragraph = _normalize_si_text("".join(texts))
            if not paragraph:
                continue
            if _should_keep_si_paragraph(paragraph):
                para_lines.append(paragraph)
                para_count += 1
    return "\n".join(table_lines + para_lines).strip()


def _extract_delimited_text(path: Path, *, delimiter: str) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            lines: list[str] = []
            for row_idx, row in enumerate(reader):
                if row_idx >= SI_TABLE_MAX_ROWS:
                    break
                cleaned = [_normalize_si_text(cell) for cell in row[:SI_TABLE_MAX_COLS]]
                while cleaned and not cleaned[-1]:
                    cleaned.pop()
                if not cleaned:
                    continue
                row_text = " | ".join(cleaned)
                if row_idx == 0 or re.search(r"\d", row_text):
                    lines.append(row_text)
            return "\n".join(lines).strip()
    except OSError:
        return ""


def _column_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref.upper())
    if not match:
        return 0
    letters = match.group(1)
    value = 0
    for char in letters:
        value = value * 26 + (ord(char) - ord("A") + 1)
    return max(value - 1, 0)


def _extract_xlsx_text(path: Path) -> str:
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    try:
        with ZipFile(path) as zip_file:
            shared_strings: list[str] = []
            try:
                shared_root = ET.fromstring(zip_file.read("xl/sharedStrings.xml"))
                for item in shared_root.findall("s:si", ns):
                    texts = [node.text or "" for node in item.findall(".//s:t", ns)]
                    shared_strings.append(_normalize_si_text("".join(texts)))
            except KeyError:
                shared_strings = []

            sheet_names = sorted(name for name in zip_file.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"))
            lines: list[str] = []
            for sheet_idx, sheet_name in enumerate(sheet_names[:SI_XLSX_MAX_SHEETS], start=1):
                try:
                    sheet_root = ET.fromstring(zip_file.read(sheet_name))
                except ET.ParseError:
                    continue
                rows = sheet_root.findall(".//s:sheetData/s:row", ns)
                if not rows:
                    continue
                lines.append(f"Sheet {sheet_idx}:")
                for row_idx, row in enumerate(rows):
                    if row_idx >= SI_TABLE_MAX_ROWS:
                        break
                    values: dict[int, str] = {}
                    for cell in row.findall("s:c", ns):
                        ref = cell.get("r") or ""
                        col_idx = _column_index(ref)
                        cell_type = cell.get("t")
                        value = ""
                        if cell_type == "s":
                            index_text = cell.findtext("s:v", default="", namespaces=ns)
                            try:
                                index = int(index_text)
                                if 0 <= index < len(shared_strings):
                                    value = shared_strings[index]
                            except ValueError:
                                value = ""
                        elif cell_type == "inlineStr":
                            texts = [node.text or "" for node in cell.findall(".//s:t", ns)]
                            value = _normalize_si_text("".join(texts))
                        else:
                            value = cell.findtext("s:v", default="", namespaces=ns).strip()
                        value = _normalize_si_text(value)
                        if value:
                            values[col_idx] = value
                    if not values:
                        continue
                    max_idx = min(max(values), SI_TABLE_MAX_COLS - 1)
                    row_cells = [values.get(idx, "") for idx in range(max_idx + 1)]
                    row_text = " | ".join(_normalize_si_text(cell) for cell in row_cells)
                    if row_idx == 0 or re.search(r"\d", row_text):
                        lines.append(row_text)
    except OSError:
        return ""
    return "\n".join(lines).strip()


def _si_source_rank(suffix: str) -> tuple[str, int]:
    if suffix == ".docx":
        return "docx", 0
    if suffix in {".xlsx", ".csv", ".tsv"}:
        return "tabular", 1
    if suffix in {".txt", ".md", ".markdown"}:
        return "text", 2
    return "other", 9


def _read_si_content(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        try:
            return path.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            return ""
    if suffix == ".docx":
        return _extract_docx_text(path)
    if suffix == ".csv":
        return _extract_delimited_text(path, delimiter=",")
    if suffix == ".tsv":
        return _extract_delimited_text(path, delimiter="\t")
    if suffix == ".xlsx":
        return _extract_xlsx_text(path)
    return ""


def _iter_si_text_entries(scientific_references: dict[str, Any]) -> list[dict[str, Any]]:
    downloads = scientific_references.get("si_downloads")
    entries = downloads.get("entries") if isinstance(downloads, dict) else None
    if not isinstance(entries, list):
        return []
    results: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        path_text = entry.get("path")
        if not isinstance(path_text, str) or not path_text.strip():
            continue
        path = Path(path_text)
        if not path.exists():
            continue
        content = _read_si_content(path)
        if not content:
            continue
        snippet = content[:SI_SNIPPET_MAX_CHARS].strip()
        if not snippet:
            continue
        source_type, source_rank = _si_source_rank(path.suffix.lower())
        results.append(
            {
                "doi": entry.get("doi"),
                "snippet": snippet,
                "source_path": str(path),
                "source_type": source_type,
                "source_rank": source_rank,
            }
        )
    return results


def _load_si_snippets(scientific_references: dict[str, Any]) -> list[dict[str, Any]]:
    entries = scientific_references.get("si_mineru_outputs")
    if not isinstance(entries, list):
        entries = []
    primary_dois: list[str] = []
    clusters = scientific_references.get(REFERENCE_CLUSTERS_KEY)
    if isinstance(clusters, dict):
        primary_id = str(clusters.get("primary_cluster_id") or "").strip()
        clusters_list = clusters.get("clusters")
        if primary_id and isinstance(clusters_list, list):
            for cluster in clusters_list:
                if not isinstance(cluster, dict):
                    continue
                if str(cluster.get("cluster_id") or "").strip() == primary_id:
                    primary_dois = [str(item).strip() for item in (cluster.get("dois") or []) if str(item).strip()]
                    break
    primary_set = {_normalize_doi(item) for item in primary_dois if item}
    candidates: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    def add_candidate(
        doi: str | None,
        snippet: str,
        source_path: str,
        *,
        source_type: str,
        source_rank: int,
    ) -> None:
        norm_doi = _normalize_doi(doi)
        key = (norm_doi, source_path)
        if key in seen_keys:
            return
        seen_keys.add(key)
        candidates.append(
            {
                "doi": doi or None,
                "normalized_doi": norm_doi,
                "snippet": snippet,
                "source_path": source_path,
                "source_type": source_type,
                "source_rank": source_rank,
            }
        )

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        output_path = entry.get("output_path")
        if not isinstance(output_path, str) or not output_path.strip():
            continue
        path = Path(output_path)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        blocks = _extract_mineru_text_blocks(payload, max_blocks=SI_SNIPPET_MAX_BLOCKS)
        snippet = _collect_snippet(blocks, max_chars=SI_SNIPPET_MAX_CHARS)
        if not snippet:
            continue
        doi = str(entry.get("doi") or "").strip() or None
        add_candidate(doi, snippet, str(path), source_type="mineru", source_rank=3)

    for entry in _iter_si_text_entries(scientific_references):
        doi = str(entry.get("doi") or "").strip() or None
        snippet = entry.get("snippet")
        source_path = entry.get("source_path")
        source_type = str(entry.get("source_type") or "text")
        source_rank = int(entry.get("source_rank") or 2)
        if not isinstance(snippet, str) or not isinstance(source_path, str):
            continue
        add_candidate(doi, snippet, source_path, source_type=source_type, source_rank=source_rank)

    if primary_set:
        candidates.sort(
            key=lambda item: (
                item.get("normalized_doi") not in primary_set,
                item.get("source_rank", 9),
                item.get("source_path") or "",
            )
        )
    else:
        candidates.sort(key=lambda item: (item.get("source_rank", 9), item.get("source_path") or ""))

    snippets: list[dict[str, Any]] = []
    seen_dois: set[str] = set()
    for candidate in candidates:
        if len(snippets) >= SI_SNIPPET_MAX_FILES:
            break
        norm_doi = candidate.get("normalized_doi") or ""
        if norm_doi and norm_doi in seen_dois:
            continue
        snippets.append(
            {
                "doi": candidate.get("doi"),
                "snippet": candidate.get("snippet"),
                "source_path": candidate.get("source_path"),
            }
        )
        if norm_doi:
            seen_dois.add(norm_doi)
    return snippets


def _format_fulltext_entries_for_prompt(
    fulltext_entries: list[dict[str, Any]],
    *,
    max_records: int = 2,
    max_chars: int = 1600,
) -> str:
    if not fulltext_entries:
        return ""
    lines = ["Fulltext References:"]
    for entry in fulltext_entries:
        if not isinstance(entry, dict):
            continue
        doi = str(entry.get("doi") or "").strip()
        records = entry.get("records") or []
        records_list = [item for item in records if isinstance(item, dict)]
        snippet = _collect_article_text(records_list, max_chars=max_chars, max_records=max_records)
        if not snippet:
            continue
        label = f"[{doi}]" if doi else "[ref]"
        lines.append(f"{label} {snippet}")
    return "\n".join(lines)


def _normalize_exchange_name(value: str | None) -> str:
    if not value:
        return ""
    text = _strip_flow_label(str(value)).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _coerce_amount_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_amount_value(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _format_amount_value(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.6g}"


def _normalize_unit_token(value: str | None) -> str:
    if not value:
        return ""
    text = str(value).strip().lower()
    text = text.replace(" ", "")
    text = text.replace("^", "")
    text = text.replace("³", "3")
    return text


_UNIT_DIMENSIONS: dict[str, tuple[str, float]] = {
    "kg": ("mass", 1.0),
    "g": ("mass", 0.001),
    "mg": ("mass", 1.0e-6),
    "t": ("mass", 1000.0),
    "lb": ("mass", 0.45359237),
    "lbav": ("mass", 0.45359237),
    "oz": ("mass", 0.028349523125),
    "mj": ("energy", 1.0),
    "gj": ("energy", 1000.0),
    "j": ("energy", 1.0e-6),
    "kwh": ("energy", 3.6),
    "mwh": ("energy", 3600.0),
    "m3": ("volume", 1.0),
    "l": ("volume", 0.001),
    "liter": ("volume", 0.001),
    "litre": ("volume", 0.001),
    "dm3": ("volume", 0.001),
    "cm3": ("volume", 1.0e-6),
    "ml": ("volume", 1.0e-6),
}

_AMBIGUOUS_BALANCE_UNITS = {
    "unit",
    "units",
    "item",
    "items",
    "piece",
    "pieces",
    "pc",
    "pcs",
    "ea",
    "count",
    "set",
    "batch",
}


def _convert_amount_simple(amount: float, from_unit: str, to_unit: str) -> float | None:
    from_key = _normalize_unit_token(from_unit)
    to_key = _normalize_unit_token(to_unit)
    if not from_key or not to_key:
        return None
    if from_key == to_key:
        return amount
    from_entry = _UNIT_DIMENSIONS.get(from_key)
    to_entry = _UNIT_DIMENSIONS.get(to_key)
    if not from_entry or not to_entry:
        return None
    if from_entry[0] != to_entry[0]:
        return None
    return amount * from_entry[1] / to_entry[1]


def _select_dataset_file(directory: Path, dataset_id: str | None) -> Path | None:
    if not dataset_id:
        return None
    try:
        return next(directory.glob(f"{dataset_id}_*.json"), None)
    except FileNotFoundError:
        return None


def _load_json_dataset(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _extract_name_text(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and str(item.get("@xml:lang") or "").strip().lower() == "en":
                text = str(item.get("#text") or "").strip()
                if text:
                    return text
        for item in value:
            if isinstance(item, dict):
                text = str(item.get("#text") or "").strip()
                if text:
                    return text
        return ""
    if isinstance(value, dict):
        text = str(value.get("#text") or "").strip()
        return text
    return str(value or "").strip()


def _get_flow_property_info(flow_property_id: str | None) -> FlowPropertyInfo | None:
    if not flow_property_id:
        return None
    cached = _FLOW_PROPERTY_CACHE.get(flow_property_id)
    if cached:
        return cached
    path = _select_dataset_file(_FLOWPROPERTY_DIR, flow_property_id)
    dataset = _load_json_dataset(path)
    if not dataset:
        return None
    flow_property = dataset.get("flowPropertyDataSet") if isinstance(dataset, dict) else None
    info = flow_property.get("flowPropertiesInformation") if isinstance(flow_property, dict) else None
    data_info = info.get("dataSetInformation") if isinstance(info, dict) else None
    name = _extract_name_text((data_info or {}).get("common:name")) or flow_property_id
    quant_ref = info.get("quantitativeReference") if isinstance(info, dict) else None
    ref_group = (quant_ref or {}).get("referenceToReferenceUnitGroup") if isinstance(quant_ref, dict) else None
    unit_group_id = str((ref_group or {}).get("@refObjectId") or "").strip() or None
    entry = FlowPropertyInfo(flow_property_id=flow_property_id, name=name, unit_group_id=unit_group_id)
    _FLOW_PROPERTY_CACHE[flow_property_id] = entry
    return entry


def _get_unit_group_info(unit_group_id: str | None) -> UnitGroupInfo | None:
    if not unit_group_id:
        return None
    cached = _UNIT_GROUP_CACHE.get(unit_group_id)
    if cached:
        return cached
    path = _select_dataset_file(_UNIT_GROUP_DIR, unit_group_id)
    dataset = _load_json_dataset(path)
    if not dataset:
        return None
    unit_group = dataset.get("unitGroupDataSet") if isinstance(dataset, dict) else None
    info = unit_group.get("unitGroupInformation") if isinstance(unit_group, dict) else None
    data_info = info.get("dataSetInformation") if isinstance(info, dict) else None
    name = _extract_name_text((data_info or {}).get("common:name")) or unit_group_id
    quant_ref = info.get("quantitativeReference") if isinstance(info, dict) else None
    ref_internal_id = str((quant_ref or {}).get("referenceToReferenceUnit") or "").strip()
    units_block = unit_group.get("units") if isinstance(unit_group, dict) else None
    units_raw = (units_block or {}).get("unit") if isinstance(units_block, dict) else None
    units_list = units_raw if isinstance(units_raw, list) else [units_raw] if isinstance(units_raw, dict) else []
    units: dict[str, float] = {}
    reference_unit = ""
    for unit in units_list:
        if not isinstance(unit, dict):
            continue
        unit_name = str(unit.get("name") or "").strip()
        if not unit_name:
            continue
        mean_value = _parse_amount_value(unit.get("meanValue"))
        if mean_value is None:
            continue
        units[_normalize_unit_token(unit_name)] = mean_value
        internal_id = str(unit.get("@dataSetInternalID") or "").strip()
        if internal_id and internal_id == ref_internal_id:
            reference_unit = unit_name
    if not reference_unit and units_list:
        reference_unit = str((units_list[0] or {}).get("name") or "").strip()
    entry = UnitGroupInfo(unit_group_id=unit_group_id, name=name, reference_unit=reference_unit, units=units)
    _UNIT_GROUP_CACHE[unit_group_id] = entry
    return entry


def _flow_reference_info_from_dataset(flow_dataset: dict[str, Any] | None) -> FlowReferenceInfo | None:
    if not isinstance(flow_dataset, dict):
        return None
    # DatabaseCrudClient.select_flow() may return either {"flowDataSet": {...}}
    # or the bare flowDataSet payload. Accept both.
    flow_data = flow_dataset.get("flowDataSet") if isinstance(flow_dataset.get("flowDataSet"), dict) else flow_dataset
    if not isinstance(flow_data, dict):
        return None
    flow_info = flow_data.get("flowInformation") if isinstance(flow_data.get("flowInformation"), dict) else None
    quant_ref = flow_info.get("quantitativeReference") if isinstance(flow_info, dict) else None
    ref_internal_id = str((quant_ref or {}).get("referenceToReferenceFlowProperty") or "").strip()
    properties_block = flow_data.get("flowProperties") if isinstance(flow_data.get("flowProperties"), dict) else None
    properties_raw = (properties_block or {}).get("flowProperty")
    properties = properties_raw if isinstance(properties_raw, list) else [properties_raw] if isinstance(properties_raw, dict) else []
    selected = None
    if properties and ref_internal_id:
        for item in properties:
            if isinstance(item, dict) and str(item.get("@dataSetInternalID") or "").strip() == ref_internal_id:
                selected = item
                break
    if selected is None and properties:
        selected = properties[0] if isinstance(properties[0], dict) else None
    if not isinstance(selected, dict):
        return None
    flow_property_ref = selected.get("referenceToFlowPropertyDataSet") if isinstance(selected.get("referenceToFlowPropertyDataSet"), dict) else None
    flow_property_id = str((flow_property_ref or {}).get("@refObjectId") or "").strip() or None
    flow_property = _get_flow_property_info(flow_property_id)
    unit_group = _get_unit_group_info(flow_property.unit_group_id) if flow_property else None
    return FlowReferenceInfo(flow_property_id=flow_property_id, unit_group=unit_group)


def _convert_amount_with_unit_group(amount: float, unit: str, unit_group: UnitGroupInfo) -> float | None:
    unit_key = _normalize_unit_token(unit)
    if not unit_key:
        return None
    factor = unit_group.units.get(unit_key)
    if factor is None:
        return None
    return amount * factor


def _unit_group_dimension(unit_group: UnitGroupInfo | None) -> str | None:
    if not unit_group or not unit_group.name:
        return None
    name = unit_group.name.strip().lower()
    if name == "units of mass":
        return "mass"
    if name == "units of energy":
        return "energy"
    return None


def _unit_group_category(unit_group: UnitGroupInfo | None) -> str | None:
    if not unit_group or not unit_group.name:
        return None
    name = unit_group.name.strip().lower()
    if name == "units of mass":
        return "mass"
    if name == "units of energy":
        return "energy"
    if name == "units of volume":
        return "volume"
    if name == "units of items":
        return "items"
    if name == "units of area":
        return "area"
    if name == "units of length":
        return "length"
    if name == "units of mole":
        return "mole"
    return None


def _unit_dimension_from_unit(unit: str | None) -> str | None:
    unit_key = _normalize_unit_token(unit)
    if not unit_key:
        return None
    entry = _UNIT_DIMENSIONS.get(unit_key)
    if not entry:
        return None
    return entry[0]


def _normalize_density_unit(value: str | None) -> str:
    if not value:
        return ""
    text = str(value).strip().lower()
    text = text.replace(" ", "")
    text = text.replace("^", "")
    text = text.replace("³", "3")
    text = text.replace("liter", "l")
    text = text.replace("litre", "l")
    return text


def _density_to_kg_per_m3(value: float, unit: str) -> float | None:
    unit_key = _normalize_density_unit(unit)
    if not unit_key:
        return None
    factors = {
        "kg/m3": 1.0,
        "g/cm3": 1000.0,
        "g/ml": 1000.0,
        "kg/l": 1000.0,
        "g/l": 1.0,
        "t/m3": 1000.0,
    }
    factor = factors.get(unit_key)
    if factor is None:
        return None
    return value * factor


def _format_balance_line(label: str, state: dict[str, Any]) -> str:
    if not state.get("count"):
        return f"{label} balance: insufficient data"
    inputs = float(state.get("inputs") or 0.0)
    outputs = float(state.get("outputs") or 0.0)
    unit = str(state.get("unit") or "").strip()
    count = int(state.get("count") or 0)
    if inputs <= 0 or outputs <= 0:
        return f"{label} balance: inputs={_format_amount_value(inputs)} {unit}, outputs={_format_amount_value(outputs)} {unit}, status=insufficient (n={count})"
    ratio = outputs / inputs
    status = "ok" if 0.8 <= ratio <= 1.2 else "check"
    return f"{label} balance: inputs={_format_amount_value(inputs)} {unit}, " f"outputs={_format_amount_value(outputs)} {unit}, ratio={ratio:.3g}, status={status} (n={count})"


def _build_balance_note(balance_state: dict[str, dict[str, Any]]) -> str | None:
    lines: list[str] = []
    for label in ("mass", "energy"):
        state = balance_state.get(label)
        if not state:
            continue
        if state.get("count") or state.get("inputs") or state.get("outputs"):
            lines.append(_format_balance_line(label, state))
    if not lines:
        return None
    return "Balance check: " + "; ".join(lines)


def _balance_state_summary(state: dict[str, Any]) -> dict[str, Any]:
    inputs = float(state.get("inputs") or 0.0)
    outputs = float(state.get("outputs") or 0.0)
    unit = str(state.get("unit") or "").strip() or None
    count = int(state.get("count") or 0)
    ratio = None
    status = "insufficient"
    if count and inputs > 0 and outputs > 0:
        ratio = outputs / inputs
        status = "ok" if 0.8 <= ratio <= 1.2 else "check"
    return {
        "inputs": inputs,
        "outputs": outputs,
        "unit": unit,
        "count": count,
        "ratio": round(ratio, 3) if ratio is not None else None,
        "status": status,
    }


def _merge_balance_status(*statuses: str | None) -> str:
    if any(status == "check" for status in statuses):
        return "check"
    if any(status == "ok" for status in statuses):
        return "ok"
    return "insufficient"


def _default_balance_unit(dimension: str) -> str:
    return "kg" if dimension == "mass" else "MJ"


def _convert_exchange_amount_for_balance(
    amount: float,
    unit: str,
    reference_info: FlowReferenceInfo | None,
) -> tuple[str | None, float | None, str | None]:
    unit_text = str(unit or "").strip()
    if not unit_text:
        return None, None, None
    if reference_info and reference_info.unit_group:
        unit_group = reference_info.unit_group
        dimension = _unit_group_dimension(unit_group)
        if dimension:
            converted = _convert_amount_with_unit_group(amount, unit_text, unit_group)
            unit_label = unit_group.reference_unit
            if converted is None and unit_label:
                converted = _convert_amount_simple(amount, unit_text, unit_label)
            if converted is not None:
                return dimension, converted, unit_label or _default_balance_unit(dimension)
    unit_key = _normalize_unit_token(unit_text)
    entry = _UNIT_DIMENSIONS.get(unit_key)
    if entry:
        dimension = entry[0]
        if dimension not in {"mass", "energy"}:
            return None, None, None
        target_unit = _default_balance_unit(dimension)
        converted = _convert_amount_simple(amount, unit_text, target_unit)
        if converted is not None:
            return dimension, converted, target_unit
    return None, None, None


def _convert_amount_from_unit_group_reference(
    amount_in_reference_unit: float,
    unit: str,
    unit_group: UnitGroupInfo,
) -> float | None:
    unit_key = _normalize_unit_token(unit)
    if not unit_key:
        return None
    factor = unit_group.units.get(unit_key)
    if factor in (None, 0):
        return None
    return amount_in_reference_unit / factor


def _convert_balance_amount_to_resolved_unit(
    amount: float,
    *,
    balance_unit: str | None,
    resolved_unit: str | None,
    reference_info: FlowReferenceInfo | None,
) -> float | None:
    balance_unit_text = str(balance_unit or "").strip()
    resolved_unit_text = str(resolved_unit or "").strip()
    if not resolved_unit_text:
        return None
    if not balance_unit_text:
        balance_unit_text = resolved_unit_text
    if _normalize_unit_token(balance_unit_text) == _normalize_unit_token(resolved_unit_text):
        return amount
    if reference_info and reference_info.unit_group:
        unit_group = reference_info.unit_group
        reference_unit = str(unit_group.reference_unit or "").strip()
        if reference_unit:
            in_reference = amount
            if _normalize_unit_token(balance_unit_text) != _normalize_unit_token(reference_unit):
                converted = _convert_amount_simple(amount, balance_unit_text, reference_unit)
                if converted is None:
                    return None
                in_reference = converted
            by_group = _convert_amount_from_unit_group_reference(in_reference, resolved_unit_text, unit_group)
            if by_group is not None:
                return by_group
    return _convert_amount_simple(amount, balance_unit_text, resolved_unit_text)


def _is_ambiguous_balance_unit(value: str | None) -> bool:
    token = _normalize_unit_token(value)
    if not token:
        return True
    return token in _AMBIGUOUS_BALANCE_UNITS


def _resolve_exchange_balance_unit(
    exchange: dict[str, Any],
    *,
    reference_info: FlowReferenceInfo | None,
    material_role: str | None,
    flow_kind: str | None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    unit_text = str(exchange.get("unit") or "").strip()
    tags = _parse_exchange_comment_tags(exchange.get("generalComment"))
    tag_unit = str(tags.get(_EXCHANGE_UNIT_TAG_NAME) or "").strip()
    normalized_role = _normalize_material_role(material_role)
    normalized_kind = _normalize_flow_type(flow_kind)

    resolved = unit_text
    if _is_ambiguous_balance_unit(resolved) and tag_unit and not _is_ambiguous_balance_unit(tag_unit):
        resolved = tag_unit
        reasons.append("uom_from_comment_tag")

    unit_group = reference_info.unit_group if isinstance(reference_info, FlowReferenceInfo) else None
    dimension = _unit_group_dimension(unit_group)
    reference_unit = unit_group.reference_unit if unit_group else ""
    if _is_ambiguous_balance_unit(resolved):
        # For core material exchanges, when the unit is ambiguous (e.g. "unit"),
        # fall back to the matched flow reference unit to keep mass balance computable.
        if reference_unit and dimension in {"mass", "energy"} and (normalized_role in _BALANCE_CORE_ROLES or normalized_kind in {"product", "waste"}):
            resolved = reference_unit
            reasons.append("assume_flow_reference_unit")

    return resolved, reasons


def _is_core_mass_exchange(
    *,
    material_role: str | None,
    flow_kind: str | None,
    balance_exclude: bool | None,
) -> bool:
    if balance_exclude is True:
        return False
    normalized_role = _normalize_material_role(material_role)
    if normalized_role in _BALANCE_EXCLUDE_ROLES:
        return False
    if normalized_role in {"energy", "emission", "service"}:
        return False
    if normalized_role in _BALANCE_CORE_ROLES:
        return True
    normalized_kind = _normalize_flow_type(flow_kind)
    return normalized_kind in {"product", "waste"}


def _assess_unit_compatibility(
    *,
    unit: str | None,
    amount: float | None,
    reference_info: FlowReferenceInfo | None,
) -> tuple[dict[str, Any], float | None, str | None]:
    unit_text = str(unit or "").strip()
    if not unit_text:
        return (
            {"status": "missing_unit", "reason": "exchange_unit_missing"},
            None,
            None,
        )
    if not reference_info or not reference_info.unit_group:
        return (
            {
                "status": "review",
                "reason": "flow_unit_group_missing",
                "exchange_unit": unit_text,
            },
            None,
            None,
        )

    unit_group = reference_info.unit_group
    reference_unit = unit_group.reference_unit
    unit_key = _normalize_unit_token(unit_text)
    reference_key = _normalize_unit_token(reference_unit)
    exchange_dimension = _unit_dimension_from_unit(unit_text)
    flow_dimension = _unit_group_category(unit_group)
    unit_check: dict[str, Any] = {
        "exchange_unit": unit_text,
        "flow_reference_unit": reference_unit or None,
        "flow_unit_group": unit_group.name,
        "exchange_dimension": exchange_dimension,
        "flow_dimension": flow_dimension,
    }

    if unit_key in unit_group.units:
        if amount is None:
            unit_check["status"] = "ok"
            return unit_check, None, reference_unit
        converted = _convert_amount_with_unit_group(amount, unit_text, unit_group)
        if converted is None:
            unit_check["status"] = "review"
            unit_check["reason"] = "conversion_failed"
            return unit_check, None, reference_unit
        if reference_key and unit_key != reference_key:
            unit_check["status"] = "converted"
        else:
            unit_check["status"] = "ok"
        return unit_check, converted, reference_unit

    if flow_dimension and exchange_dimension and flow_dimension != exchange_dimension:
        unit_check["status"] = "mismatch"
        unit_check["reason"] = f"dimension_mismatch: exchange={exchange_dimension}, flow={flow_dimension}"
        return unit_check, None, reference_unit

    if amount is not None and flow_dimension and exchange_dimension and flow_dimension == exchange_dimension and reference_unit:
        converted = _convert_amount_simple(amount, unit_text, reference_unit)
        if converted is not None:
            unit_check["status"] = "converted"
            unit_check["reason"] = "converted_by_dimension"
            return unit_check, converted, reference_unit

    unit_check["status"] = "review"
    if exchange_dimension is None:
        unit_check["reason"] = "exchange_unit_dimension_unknown"
    elif flow_dimension is None:
        unit_check["reason"] = "flow_unit_group_dimension_unknown"
    else:
        unit_check["reason"] = "unit_not_in_group"
    return unit_check, None, reference_unit


def _normalize_exchange_value_candidates(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    for proc in raw:
        if not isinstance(proc, dict):
            continue
        process_id = str(proc.get("process_id") or proc.get("processId") or "").strip()
        values = proc.get("exchanges") or proc.get("values") or []
        if not isinstance(values, list):
            continue
        cleaned_values: list[dict[str, Any]] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            name = _strip_flow_label(str(item.get("exchangeName") or item.get("exchange_name") or item.get("name") or "").strip())
            if not name:
                continue
            amount = _coerce_amount_text(item.get("amount"))
            unit = str(item.get("unit") or "").strip()
            basis_amount = _coerce_amount_text(item.get("basis_amount") or item.get("basisAmount"))
            basis_unit = str(item.get("basis_unit") or item.get("basisUnit") or "").strip()
            basis_flow = str(item.get("basis_flow") or item.get("basisFlow") or "").strip()
            source_type = _normalize_source_type(item.get("source_type") or item.get("sourceType") or (item.get("data_source") or {}).get("source_type"))
            evidence = item.get("evidence") or item.get("citations") or []
            if isinstance(evidence, str):
                evidence_list = [evidence]
            elif isinstance(evidence, list):
                evidence_list = [str(entry).strip() for entry in evidence if str(entry).strip()]
            else:
                evidence_list = []
            cleaned_values.append(
                {
                    "exchangeName": name,
                    "amount": amount,
                    "unit": unit,
                    "basis_amount": basis_amount,
                    "basis_unit": basis_unit,
                    "basis_flow": basis_flow,
                    "source_type": source_type,
                    "evidence": evidence_list,
                }
            )
        if cleaned_values:
            normalized.append({"process_id": process_id, "exchanges": cleaned_values})
    return normalized


def _apply_exchange_value_candidates(
    process_exchanges: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not candidates:
        return process_exchanges
    candidate_map: dict[tuple[str, str], dict[str, Any]] = {}
    for proc in candidates:
        if not isinstance(proc, dict):
            continue
        process_id = str(proc.get("process_id") or "").strip()
        for item in proc.get("exchanges") or []:
            if not isinstance(item, dict):
                continue
            name = _normalize_exchange_name(item.get("exchangeName"))
            if not name:
                continue
            key = (process_id, name)
            if key not in candidate_map:
                candidate_map[key] = item

    updated: list[dict[str, Any]] = []
    for proc in process_exchanges:
        if not isinstance(proc, dict):
            continue
        process_id = str(proc.get("process_id") or proc.get("processId") or "").strip()
        exchanges = proc.get("exchanges") or []
        cleaned: list[dict[str, Any]] = []
        for exchange in exchanges:
            if not isinstance(exchange, dict):
                continue
            name = _normalize_exchange_name(exchange.get("exchangeName"))
            candidate = candidate_map.get((process_id, name))
            if candidate:
                amount = candidate.get("amount")
                unit = candidate.get("unit")
                basis_amount = candidate.get("basis_amount")
                basis_unit = candidate.get("basis_unit")
                basis_flow = candidate.get("basis_flow")
                source_type = candidate.get("source_type")
                evidence = candidate.get("evidence") or []
                value_citations, value_evidence = _normalize_citations_and_evidence([], _clean_evidence_list(evidence))
                if amount:
                    exchange["amount"] = amount
                if unit:
                    exchange["unit"] = unit
                if basis_amount:
                    exchange["basis_amount"] = basis_amount
                if basis_unit:
                    exchange["basis_unit"] = basis_unit
                if basis_flow:
                    exchange["basis_flow"] = basis_flow
                existing_ds = exchange.get("data_source") if isinstance(exchange.get("data_source"), dict) else {}
                existing_evidence = _clean_evidence_list(exchange.get("evidence"))
                citations, evidence = _normalize_citations_and_evidence(
                    existing_ds.get("citations"),
                    existing_evidence + _clean_evidence_list(evidence),
                )
                data_source = dict(existing_ds)
                if source_type:
                    data_source["source_type"] = source_type
                if citations:
                    data_source["citations"] = citations
                else:
                    data_source.pop("citations", None)
                exchange["data_source"] = data_source
                exchange["evidence"] = evidence
                _merge_value_evidence(exchange, value_citations, value_evidence)
            _ensure_exchange_comment_tags(exchange)
            cleaned.append(exchange)
        updated.append({"process_id": process_id, "exchanges": cleaned})
    return updated


def _parse_quantitative_reference(value: str | None) -> tuple[float | None, str | None, str | None]:
    if not value:
        return None, None, None
    cleaned = re.sub(r"\s+", " ", str(value).strip())
    if not cleaned:
        return None, None, None
    match = _QUANT_REF_PATTERN.match(cleaned)
    if match:
        amount = _parse_amount_value(match.group("amount"))
        unit = match.group("unit").strip()
        flow = match.group("flow").strip()
        return amount, unit or None, flow or None
    parts = cleaned.split(" ", 2)
    if len(parts) < 2:
        return None, None, None
    amount = _parse_amount_value(parts[0])
    unit = parts[1].strip()
    flow = parts[2].strip() if len(parts) > 2 else ""
    return amount, unit or None, flow or None


def _flow_name_matches(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    left_norm = _normalize_exchange_name(left)
    right_norm = _normalize_exchange_name(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    return left_norm in right_norm or right_norm in left_norm


def _reference_basis_for_plan(
    plan: dict[str, Any] | None,
    exchanges: list[dict[str, Any]] | None,
) -> tuple[float | None, str | None, str | None]:
    name_parts = plan.get("name_parts") if isinstance((plan or {}).get("name_parts"), dict) else {}
    quantitative_reference = str(name_parts.get("quantitative_reference") or "").strip()
    amount, unit, flow = _parse_quantitative_reference(quantitative_reference)
    if not flow:
        flow = str((plan or {}).get("reference_flow_name") or "").strip() or None
    if amount is None or not unit:
        for exchange in exchanges or []:
            if not isinstance(exchange, dict):
                continue
            if not bool(exchange.get("is_reference_flow")):
                continue
            amount = _parse_amount_value(exchange.get("amount"))
            unit = str(exchange.get("unit") or "").strip() or unit
            if not flow:
                flow = str(exchange.get("exchangeName") or "").strip() or None
            break
    return amount, unit, flow


def _is_exchange_adjustable(exchange: dict[str, Any]) -> bool:
    data_source = exchange.get("data_source")
    if isinstance(data_source, dict):
        source_type = _normalize_source_type(data_source.get("source_type") or data_source.get("sourceType"))
        citations = _clean_evidence_list(data_source.get("citations"))
    else:
        source_type = _normalize_source_type(exchange.get("dataSource") or data_source or "")
        citations = []
    return source_type == "expert_judgement" or not citations


def _scale_exchange_amounts(
    process_exchanges: list[dict[str, Any]],
    process_plans: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    for proc in process_exchanges:
        if not isinstance(proc, dict):
            continue
        process_id = str(proc.get("process_id") or proc.get("processId") or "").strip()
        exchanges = proc.get("exchanges") or []
        if not isinstance(exchanges, list):
            continue
        plan = process_plans.get(process_id) if process_id else None
        ref_amount, ref_unit, ref_flow = _reference_basis_for_plan(plan, exchanges)
        if ref_amount is None or not ref_unit:
            continue
        for exchange in exchanges:
            if not isinstance(exchange, dict):
                continue
            if not _is_exchange_adjustable(exchange):
                continue
            amount_value = _parse_amount_value(exchange.get("amount"))
            if amount_value is None:
                continue
            basis_amount = _parse_amount_value(exchange.get("basis_amount"))
            if basis_amount is None:
                basis_amount = 1.0
            basis_unit = str(exchange.get("basis_unit") or ref_unit or "").strip()
            basis_flow = str(exchange.get("basis_flow") or ref_flow or "").strip()
            if basis_flow and ref_flow and not _flow_name_matches(basis_flow, ref_flow):
                continue
            if not basis_unit:
                continue
            basis_amount_in_ref = _convert_amount_simple(basis_amount, basis_unit, ref_unit)
            if basis_amount_in_ref is None:
                if _normalize_unit_token(basis_unit) != _normalize_unit_token(ref_unit):
                    continue
                basis_amount_in_ref = basis_amount
            if basis_amount_in_ref == 0:
                continue
            scale_factor = ref_amount / basis_amount_in_ref
            if abs(scale_factor - 1.0) < 1.0e-9:
                continue
            exchange["amount"] = _format_amount_value(amount_value * scale_factor)
            evidence = _clean_evidence_list(exchange.get("evidence"))
            note_flow = f" of {ref_flow}" if ref_flow else ""
            note = f"Scaled to reference flow {ref_amount:g} {ref_unit}{note_flow} " f"(basis {basis_amount:g} {basis_unit})."
            if note not in evidence:
                evidence.append(note)
                exchange["evidence"] = evidence
    return process_exchanges


def _normalize_steps(value: Any) -> list[str]:
    if not value:
        return []
    raw = value if isinstance(value, list) else [value]
    cleaned: list[str] = []
    for item in raw:
        text = str(item).strip().lower().replace(" ", "").replace("-", "")
        if text in {"step1", "s1", "1"}:
            cleaned.append("step1")
        elif text in {"step2", "s2", "2"}:
            cleaned.append("step2")
        elif text in {"step3", "s3", "3"}:
            cleaned.append("step3")
    return sorted(set(cleaned))


def _usage_tags_from_supported_steps(supported_steps: list[str], decision: str | None = None) -> list[str]:
    tags: list[str] = []
    normalized = {item.strip().lower() for item in supported_steps if str(item).strip()}
    if "step1" in normalized:
        tags.append("tech_route")
    if "step2" in normalized:
        tags.append("process_split")
    if "step3" in normalized:
        tags.append("exchange_values")
    if not tags and decision and str(decision).strip().lower() == "unusable":
        tags.append("background_only")
    return tags


_USAGE_TAGS_USED = {"tech_route", "process_split", "exchange_values"}
_USAGE_TAGS_ALL = _USAGE_TAGS_USED | {"background_only"}


def _normalize_usage_tags(value: Any) -> list[str]:
    if not value:
        return []
    items = value if isinstance(value, list) else [value]
    tags: list[str] = []
    for item in items:
        text = str(item).strip().lower()
        if text in _USAGE_TAGS_ALL and text not in tags:
            tags.append(text)
    if "background_only" in tags and len(tags) > 1:
        tags = [tag for tag in tags if tag != "background_only"]
    return tags


def _collect_usage_tag_map(scientific_references: dict[str, Any]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    doi_map: dict[str, set[str]] = {}
    key_map: dict[str, set[str]] = {}
    tagged_dois: set[str] = set()

    def add_tags(*, doi: str | None, key: str | None, tags: list[str]) -> None:
        if not tags:
            return
        if doi:
            doi_map.setdefault(doi, set()).update(tags)
        if key:
            key_map.setdefault(key, set()).update(tags)

    usage_tagging = scientific_references.get("usage_tagging")
    results = usage_tagging.get("results") if isinstance(usage_tagging, dict) else None
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            doi = _normalize_doi(item.get("doi"))
            key = str(item.get("key") or "").strip() or None
            tags = _normalize_usage_tags(item.get("usage_tags") or item.get("usageTags"))
            add_tags(doi=doi or None, key=key, tags=tags)
            if doi:
                tagged_dois.add(doi)

    clusters = scientific_references.get(REFERENCE_CLUSTERS_KEY)
    summaries = clusters.get("reference_summaries") if isinstance(clusters, dict) else None
    if isinstance(summaries, list):
        for summary in summaries:
            if not isinstance(summary, dict):
                continue
            doi = _normalize_doi(summary.get("doi"))
            if doi and doi in tagged_dois:
                continue
            tags = _normalize_usage_tags(summary.get("usage_tags") or summary.get("usageTags"))
            add_tags(doi=doi or None, key=None, tags=tags)

    usability = scientific_references.get("usability")
    usability_results = usability.get("results") if isinstance(usability, dict) else None
    if isinstance(usability_results, list):
        for item in usability_results:
            if not isinstance(item, dict):
                continue
            doi = _normalize_doi(item.get("doi"))
            if doi and doi in tagged_dois:
                continue
            supported_steps = _normalize_steps(item.get("supported_steps"))
            decision = item.get("decision")
            tags = _usage_tags_from_supported_steps(supported_steps, decision)
            add_tags(doi=doi or None, key=None, tags=tags)

    industry_block = scientific_references.get(INDUSTRY_AVERAGE_KEY)
    industry_refs = industry_block.get("references") if isinstance(industry_block, dict) else None
    if isinstance(industry_refs, list):
        for ref in industry_refs:
            if not isinstance(ref, dict):
                continue
            info = _reference_info_from_record(ref, origin_step=INDUSTRY_AVERAGE_KEY)
            if not info:
                continue
            doi = _normalize_doi(info.get("doi"))
            key = str(info.get("key") or "").strip() or None
            add_tags(doi=doi or None, key=key, tags=["exchange_values"])

    return doi_map, key_map


def _reference_is_used(info: dict[str, Any], doi_map: dict[str, set[str]], key_map: dict[str, set[str]]) -> bool:
    if not doi_map and not key_map:
        return True
    doi = _normalize_doi(info.get("doi"))
    key = str(info.get("key") or "").strip()
    tags: set[str] = set()
    if doi:
        tags.update(doi_map.get(doi, set()))
    if key:
        tags.update(key_map.get(key, set()))
    if not tags:
        return False
    return any(tag in _USAGE_TAGS_USED for tag in tags)


def _filter_reference_infos_by_usage(
    infos: list[dict[str, Any]],
    *,
    doi_map: dict[str, set[str]],
    key_map: dict[str, set[str]],
) -> list[dict[str, Any]]:
    if not doi_map and not key_map:
        return infos
    return [info for info in infos if _reference_is_used(info, doi_map, key_map)]


def _clean_evidence_list(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    return cleaned


def _split_citations_from_evidence(evidence: list[str]) -> tuple[list[str], list[str]]:
    citations: list[str] = []
    remaining: list[str] = []
    for item in evidence:
        text = str(item).strip()
        if not text:
            continue
        doi = _extract_doi_from_text(text)
        if doi:
            citations.append(f"DOI {doi}")
            continue
        url = _extract_reference_url(text)
        if url:
            citations.append(url)
            continue
        remaining.append(text)
    return _dedupe_flows(citations), _dedupe_flows(remaining)


def _normalize_citations_and_evidence(citations: Any, evidence: Any) -> tuple[list[str], list[str]]:
    citation_list = _clean_evidence_list(citations)
    evidence_list = _clean_evidence_list(evidence)
    citations_from_citations, remaining_from_citations = _split_citations_from_evidence(citation_list)
    citations_from_evidence, remaining_from_evidence = _split_citations_from_evidence(evidence_list)
    merged_citations = _dedupe_flows(citations_from_citations + citations_from_evidence)
    merged_evidence = _dedupe_flows(remaining_from_citations + remaining_from_evidence)
    return merged_citations, merged_evidence


def _format_evidence_for_comment(evidence: list[str]) -> str:
    if not evidence:
        return ""
    compacted = [_compact_text(item, limit=220) for item in evidence if str(item).strip()]
    if not compacted:
        return ""
    return f"Evidence: {' | '.join(compacted)}"


def _sanitize_exchange_comment_tag_value(value: Any, *, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        text = fallback
    text = re.sub(r"\s+", "_", text)
    text = text.replace("[", "").replace("]", "")
    return text or fallback


def _exchange_comment_tag_block(*, flow_kind: Any, unit: Any) -> str:
    normalized_kind = _normalize_flow_type(flow_kind) or str(flow_kind or "").strip().lower() or "unknown"
    kind_tag = _sanitize_exchange_comment_tag_value(normalized_kind, fallback="unknown")
    unit_tag = _sanitize_exchange_comment_tag_value(unit, fallback="unit")
    return f"[{_EXCHANGE_KIND_TAG_NAME}={kind_tag}] [{_EXCHANGE_UNIT_TAG_NAME}={unit_tag}]"


def _parse_exchange_comment_tags(comment: Any) -> dict[str, str]:
    text = str(comment).strip() if comment is not None else ""
    if not text:
        return {}
    parsed: dict[str, str] = {}
    for match in _EXCHANGE_COMMENT_TAG_CAPTURE_PATTERN.finditer(text):
        key = str(match.group("name") or "").strip().lower()
        value = str(match.group("value") or "").strip()
        if not key:
            continue
        parsed[key] = value
    return parsed


def _strip_exchange_comment_tags(comment: Any) -> str:
    text = str(comment).strip() if comment is not None else ""
    if not text:
        return ""
    stripped = _EXCHANGE_COMMENT_TAG_PATTERN.sub("", text)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    stripped = stripped.strip("|;").strip()
    return stripped


def _apply_exchange_comment_tags(comment: Any, *, flow_kind: Any, unit: Any) -> str:
    base = _strip_exchange_comment_tags(comment)
    tags = _exchange_comment_tag_block(flow_kind=flow_kind, unit=unit)
    if base:
        return f"{base} {tags}".strip()
    return tags


def _resolve_exchange_comment_tag_unit(
    exchange: dict[str, Any],
    *,
    reference_info: FlowReferenceInfo | None = None,
    fallback_unit: str | None = None,
) -> str:
    if isinstance(reference_info, FlowReferenceInfo):
        unit_group = reference_info.unit_group
        if isinstance(unit_group, UnitGroupInfo):
            ref_unit = str(unit_group.reference_unit or "").strip()
            if ref_unit:
                return ref_unit
    flow_search = exchange.get("flow_search") if isinstance(exchange.get("flow_search"), dict) else {}
    unit_check = flow_search.get("unit_check") if isinstance(flow_search.get("unit_check"), dict) else {}
    for key in ("flow_reference_unit", "converted_unit"):
        candidate = str(unit_check.get(key) or "").strip()
        if candidate:
            return candidate
    fallback = str(fallback_unit or "").strip()
    if fallback:
        return fallback
    unit = str(exchange.get("unit") or "").strip()
    if unit:
        return unit
    tags = _parse_exchange_comment_tags(exchange.get("generalComment"))
    tag_unit = str(tags.get(_EXCHANGE_UNIT_TAG_NAME) or "").strip()
    if tag_unit:
        return tag_unit
    return "unit"


def _ensure_exchange_comment_tags(exchange: dict[str, Any], *, preferred_unit: str | None = None) -> None:
    direction = str(exchange.get("exchangeDirection") or "").strip()
    is_reference = bool(exchange.get("is_reference_flow"))
    flow_kind = _normalize_flow_type(exchange.get("flow_type") or exchange.get("flowType"))
    if not flow_kind:
        flow_kind = _infer_flow_type(
            str(exchange.get("exchangeName") or ""),
            direction=direction,
            is_reference_flow=is_reference,
        )
    unit = str(preferred_unit or "").strip()
    if not unit:
        unit = _resolve_exchange_comment_tag_unit(exchange)
    exchange["generalComment"] = _apply_exchange_comment_tags(
        exchange.get("generalComment"),
        flow_kind=flow_kind,
        unit=unit,
    )


def _parse_industry_average_response(data: dict[str, Any]) -> tuple[str | None, str | None, list[str], str | None]:
    amount = _coerce_amount_text(data.get("amount") or data.get("mean_amount") or data.get("value"))
    unit = str(data.get("unit") or "").strip() or None
    evidence = _clean_evidence_list(data.get("evidence"))
    notes = str(data.get("notes") or "").strip() or None
    return amount, unit, evidence, notes


def _parse_density_estimate_response(data: dict[str, Any]) -> tuple[float | None, str | None, str | None, str, str | None]:
    density_value = _parse_amount_value(data.get("density_value") or data.get("densityValue"))
    density_unit = str(data.get("density_unit") or data.get("densityUnit") or "").strip() or None
    assumptions = str(data.get("assumptions") or "").strip() or None
    notes = str(data.get("notes") or "").strip() or None
    source_type = _normalize_source_type(data.get("source_type") or data.get("sourceType")) or "expert_judgement"
    if density_value is None or density_value <= 0 or not density_unit:
        return None, None, assumptions, source_type, notes
    return density_value, density_unit, assumptions, source_type, notes


def _merge_industry_average_block(
    scientific_references: dict[str, Any],
    *,
    query_entry: dict[str, Any] | None,
    references: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    updated = dict(scientific_references)
    block = dict(updated.get(INDUSTRY_AVERAGE_KEY) or {})
    existing_refs = block.get("references")
    if not isinstance(existing_refs, list):
        existing_refs = []
    new_refs = references or []
    if new_refs:
        block["references"] = _merge_reference_records(existing_refs, new_refs)
    else:
        block["references"] = existing_refs
    if query_entry:
        queries = block.get("queries")
        if not isinstance(queries, list):
            queries = []
        queries.append(query_entry)
        block["queries"] = queries
    updated[INDUSTRY_AVERAGE_KEY] = block
    return updated


def _normalize_source_type(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text in {"literature", "paper", "reference", "journal"}:
        return "literature"
    if text in {"si", "supplement", "supplementary", "supporting", "appendix"}:
        return "si"
    if text in {"expert_judgement", "expert judgement", "expert", "common_sense"}:
        return "expert_judgement"
    return text


def _infer_source_type_from_evidence(evidence: list[str]) -> str:
    for item in evidence:
        lower = item.lower()
        if "supplement" in lower or "supporting" in lower or "appendix" in lower or "si " in lower:
            return "si"
    return "literature"


def _exchange_has_evidence(exchange: dict[str, Any]) -> bool:
    evidence = _clean_evidence_list(exchange.get("evidence"))
    if evidence:
        return True
    data_source = exchange.get("data_source") or exchange.get("dataSource")
    if isinstance(data_source, dict):
        citations = _clean_evidence_list(data_source.get("citations"))
        if citations:
            return True
        source_type = _normalize_source_type(data_source.get("source_type") or data_source.get("sourceType"))
        return bool(source_type and source_type != "expert_judgement")
    if isinstance(data_source, str):
        return _normalize_source_type(data_source) not in {"", "expert_judgement"}
    return False


def _apply_exchange_evidence_defaults(
    exchange: dict[str, Any],
    *,
    use_references: bool,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    data_source = exchange.get("data_source")
    if isinstance(data_source, dict):
        data_source = dict(data_source)
    elif isinstance(data_source, str):
        data_source = {"source_type": data_source}
    else:
        legacy = exchange.get("dataSource")
        if isinstance(legacy, dict):
            data_source = dict(legacy)
        elif isinstance(legacy, str):
            data_source = {"source_type": legacy}
        else:
            data_source = {}
    evidence = _clean_evidence_list(exchange.get("evidence"))
    citations, evidence = _normalize_citations_and_evidence(data_source.get("citations"), evidence)
    source_type = _normalize_source_type(data_source.get("source_type") or data_source.get("sourceType"))
    if not source_type:
        if evidence:
            source_type = _infer_source_type_from_evidence(evidence)
        else:
            source_type = "expert_judgement" if not use_references else "expert_judgement"
    data_source = {**data_source, "source_type": source_type}
    if citations:
        data_source["citations"] = citations
    else:
        data_source.pop("citations", None)
    if source_type == "expert_judgement" and fallback_reason:
        if not data_source.get("reason"):
            data_source["reason"] = fallback_reason
        if not evidence:
            evidence = [fallback_reason]
    exchange["data_source"] = data_source
    exchange["evidence"] = evidence
    if "dataSource" in exchange:
        exchange.pop("dataSource")
    return exchange


def _merge_value_evidence(exchange: dict[str, Any], citations: list[str], evidence: list[str]) -> None:
    if not citations and not evidence:
        return
    merged_citations = _dedupe_flows(_clean_evidence_list(exchange.get("value_citations")) + citations)
    merged_evidence = _dedupe_flows(_clean_evidence_list(exchange.get("value_evidence")) + evidence)
    if merged_citations:
        exchange["value_citations"] = merged_citations
    if merged_evidence:
        exchange["value_evidence"] = merged_evidence
    if merged_citations or merged_evidence:
        exchange["value_reference_used"] = True


def _is_energy_exchange(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in _ENERGY_KEYWORDS)


def _exchange_direction_for_dedupe(exchange: dict[str, Any], *, reference_direction: str | None) -> str:
    direction = str(exchange.get("exchangeDirection") or "").strip()
    if direction not in {"Input", "Output"}:
        direction = "Input"
    if bool(exchange.get("is_reference_flow")) and reference_direction:
        direction = reference_direction
    return direction


def _exchange_flow_type_for_dedupe(exchange: dict[str, Any], *, direction: str) -> str:
    raw_flow_type = _normalize_flow_type(exchange.get("flow_type") or exchange.get("flowType"))
    if raw_flow_type:
        return raw_flow_type
    flow_search = exchange.get("flow_search")
    if isinstance(flow_search, dict):
        selected_uuid = flow_search.get("selected_uuid")
        candidates = flow_search.get("candidates")
        if isinstance(candidates, list) and selected_uuid:
            for cand in candidates:
                if isinstance(cand, dict) and cand.get("uuid") == selected_uuid:
                    cand_flow_type = _normalize_flow_type(cand.get("flow_type") or cand.get("flowType"))
                    if cand_flow_type:
                        return cand_flow_type
    name = str(exchange.get("exchangeName") or "").strip()
    return _infer_flow_type(name, direction=direction, is_reference_flow=bool(exchange.get("is_reference_flow")))


def _exchange_uuid_for_dedupe(exchange: dict[str, Any]) -> str | None:
    flow_search = exchange.get("flow_search")
    if isinstance(flow_search, dict):
        uuid_value = flow_search.get("selected_uuid")
        if isinstance(uuid_value, str) and uuid_value.strip():
            return uuid_value.strip()
    return None


def _score_product_exchange(exchange: dict[str, Any]) -> tuple[int, int, int]:
    is_reference = 1 if bool(exchange.get("is_reference_flow")) else 0
    has_evidence = 1 if _has_strong_evidence(exchange) else 0
    has_amount = 1 if _parse_amount_value(exchange.get("amount")) is not None else 0
    return (is_reference, has_evidence, has_amount)


def _dedupe_product_uuid_exchanges(
    exchanges: list[dict[str, Any]],
    *,
    reference_direction: str | None,
) -> list[dict[str, Any]]:
    if not exchanges:
        return exchanges
    seen: dict[tuple[str, str], tuple[dict[str, Any], int]] = {}
    deduped: list[dict[str, Any]] = []
    for exchange in exchanges:
        if not isinstance(exchange, dict):
            continue
        direction = _exchange_direction_for_dedupe(exchange, reference_direction=reference_direction)
        flow_type = _exchange_flow_type_for_dedupe(exchange, direction=direction)
        if flow_type != "product":
            deduped.append(exchange)
            continue
        uuid_value = _exchange_uuid_for_dedupe(exchange)
        if not uuid_value:
            deduped.append(exchange)
            continue
        key = (direction, uuid_value)
        existing = seen.get(key)
        if not existing:
            seen[key] = (exchange, len(deduped))
            deduped.append(exchange)
            continue
        existing_exchange, index = existing
        if _score_product_exchange(exchange) > _score_product_exchange(existing_exchange):
            deduped[index] = exchange
            seen[key] = (exchange, index)
    return deduped


def _has_strong_evidence(exchange: dict[str, Any]) -> bool:
    data_source = exchange.get("data_source") or exchange.get("dataSource")
    if isinstance(data_source, dict):
        source_type = _normalize_source_type(data_source.get("source_type") or data_source.get("sourceType"))
        citations = _clean_evidence_list(data_source.get("citations"))
        if citations:
            return True
        if source_type in {"literature", "si"}:
            return True
    value_evidence = _clean_evidence_list(exchange.get("value_citations")) + _clean_evidence_list(exchange.get("value_evidence"))
    return bool(value_evidence)


def _dedupe_reference_exchanges(
    exchanges: list[dict[str, Any]],
    *,
    reference_flow: str | None,
    reference_direction: str | None,
) -> list[dict[str, Any]]:
    if not exchanges or not reference_flow:
        return exchanges
    candidates = [item for item in exchanges if _flow_name_matches(str(item.get("exchangeName") or ""), reference_flow)]
    if not candidates:
        return exchanges
    for item in candidates:
        if reference_direction:
            item["exchangeDirection"] = reference_direction
        item["is_reference_flow"] = True
    if len(candidates) == 1:
        return exchanges

    def score(item: dict[str, Any]) -> tuple[int, int]:
        has_evidence = 1 if _has_strong_evidence(item) else 0
        has_amount = 1 if _parse_amount_value(item.get("amount")) is not None else 0
        return (has_evidence, has_amount)

    primary = max(candidates, key=score)
    return [item for item in exchanges if (item is primary) or (item not in candidates)]


def _should_keep_exchange(
    exchange: dict[str, Any],
    *,
    structure_inputs: set[str],
    structure_outputs: set[str],
    reference_flow: str | None,
) -> bool:
    name = str(exchange.get("exchangeName") or "").strip()
    if reference_flow and _flow_name_matches(name, reference_flow):
        return True
    flow_type = str(exchange.get("flow_type") or "").strip().lower()
    if flow_type == "elementary":
        return True
    if flow_type == "service":
        return True
    if _is_energy_exchange(name):
        return True
    direction = str(exchange.get("exchangeDirection") or "").strip()
    name_key = _normalize_exchange_name(name)
    if direction == "Input" and not structure_inputs:
        return True
    if direction == "Output" and not structure_outputs:
        return True
    if direction == "Input" and name_key in structure_inputs:
        return True
    if direction == "Output" and name_key in structure_outputs:
        return True
    if _has_strong_evidence(exchange):
        return True
    return False


def _is_key_exchange(exchange: dict[str, Any]) -> bool:
    flag = exchange.get("is_key_exchange")
    if isinstance(flag, bool):
        return flag
    flag = exchange.get("isKeyExchange")
    if isinstance(flag, bool):
        return flag
    if bool(exchange.get("is_reference_flow")):
        return True
    flow_type = str(exchange.get("flow_type") or "").strip().lower()
    direction = str(exchange.get("exchangeDirection") or "").strip().lower()
    name = str(exchange.get("exchangeName") or "").strip().lower()
    if flow_type == "elementary":
        return True
    if direction == "input" and any(keyword in name for keyword in _ENERGY_KEYWORDS):
        return True
    return False


def _compute_coverage_metrics(process_exchanges: list[dict[str, Any]]) -> dict[str, Any]:
    total_processes = 0
    covered_processes = 0
    key_total = 0
    key_covered = 0
    total_exchanges = 0
    total_covered = 0
    for proc in process_exchanges:
        if not isinstance(proc, dict):
            continue
        exchanges = proc.get("exchanges") or []
        if not isinstance(exchanges, list):
            exchanges = []
        total_processes += 1
        process_has_evidence = False
        for exchange in exchanges:
            if not isinstance(exchange, dict):
                continue
            total_exchanges += 1
            has_evidence = _exchange_has_evidence(exchange)
            if has_evidence:
                total_covered += 1
                process_has_evidence = True
            if _is_key_exchange(exchange):
                key_total += 1
                if has_evidence:
                    key_covered += 1
        if process_has_evidence:
            covered_processes += 1
    if key_total == 0 and total_exchanges:
        key_total = total_exchanges
        key_covered = total_covered
    process_coverage = covered_processes / total_processes if total_processes else 0.0
    exchange_value_coverage = key_covered / key_total if key_total else 0.0
    return {
        "process_total": total_processes,
        "process_covered": covered_processes,
        "process_coverage": round(process_coverage, 3),
        "key_exchange_total": key_total,
        "key_exchange_covered": key_covered,
        "exchange_value_coverage": round(exchange_value_coverage, 3),
    }


def _append_coverage_history(state: ProcessFromFlowState, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    history = state.get("coverage_history")
    if not isinstance(history, list):
        history = []
    entry = dict(metrics)
    entry["evaluated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    history.append(entry)
    return history


def _should_fallback_to_expert(scientific_references: dict[str, Any] | None) -> bool:
    if not isinstance(scientific_references, dict):
        return False
    usability = scientific_references.get("usability")
    results = usability.get("results") if isinstance(usability, dict) else None
    if not isinstance(results, list) or not results:
        return False
    decisions = [str(item.get("decision") or "").strip().lower() for item in results if isinstance(item, dict)]
    hints = [str(item.get("si_hint") or "").strip().lower() for item in results if isinstance(item, dict)]
    if decisions and all(item == "unusable" for item in decisions) and hints and all(item == "none" for item in hints):
        return True
    return False


def _evaluate_stop_rules(state: ProcessFromFlowState, metrics: dict[str, Any]) -> dict[str, Any]:
    process_coverage = float(metrics.get("process_coverage") or 0.0)
    exchange_coverage = float(metrics.get("exchange_value_coverage") or 0.0)
    history = state.get("coverage_history")
    delta = None
    if isinstance(history, list) and history:
        prev = history[-1] if isinstance(history[-1], dict) else {}
        prev_process = float(prev.get("process_coverage") or 0.0)
        prev_exchange = float(prev.get("exchange_value_coverage") or 0.0)
        delta = max(process_coverage, exchange_coverage) - max(prev_process, prev_exchange)
    fallback_to_expert = _should_fallback_to_expert(state.get("scientific_references"))
    if fallback_to_expert:
        return {
            "should_stop": True,
            "action": "expert_judgement",
            "reason": "no_usable_references_and_no_si_hint",
            "process_coverage": process_coverage,
            "exchange_value_coverage": exchange_coverage,
            "coverage_delta": delta,
        }
    if process_coverage >= STOP_RULE_PROCESS_COVERAGE and exchange_coverage >= STOP_RULE_EXCHANGE_COVERAGE:
        return {
            "should_stop": True,
            "action": "stop_retrieval",
            "reason": "coverage_threshold_met",
            "process_coverage": process_coverage,
            "exchange_value_coverage": exchange_coverage,
            "coverage_delta": delta,
        }
    if delta is not None and delta < STOP_RULE_MIN_DELTA:
        return {
            "should_stop": True,
            "action": "stop_retrieval",
            "reason": "coverage_delta_below_threshold",
            "process_coverage": process_coverage,
            "exchange_value_coverage": exchange_coverage,
            "coverage_delta": delta,
        }
    return {
        "should_stop": False,
        "action": "continue_retrieval",
        "reason": "coverage_below_threshold",
        "process_coverage": process_coverage,
        "exchange_value_coverage": exchange_coverage,
        "coverage_delta": delta,
    }


def _cluster_scientific_references(
    *,
    llm: LanguageModelProtocol,
    state: ProcessFromFlowState,
    fulltext_entries: list[dict[str, Any]],
    flow_summary: dict[str, Any],
    operation: str,
) -> dict[str, Any] | None:
    summaries = _build_reference_cluster_summaries(
        fulltext_entries,
        usability_map=_reference_usability_map(state),
    )
    summaries = [item for item in summaries if item.get("doi")]
    if not summaries:
        return None

    payload = {
        "prompt": REFERENCE_CLUSTER_PROMPT,
        "context": {
            "operation": operation,
            "flow": flow_summary,
            "reference_summaries": summaries,
        },
        "response_format": {"type": "json_object"},
    }
    raw = llm.invoke(payload)
    data = _ensure_dict(raw)

    clusters_raw = data.get("clusters")
    clusters: list[dict[str, Any]] = []
    if isinstance(clusters_raw, list):
        for idx, cluster in enumerate(clusters_raw, start=1):
            if not isinstance(cluster, dict):
                continue
            cluster_id = str(cluster.get("cluster_id") or f"C{idx}").strip()
            dois = [str(item).strip() for item in (cluster.get("dois") or []) if str(item).strip()]
            if not dois:
                continue
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "dois": dois,
                    "system_boundary": str(cluster.get("system_boundary") or "unspecified").strip(),
                    "granularity": str(cluster.get("granularity") or "unknown").strip(),
                    "key_process_chain": [str(item).strip() for item in (cluster.get("key_process_chain") or []) if str(item).strip()],
                    "key_intermediate_flows": [str(item).strip() for item in (cluster.get("key_intermediate_flows") or []) if str(item).strip()],
                    "supported_steps": _normalize_steps(cluster.get("supported_steps")),
                    "recommendation": str(cluster.get("recommendation") or "supplement").strip(),
                    "reason": str(cluster.get("reason") or "").strip(),
                }
            )

    if not clusters:
        all_dois = [item.get("doi") for item in summaries if item.get("doi")]
        clusters = [
            {
                "cluster_id": "C1",
                "dois": all_dois,
                "system_boundary": "unspecified",
                "granularity": "unknown",
                "key_process_chain": [],
                "key_intermediate_flows": [],
                "supported_steps": [],
                "recommendation": "primary",
                "reason": "Fallback: unable to cluster references reliably.",
            }
        ]

    primary_cluster_id = str(data.get("primary_cluster_id") or clusters[0]["cluster_id"]).strip()
    selection_guidance = str(data.get("selection_guidance") or "").strip()

    return {
        "source_step": REFERENCE_FULLTEXT_KEY,
        "input_dois": [item.get("doi") for item in summaries if item.get("doi")],
        "reference_summaries": summaries,
        "clusters": clusters,
        "primary_cluster_id": primary_cluster_id,
        "selection_guidance": selection_guidance,
    }


def _classification_terms(classification: list[dict[str, Any]] | None, *, max_items: int = 3) -> str:
    if not classification:
        return ""
    texts = [str(item.get("#text") or "").strip() for item in classification if isinstance(item, dict)]
    texts = [item for item in texts if item]
    if not texts:
        return ""
    if len(texts) > max_items:
        texts = texts[-max_items:]
    return " ".join(texts)


def _flow_reference_context(flow_summary: dict[str, Any], *, include_comment: bool = True) -> str:
    treatment = _first_nonempty(flow_summary.get("treatment_en"), flow_summary.get("treatment_zh"))
    mix = _first_nonempty(flow_summary.get("mix_en"), flow_summary.get("mix_zh"))
    general = ""
    if include_comment:
        general = _first_nonempty(flow_summary.get("general_comment_en"), flow_summary.get("general_comment_zh"))
    classification = _classification_terms(flow_summary.get("classification"))
    parts = [treatment, mix, general, classification]
    cleaned = [_compact_text(item, limit=160) for item in parts if item]
    return " ".join(cleaned)


def _compose_industry_average_query(
    *,
    exchange: dict[str, Any],
    process_plan: dict[str, Any] | None,
    flow_summary: dict[str, Any],
    operation: str,
) -> str:
    exchange_name = str(exchange.get("exchangeName") or "").strip()
    unit = str(exchange.get("unit") or "").strip() or "unit"
    process_name = str((process_plan or {}).get("name") or "").strip()
    reference_flow = str((process_plan or {}).get("reference_flow_name") or "").strip()
    name_parts = (process_plan or {}).get("name_parts") if isinstance((process_plan or {}).get("name_parts"), dict) else {}
    quantitative_reference = str(name_parts.get("quantitative_reference") or "").strip()
    mix_location = str(name_parts.get("mix_and_location") or "").strip()
    structure = (process_plan or {}).get("structure") if isinstance((process_plan or {}).get("structure"), dict) else {}
    boundary = str(structure.get("boundary") or "").strip()
    base_flow = _first_nonempty(flow_summary.get("base_name_en"), flow_summary.get("base_name_zh"))
    flow_context = _flow_reference_context(flow_summary, include_comment=False)
    parts = [
        exchange_name,
        unit,
        "industry average",
        "LCI",
        base_flow,
        operation,
        process_name,
        reference_flow,
        quantitative_reference,
        mix_location,
        boundary,
        flow_context,
    ]
    cleaned = [item for item in (part.strip() for part in parts) if item]
    return " ".join(cleaned)


def _update_scientific_references(
    state: ProcessFromFlowState,
    *,
    step: str,
    query: str,
    references: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing = state.get("scientific_references")
    updated: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    payload: dict[str, Any] = {
        "query": query,
        "references": references,
    }
    if extra:
        payload.update(extra)
    updated[step] = payload
    return updated


FlowSearchFn = Callable[[FlowQuery], tuple[list[FlowCandidate], list[object]]]

_DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s\])>,;\"']+", re.IGNORECASE)
_CAS_NUMBER_PATTERN = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_URL_PATTERN = re.compile(r"https?://[^\s\])>]+", re.IGNORECASE)
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_FLOW_LABEL_PATTERN = re.compile(r"^f\\d+\\s*[:\\-]\\s*", re.IGNORECASE)
_QUANT_REF_PATTERN = re.compile(
    r"^(?P<amount>\\d+(?:\\.\\d+)?)\\s*(?P<unit>[^\\s]+)\\s+of\\s+(?P<flow>.+)$",
    re.IGNORECASE,
)
_EXCHANGE_KIND_TAG_NAME = "tg_io_kind_tag"
_EXCHANGE_UNIT_TAG_NAME = "tg_io_uom_tag"
_EXCHANGE_COMMENT_TAG_PATTERN = re.compile(
    rf"\[(?:{_EXCHANGE_KIND_TAG_NAME}|{_EXCHANGE_UNIT_TAG_NAME})=[^\]]*\]",
    re.IGNORECASE,
)
_EXCHANGE_COMMENT_TAG_CAPTURE_PATTERN = re.compile(
    rf"\[(?P<name>{_EXCHANGE_KIND_TAG_NAME}|{_EXCHANGE_UNIT_TAG_NAME})=(?P<value>[^\]]*)\]",
    re.IGNORECASE,
)
_FLOW_NAME_FIELDS = ("baseName", "treatmentStandardsRoutes", "mixAndLocationTypes")
_FLOW_QUALIFIER_FIELDS = ("flowProperties",)
_MEDIA_SUFFIXES = ("to air", "to water", "to soil")
_EMISSION_KEYWORDS = (
    "emission",
    "methane",
    "nitrous oxide",
    "ammonia",
    "carbon dioxide",
    "co2",
    "ch4",
    "n2o",
    "dust",
    "particulate",
    "pm",
    "so2",
    "nox",
    "voc",
)
_WATER_KEYWORDS = ("water", "wastewater", "runoff", "leaching", "leachate", "effluent", "drainage")
_SOIL_KEYWORDS = ("soil", "land", "ground", "field", "sediment")
_ENERGY_KEYWORDS = ("electricity", "diesel", "gasoline", "natural gas", "steam", "heat", "fuel", "coal")
_MATERIAL_ROLES = {
    "raw_material",
    "auxiliary",
    "catalyst",
    "energy",
    "emission",
    "product",
    "waste",
    "service",
    "unknown",
}
_MATERIAL_ROLE_ALIASES = {
    "raw": "raw_material",
    "raw material": "raw_material",
    "feedstock": "raw_material",
    "auxiliary material": "auxiliary",
    "additive": "auxiliary",
    "catalyst": "catalyst",
    "energy": "energy",
    "emission": "emission",
    "product": "product",
    "waste": "waste",
    "service": "service",
    "unknown": "unknown",
}
_BALANCE_EXCLUDE_ROLES = {"auxiliary", "catalyst"}
_BALANCE_CORE_ROLES = {"raw_material", "product", "waste"}


class ProcessFromFlowState(TypedDict, total=False):
    flow_path: str
    flow_dataset: dict[str, Any]
    flow_summary: dict[str, Any]
    operation: str
    stop_after: str
    allow_density_conversion: bool
    auto_balance_revise: bool
    technical_description: str
    assumptions: list[str]
    scope: str
    technology_routes: list[dict[str, Any]]
    process_routes: list[dict[str, Any]]
    selected_route_id: str
    intended_applications: dict[str, dict[str, str]] | dict[str, str] | list[str]
    processes: list[dict[str, Any]]
    process_exchanges: list[dict[str, Any]]
    exchange_value_candidates: list[dict[str, Any]]
    exchange_values_applied: bool
    matched_process_exchanges: list[dict[str, Any]]
    process_datasets: list[dict[str, Any]]
    source_datasets: list[dict[str, Any]]
    source_references: list[dict[str, Any]]
    placeholder_precheck: dict[str, Any]
    placeholder_report: list[dict[str, Any]]
    placeholder_resolutions: list[dict[str, Any]]
    placeholder_resolution_applied: bool
    unit_alignment_applied: bool
    unit_alignment_summary: dict[str, Any]
    density_conversion_applied: bool
    density_conversion_summary: dict[str, Any]
    balance_review: list[dict[str, Any]]
    balance_review_initial: list[dict[str, Any]]
    balance_review_summary: dict[str, Any]
    balance_review_summary_initial: dict[str, Any]
    balance_revise_applied: bool
    balance_revise_summary: dict[str, Any]
    data_cut_off_and_completeness_principles: dict[str, dict[str, str]] | dict[str, str] | list[str]
    data_treatment_and_extrapolations_principles: dict[str, dict[str, str]] | dict[str, str] | list[str]
    data_treatment_principles_applied: bool
    data_treatment_summary: dict[str, Any]
    data_cutoff_principles_applied: bool
    data_cutoff_summary: dict[str, Any]
    step_markers: dict[str, bool]
    scientific_references: dict[str, Any]
    coverage_metrics: dict[str, Any]
    coverage_history: list[dict[str, Any]]
    stop_rule_decision: dict[str, Any]


def _ensure_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = parse_json_response(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Expected a JSON object")


def _language_entry(text: str, lang: str = "en") -> dict[str, str]:
    return {"@xml:lang": lang, "#text": text}


def _normalize_uuid(value: str | None) -> str:
    if not value:
        return str(uuid4())
    try:
        return str(UUID(str(value)))
    except Exception:
        return str(uuid4())


def _pick_lang(value: Any, *, prefer: str = "en") -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        text = value.get("#text") or value.get("text") or value.get("@value")
        if isinstance(text, str) and text.strip():
            return text.strip()
        for nested in value.values():
            candidate = _pick_lang(nested, prefer=prefer)
            if candidate:
                return candidate
        return None
    if isinstance(value, list):
        preferred = None
        fallback = None
        for item in value:
            if isinstance(item, dict):
                lang = str(item.get("@xml:lang") or "").strip().lower()
                text = item.get("#text")
                if isinstance(text, str) and text.strip():
                    if lang == prefer.lower() and preferred is None:
                        preferred = text.strip()
                    if fallback is None:
                        fallback = text.strip()
            else:
                if fallback is None:
                    fallback = _pick_lang(item, prefer=prefer)
        return preferred or fallback
    return str(value).strip() or None


def _flow_summary(flow_dataset: dict[str, Any]) -> dict[str, Any]:
    flow = flow_dataset.get("flowDataSet") if isinstance(flow_dataset.get("flowDataSet"), dict) else flow_dataset
    info = flow.get("flowInformation", {}) if isinstance(flow, dict) else {}
    data_info = info.get("dataSetInformation", {}) if isinstance(info, dict) else {}
    name_block = data_info.get("name", {}) if isinstance(data_info, dict) else {}
    admin = flow.get("administrativeInformation", {}) if isinstance(flow, dict) else {}
    publication = admin.get("publicationAndOwnership", {}) if isinstance(admin, dict) else {}

    base_name_en = _pick_lang(name_block.get("baseName"), prefer="en")
    base_name_zh = _pick_lang(name_block.get("baseName"), prefer="zh")
    treatment_en = _pick_lang(name_block.get("treatmentStandardsRoutes"), prefer="en")
    treatment_zh = _pick_lang(name_block.get("treatmentStandardsRoutes"), prefer="zh")
    mix_en = _pick_lang(name_block.get("mixAndLocationTypes"), prefer="en")
    mix_zh = _pick_lang(name_block.get("mixAndLocationTypes"), prefer="zh")
    general_en = _pick_lang(data_info.get("common:generalComment"), prefer="en")
    general_zh = _pick_lang(data_info.get("common:generalComment"), prefer="zh")

    classification: list[dict[str, Any]] = []
    classification_info = data_info.get("classificationInformation") if isinstance(data_info, dict) else None
    if isinstance(classification_info, dict):
        carrier = classification_info.get("common:classification")
        if isinstance(carrier, dict):
            classes = carrier.get("common:class")
            if isinstance(classes, list):
                classification = [item for item in classes if isinstance(item, dict)]

    return {
        "uuid": str(data_info.get("common:UUID") or "").strip() or None,
        "version": str(publication.get("common:dataSetVersion") or "").strip() or None,
        "base_name_en": base_name_en,
        "base_name_zh": base_name_zh,
        "treatment_en": treatment_en,
        "treatment_zh": treatment_zh,
        "mix_en": mix_en,
        "mix_zh": mix_zh,
        "general_comment_en": general_en,
        "general_comment_zh": general_zh,
        "classification": classification,
    }


def _as_multilang_list(value: Any, *, default_lang: str = "en") -> MultiLangList:
    if isinstance(value, MultiLangList):
        return value
    if value is None:
        return MultiLangList()
    if isinstance(value, list):
        out: list[dict[str, str]] = []
        for item in value:
            if isinstance(item, dict) and "#text" in item:
                out.append(_language_entry(str(item.get("#text") or ""), str(item.get("@xml:lang") or default_lang) or default_lang))
            else:
                text = str(item).strip()
                if text:
                    out.append(_language_entry(text, default_lang))
        return MultiLangList([entry for entry in out if entry.get("#text")])
    if isinstance(value, dict) and "#text" in value:
        text = str(value.get("#text") or "").strip()
        if not text:
            return MultiLangList()
        lang = str(value.get("@xml:lang") or default_lang) or default_lang
        return MultiLangList([_language_entry(text, lang)])
    text = str(value).strip()
    return MultiLangList([_language_entry(text, default_lang)]) if text else MultiLangList()


def _contains_chinese(text: str) -> bool:
    return bool(_CJK_PATTERN.search(text or ""))


def _coerce_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict) and "#text" in value:
        text = str(value.get("#text") or "").strip()
        return [text] if text else []
    return _clean_string_list(value)


def _join_texts(values: Any) -> str | None:
    parts = _coerce_text_list(values)
    if not parts:
        return None
    parts = _dedupe_flows([text for text in parts if text])
    return "; ".join(parts) if parts else None


def _split_bilingual_values(values: list[str]) -> dict[str, str]:
    en_parts: list[str] = []
    zh_parts: list[str] = []
    for item in values:
        if _contains_chinese(item):
            zh_parts.append(item)
        else:
            en_parts.append(item)
    result: dict[str, str] = {}
    en_text = _join_texts(en_parts)
    zh_text = _join_texts(zh_parts)
    if en_text:
        result["en"] = en_text
    if zh_text:
        result["zh"] = zh_text
    return result


def _normalize_bilingual_text(value: Any) -> dict[str, str]:
    texts: dict[str, str] = {}
    if isinstance(value, dict):
        for key in ("en", "zh", "zh-cn", "zh-hans", "zh_cn", "zh-hant"):
            if key in value:
                text = _join_texts(value.get(key))
                if text:
                    lang = "zh" if key.startswith("zh") else "en"
                    texts[lang] = text
        if texts:
            return texts
        if "@xml:lang" in value and "#text" in value:
            lang_raw = str(value.get("@xml:lang") or "").strip().lower()
            lang = "zh" if lang_raw.startswith("zh") else (lang_raw or "en")
            text = _join_texts(value)
            if text:
                texts[lang] = text
            return texts
        for key in (
            "intended_applications",
            "intendedApplications",
            "data_cut_off_and_completeness_principles",
            "dataCutOffAndCompletenessPrinciples",
        ):
            if key in value:
                nested = _normalize_bilingual_text(value.get(key))
                if nested:
                    return nested
    if isinstance(value, list):
        if all(isinstance(item, dict) and "@xml:lang" in item and "#text" in item for item in value):
            for item in value:
                lang_raw = str(item.get("@xml:lang") or "").strip().lower()
                lang = "zh" if lang_raw.startswith("zh") else (lang_raw or "en")
                text = _join_texts(item.get("#text"))
                if not text:
                    continue
                if lang in texts:
                    texts[lang] = f"{texts[lang]}; {text}"
                else:
                    texts[lang] = text
            return texts
        return _split_bilingual_values(_clean_string_list(value))
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        if text:
            if _contains_chinese(text):
                texts["zh"] = text
            else:
                texts["en"] = text
    return texts


def _coerce_bilingual_multilang(
    value: Any,
    *,
    translator: Translator | None = None,
    fallback_en: str | None = None,
) -> MultiLangList:
    texts = _normalize_bilingual_text(value)
    if fallback_en and not texts.get("en"):
        texts["en"] = fallback_en
    en_text = texts.get("en") or ""
    zh_text = texts.get("zh") or ""
    if not zh_text and translator and en_text:
        translated = translator.translate(en_text, "zh")
        if translated:
            zh_text = translated.strip()
    entries: list[dict[str, str]] = []
    if en_text:
        entries.append(_language_entry(en_text, "en"))
    if zh_text:
        entries.append(_language_entry(zh_text, "zh"))
    return MultiLangList(entries)


def _coerce_bilingual_payload(
    value: Any,
    *,
    translator: Translator | None = None,
    fallback_en: str | None = None,
) -> list[dict[str, str]]:
    return _coerce_bilingual_multilang(value, translator=translator, fallback_en=fallback_en).to_plain_list()


def _global_reference(
    *,
    ref_type: str,
    ref_object_id: str,
    version: str,
    uri: str,
    short_description: Any,
    extra_fields: dict[str, Any] | None = None,
) -> GlobalReferenceTypeVariant0:
    reference = GlobalReferenceTypeVariant0(
        type=ref_type,
        ref_object_id=ref_object_id,
        version=version,
        uri=uri,
        common_short_description=_as_multilang_list(short_description),
    )
    if extra_fields:
        for key, value in extra_fields.items():
            setattr(reference, key, value)
    return reference


def _as_classification_items(entries: list[dict[str, Any]]) -> list[CommonClassItemOption0]:
    items: list[CommonClassItemOption0] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        level = str(entry.get("@level") or entry.get("level") or "0").strip()
        if not (level.isdigit() and len(level) == 1):
            level = "0"
        class_id = str(entry.get("@classId") or entry.get("class_id") or entry.get("classId") or "C").strip() or "C"
        text = str(entry.get("#text") or entry.get("text") or "").strip()
        if not text:
            continue
        items.append(CommonClassItemOption0(level=level, class_id=class_id, text=text))
    if not items:
        items = [CommonClassItemOption0(level="0", class_id="C", text="Manufacturing")]
    return items


def _contact_reference() -> GlobalReferenceTypeVariant0:
    ref_object_id = "f4b4c314-8c4c-4c83-968f-5b3c7724f6a8"
    version = "01.00.000"
    return _global_reference(
        ref_type="contact data set",
        ref_object_id=ref_object_id,
        version=version,
        uri=build_local_dataset_uri("contact data set", ref_object_id, version),
        short_description=[
            _language_entry("Tiangong LCA Data Working Group", "en"),
            _language_entry("天工LCA数据团队", "zh"),
        ],
    )


def _entry_level_compliance_reference() -> GlobalReferenceTypeVariant0:
    return _global_reference(
        ref_type="source data set",
        ref_object_id=ILCD_ENTRY_LEVEL_REFERENCE_ID,
        version=ILCD_ENTRY_LEVEL_REFERENCE_VERSION,
        uri=build_local_dataset_uri(
            "source data set",
            ILCD_ENTRY_LEVEL_REFERENCE_ID,
            ILCD_ENTRY_LEVEL_REFERENCE_VERSION,
        ),
        short_description=[_language_entry("ILCD Data Network - Entry-level", "en")],
    )


def _compliance_declarations() -> ProcessDataSetModellingAndValidationComplianceDeclarations:
    mapped_fields = {
        "common_approval_of_overall_compliance": "common:approvalOfOverallCompliance",
        "common_nomenclature_compliance": "common:nomenclatureCompliance",
        "common_methodological_compliance": "common:methodologicalCompliance",
        "common_review_compliance": "common:reviewCompliance",
        "common_documentation_compliance": "common:documentationCompliance",
        "common_quality_compliance": "common:qualityCompliance",
    }
    values: dict[str, str] = {}
    for field_name, source_key in mapped_fields.items():
        values[field_name] = COMPLIANCE_DEFAULT_PREFERENCES.get(source_key) or "Not defined"
    compliance = ComplianceDeclarationsComplianceOption0(
        common_reference_to_compliance_system=_entry_level_compliance_reference(),
        **values,
    )
    return ProcessDataSetModellingAndValidationComplianceDeclarations(compliance=compliance)


def _dataset_format_reference() -> GlobalReferenceTypeVariant0:
    return _global_reference(
        ref_type="source data set",
        ref_object_id=ILCD_FORMAT_SOURCE_UUID,
        version=ILCD_FORMAT_SOURCE_VERSION,
        uri=ILCD_FORMAT_SOURCE_URI,
        short_description=[ILCD_FORMAT_SOURCE_SHORT_DESCRIPTION],
    )


def _candidate_reference(
    candidate: FlowCandidate,
    *,
    translator: Translator | None = None,
    short_description: Any | None = None,
) -> GlobalReferenceTypeVariant0:
    version = candidate.version or "01.01.000"
    uuid_value = _normalize_uuid(candidate.uuid)
    uri = build_portal_uri("flow", uuid_value, version)
    name = str(candidate.base_name or "Unnamed flow").strip() or "Unnamed flow"
    short_desc = short_description or _build_multilang_entries(name, translator=translator)
    if not short_desc:
        short_desc = [_language_entry(name, "en")]
    return _global_reference(
        ref_type="flow data set",
        ref_object_id=uuid_value,
        version=version,
        uri=uri,
        short_description=short_desc,
    )


def _placeholder_flow_reference(name: str, *, translator: Translator | None = None) -> GlobalReferenceTypeVariant0:
    identifier = _normalize_uuid(None)
    version = "00.00.000"
    uri = build_portal_uri("flow", identifier, version)
    short_desc = _build_multilang_entries(name or "Unnamed flow", translator=translator)
    if not short_desc:
        short_desc = [_language_entry(name or "Unnamed flow", "en")]
    return _global_reference(
        ref_type="flow data set",
        ref_object_id=identifier,
        version=version,
        uri=uri,
        short_description=short_desc,
        extra_fields={"unmatched:placeholder": True},
    )


def _default_exchange_amount() -> str:
    return "1.0"


def _reference_direction(operation: str | None) -> str:
    op = str(operation or "produce").strip().lower()
    if op in {"treat", "dispose", "disposal", "treatment"}:
        return "Input"
    return "Output"


def _build_multilang_entries(
    text: str | None,
    *,
    translator: Translator | None = None,
    zh_text: str | None = None,
) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    base = str(text).strip() if text else ""
    if base:
        entries.append(_language_entry(base, "en"))
    zh_value = str(zh_text).strip() if zh_text else ""
    if not zh_value and translator and base:
        translated = translator.translate(base, "zh")
        if translated:
            zh_value = translated.strip()
    if zh_value and zh_value != base:
        entries.append(_language_entry(zh_value, "zh"))
    return entries


def _merge_lang_maps(target: dict[str, list[str]], source: dict[str, list[str]]) -> dict[str, list[str]]:
    for lang, values in source.items():
        bucket = target.setdefault(lang, [])
        for value in values:
            if value and value not in bucket:
                bucket.append(value)
    return target


def _extract_lang_texts(value: Any) -> dict[str, list[str]]:
    if value is None:
        return {}
    if isinstance(value, list):
        merged: dict[str, list[str]] = {}
        for item in value:
            merged = _merge_lang_maps(merged, _extract_lang_texts(item))
        return merged
    if isinstance(value, dict):
        text = value.get("#text") or value.get("text") or value.get("@value")
        if isinstance(text, str) and text.strip():
            lang = str(value.get("@xml:lang") or value.get("@lang") or "en").strip() or "en"
            return {lang: [text.strip()]}
        merged: dict[str, list[str]] = {}
        for nested in value.values():
            merged = _merge_lang_maps(merged, _extract_lang_texts(nested))
        return merged
    text = str(value).strip()
    return {"en": [text]} if text else {}


def _field_text_by_lang(value: Any, *, separator: str = ", ") -> dict[str, str]:
    lang_map = _extract_lang_texts(value)
    return {lang: separator.join([text for text in values if text]) for lang, values in lang_map.items() if any(values)}


def _compose_flow_name_parts(name_block: dict[str, Any]) -> dict[str, str]:
    parts_by_lang: dict[str, list[str]] = {}
    for field in _FLOW_NAME_FIELDS:
        for lang, text in _field_text_by_lang(name_block.get(field)).items():
            bucket = parts_by_lang.setdefault(lang, [])
            if text and text not in bucket:
                bucket.append(text)
    for qualifier_field in _FLOW_QUALIFIER_FIELDS:
        qualifier_map = _field_text_by_lang(name_block.get(qualifier_field))
        if qualifier_map:
            for lang, text in qualifier_map.items():
                bucket = parts_by_lang.setdefault(lang, [])
                if text and text not in bucket:
                    bucket.append(text)
            break
    return {lang: "; ".join(parts) for lang, parts in parts_by_lang.items() if parts}


def _flow_short_description_from_dataset(flow_dataset: dict[str, Any]) -> list[dict[str, str]] | None:
    flow = flow_dataset.get("flowDataSet") if isinstance(flow_dataset.get("flowDataSet"), dict) else flow_dataset
    info = flow.get("flowInformation", {}) if isinstance(flow, dict) else {}
    data_info = info.get("dataSetInformation", {}) if isinstance(info, dict) else {}
    name_block = data_info.get("name") if isinstance(data_info, dict) else None
    if not isinstance(name_block, dict):
        return None
    parts_by_lang = _compose_flow_name_parts(name_block)
    if not parts_by_lang:
        return None
    en_text = parts_by_lang.get("en")
    zh_text = parts_by_lang.get("zh")
    if en_text:
        entries = [_language_entry(en_text, "en")]
        if zh_text:
            entries.append(_language_entry(zh_text, "zh"))
        return entries
    if zh_text:
        return [_language_entry(zh_text, "zh")]
    for lang, text in parts_by_lang.items():
        if text:
            return [_language_entry(text, lang)]
    return None


def _flow_dataset_version(flow_dataset: dict[str, Any]) -> str | None:
    flow = flow_dataset.get("flowDataSet") if isinstance(flow_dataset.get("flowDataSet"), dict) else flow_dataset
    admin = flow.get("administrativeInformation", {}) if isinstance(flow, dict) else {}
    publication = admin.get("publicationAndOwnership", {}) if isinstance(admin, dict) else {}
    version = str(publication.get("common:dataSetVersion") or "").strip()
    return version or None


def _build_reference_from_selected_flow(
    *,
    selected_uuid: str,
    selected_version: str | None,
    selected_base_name: str | None,
    fallback_name: str,
    crud_client: DatabaseCrudClient | None,
    flow_cache: dict[tuple[str, str | None], dict[str, Any]],
    translator: Translator | None = None,
    stage: str = "process_from_flow",
) -> tuple[GlobalReferenceTypeVariant0, FlowReferenceInfo | None, str | None]:
    flow_uuid = str(selected_uuid or "").strip()
    if not flow_uuid:
        raise ValueError("Selected flow UUID is required.")

    version_hint = str(selected_version).strip() if selected_version is not None else None
    if version_hint == "":
        version_hint = None

    cache_key = (flow_uuid, version_hint)
    cached = flow_cache.get(cache_key)
    if cached is None:
        flow_dataset = None
        if crud_client:
            try:
                flow_dataset = crud_client.select_flow(flow_uuid, version=version_hint)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning(
                    "process_from_flow.flow_select_failed",
                    flow_id=flow_uuid,
                    version=version_hint,
                    stage=stage,
                    error=str(exc),
                )
        short_desc = _flow_short_description_from_dataset(flow_dataset) if flow_dataset else None
        version_override = _flow_dataset_version(flow_dataset) if flow_dataset else None
        reference_info = _flow_reference_info_from_dataset(flow_dataset)
        cached = {
            "short_description": short_desc,
            "version": version_override,
            "reference_info": reference_info,
        }
        flow_cache[cache_key] = cached

    resolved_version = str(cached.get("version") or version_hint or "").strip() or None
    candidate = FlowCandidate(
        uuid=flow_uuid,
        base_name=str(selected_base_name or fallback_name or "Unnamed flow"),
        version=resolved_version,
    )
    reference = _candidate_reference(
        candidate,
        translator=translator,
        short_description=cached.get("short_description"),
    )
    reference_info = cached.get("reference_info")
    if not isinstance(reference_info, FlowReferenceInfo):
        reference_info = None
    return reference, reference_info, resolved_version


def _update_step_markers(state: ProcessFromFlowState, step_name: str) -> dict[str, bool]:
    markers = dict(state.get("step_markers") or {})
    markers[step_name] = True
    return markers


def _resolve_runtime_state_path() -> Path | None:
    explicit_path = str(os.getenv(_PFF_RUNTIME_STATE_PATH_ENV) or "").strip()
    if explicit_path:
        return Path(explicit_path)
    run_id = str(os.getenv(_PFF_RUNTIME_RUN_ID_ENV) or "").strip()
    if not run_id:
        return None
    return _PFF_RUNTIME_ARTIFACTS_ROOT / run_id / "cache" / "process_from_flow_state.json"


def _persist_runtime_state(state: ProcessFromFlowState, *, reason: str) -> None:
    state_path = _resolve_runtime_state_path()
    if state_path is None:
        return
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state, ensure_ascii=False, indent=2, default=str)
        state_path.write_text(payload, encoding="utf-8")
        LOGGER.info("process_from_flow.runtime_state_persisted", reason=reason, path=str(state_path))
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning(
            "process_from_flow.runtime_state_persist_failed",
            reason=reason,
            path=str(state_path),
            error=str(exc),
        )


def _empty_placeholder_precheck() -> dict[str, Any]:
    return {
        "placeholder_total": 0,
        "matched_placeholder_total": 0,
        "unmatched_placeholder_total": 0,
        "placeholders": [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _build_placeholder_precheck(
    placeholders: list[dict[str, Any]],
    match_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    if not placeholders:
        return _empty_placeholder_precheck()

    matched_total = 0
    unmatched_total = 0
    precheck_entries: list[dict[str, Any]] = []
    for entry in placeholders:
        matched_exchange = _match_placeholder_exchange(entry, match_index)
        exchange_name = str(entry.get("exchange_name") or "").strip() or None
        comment = ""
        flow_type = None
        direction = str(entry.get("exchange_direction") or "").strip() or None
        unit = None
        expected_compartment = None
        if isinstance(matched_exchange, dict):
            matched_total += 1
            exchange_name = str(matched_exchange.get("exchangeName") or exchange_name or "").strip() or exchange_name
            comment = _strip_exchange_comment_tags(matched_exchange.get("generalComment"))
            flow_type = _normalize_flow_type(matched_exchange.get("flow_type")) or str(matched_exchange.get("flow_type") or "").strip() or None
            direction = str(matched_exchange.get("exchangeDirection") or direction or "").strip() or None
            unit = str(matched_exchange.get("unit") or "").strip() or None
            expected_compartment = _infer_media_suffix(f"{exchange_name or ''} {comment}")
        else:
            unmatched_total += 1

        aliases: list[str] = []
        for alias in entry.get("exchange_names") or []:
            if not isinstance(alias, str):
                continue
            text = alias.strip()
            if text and text not in aliases:
                aliases.append(text)
        if exchange_name and exchange_name not in aliases:
            aliases.append(exchange_name)

        precheck_entries.append(
            {
                "process_id": entry.get("process_id"),
                "process_uuid": entry.get("process_uuid"),
                "process_name": entry.get("process_name"),
                "exchange_index": entry.get("exchange_index"),
                "exchange_name": exchange_name,
                "exchange_aliases": aliases,
                "flow_type": flow_type,
                "exchange_direction": direction,
                "unit": unit,
                "expected_compartment": expected_compartment,
                "status": "matched_context" if isinstance(matched_exchange, dict) else "missing_match_context",
            }
        )

    return {
        "placeholder_total": len(placeholders),
        "matched_placeholder_total": matched_total,
        "unmatched_placeholder_total": unmatched_total,
        "placeholders": precheck_entries,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _extract_cas_number(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        for item in value:
            cas = _extract_cas_number(item)
            if cas:
                return cas
        return None
    text = str(value).strip()
    if not text:
        return None
    match = _CAS_NUMBER_PATTERN.search(text)
    if not match:
        return None
    return match.group(0)


def _normalize_exchange_direction_value(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text.startswith("in"):
        return "Input"
    if text.startswith("out"):
        return "Output"
    return None


def _normalize_compartment_value(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"air", "water", "soil"}:
        return text
    return None


def _coerce_classification_hints(value: Any, *, max_items: int = 6) -> list[str]:
    hints: list[str] = []
    if isinstance(value, str):
        parts = re.split(r"[;,|/]", value)
        for part in parts:
            text = str(part).strip()
            if text and text not in hints:
                hints.append(text)
                if len(hints) >= max_items:
                    break
        return hints
    if isinstance(value, list):
        for item in value:
            for text in _coerce_classification_hints(item, max_items=max_items):
                if text not in hints:
                    hints.append(text)
                    if len(hints) >= max_items:
                        return hints
        return hints
    if isinstance(value, dict):
        for key in ("classification_hints", "classification", "hints", "categories"):
            nested = value.get(key)
            if nested is None:
                continue
            for text in _coerce_classification_hints(nested, max_items=max_items):
                if text not in hints:
                    hints.append(text)
                    if len(hints) >= max_items:
                        return hints
    return hints


def _compose_placeholder_query_text(
    *,
    exchange_name: str,
    description: str | None,
    cas: str | None,
    classification_hints: list[str] | None,
    flow_type: str | None,
    direction: str | None,
    unit: str | None,
    compartment: str | None,
) -> str:
    parts: list[str] = []
    name = str(exchange_name or "").strip()
    if name:
        parts.append(f"flow_name: {name}")
        parts.append(f"exchange: {name}")

    description_text = str(description or "").strip() or None
    constraint_bits: list[str] = []
    if flow_type:
        constraint_bits.append(f"flow_type={flow_type}")
    if direction:
        constraint_bits.append(f"direction={direction}")
    if unit:
        constraint_bits.append(f"unit={unit}")
    if compartment:
        constraint_bits.append(f"compartment={compartment}")
    if classification_hints:
        non_empty = [item for item in classification_hints if str(item).strip()]
        if non_empty:
            constraint_bits.append(f"classification_hints={', '.join(non_empty)}")
    if constraint_bits:
        constraint_text = "; ".join(constraint_bits)
        description_text = f"{description_text} | constraints: {constraint_text}" if description_text else f"constraints: {constraint_text}"
    if description_text:
        parts.append(f"description: {description_text}")

    if cas:
        parts.append(f"cas: {cas}")
    if flow_type:
        parts.append(f"flow_type: {flow_type}")
    if direction:
        parts.append(f"direction: {direction}")
    if unit:
        parts.append(f"unit: {unit}")
    if compartment:
        parts.append(f"compartment: {compartment}")
    return " \n".join(parts)


def _build_placeholder_one_shot_query_payload(
    *,
    llm: LanguageModelProtocol | None,
    exchange_name: str,
    comment: str | None,
    flow_type: str | None,
    direction: str | None,
    unit: str | None,
    expected_compartment: str | None,
    search_hints: list[str] | None,
) -> dict[str, Any]:
    fallback_flow_type = _normalize_flow_type(flow_type)
    fallback_direction = _normalize_exchange_direction_value(direction)
    fallback_unit = str(unit or "").strip() or None
    fallback_compartment = _normalize_compartment_value(expected_compartment) or _infer_media_suffix(f"{exchange_name} {comment or ''}")
    fallback_cas = _extract_cas_number([exchange_name, comment, search_hints or []])
    fallback_hints = _coerce_classification_hints(search_hints or [])

    payload: dict[str, Any] = {
        "exchange_name": str(exchange_name or "").strip(),
        "description": str(comment or "").strip() or None,
        "cas": fallback_cas,
        "classification_hints": fallback_hints,
        "flow_type": fallback_flow_type,
        "direction": fallback_direction,
        "unit": fallback_unit,
        "compartment": fallback_compartment,
    }
    if llm is not None:
        llm_input = {
            "prompt": PLACEHOLDER_QUERY_BUILDER_PROMPT,
            "context": {
                "exchange_name": exchange_name,
                "general_comment": comment,
                "flow_type": fallback_flow_type,
                "direction": fallback_direction,
                "unit": fallback_unit,
                "compartment": fallback_compartment,
                "search_hints": search_hints or [],
                "cas_hint": fallback_cas,
                "classification_hints": fallback_hints,
            },
            "response_format": {"type": "json_object"},
        }
        try:
            raw = llm.invoke(llm_input)
            data = _ensure_dict(raw)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("process_from_flow.placeholder_query_builder_failed", exchange=exchange_name, error=str(exc))
            data = {}

        payload["exchange_name"] = str(data.get("exchange_name") or payload["exchange_name"] or "").strip() or payload["exchange_name"]
        payload["description"] = str(data.get("description") or payload["description"] or "").strip() or payload["description"]
        payload["cas"] = _extract_cas_number(data.get("cas")) or payload["cas"]
        payload["classification_hints"] = _coerce_classification_hints(data.get("classification_hints") or data.get("classification") or payload["classification_hints"])
        payload["flow_type"] = _normalize_flow_type(data.get("flow_type")) or payload["flow_type"]
        payload["direction"] = _normalize_exchange_direction_value(data.get("direction")) or payload["direction"]
        payload["unit"] = str(data.get("unit") or payload["unit"] or "").strip() or payload["unit"]
        payload["compartment"] = _normalize_compartment_value(data.get("compartment")) or payload["compartment"]

    payload["query_text"] = _compose_placeholder_query_text(
        exchange_name=str(payload.get("exchange_name") or exchange_name or "").strip(),
        description=payload.get("description"),
        cas=payload.get("cas"),
        classification_hints=payload.get("classification_hints") if isinstance(payload.get("classification_hints"), list) else [],
        flow_type=payload.get("flow_type"),
        direction=payload.get("direction"),
        unit=payload.get("unit"),
        compartment=payload.get("compartment"),
    )
    return payload


def _coerce_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return 0.0
    if number > 1:
        return 1.0
    return number


def _select_placeholder_uuid_with_llm(
    *,
    llm: LanguageModelProtocol | None,
    exchange_context: dict[str, Any],
    query_payload: dict[str, Any],
    candidates: list[FlowCandidate],
) -> tuple[str | None, str | None, float | None]:
    if llm is None:
        return None, None, None
    candidate_payload = _serialize_candidate_list(candidates, limit=10)
    llm_input = {
        "prompt": PLACEHOLDER_UUID_SELECTOR_PROMPT,
        "context": {
            "exchange": exchange_context,
            "query_payload": query_payload,
            "candidates": candidate_payload,
        },
        "response_format": {"type": "json_object"},
    }
    try:
        raw = llm.invoke(llm_input)
        data = _ensure_dict(raw)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning(
            "process_from_flow.placeholder_uuid_selector_failed",
            exchange=str(exchange_context.get("exchangeName") or ""),
            error=str(exc),
        )
        return None, "LLM selector failed.", None

    selected_uuid = str(data.get("selected_uuid") or data.get("uuid") or "").strip() or None
    if selected_uuid is None and isinstance(data.get("best_index"), int):
        idx = int(data.get("best_index"))
        if 0 <= idx < len(candidates):
            selected_uuid = str(candidates[idx].uuid or "").strip() or None
    reason = str(data.get("reason") or "").strip() or None
    confidence = _coerce_confidence(data.get("confidence"))
    if selected_uuid:
        candidate_uuids = {str(item.uuid).strip() for item in candidates if str(item.uuid).strip()}
        if selected_uuid not in candidate_uuids:
            return None, "LLM selected UUID not present in candidate list.", confidence
    return selected_uuid, reason, confidence


def _one_shot_flow_search_candidates(client: FlowSearchClient, query_text: str) -> list[FlowCandidate]:
    raw_candidates = client.search_query_text(query_text)
    hydrated: list[FlowCandidate] = []
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        try:
            hydrated.append(hydrate_candidate(item))
        except Exception:  # pylint: disable=broad-except
            continue
    return hydrated


def _clean_string_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, int, float)):
        text = str(values).strip()
        return [text] if text else []
    if isinstance(values, list):
        cleaned: list[str] = []
        for item in values:
            text = str(item).strip()
            if text:
                cleaned.append(text)
        return cleaned
    return []


def _dedupe_flows(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _fallback_intended_applications(
    *,
    flow_summary: dict[str, Any],
    technical_description: str,
    scope: str,
    assumptions: list[str],
    operation: str | None,
    process_name: str | None = None,
) -> str:
    flow_name = flow_summary.get("base_name_en") or flow_summary.get("base_name_zh") or "reference flow"
    op = str(operation or "produce").strip().lower()
    activity = "treatment/disposal" if op in {"treat", "dispose", "disposal", "treatment"} else "production"
    process_label = str(process_name or "").strip()
    basis = _compact_text(technical_description or scope, limit=180)
    if basis:
        if process_label:
            return f"Intended for unit-process LCA modelling of {process_label} ({activity} of {flow_name}), based on {basis}."
        return f"Intended for unit-process LCA modelling of {activity} of {flow_name}, based on {basis}."
    assumption_text = _compact_text("; ".join([text for text in assumptions if text]), limit=140)
    if assumption_text:
        if process_label:
            return f"Intended for unit-process LCA modelling of {process_label} ({activity} of {flow_name}), " f"based on stated assumptions ({assumption_text})."
        return f"Intended for unit-process LCA modelling of {activity} of {flow_name}, based on stated assumptions ({assumption_text})."
    if process_label:
        return f"Intended for unit-process LCA modelling of {process_label} ({activity} of {flow_name})."
    return f"Intended for unit-process LCA modelling of {activity} of {flow_name}."


def _summarize_cutoff_inputs(state: ProcessFromFlowState, *, process_id: str | None = None) -> dict[str, Any]:
    matched = state.get("matched_process_exchanges")
    process_count = 0
    exchange_total = 0
    missing_amount = 0
    missing_unit = 0
    unit_converted = 0
    unit_mismatch = 0
    unit_review = 0
    unit_missing = 0
    density_converted = 0
    source_type_counts = {"literature": 0, "si": 0, "expert_judgement": 0, "unknown": 0}
    if isinstance(matched, list):
        for proc in matched:
            if not isinstance(proc, dict):
                continue
            proc_id = str(proc.get("process_id") or proc.get("processId") or "").strip()
            if process_id and proc_id != process_id:
                continue
            process_count += 1
            exchanges = proc.get("exchanges")
            if not isinstance(exchanges, list):
                continue
            for exchange in exchanges:
                if not isinstance(exchange, dict):
                    continue
                exchange_total += 1
                amount = exchange.get("amount")
                if amount in (None, ""):
                    missing_amount += 1
                unit = str(exchange.get("unit") or "").strip()
                if not unit:
                    missing_unit += 1
                flow_search = exchange.get("flow_search") if isinstance(exchange.get("flow_search"), dict) else {}
                unit_check = flow_search.get("unit_check") if isinstance(flow_search.get("unit_check"), dict) else None
                density_marked = False
                if unit_check:
                    status = str(unit_check.get("status") or "").strip().lower()
                    if status in {"converted", "converted_by_dimension"}:
                        unit_converted += 1
                    elif status == "mismatch":
                        unit_mismatch += 1
                    elif status == "review":
                        unit_review += 1
                    elif status == "missing_unit":
                        unit_missing += 1
                    elif status == "converted_by_density":
                        density_converted += 1
                        density_marked = True
                if isinstance(exchange.get("density_used"), dict) and not density_marked:
                    density_converted += 1
                data_source = exchange.get("data_source") if isinstance(exchange.get("data_source"), dict) else {}
                source_type = _normalize_source_type(data_source.get("source_type"))
                if source_type in source_type_counts:
                    source_type_counts[source_type] += 1
                else:
                    source_type_counts["unknown"] += 1

    placeholder_report = state.get("placeholder_report")
    placeholder_total = 0
    placeholder_resolved = 0
    placeholder_unresolved = 0
    if isinstance(placeholder_report, list):
        placeholder_total = len(placeholder_report)
        for entry in placeholder_report:
            if not isinstance(entry, dict):
                continue
            entry_process_id = str(entry.get("process_id") or "").strip()
            if process_id and entry_process_id != process_id:
                continue
            status = str(entry.get("resolution_status") or "").strip().lower()
            if status == "resolved":
                placeholder_resolved += 1
            elif status:
                placeholder_unresolved += 1

    unresolved_entries = _collect_placeholder_entries(state)
    if process_id:
        unresolved_entries = [entry for entry in unresolved_entries if str(entry.get("process_id") or "").strip() == process_id]
    unresolved_current = len(unresolved_entries)
    balance_review = state.get("balance_review") if isinstance(state.get("balance_review"), list) else []
    balance_entry = None
    if process_id:
        for entry in balance_review:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("process_id") or "").strip() == process_id:
                balance_entry = entry
                break

    return {
        "process_count": process_count,
        "exchange_total": exchange_total,
        "exchange_missing_amount": missing_amount,
        "exchange_missing_unit": missing_unit,
        "placeholder_total": placeholder_total,
        "placeholder_resolved": placeholder_resolved,
        "placeholder_unresolved": placeholder_unresolved,
        "placeholder_unresolved_current": unresolved_current,
        "unit_converted": unit_converted,
        "unit_mismatch": unit_mismatch,
        "unit_review": unit_review,
        "unit_missing": unit_missing,
        "density_converted": density_converted,
        "balance_review_entry": balance_entry,
        "source_type_counts": source_type_counts,
    }


def _fallback_cutoff_principles(summary: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    missing_amount = int(summary.get("exchange_missing_amount") or 0)
    unresolved = int(summary.get("placeholder_unresolved_current") or 0)
    unit_mismatch = int(summary.get("unit_mismatch") or 0)
    density_converted = int(summary.get("density_converted") or 0)
    if missing_amount:
        notes.append(f"{missing_amount} exchange amounts were unavailable and kept as placeholders or expert judgement estimates.")
    if unresolved:
        notes.append(f"{unresolved} exchanges still reference placeholder flows after secondary matching.")
    if unit_mismatch or density_converted:
        detail = "Unit compatibility checks were applied"
        if density_converted:
            detail += f", including {density_converted} density-based conversions"
        if unit_mismatch:
            detail += f"; {unit_mismatch} exchanges remain flagged for unit mismatch review."
        else:
            detail += "."
        notes.append(detail if detail.endswith(".") else f"{detail}.")
    if not notes:
        notes.append("Data completeness reflects the listed exchanges; no additional cut-off rules are recorded beyond the documented inventory.")
    return notes[:3]


def _format_balance_metric_for_treatment(label: str, metric: dict[str, Any]) -> str:
    status = str(metric.get("status") or "insufficient").strip() or "insufficient"
    inputs = _parse_amount_value(metric.get("inputs")) or 0.0
    outputs = _parse_amount_value(metric.get("outputs")) or 0.0
    ratio = _parse_amount_value(metric.get("ratio"))
    count = int(metric.get("count") or 0)
    unit = str(metric.get("unit") or "").strip()
    suffix = f" {unit}" if unit else ""
    ratio_text = f"{ratio:.3g}" if ratio is not None else "n/a"
    return f"{label}: status={status}, inputs={_format_amount_value(inputs)}{suffix}, " f"outputs={_format_amount_value(outputs)}{suffix}, ratio={ratio_text}, n={count}"


def _balance_entry_snapshot(balance_entry: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(balance_entry, dict):
        return {"status": "missing"}
    return {
        "status": str(balance_entry.get("status") or "insufficient").strip() or "insufficient",
        "mass_status": str((balance_entry.get("mass") or {}).get("status") or "insufficient").strip() or "insufficient",
        "mass_core_status": str((balance_entry.get("mass_core") or {}).get("status") or "insufficient").strip() or "insufficient",
        "energy_status": str((balance_entry.get("energy") or {}).get("status") or "insufficient").strip() or "insufficient",
        "unit_mismatch_count": int(balance_entry.get("unit_mismatch_count") or 0),
        "unit_assumption_count": int(balance_entry.get("unit_assumption_count") or 0),
        "density_estimate_count": int(balance_entry.get("density_estimate_count") or 0),
        "mapping_conflict_count": int(balance_entry.get("mapping_conflict_count") or 0),
        "role_missing_count": int(balance_entry.get("role_missing_count") or 0),
        "balance_excluded_count": int(balance_entry.get("balance_excluded_count") or 0),
        "core_exchange_count": int(balance_entry.get("core_exchange_count") or 0),
        "exchange_count": int(balance_entry.get("exchange_count") or 0),
    }


def _fallback_data_treatment_principles(
    *,
    process_id: str,
    balance_entry: dict[str, Any] | None,
    balance_summary: dict[str, Any] | None,
) -> list[str]:
    notes: list[str] = []
    snapshot = _balance_entry_snapshot(balance_entry)
    status = str(snapshot.get("status") or "insufficient")
    notes.append(
        f"Balance review outcome ({process_id}): status={status}, " f"core_exchange_count={snapshot.get('core_exchange_count', 0)}, exchange_count={snapshot.get('exchange_count', 0)}."
    )
    if isinstance(balance_entry, dict):
        for label in ("mass_core", "mass", "energy"):
            metric = balance_entry.get(label)
            if isinstance(metric, dict):
                notes.append(_format_balance_metric_for_treatment(label, metric))
        risk_fields = (
            "unit_mismatch_count",
            "unit_assumption_count",
            "density_estimate_count",
            "mapping_conflict_count",
            "role_missing_count",
            "balance_excluded_count",
        )
        risk_parts = [f"{name}={snapshot.get(name, 0)}" for name in risk_fields if int(snapshot.get(name, 0) or 0) > 0]
        if risk_parts:
            notes.append("Review flags: " + ", ".join(risk_parts) + ".")
        else:
            notes.append("Review flags: none.")
    else:
        notes.append("Balance review details unavailable; keep exchange-level assumptions and source traceability for manual verification.")
    if isinstance(balance_summary, dict) and balance_summary:
        notes.append(
            "Run balance summary: "
            f"mass_core_check_processes={int(balance_summary.get('mass_core_check_processes') or 0)}, "
            f"unit_mismatch_total={int(balance_summary.get('unit_mismatch_total') or 0)}, "
            f"mapping_conflict_total={int(balance_summary.get('mapping_conflict_total') or 0)}, "
            f"role_missing_total={int(balance_summary.get('role_missing_total') or 0)}."
        )
    return notes[:5]


def _normalize_quantitative_reference(value: Any, fallback_flow_name: str | None) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        fallback = fallback_flow_name or "reference flow"
        return f"1 unit of {fallback}"
    if any(ch.isdigit() for ch in text):
        return text
    fallback = fallback_flow_name or text
    return f"1 unit of {fallback}"


def _strip_flow_label(value: str) -> str:
    return _FLOW_LABEL_PATTERN.sub("", value).strip()


def _label_flows(values: list[str], *, prefix: str = "f") -> list[str]:
    labeled: list[str] = []
    for idx, value in enumerate(values, start=1):
        raw = _strip_flow_label(value)
        if not raw:
            continue
        labeled.append(f"{prefix}{idx}: {raw}")
    return labeled


def _normalize_flow_type(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    normalized = {
        "elementary flow": "elementary",
        "emission": "elementary",
        "resource": "elementary",
        "waste flow": "waste",
        "service flow": "service",
    }.get(text, text)
    if normalized in {"product", "elementary", "waste", "service"}:
        return normalized
    return None


def _normalize_material_role(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    normalized = _MATERIAL_ROLE_ALIASES.get(text, text)
    if normalized in _MATERIAL_ROLES:
        return normalized
    return None


def _normalize_balance_exclude(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1"}:
            return True
        if text in {"false", "no", "n", "0"}:
            return False
    return None


def _infer_flow_type(name: str, *, direction: str, is_reference_flow: bool) -> str:
    if is_reference_flow:
        return "product"
    lower = name.lower().strip()
    if any(token in lower for token in _MEDIA_SUFFIXES):
        return "elementary"
    if "labor" in lower or "labour" in lower:
        return "service"
    if direction == "Output" and ("waste" in lower or "residue" in lower or "sludge" in lower):
        return "waste"
    if direction == "Input" and any(token in lower for token in _WATER_KEYWORDS + _SOIL_KEYWORDS):
        return "elementary"
    if direction == "Output" and any(token in lower for token in _EMISSION_KEYWORDS + _WATER_KEYWORDS):
        return "elementary"
    return "product"


def _infer_media_suffix(name: str) -> str | None:
    lower = name.lower()
    if any(token in lower for token in _WATER_KEYWORDS):
        return "water"
    if any(token in lower for token in _SOIL_KEYWORDS):
        return "soil"
    if any(token in lower for token in _EMISSION_KEYWORDS):
        return "air"
    return None


def _ensure_media_suffix(name: str, *, direction: str, flow_type: str, is_reference_flow: bool) -> str:
    if flow_type != "elementary":
        return name
    if is_reference_flow:
        return name
    if direction != "Output":
        return name
    lower = name.lower()
    if any(suffix in lower for suffix in _MEDIA_SUFFIXES):
        return name
    medium = _infer_media_suffix(name)
    return f"{name}, to {medium}" if medium else name


def _normalize_exchange_label(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _extract_short_description_texts(value: Any) -> list[str]:
    lang_map = _extract_lang_texts(value)
    texts: list[str] = []
    for items in lang_map.values():
        for text in items:
            if text and text not in texts:
                texts.append(text)
    return texts


def _extract_process_dataset_name(dataset: dict[str, Any]) -> str | None:
    info = dataset.get("processInformation") if isinstance(dataset.get("processInformation"), dict) else {}
    data_info = info.get("dataSetInformation") if isinstance(info.get("dataSetInformation"), dict) else {}
    name_block = data_info.get("name") if isinstance(data_info.get("name"), dict) else {}
    base_name = _pick_lang(name_block.get("baseName"), prefer="en") or _pick_lang(name_block.get("baseName"), prefer="zh")
    return base_name


def _extract_exchange_name_text(value: Any) -> str:
    return _pick_lang(value, prefer="en") or _pick_lang(value, prefer="zh") or ""


def _effective_exchange_direction(exchange: dict[str, Any], *, reference_direction: str) -> str:
    direction = str(exchange.get("exchangeDirection") or "").strip()
    if direction not in {"Input", "Output"}:
        direction = "Input"
    if bool(exchange.get("is_reference_flow")):
        return reference_direction
    return direction


def _process_dataset_lookup_by_process_id(
    state: ProcessFromFlowState,
    process_datasets: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    dataset_by_process_id: dict[str, dict[str, Any]] = {}
    dataset_index_by_process_id: dict[str, int] = {}
    process_list = [item for item in (state.get("processes") or []) if isinstance(item, dict)]
    process_ids = [str(item.get("process_id") or "").strip() for item in process_list]
    for idx, dataset in enumerate(process_datasets):
        if not isinstance(dataset, dict):
            continue
        process_id = process_ids[idx] if idx < len(process_ids) else ""
        if not process_id:
            continue
        dataset_by_process_id[process_id] = dataset
        dataset_index_by_process_id[process_id] = idx
    return dataset_by_process_id, dataset_index_by_process_id


def _update_dataset_exchange_amount_from_match(
    dataset_payload: dict[str, Any],
    *,
    matched_exchange: dict[str, Any],
    exchange_index_hint: int | None,
    amount_text: str,
) -> bool:
    process_block = dataset_payload.get("processDataSet") if isinstance(dataset_payload.get("processDataSet"), dict) else dataset_payload
    if not isinstance(process_block, dict):
        return False
    exchanges_block = process_block.get("exchanges")
    if not isinstance(exchanges_block, dict):
        return False
    exchanges = exchanges_block.get("exchange")
    if not isinstance(exchanges, list):
        return False

    target_name = _normalize_exchange_label(str(matched_exchange.get("exchangeName") or "").strip())
    target_direction = str(matched_exchange.get("exchangeDirection") or "").strip()
    target_is_reference = bool(matched_exchange.get("is_reference_flow"))

    def _match_index(idx: int) -> bool:
        if idx < 0 or idx >= len(exchanges):
            return False
        item = exchanges[idx]
        if not isinstance(item, dict):
            return False
        name = _normalize_exchange_label(_extract_exchange_name_text(item.get("exchangeName")))
        direction = str(item.get("exchangeDirection") or "").strip()
        internal_ref = bool(item.get("referenceToFlowDataSet", {}).get("unmatched:placeholder")) if isinstance(item.get("referenceToFlowDataSet"), dict) else False
        if target_name and name and target_name != name:
            return False
        if target_direction and direction and target_direction != direction:
            return False
        # If matched exchange is reference flow, prefer dataset exchange carrying internalRef id mapping.
        if target_is_reference and not internal_ref and target_name and name != target_name:
            return False
        return True

    selected_index: int | None = None
    if isinstance(exchange_index_hint, int) and _match_index(exchange_index_hint):
        selected_index = exchange_index_hint
    else:
        for idx in range(len(exchanges)):
            if _match_index(idx):
                selected_index = idx
                break
    if selected_index is None:
        return False
    target_exchange = exchanges[selected_index]
    if not isinstance(target_exchange, dict):
        return False
    target_exchange["meanAmount"] = amount_text
    target_exchange["resultingAmount"] = amount_text
    return True


def _collect_placeholder_entries(
    state: ProcessFromFlowState,
    *,
    process_datasets: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    datasets = process_datasets if isinstance(process_datasets, list) else state.get("process_datasets")
    if not isinstance(datasets, list) or not datasets:
        return []
    process_list = [item for item in (state.get("processes") or []) if isinstance(item, dict)]
    process_ids = [str(item.get("process_id") or "").strip() or None for item in process_list]
    entries: list[dict[str, Any]] = []
    for proc_idx, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            continue
        process_block = dataset.get("processDataSet") if isinstance(dataset.get("processDataSet"), dict) else {}
        info = process_block.get("processInformation") if isinstance(process_block.get("processInformation"), dict) else {}
        data_info = info.get("dataSetInformation") if isinstance(info.get("dataSetInformation"), dict) else {}
        proc_uuid = str(data_info.get("common:UUID") or "").strip() or None
        proc_name = _extract_process_dataset_name(process_block)
        process_id = process_ids[proc_idx] if proc_idx < len(process_ids) else None
        exchanges = process_block.get("exchanges", {}) if isinstance(process_block.get("exchanges"), dict) else {}
        for ex_idx, exchange in enumerate(exchanges.get("exchange") or []):
            if not isinstance(exchange, dict):
                continue
            reference = exchange.get("referenceToFlowDataSet")
            if not isinstance(reference, dict) or not reference.get("unmatched:placeholder"):
                continue
            short_desc = reference.get("common:shortDescription")
            names = _extract_short_description_texts(short_desc)
            entries.append(
                {
                    "process_index": proc_idx,
                    "process_id": process_id,
                    "process_uuid": proc_uuid,
                    "process_name": proc_name,
                    "exchange_index": ex_idx,
                    "exchange_direction": exchange.get("exchangeDirection"),
                    "exchange_names": names,
                    "exchange_name": names[0] if names else None,
                }
            )
    return entries


def _index_matched_exchanges(matched: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for proc in matched:
        if not isinstance(proc, dict):
            continue
        process_id = str(proc.get("process_id") or "").strip()
        for exchange in proc.get("exchanges") or []:
            if not isinstance(exchange, dict):
                continue
            name = _normalize_exchange_label(exchange.get("exchangeName"))
            if not name or not process_id:
                continue
            index.setdefault((process_id, name), []).append(exchange)
    return index


def _match_placeholder_exchange(
    entry: dict[str, Any],
    match_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any] | None:
    process_id = str(entry.get("process_id") or "").strip()
    if not process_id:
        return None
    direction = str(entry.get("exchange_direction") or "").strip()
    names = [name for name in (entry.get("exchange_names") or []) if isinstance(name, str)]
    matches: list[dict[str, Any]] = []
    for name in names:
        key = (process_id, _normalize_exchange_label(name))
        matches.extend(match_index.get(key, []))
    if direction:
        matches = [item for item in matches if str(item.get("exchangeDirection") or "").strip() == direction] or matches
    return matches[0] if matches else None


def _version_sort_key(version: str | None) -> tuple[int, ...]:
    if not version:
        return (-1, -1, -1)
    parts: list[int] = []
    for token in str(version).split("."):
        token = token.strip()
        if not token:
            parts.append(0)
            continue
        try:
            parts.append(int(token))
        except ValueError:
            digits = "".join(ch for ch in token if ch.isdigit())
            parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _dedupe_candidates_by_uuid_version(candidates: list[FlowCandidate]) -> list[FlowCandidate]:
    if not candidates:
        return []
    ordered: list[FlowCandidate] = []
    best_by_uuid: dict[str, tuple[FlowCandidate, tuple[int, ...], int]] = {}
    for candidate in candidates:
        uuid_value = (candidate.uuid or "").strip()
        if not uuid_value:
            ordered.append(candidate)
            continue
        version_key = _version_sort_key(candidate.version)
        existing = best_by_uuid.get(uuid_value)
        if existing is None:
            best_by_uuid[uuid_value] = (candidate, version_key, len(ordered))
            ordered.append(candidate)
            continue
        _, best_key, index = existing
        if version_key > best_key:
            best_by_uuid[uuid_value] = (candidate, version_key, index)
            ordered[index] = candidate
    return ordered


def _serialize_candidate_list(candidates: list[FlowCandidate], *, limit: int = 10) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for candidate in candidates[:limit]:
        classification = []
        for item in candidate.classification or []:
            if isinstance(item, dict):
                text = str(item.get("#text") or item.get("text") or "").strip()
                if text:
                    classification.append(text)
        serialized.append(
            {
                "uuid": candidate.uuid,
                "base_name": candidate.base_name,
                "flow_type": candidate.flow_type,
                "version": candidate.version,
                "classification_path": classification,
                "category_path": candidate.category_path,
                "cas": candidate.cas,
            }
        )
    return serialized


def _serialize_flow_search_candidates(candidates: list[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for candidate in candidates[:limit]:
        if not isinstance(candidate, dict):
            continue
        classification = []
        for item in candidate.get("classification") or []:
            if isinstance(item, dict):
                text = str(item.get("#text") or item.get("text") or "").strip()
                if text:
                    classification.append(text)
        serialized.append(
            {
                "uuid": candidate.get("uuid"),
                "base_name": candidate.get("base_name"),
                "flow_type": candidate.get("flow_type"),
                "version": candidate.get("version"),
                "classification_path": classification,
                "category_path": candidate.get("category_path"),
                "cas": candidate.get("cas"),
            }
        )
    return serialized


def _build_placeholder_report(state: ProcessFromFlowState) -> list[dict[str, Any]]:
    if isinstance(state.get("placeholder_resolutions"), list) and state.get("placeholder_resolutions"):
        return state.get("placeholder_resolutions") or []
    process_datasets = state.get("process_datasets")
    matched = state.get("matched_process_exchanges")
    if not isinstance(process_datasets, list) or not isinstance(matched, list):
        return []
    entries = _collect_placeholder_entries(state, process_datasets=process_datasets)
    if not entries:
        return []
    match_index = _index_matched_exchanges(matched)
    report: list[dict[str, Any]] = []
    for entry in entries:
        matched_exchange = _match_placeholder_exchange(entry, match_index)
        flow_search = matched_exchange.get("flow_search") if isinstance(matched_exchange, dict) else {}
        candidates = flow_search.get("candidates") if isinstance(flow_search, dict) else None
        report.append(
            {
                "process_id": entry.get("process_id"),
                "process_uuid": entry.get("process_uuid"),
                "process_name": entry.get("process_name"),
                "exchange_name": matched_exchange.get("exchangeName") if isinstance(matched_exchange, dict) else entry.get("exchange_name"),
                "exchange_direction": entry.get("exchange_direction"),
                "flow_type": matched_exchange.get("flow_type") if isinstance(matched_exchange, dict) else None,
                "unit": matched_exchange.get("unit") if isinstance(matched_exchange, dict) else None,
                "candidate_list": _serialize_flow_search_candidates(candidates) if isinstance(candidates, list) else [],
                "selected_uuid": flow_search.get("selected_uuid") if isinstance(flow_search, dict) else None,
                "selected_reason": flow_search.get("selected_reason") if isinstance(flow_search, dict) else None,
            }
        )
    return report


def _apply_balance_auto_revisions(
    state: ProcessFromFlowState,
    *,
    process_exchanges: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    settings: Settings,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, dict[str, Any]]:
    revised_matches = copy.deepcopy(process_exchanges)
    process_datasets = state.get("process_datasets")
    revised_datasets = copy.deepcopy(process_datasets) if isinstance(process_datasets, list) else None
    dataset_by_process_id: dict[str, dict[str, Any]] = {}
    if isinstance(revised_datasets, list):
        dataset_by_process_id, _ = _process_dataset_lookup_by_process_id(state, revised_datasets)

    review_index: dict[str, dict[str, Any]] = {}
    for item in reviews:
        if not isinstance(item, dict):
            continue
        process_id = str(item.get("process_id") or "").strip()
        if process_id:
            review_index[process_id] = item

    summary: dict[str, Any] = {
        "enabled": True,
        "candidate_processes": 0,
        "revised_processes": 0,
        "revised_exchanges": 0,
        "skipped_processes": [],
        "changes": [],
    }
    reference_direction = _reference_direction(state.get("operation"))
    flow_cache: dict[str, FlowReferenceInfo | None] = {}
    crud_client: DatabaseCrudClient | None = None
    should_close_crud = False
    needs_crud = False
    for proc in revised_matches:
        exchanges = proc.get("exchanges") if isinstance(proc, dict) else None
        if not isinstance(exchanges, list):
            continue
        for exchange in exchanges:
            if not isinstance(exchange, dict):
                continue
            flow_search = exchange.get("flow_search") if isinstance(exchange.get("flow_search"), dict) else {}
            if flow_search.get("selected_uuid"):
                needs_crud = True
                break
        if needs_crud:
            break
    if needs_crud:
        try:
            crud_client = DatabaseCrudClient(settings)
            should_close_crud = True
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("process_from_flow.balance_revise_crud_init_failed", error=str(exc))
            crud_client = None

    try:
        for proc_idx, proc in enumerate(revised_matches):
            if not isinstance(proc, dict):
                continue
            process_id = str(proc.get("process_id") or proc.get("processId") or "").strip()
            review = review_index.get(process_id) if process_id else None
            if not isinstance(review, dict):
                continue
            mass_core = review.get("mass_core")
            if not isinstance(mass_core, dict):
                continue
            if str(mass_core.get("status") or "").strip().lower() != "check":
                continue
            inputs = _parse_amount_value(mass_core.get("inputs"))
            outputs = _parse_amount_value(mass_core.get("outputs"))
            if inputs is None or outputs is None or inputs <= 0 or outputs <= 0:
                summary["skipped_processes"].append(
                    {
                        "process_id": process_id or f"process_{proc_idx + 1}",
                        "reason": "mass_core_inputs_or_outputs_missing",
                    }
                )
                continue
            if outputs > inputs:
                target_direction = "Output"
                imbalance = outputs - inputs
            elif inputs > outputs:
                target_direction = "Input"
                imbalance = inputs - outputs
            else:
                continue
            if imbalance <= 0:
                continue
            summary["candidate_processes"] += 1

            exchanges = proc.get("exchanges") if isinstance(proc.get("exchanges"), list) else []
            adjustable: list[dict[str, Any]] = []
            for ex_idx, exchange in enumerate(exchanges):
                if not isinstance(exchange, dict):
                    continue
                direction = _effective_exchange_direction(exchange, reference_direction=reference_direction)
                if direction != target_direction:
                    continue
                if bool(exchange.get("is_reference_flow")):
                    continue
                material_role = _normalize_material_role(exchange.get("material_role") or exchange.get("materialRole"))
                balance_exclude = _normalize_balance_exclude(exchange.get("balance_exclude") or exchange.get("balanceExclude"))
                comment_tags = _parse_exchange_comment_tags(exchange.get("generalComment"))
                flow_kind = _normalize_flow_type(comment_tags.get(_EXCHANGE_KIND_TAG_NAME))
                if not flow_kind:
                    flow_kind = _exchange_flow_type_for_dedupe(exchange, direction=direction)
                if not _is_core_mass_exchange(
                    material_role=material_role,
                    flow_kind=flow_kind,
                    balance_exclude=balance_exclude,
                ):
                    continue

                amount_value = _parse_amount_value(exchange.get("amount"))
                if amount_value is None:
                    continue
                flow_search = exchange.get("flow_search") if isinstance(exchange.get("flow_search"), dict) else {}
                selected_uuid = str(flow_search.get("selected_uuid") or "").strip() or None
                reference_info = None
                if selected_uuid:
                    if selected_uuid in flow_cache:
                        reference_info = flow_cache[selected_uuid]
                    else:
                        flow_dataset = None
                        if crud_client:
                            try:
                                flow_dataset = crud_client.select_flow(selected_uuid)
                            except Exception as exc:  # pylint: disable=broad-except
                                LOGGER.warning(
                                    "process_from_flow.balance_revise_flow_select_failed",
                                    flow_id=selected_uuid,
                                    error=str(exc),
                                )
                        reference_info = _flow_reference_info_from_dataset(flow_dataset) if flow_dataset else None
                        flow_cache[selected_uuid] = reference_info
                resolved_unit, _ = _resolve_exchange_balance_unit(
                    exchange,
                    reference_info=reference_info,
                    material_role=material_role,
                    flow_kind=flow_kind,
                )
                if not resolved_unit:
                    continue
                dimension, converted, converted_unit = _convert_exchange_amount_for_balance(
                    amount_value,
                    resolved_unit,
                    reference_info,
                )
                if dimension != "mass" or converted is None or converted <= 0:
                    continue
                adjustable.append(
                    {
                        "exchange": exchange,
                        "exchange_index": ex_idx,
                        "matched_name": str(exchange.get("exchangeName") or "").strip(),
                        "matched_direction": direction,
                        "converted_amount": float(converted),
                        "converted_unit": str(converted_unit or "").strip() or None,
                        "resolved_unit": resolved_unit,
                        "reference_info": reference_info,
                        "selected_uuid": selected_uuid,
                    }
                )

            total_adjustable = sum(item["converted_amount"] for item in adjustable)
            if total_adjustable <= 0:
                summary["skipped_processes"].append(
                    {
                        "process_id": process_id or f"process_{proc_idx + 1}",
                        "reason": f"no_adjustable_core_mass_exchanges_on_{target_direction.lower()}_side",
                    }
                )
                continue

            reduction = min(imbalance, total_adjustable)
            ratio = 1.0 - (reduction / total_adjustable)
            revised_count_for_process = 0
            dataset_payload = dataset_by_process_id.get(process_id) if process_id else None
            if dataset_payload is None and isinstance(revised_datasets, list) and 0 <= proc_idx < len(revised_datasets):
                candidate = revised_datasets[proc_idx]
                if isinstance(candidate, dict):
                    dataset_payload = candidate

            for item in adjustable:
                exchange = item["exchange"]
                old_amount = _parse_amount_value(exchange.get("amount"))
                if old_amount is None:
                    continue
                new_converted = item["converted_amount"] * ratio
                new_amount = _convert_balance_amount_to_resolved_unit(
                    new_converted,
                    balance_unit=item["converted_unit"],
                    resolved_unit=item["resolved_unit"],
                    reference_info=item["reference_info"],
                )
                if new_amount is None or new_amount < 0:
                    continue
                old_text = _format_amount_value(old_amount)
                new_text = _format_amount_value(new_amount)
                if old_text == new_text:
                    continue
                exchange["amount"] = new_text
                exchange["balance_auto_revise"] = {
                    "applied": True,
                    "old_amount": old_text,
                    "new_amount": new_text,
                    "resolved_unit_basis": item["resolved_unit"],
                    "balance_unit": item["converted_unit"],
                    "process_direction_adjusted": target_direction,
                }
                dataset_updated = False
                if isinstance(dataset_payload, dict):
                    dataset_updated = _update_dataset_exchange_amount_from_match(
                        dataset_payload,
                        matched_exchange=exchange,
                        exchange_index_hint=item["exchange_index"],
                        amount_text=new_text,
                    )
                summary["changes"].append(
                    {
                        "process_id": process_id or f"process_{proc_idx + 1}",
                        "exchange_name": item["matched_name"],
                        "exchange_direction": item["matched_direction"],
                        "flow_uuid": item["selected_uuid"],
                        "old_amount": old_text,
                        "new_amount": new_text,
                        "resolved_unit_basis": item["resolved_unit"],
                        "dataset_synced": dataset_updated,
                    }
                )
                revised_count_for_process += 1

            if revised_count_for_process:
                summary["revised_processes"] += 1
                summary["revised_exchanges"] += revised_count_for_process
            else:
                summary["skipped_processes"].append(
                    {
                        "process_id": process_id or f"process_{proc_idx + 1}",
                        "reason": "adjustable_exchanges_found_but_no_amount_changes_applied",
                    }
                )
    finally:
        if should_close_crud and crud_client:
            crud_client.close()

    return revised_matches, revised_datasets, summary


def _normalize_route_processes(
    processes: list[dict[str, Any]],
    *,
    flow_summary: dict[str, Any],
    route_name: str,
) -> list[dict[str, Any]]:
    flow_name = str(flow_summary.get("base_name_en") or "reference flow").strip() or "reference flow"
    normalized: list[dict[str, Any]] = []

    for idx, proc in enumerate(processes, start=1):
        process_id = str(proc.get("process_id") or "").strip() or f"P{idx}"
        is_reference_flow_process = bool(proc.get("is_reference_flow_process"))
        name_parts = proc.get("name_parts") if isinstance(proc.get("name_parts"), dict) else {}
        structure = proc.get("structure") if isinstance(proc.get("structure"), dict) else {}
        geography = proc.get("geography") if isinstance(proc.get("geography"), dict) else {}
        structure_inputs = [_strip_flow_label(val) for val in _clean_string_list(structure.get("inputs"))]
        structure_outputs = [_strip_flow_label(val) for val in _clean_string_list(structure.get("outputs"))]
        structure_assumptions = _clean_string_list(structure.get("assumptions"))

        reference_flow_name = str(proc.get("reference_flow_name") or proc.get("referenceFlowName") or "").strip()
        if is_reference_flow_process:
            reference_flow_name = flow_name
        if not reference_flow_name:
            candidate_outputs = structure_outputs
            if candidate_outputs:
                reference_flow_name = candidate_outputs[0]
        if not reference_flow_name:
            base_name_fallback = str(name_parts.get("base_name") or proc.get("name") or "").strip()
            reference_flow_name = f"intermediate product from {base_name_fallback}" if base_name_fallback else "intermediate product"

        if reference_flow_name and reference_flow_name not in structure_outputs:
            structure_outputs.insert(0, reference_flow_name)

        base_name = str(name_parts.get("base_name") or "").strip()
        if not base_name:
            base_name = str(proc.get("name") or "").strip() or reference_flow_name
        treatment_and_route = str(name_parts.get("treatment_and_route") or route_name or "").strip()
        if not treatment_and_route:
            treatment_and_route = "Unspecified route"
        mix_and_location = str(name_parts.get("mix_and_location") or flow_summary.get("mix_en") or "Unspecified mix/location").strip()
        quantitative_reference = _normalize_quantitative_reference(name_parts.get("quantitative_reference"), reference_flow_name)

        name_parts = {
            "base_name": base_name,
            "treatment_and_route": treatment_and_route,
            "mix_and_location": mix_and_location,
            "quantitative_reference": quantitative_reference,
        }
        name = " | ".join([base_name, treatment_and_route, mix_and_location, quantitative_reference])

        description = str(proc.get("description") or "").strip()
        if not description and structure:
            tech = str(structure.get("technology") or "").strip()
            inputs = ", ".join([val for val in structure_inputs if val])
            outputs = ", ".join([val for val in structure_outputs if val])
            boundary = str(structure.get("boundary") or "").strip()
            assumptions = ", ".join([val for val in structure_assumptions if val])
            parts = []
            if tech:
                parts.append(f"Technology: {tech}")
            if inputs:
                parts.append(f"Inputs: {inputs}")
            if outputs:
                parts.append(f"Outputs: {outputs}")
            if boundary:
                parts.append(f"Boundary: {boundary}")
            if assumptions:
                parts.append(f"Assumptions: {assumptions}")
            description = "; ".join(parts)

        normalized.append(
            {
                "process_id": process_id,
                "name": name,
                "description": description,
                "is_reference_flow_process": is_reference_flow_process,
                "reference_flow_name": reference_flow_name,
                "name_parts": name_parts,
                "geography": geography,
                "structure": {
                    **structure,
                    "inputs": structure_inputs,
                    "outputs": structure_outputs,
                    "assumptions": structure_assumptions,
                },
            }
        )

    for idx in range(len(normalized) - 1):
        chain_flow = normalized[idx].get("reference_flow_name")
        if not chain_flow:
            continue
        next_proc = normalized[idx + 1]
        next_structure = next_proc.get("structure") if isinstance(next_proc.get("structure"), dict) else {}
        next_inputs = _clean_string_list(next_structure.get("inputs"))
        if chain_flow not in next_inputs:
            next_inputs.insert(0, chain_flow)
        next_proc["structure"] = {**next_structure, "inputs": next_inputs}

    for proc in normalized:
        structure = proc.get("structure") if isinstance(proc.get("structure"), dict) else {}
        inputs = [_strip_flow_label(val) for val in _clean_string_list(structure.get("inputs"))]
        outputs = [_strip_flow_label(val) for val in _clean_string_list(structure.get("outputs"))]
        proc["exchange_keywords"] = {
            "inputs": _dedupe_flows(inputs),
            "outputs": _dedupe_flows(outputs),
        }
        inputs = _clean_string_list(inputs)
        outputs = _clean_string_list(outputs)
        proc["structure"] = {
            **structure,
            "inputs": _label_flows(inputs, prefix="f"),
            "outputs": _label_flows(outputs, prefix="f"),
        }

    return normalized


def _build_langgraph(
    *,
    llm: LanguageModelProtocol | None,
    settings: Settings,
    flow_search_fn: FlowSearchFn,
    selector: CandidateSelector,
    translator: Translator | None,
    mcp_client: MCPToolClient | None = None,
) -> Any:
    graph = StateGraph(ProcessFromFlowState)
    # Create or use provided MCP client for scientific literature search
    use_mcp_client = mcp_client

    def load_flow(state: ProcessFromFlowState) -> ProcessFromFlowState:
        path = Path(state["flow_path"])
        dataset = json.loads(path.read_text(encoding="utf-8"))
        summary = _flow_summary(dataset)
        LOGGER.info("process_from_flow.load_flow", path=str(path), uuid=summary.get("uuid"))
        return {"flow_dataset": dataset, "flow_summary": summary}

    def describe_technology(state: ProcessFromFlowState) -> ProcessFromFlowState:
        if state.get("technology_routes"):
            return {"step_markers": _update_step_markers(state, "step1")}
        if state.get("technical_description"):
            route = {
                "route_id": "R1",
                "route_name": "Default route",
                "route_summary": str(state.get("technical_description") or "").strip(),
                "key_unit_processes": [],
                "key_inputs": [],
                "key_outputs": [],
                "assumptions": [str(item) for item in (state.get("assumptions") or []) if str(item).strip()],
                "scope": str(state.get("scope") or "").strip(),
                "supported_dois": [],
                "route_evidence": {
                    "source_type": "expert_judgement",
                    "citations": [],
                    "notes": "Route summary provided without literature evidence.",
                },
            }
            return {
                "technology_routes": [route],
                "step_markers": _update_step_markers(state, "step1"),
            }
        if llm is None:
            summary = state.get("flow_summary") or {}
            base_name = summary.get("base_name_en") or "reference flow"
            operation = str(state.get("operation") or "produce").strip().lower()
            verb = "treatment/disposal" if operation in {"treat", "dispose", "disposal", "treatment"} else "production"
            route_summary = f"Generic {verb} of {base_name}. Assumptions: unspecified technology route; generic foreground process."
            route = {
                "route_id": "R1",
                "route_name": f"Typical {verb} route",
                "route_summary": route_summary,
                "key_unit_processes": [f"{verb.title()} of {base_name}"],
                "key_inputs": [],
                "key_outputs": [base_name],
                "assumptions": ["No quantified inventory available; amounts are placeholders."],
                "scope": "Generic scope",
                "supported_dois": [],
                "route_evidence": {
                    "source_type": "expert_judgement",
                    "citations": [],
                    "notes": "No LLM available; route uses generic assumptions.",
                },
            }
            return {
                "technical_description": route_summary,
                "assumptions": route["assumptions"],
                "scope": route["scope"],
                "technology_routes": [route],
                "step_markers": _update_step_markers(state, "step1"),
            }
        # Search for scientific references before invoking LLM
        flow_summary = state.get("flow_summary") or {}
        flow_name = flow_summary.get("base_name_en") or flow_summary.get("base_name_zh") or "reference flow"
        operation = str(state.get("operation") or "produce").strip()

        scientific_references = state.get("scientific_references") if isinstance(state.get("scientific_references"), dict) else {}
        references_text = ""
        references: list[dict[str, Any]] = []
        fulltext_entries: list[dict[str, Any]] = []

        # Build search query for scientific literature
        search_query = f"{operation} {flow_name} technology process route LCA life cycle assessment"
        flow_context = _flow_reference_context(flow_summary, include_comment=True)
        if flow_context:
            search_query = f"{search_query} {flow_context}"

        existing_search = scientific_references.get(REFERENCE_SEARCH_KEY)
        if isinstance(existing_search, dict):
            stored = existing_search.get("references")
            if isinstance(stored, list):
                references = stored
        if not references:
            references = _search_scientific_references(
                search_query,
                mcp_client=use_mcp_client,
                top_k=SCIENTIFIC_REFERENCE_TOP_K,
                country_preference=REFERENCE_COUNTRY_PREFERENCE,
                country_aliases=REFERENCE_COUNTRY_ALIASES,
            )
            scientific_references = _update_scientific_references(
                state,
                step=REFERENCE_SEARCH_KEY,
                query=search_query,
                references=references,
            )

        existing_fulltext = scientific_references.get(REFERENCE_FULLTEXT_KEY)
        if isinstance(existing_fulltext, dict):
            stored_fulltext = existing_fulltext.get("references")
            if isinstance(stored_fulltext, list):
                fulltext_entries = stored_fulltext
        if not fulltext_entries and references:
            fulltext_entries, missing_doi = _fetch_fulltext_references(
                references,
                mcp_client=use_mcp_client,
                fallback_query=search_query,
            )
            scientific_references = _update_scientific_references(
                {"scientific_references": scientific_references},
                step=REFERENCE_FULLTEXT_KEY,
                query=search_query,
                references=fulltext_entries,
                extra={"missing_doi": missing_doi} if missing_doi else None,
            )
        if llm is not None and fulltext_entries:
            if not _reference_clusters(scientific_references):
                cluster_result = _cluster_scientific_references(
                    llm=llm,
                    state={"scientific_references": scientific_references},
                    fulltext_entries=fulltext_entries,
                    flow_summary=flow_summary,
                    operation=operation,
                )
                if cluster_result:
                    scientific_references = dict(scientific_references)
                    scientific_references[REFERENCE_CLUSTERS_KEY] = cluster_result
        si_snippets = _load_si_snippets(scientific_references)
        if si_snippets:
            scientific_references = dict(scientific_references)
            scientific_references["si_snippets"] = si_snippets
        use_references = _references_usable(scientific_references)
        primary_dois = _primary_cluster_dois(scientific_references)
        if use_references and references:
            references_text = _format_references_for_prompt(references)
        stop_after = str(state.get("stop_after") or "").strip().lower()
        if stop_after in {"references", "reference", "refs", "papers", "sci"}:
            return {
                "scientific_references": scientific_references,
                "step_markers": _update_step_markers(state, "step1"),
            }

        # Build prompt with references
        enhanced_prompt = TECH_DESCRIPTION_PROMPT
        if references_text:
            enhanced_prompt = f"{TECH_DESCRIPTION_PROMPT}\n\n" f"Use the following scientific references as primary evidence for technology routes:\n" f"{references_text}\n"

        payload = {
            "prompt": enhanced_prompt,
            "context": {
                "operation": operation,
                "flow": flow_summary,
                "step_1c_reference_clusters": _reference_clusters(scientific_references) or {},
                "si_snippets": si_snippets,
            },
            "response_format": {"type": "json_object"},
        }
        raw = llm.invoke(payload)
        data = _ensure_dict(raw)
        routes = data.get("routes")
        cleaned_routes: list[dict[str, Any]] = []
        if isinstance(routes, list):
            for idx, route in enumerate(routes, start=1):
                if not isinstance(route, dict):
                    continue
                route_id = str(route.get("route_id") or route.get("routeId") or "").strip() or f"R{idx}"
                route_name = str(route.get("route_name") or route.get("routeName") or "").strip() or f"Route {route_id}"
                route_summary = str(route.get("route_summary") or route.get("routeSummary") or "").strip()
                key_unit_processes = [str(item).strip() for item in (route.get("key_unit_processes") or route.get("keyUnitProcesses") or []) if str(item).strip()]
                key_inputs = [str(item).strip() for item in (route.get("key_inputs") or route.get("keyInputs") or []) if str(item).strip()]
                key_outputs = [str(item).strip() for item in (route.get("key_outputs") or route.get("keyOutputs") or []) if str(item).strip()]
                assumptions = [str(item).strip() for item in (route.get("assumptions") or []) if str(item).strip()]
                scope = str(route.get("scope") or "").strip()
                supported_dois = _clean_string_list(route.get("supported_dois") or route.get("supportedDois"))
                if not supported_dois and primary_dois:
                    supported_dois = list(primary_dois)
                route_evidence = route.get("route_evidence") or route.get("routeEvidence")
                if not isinstance(route_evidence, dict):
                    route_evidence = {}
                citations = _clean_evidence_list(route_evidence.get("citations"))
                if not citations:
                    citations = _clean_evidence_list(route.get("citations"))
                if citations:
                    route_evidence["citations"] = citations
                source_type = _normalize_source_type(route_evidence.get("source_type") or route_evidence.get("sourceType"))
                if not source_type:
                    if citations:
                        source_type = _infer_source_type_from_evidence(citations)
                    else:
                        source_type = "literature" if use_references else "expert_judgement"
                route_evidence["source_type"] = source_type
                notes = route_evidence.get("notes")
                if not notes and not citations and source_type == "expert_judgement":
                    route_evidence["notes"] = "No usable references; route inferred from context."
                cleaned_routes.append(
                    {
                        "route_id": route_id,
                        "route_name": route_name,
                        "route_summary": route_summary,
                        "key_unit_processes": key_unit_processes,
                        "key_inputs": key_inputs,
                        "key_outputs": key_outputs,
                        "assumptions": assumptions,
                        "scope": scope,
                        "supported_dois": supported_dois,
                        "route_evidence": route_evidence,
                    }
                )
        if cleaned_routes:
            primary = cleaned_routes[0]
            return {
                "technical_description": primary.get("route_summary") or "",
                "assumptions": primary.get("assumptions") or [],
                "scope": primary.get("scope") or "",
                "technology_routes": cleaned_routes,
                "scientific_references": scientific_references,
                "step_markers": _update_step_markers(state, "step1"),
            }
        technical_description = str(data.get("technical_description") or "").strip()
        assumptions = [str(item) for item in (data.get("assumptions") or []) if str(item).strip()]
        scope = str(data.get("scope") or "").strip()
        fallback_route = {
            "route_id": "R1",
            "route_name": "Default route",
            "route_summary": technical_description,
            "key_unit_processes": [],
            "key_inputs": [],
            "key_outputs": [],
            "assumptions": assumptions,
            "scope": scope,
            "supported_dois": [],
            "route_evidence": {
                "source_type": "expert_judgement",
                "citations": [],
                "notes": "No route candidates returned; route inferred from context.",
            },
        }
        return {
            "technical_description": technical_description,
            "assumptions": assumptions,
            "scope": scope,
            "technology_routes": [fallback_route],
            "scientific_references": scientific_references,
            "step_markers": _update_step_markers(state, "step1"),
        }

    def split_processes(state: ProcessFromFlowState) -> ProcessFromFlowState:
        if state.get("processes"):
            if not state.get("process_routes"):
                normalized = _normalize_route_processes(
                    [item for item in (state.get("processes") or []) if isinstance(item, dict)],
                    flow_summary=state.get("flow_summary") or {},
                    route_name="Default route",
                )
                default_route = {
                    "route_id": "R1",
                    "route_name": "Default route",
                    "processes": normalized,
                }
                return {
                    "process_routes": [default_route],
                    "selected_route_id": "R1",
                    "processes": normalized,
                    "step_markers": _update_step_markers(state, "step2"),
                }
            return {"step_markers": _update_step_markers(state, "step2")}
        if llm is None:
            summary = state.get("flow_summary") or {}
            flow_name = summary.get("base_name_en") or "reference flow"
            operation = str(state.get("operation") or "produce").strip().lower()
            prefix = "Treatment of" if operation in {"treat", "dispose", "disposal", "treatment"} else "Production of"
            name_parts = {
                "base_name": f"{prefix} {flow_name}",
                "treatment_and_route": "Generic route",
                "mix_and_location": summary.get("mix_en") or "Unspecified mix/location",
                "quantitative_reference": _normalize_quantitative_reference(None, flow_name),
            }
            process_entry = {
                "process_id": "P1",
                "name": f"{name_parts['base_name']} | {name_parts['treatment_and_route']} | {name_parts['mix_and_location']} | {name_parts['quantitative_reference']}",
                "description": state.get("technical_description") or "",
                "is_reference_flow_process": True,
                "reference_flow_name": flow_name,
                "name_parts": name_parts,
                "geography": {},
                "structure": {
                    "technology": state.get("technical_description") or "",
                    "inputs": [],
                    "outputs": [flow_name],
                    "boundary": state.get("scope") or "",
                    "assumptions": state.get("assumptions") or [],
                },
                "exchange_keywords": {"inputs": [], "outputs": _dedupe_flows([flow_name])},
            }
            return {
                "processes": [process_entry],
                "process_routes": [{"route_id": "R1", "route_name": "Default route", "processes": [process_entry]}],
                "selected_route_id": "R1",
                "step_markers": _update_step_markers(state, "step2"),
            }
        # Search for scientific references for process splitting
        flow_summary = state.get("flow_summary") or {}
        flow_name = flow_summary.get("base_name_en") or flow_summary.get("base_name_zh") or "reference flow"
        tech_desc = state.get("technical_description") or ""
        operation = state.get("operation") or "produce"

        # Build search query focusing on unit processes and process decomposition
        search_query = f"{flow_name} {operation} unit process decomposition inventory LCA"
        if tech_desc:
            # Add key technical terms from description
            tech_preview = tech_desc[:100].strip()
            search_query = f"{search_query} {tech_preview}"

        scientific_references = state.get("scientific_references") if isinstance(state.get("scientific_references"), dict) else {}
        si_snippets = _load_si_snippets(scientific_references)
        if si_snippets:
            scientific_references = dict(scientific_references)
            scientific_references["si_snippets"] = si_snippets
        use_references = _references_usable(scientific_references)
        references: list[dict[str, Any]] = []
        references_text = ""
        if use_references:
            existing_step = scientific_references.get("step2")
            if isinstance(existing_step, dict):
                stored = existing_step.get("references")
                if isinstance(stored, list):
                    references = stored
            if not references:
                references = _search_scientific_references(
                    search_query,
                    mcp_client=use_mcp_client,
                    top_k=SCIENTIFIC_REFERENCE_TOP_K,
                    country_preference=REFERENCE_COUNTRY_PREFERENCE,
                    country_aliases=REFERENCE_COUNTRY_ALIASES,
                )
                scientific_references = _update_scientific_references(
                    {"scientific_references": scientific_references},
                    step="step2",
                    query=search_query,
                    references=references,
                )
            if references:
                references_text = _format_references_for_prompt(references)
        reference_clusters = _reference_clusters(scientific_references) if use_references else None

        # Build enhanced prompt with references
        enhanced_prompt = PROCESS_SPLIT_PROMPT
        if references_text:
            enhanced_prompt = f"{PROCESS_SPLIT_PROMPT}\n\n" f"Use the following scientific references to identify and split unit processes:\n" f"{references_text}\n"

        payload = {
            "prompt": enhanced_prompt,
            "context": {
                "flow": flow_summary,
                "technical_description": tech_desc,
                "routes": state.get("technology_routes") or [],
                "operation": operation,
                "step_1c_reference_clusters": reference_clusters or {},
                "si_snippets": si_snippets,
            },
            "response_format": {"type": "json_object"},
        }
        raw = llm.invoke(payload)
        data = _ensure_dict(raw)
        routes = data.get("routes")
        cleaned_routes: list[dict[str, Any]] = []
        if isinstance(routes, list):
            for route_idx, route in enumerate(routes, start=1):
                if not isinstance(route, dict):
                    continue
                route_id = str(route.get("route_id") or route.get("routeId") or "").strip() or f"R{route_idx}"
                route_name = str(route.get("route_name") or route.get("routeName") or "").strip() or f"Route {route_id}"
                processes = route.get("processes") or []
                if not isinstance(processes, list):
                    continue
                cleaned_processes: list[dict[str, Any]] = []
                for proc_idx, item in enumerate(processes, start=1):
                    if not isinstance(item, dict):
                        continue
                    process_id = str(item.get("process_id") or item.get("processId") or "").strip() or f"P{proc_idx}"
                    name_parts = item.get("name_parts") if isinstance(item.get("name_parts"), dict) else {}
                    name = str(item.get("name") or "").strip()
                    description = str(item.get("description") or "").strip()
                    structure = item.get("structure") if isinstance(item.get("structure"), dict) else {}
                    geo_raw = item.get("geography")
                    geography: dict[str, Any] = {}
                    if isinstance(geo_raw, dict):
                        geography = geo_raw
                    elif isinstance(geo_raw, str) and geo_raw.strip():
                        geography = {"description_of_restrictions_en": geo_raw.strip()}
                    reference_flow_name = str(item.get("reference_flow_name") or item.get("referenceFlowName") or "").strip()
                    cleaned_processes.append(
                        {
                            "process_id": process_id,
                            "name": name,
                            "description": description,
                            "is_reference_flow_process": bool(item.get("is_reference_flow_process")),
                            "reference_flow_name": reference_flow_name,
                            "name_parts": name_parts,
                            "structure": structure,
                            "geography": geography,
                        }
                    )
                if not cleaned_processes:
                    continue
                if sum(1 for proc in cleaned_processes if proc.get("is_reference_flow_process")) != 1:
                    for proc in cleaned_processes:
                        proc["is_reference_flow_process"] = False
                    cleaned_processes[-1]["is_reference_flow_process"] = True
                cleaned_processes = _normalize_route_processes(
                    cleaned_processes,
                    flow_summary=state.get("flow_summary") or {},
                    route_name=route_name,
                )
                cleaned_routes.append({"route_id": route_id, "route_name": route_name, "processes": cleaned_processes})
        selected_route_id = str(data.get("selected_route_id") or data.get("selectedRouteId") or "").strip()
        selected_route: dict[str, Any] | None = None
        if cleaned_routes:
            if selected_route_id:
                for route in cleaned_routes:
                    if route.get("route_id") == selected_route_id:
                        selected_route = route
                        break
            if selected_route is None:
                selected_route = cleaned_routes[0]
                selected_route_id = str(selected_route.get("route_id") or "")
        if selected_route:
            processes = selected_route.get("processes") or []
            tech_routes = state.get("technology_routes") or []
            selected_summary = None
            if isinstance(tech_routes, list):
                for route in tech_routes:
                    if isinstance(route, dict) and route.get("route_id") == selected_route_id:
                        selected_summary = route.get("route_summary")
                        break
            update: dict[str, Any] = {
                "process_routes": cleaned_routes,
                "selected_route_id": selected_route_id,
                "processes": processes,
                "scientific_references": scientific_references,
                "step_markers": _update_step_markers(state, "step2"),
            }
            if selected_summary and not state.get("technical_description"):
                update["technical_description"] = str(selected_summary).strip()
            return update

        processes = data.get("processes")
        if not isinstance(processes, list):
            raise ValueError("LLM did not return routes[] or processes[]")
        cleaned: list[dict[str, Any]] = []
        for item in processes:
            if not isinstance(item, dict):
                continue
            process_id = str(item.get("process_id") or item.get("processId") or "").strip()
            if not process_id:
                continue
            geo_raw = item.get("geography")
            geography: dict[str, Any] = {}
            if isinstance(geo_raw, dict):
                geography = geo_raw
            elif isinstance(geo_raw, str) and geo_raw.strip():
                geography = {"description_of_restrictions_en": geo_raw.strip()}
            cleaned.append(
                {
                    "process_id": process_id,
                    "name": str(item.get("name") or "").strip(),
                    "description": str(item.get("description") or "").strip(),
                    "is_reference_flow_process": bool(item.get("is_reference_flow_process")),
                    "geography": geography,
                }
            )
        if not cleaned:
            raise ValueError("No valid process entries returned by LLM")
        if sum(1 for proc in cleaned if proc.get("is_reference_flow_process")) != 1:
            cleaned[0]["is_reference_flow_process"] = True
            for proc in cleaned[1:]:
                proc["is_reference_flow_process"] = False
        return {
            "processes": cleaned,
            "scientific_references": scientific_references,
            "step_markers": _update_step_markers(state, "step2"),
        }

    def generate_exchanges(state: ProcessFromFlowState) -> ProcessFromFlowState:
        if state.get("process_exchanges"):
            updates: dict[str, Any] = {"step_markers": _update_step_markers(state, "step3")}
            coverage_metrics = state.get("coverage_metrics")
            if not isinstance(coverage_metrics, dict):
                coverage_metrics = _compute_coverage_metrics(state.get("process_exchanges") or [])
                updates["coverage_metrics"] = coverage_metrics
            if not isinstance(state.get("stop_rule_decision"), dict):
                updates["stop_rule_decision"] = _evaluate_stop_rules(state, coverage_metrics)
            if not isinstance(state.get("coverage_history"), list):
                updates["coverage_history"] = _append_coverage_history(state, coverage_metrics)
            return updates
        if llm is None:
            summary = state.get("flow_summary") or {}
            base_name = summary.get("base_name_en") or "reference flow"
            direction = _reference_direction(state.get("operation"))
            exchange = {
                "exchangeDirection": direction,
                "exchangeName": base_name,
                "generalComment": summary.get("general_comment_en") or "",
                "unit": None,
                "amount": None,
                "is_reference_flow": True,
            }
            exchange = _apply_exchange_evidence_defaults(
                exchange,
                use_references=False,
                fallback_reason="No LLM available; expert judgement applied.",
            )
            data_source = exchange.get("data_source")
            citations = _clean_evidence_list(data_source.get("citations")) if isinstance(data_source, dict) else []
            evidence = _clean_evidence_list(exchange.get("evidence"))
            _merge_value_evidence(exchange, citations, evidence)
            _ensure_exchange_comment_tags(exchange)
            process_exchanges = [{"process_id": "P1", "exchanges": [exchange]}]
            coverage_metrics = _compute_coverage_metrics(process_exchanges)
            stop_rule_decision = _evaluate_stop_rules(state, coverage_metrics)
            coverage_history = _append_coverage_history(state, coverage_metrics)
            return {
                "process_exchanges": process_exchanges,
                "coverage_metrics": coverage_metrics,
                "coverage_history": coverage_history,
                "stop_rule_decision": stop_rule_decision,
                "step_markers": _update_step_markers(state, "step3"),
            }
        # Search for scientific references for exchange generation
        flow_summary = state.get("flow_summary") or {}
        flow_name = flow_summary.get("base_name_en") or flow_summary.get("base_name_zh") or "reference flow"
        tech_desc = state.get("technical_description") or ""
        operation = str(state.get("operation") or "produce").strip()
        processes = state.get("processes") or []

        # Build search query focusing on inventory exchanges, inputs, outputs
        search_query = f"{flow_name} {operation} inventory exchanges inputs outputs emissions resources LCA"
        if tech_desc:
            search_query = f"{search_query} {_compact_text(tech_desc, limit=200)}"
        flow_context = _flow_reference_context(flow_summary, include_comment=False)
        if flow_context:
            search_query = f"{search_query} {flow_context}"
        if processes and isinstance(processes, list):
            # Add process names to search context
            process_names = " ".join([str(p.get("name") or "")[:50] for p in processes[:3] if isinstance(p, dict)])
            search_query = f"{search_query} {process_names}"

        scientific_references = state.get("scientific_references") if isinstance(state.get("scientific_references"), dict) else {}
        si_snippets = _load_si_snippets(scientific_references)
        if si_snippets:
            scientific_references = dict(scientific_references)
            scientific_references["si_snippets"] = si_snippets
        use_references = _references_usable(scientific_references)
        fallback_to_expert = _should_fallback_to_expert(scientific_references)
        default_reason = None
        if fallback_to_expert:
            default_reason = "No usable references and no SI hints; expert judgement applied."
        elif not use_references:
            default_reason = "No usable references; expert judgement applied."
        references: list[dict[str, Any]] = []
        references_text = ""
        if use_references:
            existing_step = scientific_references.get("step3")
            if isinstance(existing_step, dict):
                stored = existing_step.get("references")
                if isinstance(stored, list):
                    references = stored
            if not references:
                references = _search_scientific_references(
                    search_query,
                    mcp_client=use_mcp_client,
                    top_k=SCIENTIFIC_REFERENCE_TOP_K,
                    country_preference=REFERENCE_COUNTRY_PREFERENCE,
                    country_aliases=REFERENCE_COUNTRY_ALIASES,
                )
                scientific_references = _update_scientific_references(
                    {"scientific_references": scientific_references},
                    step="step3",
                    query=search_query,
                    references=references,
                )
            if references:
                references_text = _format_references_for_prompt(references)
        reference_clusters = _reference_clusters(scientific_references) if use_references else None

        # Build enhanced prompt with references
        enhanced_prompt = EXCHANGES_PROMPT
        if references_text:
            enhanced_prompt = f"{EXCHANGES_PROMPT}\n\n" f"Use the following scientific references to confirm exchange flow names and amounts:\n" f"{references_text}\n"

        payload = {
            "prompt": enhanced_prompt,
            "context": {
                "flow": flow_summary,
                "technical_description": tech_desc,
                "processes": processes,
                "operation": operation,
                "step_1c_reference_clusters": reference_clusters or {},
                "si_snippets": si_snippets,
            },
            "response_format": {"type": "json_object"},
        }
        raw = llm.invoke(payload)
        data = _ensure_dict(raw)
        processes = data.get("processes")
        if not isinstance(processes, list):
            raise ValueError("LLM did not return processes[] for exchanges")
        process_plan_index = {str(item.get("process_id") or ""): item for item in (state.get("processes") or []) if isinstance(item, dict)}
        target_flow_name = str((state.get("flow_summary") or {}).get("base_name_en") or "reference flow").strip()
        reference_direction = _reference_direction(state.get("operation"))
        cleaned_processes: list[dict[str, Any]] = []
        for proc in processes:
            if not isinstance(proc, dict):
                continue
            process_id = str(proc.get("process_id") or proc.get("processId") or "").strip()
            exchanges = proc.get("exchanges") or []
            if not isinstance(exchanges, list):
                exchanges = []
            plan = process_plan_index.get(process_id) or {}
            plan_reference_flow = str(plan.get("reference_flow_name") or "").strip()
            is_reference_flow_process = bool(plan.get("is_reference_flow_process"))
            if is_reference_flow_process:
                plan_reference_flow = target_flow_name
            structure = plan.get("structure") if isinstance(plan.get("structure"), dict) else {}
            structure_inputs = {_normalize_exchange_name(_strip_flow_label(value)) for value in _clean_string_list(structure.get("inputs")) if _strip_flow_label(value).strip()}
            structure_outputs = {_normalize_exchange_name(_strip_flow_label(value)) for value in _clean_string_list(structure.get("outputs")) if _strip_flow_label(value).strip()}
            cleaned_exchanges: list[dict[str, Any]] = []
            matched_reference = False
            for exchange in exchanges:
                if not isinstance(exchange, dict):
                    continue
                name = _strip_flow_label(str(exchange.get("exchangeName") or "").strip())
                raw_flow_type = _normalize_flow_type(exchange.get("flow_type") or exchange.get("flowType"))
                unit = str(exchange.get("unit") or "").strip() or "unit"
                amount = _coerce_amount_text(exchange.get("amount"))
                exchange_direction = str(exchange.get("exchangeDirection") or "").strip()
                name_key = _normalize_exchange_name(name)
                if name_key:
                    in_inputs = name_key in structure_inputs
                    in_outputs = name_key in structure_outputs
                    if in_inputs and not in_outputs:
                        exchange_direction = "Input"
                    elif in_outputs and not in_inputs:
                        exchange_direction = "Output"
                is_reference = False
                if plan_reference_flow:
                    is_reference = _flow_name_matches(name, plan_reference_flow)
                if is_reference:
                    matched_reference = True
                if is_reference and reference_direction:
                    exchange_direction = reference_direction
                flow_type = raw_flow_type or _infer_flow_type(
                    name,
                    direction=exchange_direction or "",
                    is_reference_flow=is_reference,
                )
                name = _ensure_media_suffix(
                    name,
                    direction=exchange_direction or "",
                    flow_type=flow_type,
                    is_reference_flow=is_reference,
                )
                cleaned_exchange = {
                    **exchange,
                    "exchangeName": name,
                    "unit": unit,
                    "amount": amount,
                    "is_reference_flow": is_reference,
                    "exchangeDirection": exchange_direction,
                    "flow_type": flow_type,
                }
                material_role = _normalize_material_role(exchange.get("material_role") or exchange.get("materialRole"))
                if material_role:
                    cleaned_exchange["material_role"] = material_role
                role_reason = str(exchange.get("role_reason") or exchange.get("roleReason") or "").strip()
                if role_reason:
                    cleaned_exchange["role_reason"] = role_reason
                balance_exclude = _normalize_balance_exclude(exchange.get("balance_exclude") or exchange.get("balanceExclude"))
                if balance_exclude is not None:
                    cleaned_exchange["balance_exclude"] = balance_exclude
                cleaned_exchange.pop("materialRole", None)
                cleaned_exchange.pop("roleReason", None)
                cleaned_exchange.pop("balanceExclude", None)
                cleaned_exchange = _apply_exchange_evidence_defaults(
                    cleaned_exchange,
                    use_references=use_references,
                    fallback_reason=default_reason,
                )
                data_source = cleaned_exchange.get("data_source")
                citations = _clean_evidence_list(data_source.get("citations")) if isinstance(data_source, dict) else []
                evidence = _clean_evidence_list(cleaned_exchange.get("evidence"))
                _merge_value_evidence(cleaned_exchange, citations, evidence)
                _ensure_exchange_comment_tags(cleaned_exchange)
                cleaned_exchanges.append(cleaned_exchange)
            if plan_reference_flow and not matched_reference:
                reference_exchange = {
                    "exchangeDirection": reference_direction,
                    "exchangeName": plan_reference_flow,
                    "generalComment": "Reference flow for this unit process.",
                    "unit": "unit",
                    "amount": "1",
                    "is_reference_flow": True,
                    "flow_type": "product",
                }
                reference_exchange = _apply_exchange_evidence_defaults(
                    reference_exchange,
                    use_references=use_references,
                    fallback_reason=default_reason,
                )
                data_source = reference_exchange.get("data_source")
                citations = _clean_evidence_list(data_source.get("citations")) if isinstance(data_source, dict) else []
                evidence = _clean_evidence_list(reference_exchange.get("evidence"))
                _merge_value_evidence(reference_exchange, citations, evidence)
                _ensure_exchange_comment_tags(reference_exchange)
                cleaned_exchanges.append(reference_exchange)
            cleaned_exchanges = _dedupe_reference_exchanges(
                cleaned_exchanges,
                reference_flow=plan_reference_flow or "",
                reference_direction=reference_direction,
            )
            if structure_inputs or structure_outputs:
                filtered = [
                    item
                    for item in cleaned_exchanges
                    if _should_keep_exchange(
                        item,
                        structure_inputs=structure_inputs,
                        structure_outputs=structure_outputs,
                        reference_flow=plan_reference_flow or "",
                    )
                ]
                if filtered:
                    cleaned_exchanges = filtered
            cleaned_processes.append({"process_id": process_id, "exchanges": cleaned_exchanges})
        coverage_metrics = _compute_coverage_metrics(cleaned_processes)
        stop_state = dict(state)
        stop_state["scientific_references"] = scientific_references
        stop_rule_decision = _evaluate_stop_rules(stop_state, coverage_metrics)
        coverage_history = _append_coverage_history(stop_state, coverage_metrics)
        return {
            "process_exchanges": cleaned_processes,
            "scientific_references": scientific_references,
            "coverage_metrics": coverage_metrics,
            "coverage_history": coverage_history,
            "stop_rule_decision": stop_rule_decision,
            "step_markers": _update_step_markers(state, "step3"),
        }

    def enrich_exchange_amounts(state: ProcessFromFlowState) -> ProcessFromFlowState:
        if not state.get("process_exchanges"):
            return {}
        if state.get("exchange_values_applied"):
            return {}
        if llm is None:
            return {"exchange_values_applied": True}

        scientific_references = state.get("scientific_references") if isinstance(state.get("scientific_references"), dict) else {}
        si_snippets = _load_si_snippets(scientific_references)
        if si_snippets:
            scientific_references = dict(scientific_references)
            scientific_references["si_snippets"] = si_snippets

        fulltext_entries = scientific_references.get(REFERENCE_FULLTEXT_KEY, {}).get("references")
        fulltext_list = fulltext_entries if isinstance(fulltext_entries, list) else []
        fulltext_text = _format_fulltext_entries_for_prompt(fulltext_list)
        candidates: list[dict[str, Any]] = []
        updated_exchanges = state.get("process_exchanges") or []
        if fulltext_text or si_snippets:
            payload = {
                "prompt": EXCHANGE_VALUE_PROMPT,
                "context": {
                    "flow": state.get("flow_summary") or {},
                    "operation": state.get("operation") or "produce",
                    "process_exchanges": state.get("process_exchanges") or [],
                    "fulltext_references": fulltext_text,
                    "si_snippets": si_snippets,
                },
                "response_format": {"type": "json_object"},
            }
            raw = llm.invoke(payload)
            data = _ensure_dict(raw)
            candidates = _normalize_exchange_value_candidates(data.get("processes"))
            updated_exchanges = _apply_exchange_value_candidates(state.get("process_exchanges") or [], candidates)
        process_plan_index = {str(item.get("process_id") or ""): item for item in (state.get("processes") or []) if isinstance(item, dict)}
        flow_summary = state.get("flow_summary") or {}
        operation = str(state.get("operation") or "produce").strip()

        def estimate_industry_average(
            *,
            exchange: dict[str, Any],
            process_plan: dict[str, Any] | None,
            references_text: str,
            allow_estimate: bool,
        ) -> tuple[str | None, str | None, list[str], str | None]:
            name_parts = (process_plan or {}).get("name_parts") if isinstance((process_plan or {}).get("name_parts"), dict) else {}
            quantitative_reference = str(name_parts.get("quantitative_reference") or "").strip()
            structure = (process_plan or {}).get("structure") if isinstance((process_plan or {}).get("structure"), dict) else {}
            boundary = str(structure.get("boundary") or "").strip()
            payload = {
                "prompt": INDUSTRY_AVERAGE_PROMPT,
                "context": {
                    "flow": flow_summary,
                    "operation": operation,
                    "process": process_plan or {},
                    "exchange": exchange,
                    "boundary": boundary,
                    "quantitative_reference": quantitative_reference,
                    "references": references_text,
                    "allow_estimate_without_references": allow_estimate,
                },
                "response_format": {"type": "json_object"},
            }
            raw = llm.invoke(payload)
            data = _ensure_dict(raw)
            return _parse_industry_average_response(data)

        for proc in updated_exchanges:
            if not isinstance(proc, dict):
                continue
            process_id = str(proc.get("process_id") or proc.get("processId") or "").strip()
            exchanges = proc.get("exchanges") or []
            if not isinstance(exchanges, list):
                continue
            plan = process_plan_index.get(process_id) or {}
            name_parts = plan.get("name_parts") if isinstance(plan.get("name_parts"), dict) else {}
            quantitative_reference = str(name_parts.get("quantitative_reference") or "").strip()
            structure = plan.get("structure") if isinstance(plan.get("structure"), dict) else {}
            boundary = str(structure.get("boundary") or "").strip()
            for exchange in exchanges:
                if not isinstance(exchange, dict):
                    continue
                amount = exchange.get("amount")
                if amount not in (None, "", 0):
                    continue
                if not quantitative_reference or not boundary:
                    continue
                query = _compose_industry_average_query(
                    exchange=exchange,
                    process_plan=plan,
                    flow_summary=flow_summary,
                    operation=operation,
                )
                kb_refs = _search_scientific_references(
                    query,
                    mcp_client=use_mcp_client,
                    top_k=INDUSTRY_AVERAGE_TOP_K,
                    country_preference=REFERENCE_COUNTRY_PREFERENCE,
                    country_aliases=REFERENCE_COUNTRY_ALIASES,
                )
                references_text = _format_references_for_prompt(kb_refs) if kb_refs else ""
                avg_amount, avg_unit, avg_evidence, _ = estimate_industry_average(
                    exchange=exchange,
                    process_plan=plan,
                    references_text=references_text,
                    allow_estimate=bool(kb_refs),
                )
                if not avg_amount:
                    avg_amount, avg_unit, avg_evidence, _ = estimate_industry_average(
                        exchange=exchange,
                        process_plan=plan,
                        references_text="",
                        allow_estimate=True,
                    )
                if not avg_amount:
                    continue
                exchange["amount"] = avg_amount
                if avg_unit:
                    exchange["unit"] = avg_unit
                citations, evidence = _split_citations_from_evidence(_clean_evidence_list(avg_evidence))
                if not citations:
                    ref_citations, ref_evidence = _split_citations_from_evidence(_extract_reference_evidence(kb_refs))
                    citations = ref_citations
                    if ref_evidence:
                        evidence = evidence + [item for item in ref_evidence if item not in evidence]
                if not evidence:
                    evidence = ["Industry average estimate (expert judgement)"]
                existing_ds = exchange.get("data_source") if isinstance(exchange.get("data_source"), dict) else {}
                existing_evidence = _clean_evidence_list(exchange.get("evidence"))
                merged_citations, merged_evidence = _normalize_citations_and_evidence(
                    _clean_evidence_list(existing_ds.get("citations")) + citations,
                    existing_evidence + evidence,
                )
                data_source = {"source_type": "expert_judgement"}
                if merged_citations:
                    data_source["citations"] = merged_citations
                exchange["data_source"] = data_source
                exchange["evidence"] = merged_evidence
                _merge_value_evidence(exchange, citations, evidence)
                if kb_refs:
                    scientific_references = _merge_industry_average_block(
                        scientific_references,
                        query_entry={
                            "process_id": process_id,
                            "exchange": str(exchange.get("exchangeName") or "").strip(),
                            "query": query,
                        },
                        references=kb_refs,
                    )
        updated_exchanges = _scale_exchange_amounts(updated_exchanges, process_plan_index)
        return {
            "process_exchanges": updated_exchanges,
            "exchange_value_candidates": candidates,
            "exchange_values_applied": True,
            "scientific_references": scientific_references,
        }

    def match_flows(state: ProcessFromFlowState) -> ProcessFromFlowState:
        if state.get("matched_process_exchanges"):
            return {}
        matched: list[dict[str, Any]] = []
        flow_summary = state.get("flow_summary") or {}
        reference_name = flow_summary.get("base_name_en") or ""
        raw_processes = state.get("process_exchanges") or []

        process_entries: list[tuple[str, list[dict[str, Any]]]] = []
        for proc in raw_processes:
            if not isinstance(proc, dict):
                continue
            process_id = str(proc.get("process_id") or proc.get("processId") or "").strip()
            exchanges = proc.get("exchanges") or []
            if not process_id or not isinstance(exchanges, list):
                continue
            valid_exchanges = [item for item in exchanges if isinstance(item, dict)]
            if not valid_exchanges:
                continue
            process_entries.append((process_id, valid_exchanges))

        total_processes = len(process_entries)
        total_exchanges = sum(len(exchanges) for _, exchanges in process_entries)
        progress_start = time.perf_counter()
        completed_exchanges = 0

        LOGGER.info(
            "process_from_flow.match_flows_started",
            process_total=total_processes,
            exchange_total=total_exchanges,
        )

        for process_index, (process_id, exchanges) in enumerate(process_entries, start=1):
            reference_flow_name = None
            for exchange in exchanges:
                if exchange.get("is_reference_flow"):
                    reference_flow_name = str(exchange.get("exchangeName") or "").strip() or None
                    if reference_flow_name:
                        break
            matched_exchanges: list[dict[str, Any]] = []
            process_exchange_total = len(exchanges)
            for exchange_index, exchange in enumerate(exchanges, start=1):
                name = str(exchange.get("exchangeName") or "").strip()
                comment = _strip_exchange_comment_tags(exchange.get("generalComment")) or None
                flow_type = _normalize_flow_type(exchange.get("flow_type")) or None
                direction = str(exchange.get("exchangeDirection") or "").strip() or None
                unit = str(exchange.get("unit") or "").strip() or None
                search_hints = exchange.get("search_hints") or []
                expected_compartment = _infer_media_suffix(f"{name} {comment or ''}")

                constraint_bits: list[str] = []
                if flow_type:
                    constraint_bits.append(f"flow_type={flow_type}")
                if direction:
                    constraint_bits.append(f"direction={direction}")
                if unit:
                    constraint_bits.append(f"unit={unit}")
                if expected_compartment:
                    constraint_bits.append(f"compartment={expected_compartment}")
                if isinstance(search_hints, list) and search_hints:
                    hint_text = ", ".join([str(item) for item in search_hints if str(item).strip()])
                    if hint_text:
                        constraint_bits.append(f"search_hints={hint_text}")

                query_desc = comment
                if constraint_bits:
                    constraint_text = "; ".join(constraint_bits)
                    query_desc = f"{query_desc} | constraints: {constraint_text}" if query_desc else f"constraints: {constraint_text}"

                query = FlowQuery(exchange_name=name or reference_name or "unknown_exchange", description=query_desc)
                candidates, unmatched = flow_search_fn(query)
                candidates = _dedupe_candidates_by_uuid_version(candidates)
                candidates = candidates[:10]
                # Build a minimal exchange dict for selector context.
                selector_exchange = {
                    "exchangeName": query.exchange_name,
                    "generalComment": comment,
                    "exchangeDirection": exchange.get("exchangeDirection"),
                    "is_reference_flow": exchange.get("is_reference_flow"),
                    "reference_flow_name": reference_flow_name,
                    "flow_type": exchange.get("flow_type"),
                    "material_role": exchange.get("material_role"),
                    "search_hints": exchange.get("search_hints") or [],
                }
                decision = selector.select(query, selector_exchange, candidates)
                selected = decision.candidate
                selected_reason = decision.reasoning
                if not selected_reason:
                    if selected is not None:
                        selected_reason = "Selected by LLM."
                    else:
                        selected_reason = "No suitable candidate selected by LLM."
                matched_exchanges.append(
                    {
                        **exchange,
                        "flow_search": {
                            "query": {"exchange_name": query.exchange_name, "description": query_desc},
                            "candidates": [
                                {
                                    "uuid": cand.uuid,
                                    "base_name": cand.base_name,
                                    "treatment_standards_routes": cand.treatment_standards_routes,
                                    "mix_and_location_types": cand.mix_and_location_types,
                                    "flow_properties": cand.flow_properties,
                                    "flow_type": cand.flow_type,
                                    "version": cand.version,
                                    "geography": cand.geography,
                                    "classification": cand.classification,
                                    "category_path": cand.category_path,
                                    "cas": cand.cas,
                                    "general_comment": cand.general_comment,
                                }
                                for cand in candidates
                            ],
                            "selected_uuid": selected.uuid if selected else None,
                            "selected_reason": selected_reason,
                            "selector": decision.strategy,
                            "unmatched": [getattr(item, "base_name", None) for item in (unmatched or [])],
                        },
                    }
                )

                completed_exchanges += 1
                elapsed_seconds = time.perf_counter() - progress_start
                remaining = max(total_exchanges - completed_exchanges, 0)
                eta_seconds = (elapsed_seconds / completed_exchanges * remaining) if completed_exchanges > 0 else None
                LOGGER.info(
                    "process_from_flow.match_flows_progress",
                    process_id=process_id,
                    process_index=process_index,
                    process_total=total_processes,
                    exchange_index=exchange_index,
                    exchange_total=process_exchange_total,
                    completed=completed_exchanges,
                    total=total_exchanges,
                    elapsed_seconds=round(elapsed_seconds, 2),
                    eta_seconds=round(float(eta_seconds), 2) if eta_seconds is not None else None,
                )
            matched.append({"process_id": process_id, "exchanges": matched_exchanges})

        elapsed_total = time.perf_counter() - progress_start
        LOGGER.info(
            "process_from_flow.match_flows_completed",
            process_total=total_processes,
            exchange_total=total_exchanges,
            elapsed_seconds=round(elapsed_total, 2),
        )
        return {"matched_process_exchanges": matched}

    def align_exchange_units(state: ProcessFromFlowState) -> ProcessFromFlowState:
        if state.get("unit_alignment_applied"):
            return {}
        matched_process_exchanges = state.get("matched_process_exchanges")
        summary = {
            "exchange_checked": 0,
            "exchange_converted": 0,
            "exchange_mismatch": 0,
            "exchange_review": 0,
            "exchange_missing_unit": 0,
        }
        if not isinstance(matched_process_exchanges, list):
            return {
                "unit_alignment_applied": True,
                "unit_alignment_summary": summary,
                "step_markers": _update_step_markers(state, "unit_alignment"),
            }

        updated_matches = copy.deepcopy(matched_process_exchanges)
        flow_cache: dict[str, FlowReferenceInfo | None] = {}
        crud_client: DatabaseCrudClient | None = None
        should_close_crud = False
        needs_crud = False
        for proc in updated_matches:
            exchanges = proc.get("exchanges") if isinstance(proc, dict) else None
            if not isinstance(exchanges, list):
                continue
            for exchange in exchanges:
                if not isinstance(exchange, dict):
                    continue
                flow_search = exchange.get("flow_search") if isinstance(exchange.get("flow_search"), dict) else {}
                if flow_search.get("selected_uuid"):
                    needs_crud = True
                    break
            if needs_crud:
                break
        if needs_crud:
            try:
                crud_client = DatabaseCrudClient(settings)
                should_close_crud = True
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("process_from_flow.unit_alignment_crud_init_failed", error=str(exc))
                crud_client = None

        try:
            for proc in updated_matches:
                exchanges = proc.get("exchanges") if isinstance(proc, dict) else None
                if not isinstance(exchanges, list):
                    continue
                for exchange in exchanges:
                    if not isinstance(exchange, dict):
                        continue
                    flow_search = exchange.get("flow_search") if isinstance(exchange.get("flow_search"), dict) else {}
                    selected_uuid = str(flow_search.get("selected_uuid") or "").strip() or None
                    if not selected_uuid:
                        continue
                    unit_text = str(exchange.get("unit") or "").strip()
                    amount_value = _parse_amount_value(exchange.get("amount"))
                    reference_info = None
                    if selected_uuid in flow_cache:
                        reference_info = flow_cache[selected_uuid]
                    else:
                        flow_dataset = None
                        if crud_client:
                            try:
                                flow_dataset = crud_client.select_flow(selected_uuid)
                            except Exception as exc:  # pylint: disable=broad-except
                                LOGGER.warning(
                                    "process_from_flow.unit_alignment_flow_select_failed",
                                    flow_id=selected_uuid,
                                    error=str(exc),
                                )
                        reference_info = _flow_reference_info_from_dataset(flow_dataset) if flow_dataset else None
                        flow_cache[selected_uuid] = reference_info

                    unit_check, converted_amount, reference_unit = _assess_unit_compatibility(
                        unit=unit_text,
                        amount=amount_value,
                        reference_info=reference_info,
                    )
                    unit_check["original_unit"] = unit_text or None
                    original_amount = exchange.get("amount")
                    unit_check["original_amount"] = str(original_amount).strip() if original_amount is not None else None
                    unit_check["review_required"] = unit_check.get("status") in {"mismatch", "review", "missing_unit"}

                    summary["exchange_checked"] += 1
                    status = unit_check.get("status")
                    if status == "converted" and converted_amount is not None and reference_unit:
                        exchange["amount"] = _format_amount_value(converted_amount)
                        exchange["unit"] = reference_unit
                        unit_check["converted_amount"] = exchange["amount"]
                        unit_check["converted_unit"] = reference_unit
                        summary["exchange_converted"] += 1
                    elif status == "mismatch":
                        summary["exchange_mismatch"] += 1
                        LOGGER.warning(
                            "process_from_flow.unit_alignment_mismatch",
                            exchange_name=str(exchange.get("exchangeName") or "").strip(),
                            exchange_unit=unit_check.get("exchange_unit"),
                            flow_unit_group=unit_check.get("flow_unit_group"),
                            reason=unit_check.get("reason"),
                        )
                    elif status == "review":
                        summary["exchange_review"] += 1
                    elif status == "missing_unit":
                        summary["exchange_missing_unit"] += 1

                    flow_search["unit_check"] = unit_check
                    exchange["flow_search"] = flow_search
                    tag_unit = _resolve_exchange_comment_tag_unit(
                        exchange,
                        reference_info=reference_info,
                        fallback_unit=reference_unit,
                    )
                    _ensure_exchange_comment_tags(exchange, preferred_unit=tag_unit)
        finally:
            if should_close_crud and crud_client:
                crud_client.close()

        LOGGER.info("process_from_flow.unit_alignment_completed", **summary)
        return {
            "matched_process_exchanges": updated_matches,
            "unit_alignment_applied": True,
            "unit_alignment_summary": summary,
            "step_markers": _update_step_markers(state, "unit_alignment"),
        }

    def density_conversion(state: ProcessFromFlowState) -> ProcessFromFlowState:
        if state.get("density_conversion_applied"):
            return {}
        summary = {
            "exchange_considered": 0,
            "exchange_converted": 0,
            "exchange_no_density": 0,
            "exchange_failed": 0,
            "exchange_skipped": 0,
        }
        if not state.get("allow_density_conversion"):
            summary["exchange_skipped"] = 0
            summary["skipped_reason"] = "disabled"
            return {
                "density_conversion_applied": True,
                "density_conversion_summary": summary,
                "step_markers": _update_step_markers(state, "density_conversion"),
            }
        if llm is None:
            summary["exchange_skipped"] = 0
            summary["skipped_reason"] = "llm_unavailable"
            return {
                "density_conversion_applied": True,
                "density_conversion_summary": summary,
                "step_markers": _update_step_markers(state, "density_conversion"),
            }
        matched_process_exchanges = state.get("matched_process_exchanges")
        if not isinstance(matched_process_exchanges, list):
            return {
                "density_conversion_applied": True,
                "density_conversion_summary": summary,
                "step_markers": _update_step_markers(state, "density_conversion"),
            }

        updated_matches = copy.deepcopy(matched_process_exchanges)
        flow_summary = state.get("flow_summary") or {}
        technical_description = str(state.get("technical_description") or "").strip()
        assumptions = [str(item).strip() for item in (state.get("assumptions") or []) if str(item).strip()]
        reference_direction = _reference_direction(state.get("operation"))
        process_plans = {str(item.get("process_id") or item.get("processId") or ""): item for item in (state.get("processes") or []) if isinstance(item, dict)}
        flow_cache: dict[str, FlowReferenceInfo | None] = {}
        crud_client: DatabaseCrudClient | None = None
        should_close_crud = False
        needs_crud = False
        for proc in updated_matches:
            exchanges = proc.get("exchanges") if isinstance(proc, dict) else None
            if not isinstance(exchanges, list):
                continue
            for exchange in exchanges:
                if not isinstance(exchange, dict):
                    continue
                flow_search = exchange.get("flow_search") if isinstance(exchange.get("flow_search"), dict) else {}
                if flow_search.get("selected_uuid"):
                    needs_crud = True
                    break
            if needs_crud:
                break
        if needs_crud:
            try:
                crud_client = DatabaseCrudClient(settings)
                should_close_crud = True
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("process_from_flow.density_conversion_crud_init_failed", error=str(exc))
                crud_client = None

        try:
            for proc in updated_matches:
                if not isinstance(proc, dict):
                    continue
                process_id = str(proc.get("process_id") or proc.get("processId") or "").strip()
                process_plan = process_plans.get(process_id) or {}
                exchanges = proc.get("exchanges") if isinstance(proc.get("exchanges"), list) else []
                for exchange in exchanges:
                    if not isinstance(exchange, dict):
                        continue
                    flow_search = exchange.get("flow_search") if isinstance(exchange.get("flow_search"), dict) else {}
                    selected_uuid = str(flow_search.get("selected_uuid") or "").strip() or None
                    if not selected_uuid:
                        continue
                    unit_check = flow_search.get("unit_check") if isinstance(flow_search.get("unit_check"), dict) else None
                    amount_value = _parse_amount_value(exchange.get("amount"))
                    unit_text = str(exchange.get("unit") or "").strip()
                    if unit_check is None:
                        reference_info = flow_cache.get(selected_uuid)
                        if reference_info is None:
                            flow_dataset = None
                            if crud_client:
                                try:
                                    flow_dataset = crud_client.select_flow(selected_uuid)
                                except Exception as exc:  # pylint: disable=broad-except
                                    LOGGER.warning(
                                        "process_from_flow.density_conversion_flow_select_failed",
                                        flow_id=selected_uuid,
                                        error=str(exc),
                                    )
                            reference_info = _flow_reference_info_from_dataset(flow_dataset) if flow_dataset else None
                            flow_cache[selected_uuid] = reference_info
                        unit_check, _, _ = _assess_unit_compatibility(
                            unit=unit_text,
                            amount=amount_value,
                            reference_info=reference_info,
                        )
                        unit_check["original_unit"] = unit_text or None
                        original_amount = exchange.get("amount")
                        unit_check["original_amount"] = str(original_amount).strip() if original_amount is not None else None
                    if unit_check.get("status") != "mismatch":
                        continue
                    exchange_dimension = unit_check.get("exchange_dimension")
                    flow_dimension = unit_check.get("flow_dimension")
                    if {exchange_dimension, flow_dimension} != {"mass", "volume"}:
                        continue
                    direction = _exchange_direction_for_dedupe(exchange, reference_direction=reference_direction)
                    flow_type = _exchange_flow_type_for_dedupe(exchange, direction=direction)
                    if flow_type not in {"product", "waste"}:
                        continue
                    if amount_value is None or not unit_text:
                        summary["exchange_skipped"] += 1
                        continue

                    reference_info = flow_cache.get(selected_uuid)
                    if reference_info is None:
                        flow_dataset = None
                        if crud_client:
                            try:
                                flow_dataset = crud_client.select_flow(selected_uuid)
                            except Exception as exc:  # pylint: disable=broad-except
                                LOGGER.warning(
                                    "process_from_flow.density_conversion_flow_select_failed",
                                    flow_id=selected_uuid,
                                    error=str(exc),
                                )
                        reference_info = _flow_reference_info_from_dataset(flow_dataset) if flow_dataset else None
                        flow_cache[selected_uuid] = reference_info
                    unit_group = reference_info.unit_group if reference_info else None
                    flow_reference_unit = unit_check.get("flow_reference_unit") or (unit_group.reference_unit if unit_group else None)
                    if not flow_reference_unit:
                        summary["exchange_failed"] += 1
                        continue

                    context_exchange = {
                        "exchangeName": exchange.get("exchangeName"),
                        "generalComment": exchange.get("generalComment"),
                        "unit": unit_text,
                        "amount": exchange.get("amount"),
                        "exchangeDirection": exchange.get("exchangeDirection"),
                        "flow_type": flow_type,
                    }
                    payload = {
                        "prompt": DENSITY_ESTIMATE_PROMPT,
                        "context": {
                            "flow": flow_summary,
                            "process": process_plan,
                            "exchange": context_exchange,
                            "technical_description": technical_description,
                            "assumptions": assumptions,
                            "unit_mismatch": unit_check,
                        },
                        "response_format": {"type": "json_object"},
                    }
                    summary["exchange_considered"] += 1
                    try:
                        raw = llm.invoke(payload)
                        data = _ensure_dict(raw)
                    except Exception as exc:  # pylint: disable=broad-except
                        summary["exchange_failed"] += 1
                        LOGGER.warning(
                            "process_from_flow.density_conversion_failed",
                            exchange_name=str(exchange.get("exchangeName") or "").strip(),
                            error=str(exc),
                        )
                        continue
                    density_value, density_unit, density_assumptions, source_type, notes = _parse_density_estimate_response(data)
                    if density_value is None or not density_unit:
                        summary["exchange_no_density"] += 1
                        continue
                    density_kg_per_m3 = _density_to_kg_per_m3(density_value, density_unit)
                    if density_kg_per_m3 is None or density_kg_per_m3 <= 0:
                        summary["exchange_failed"] += 1
                        continue

                    converted_amount = None
                    conversion_direction = "volume_to_mass" if exchange_dimension == "volume" else "mass_to_volume"
                    if exchange_dimension == "volume" and flow_dimension == "mass":
                        volume_m3 = _convert_amount_simple(amount_value, unit_text, "m3")
                        if volume_m3 is None:
                            summary["exchange_failed"] += 1
                            continue
                        mass_kg = volume_m3 * density_kg_per_m3
                        if unit_group:
                            converted_amount = _convert_amount_with_unit_group(mass_kg, "kg", unit_group)
                        if converted_amount is None:
                            converted_amount = _convert_amount_simple(mass_kg, "kg", flow_reference_unit)
                    elif exchange_dimension == "mass" and flow_dimension == "volume":
                        mass_kg = _convert_amount_simple(amount_value, unit_text, "kg")
                        if mass_kg is None:
                            summary["exchange_failed"] += 1
                            continue
                        volume_m3 = mass_kg / density_kg_per_m3
                        if unit_group:
                            converted_amount = _convert_amount_with_unit_group(volume_m3, "m3", unit_group)
                        if converted_amount is None:
                            converted_amount = _convert_amount_simple(volume_m3, "m3", flow_reference_unit)

                    if converted_amount is None:
                        summary["exchange_failed"] += 1
                        continue

                    converted_text = _format_amount_value(converted_amount)
                    density_value_text = _format_amount_value(density_value)
                    density_kg_text = _format_amount_value(density_kg_per_m3)
                    original_amount_text = str(exchange.get("amount") or "").strip() or None
                    original_unit_text = unit_text or None
                    exchange["amount"] = converted_text
                    exchange["unit"] = flow_reference_unit
                    assumption_text = density_assumptions or ""
                    if notes:
                        assumption_text = f"{assumption_text} {notes}".strip() if assumption_text else notes
                    density_used = {
                        "density_value": density_value_text,
                        "density_unit": density_unit,
                        "density_kg_per_m3": density_kg_text,
                        "assumptions": assumption_text or None,
                        "notes": notes,
                        "source_type": source_type,
                        "conversion_direction": conversion_direction,
                        "original_amount": original_amount_text,
                        "original_unit": original_unit_text,
                        "converted_amount": converted_text,
                        "converted_unit": flow_reference_unit,
                        "flow_uuid": selected_uuid,
                    }
                    exchange["density_used"] = density_used

                    unit_check["status"] = "converted_by_density"
                    unit_check["density_value"] = density_value_text
                    unit_check["density_unit"] = density_unit
                    unit_check["density_kg_per_m3"] = density_kg_text
                    unit_check["density_assumptions"] = density_assumptions or None
                    unit_check["density_source_type"] = source_type
                    unit_check["density_conversion_direction"] = conversion_direction
                    unit_check["converted_amount"] = converted_text
                    unit_check["converted_unit"] = flow_reference_unit
                    unit_check["reason"] = "converted_by_density_estimate"
                    unit_check["review_required"] = True
                    flow_search["unit_check"] = unit_check
                    exchange["flow_search"] = flow_search
                    tag_unit = _resolve_exchange_comment_tag_unit(
                        exchange,
                        reference_info=reference_info,
                        fallback_unit=flow_reference_unit,
                    )
                    _ensure_exchange_comment_tags(exchange, preferred_unit=tag_unit)
                    summary["exchange_converted"] += 1
        finally:
            if should_close_crud and crud_client:
                crud_client.close()

        LOGGER.info("process_from_flow.density_conversion_completed", **summary)
        return {
            "matched_process_exchanges": updated_matches,
            "density_conversion_applied": True,
            "density_conversion_summary": summary,
            "step_markers": _update_step_markers(state, "density_conversion"),
        }

    def build_sources(state: ProcessFromFlowState) -> ProcessFromFlowState:
        if state.get("source_datasets") or state.get("source_references"):
            return {}
        scientific_references = state.get("scientific_references") if isinstance(state.get("scientific_references"), dict) else {}
        doi_map, key_map = _collect_usage_tag_map(scientific_references)
        if doi_map or key_map:
            infos = _collect_reference_infos(scientific_references)
            infos = _filter_reference_infos_by_usage(infos, doi_map=doi_map, key_map=key_map)
            source_datasets, source_references = _build_source_datasets_from_references(
                scientific_references,
                translator=translator,
                reference_infos=infos,
            )
        else:
            source_datasets, source_references = _build_source_datasets_from_references(scientific_references, translator=translator)
        if not source_datasets:
            return {}
        LOGGER.info("process_from_flow.build_sources", count=len(source_datasets))
        return {"source_datasets": source_datasets, "source_references": source_references}

    def generate_intended_applications(state: ProcessFromFlowState) -> ProcessFromFlowState:
        if state.get("intended_applications"):
            return {}
        flow_summary = state.get("flow_summary") or {}
        technical_description = str(state.get("technical_description") or "").strip()
        scope = str(state.get("scope") or "").strip()
        assumptions = [str(item).strip() for item in (state.get("assumptions") or []) if str(item).strip()]
        processes = [item for item in (state.get("processes") or []) if isinstance(item, dict)]
        intended_map: dict[str, dict[str, str]] = {}
        for idx, proc in enumerate(processes):
            process_id = str(proc.get("process_id") or proc.get("processId") or "").strip() or f"process_{idx + 1}"
            process_name = str(proc.get("name") or "").strip()
            process_desc = str(proc.get("description") or "").strip() or technical_description
            structure = proc.get("structure") if isinstance(proc.get("structure"), dict) else {}
            process_scope = str(structure.get("boundary") or "").strip() or scope
            process_assumptions = _clean_string_list(structure.get("assumptions") or assumptions)
            fallback_en = _fallback_intended_applications(
                flow_summary=flow_summary,
                technical_description=process_desc,
                scope=process_scope,
                assumptions=process_assumptions,
                operation=state.get("operation"),
                process_name=process_name or None,
            )
            if llm is None:
                intended_map[process_id] = {"en": fallback_en}
                continue
            payload = {
                "prompt": INTENDED_APPLICATIONS_PROMPT,
                "context": {
                    "technical_description": process_desc,
                    "scope": process_scope,
                    "assumptions": process_assumptions,
                    "process": {
                        "process_id": process_id,
                        "process_name": process_name,
                    },
                },
                "response_format": {"type": "json_object"},
            }
            try:
                raw = llm.invoke(payload)
                data = _ensure_dict(raw)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning(
                    "process_from_flow.intended_applications_failed",
                    process_id=process_id,
                    error=str(exc),
                )
                data = {}
            applications = _normalize_bilingual_text(data.get("intended_applications") or data.get("intendedApplications") or data)
            if not applications:
                en_text = _join_texts(data.get("intended_applications_en") or data.get("intendedApplicationsEn"))
                zh_text = _join_texts(data.get("intended_applications_zh") or data.get("intendedApplicationsZh"))
                if en_text:
                    applications["en"] = en_text
                if zh_text:
                    applications["zh"] = zh_text
            if not applications:
                fallback_list = _clean_string_list(data.get("intended_applications") or data.get("intendedApplications"))
                if fallback_list:
                    applications = _split_bilingual_values(_dedupe_flows(fallback_list))
            if not applications.get("en"):
                applications["en"] = fallback_en
            if not applications.get("zh") and translator and applications.get("en"):
                translated = translator.translate(applications["en"], "zh")
                if translated:
                    applications["zh"] = translated.strip()
            intended_map[process_id] = applications
        if not intended_map:
            fallback_en = _fallback_intended_applications(
                flow_summary=flow_summary,
                technical_description=technical_description,
                scope=scope,
                assumptions=assumptions,
                operation=state.get("operation"),
            )
            intended_map["global"] = {"en": fallback_en}
        return {
            "intended_applications": intended_map,
            "step_markers": _update_step_markers(state, "intended_applications"),
        }

    def build_process_datasets(state: ProcessFromFlowState) -> ProcessFromFlowState:
        if state.get("process_datasets"):
            return {}
        flow_summary = state.get("flow_summary") or {}
        target_flow_name = flow_summary.get("base_name_en") or "reference flow"
        target_flow_name_zh = flow_summary.get("base_name_zh")
        tech_description = state.get("technical_description") or ""
        scope = state.get("scope") or ""
        assumptions = state.get("assumptions") or []
        intended_source = state.get("intended_applications")
        reference_direction = _reference_direction(state.get("operation"))
        source_datasets = [item for item in (state.get("source_datasets") or []) if isinstance(item, dict)]
        source_references = [item for item in (state.get("source_references") or []) if isinstance(item, dict)]
        if not source_references:
            source_references = _build_source_reference_entries(source_datasets)
        source_reference_items = _coerce_global_reference_items(source_references)
        default_reference_items = source_reference_items or _entry_level_compliance_reference()
        source_reference_index = _build_source_reference_index(source_datasets, source_references) if source_datasets and source_references else {}

        process_plans = {str(item.get("process_id") or ""): item for item in (state.get("processes") or []) if isinstance(item, dict)}
        exchange_plans = {str(item.get("process_id") or ""): item for item in (state.get("matched_process_exchanges") or []) if isinstance(item, dict)}
        results: list[dict[str, Any]] = []
        crud_client: DatabaseCrudClient | None = None
        flow_cache: dict[tuple[str, str | None], dict[str, Any]] = {}
        if exchange_plans:
            crud_client = DatabaseCrudClient(settings)

        try:
            for process_id, plan in process_plans.items():
                name_parts = plan.get("name_parts") if isinstance(plan.get("name_parts"), dict) else {}
                process_name = str(plan.get("name") or "").strip()
                base_name = str(name_parts.get("base_name") or process_name or f"Process {process_id}").strip()
                treatment_route = str(name_parts.get("treatment_and_route") or scope or "Unspecified treatment").strip()
                mix_location = str(name_parts.get("mix_and_location") or flow_summary.get("mix_en") or "Unspecified mix/location").strip()
                quantitative_ref = str(name_parts.get("quantitative_reference") or "").strip()
                if name_parts:
                    name_bits = [bit for bit in [base_name, treatment_route, mix_location, quantitative_ref] if bit]
                    process_name = " | ".join(name_bits) if name_bits else base_name
                if not process_name:
                    process_name = base_name
                base_name_for_dataset = base_name or process_name or f"Process {process_id}"
                process_desc = str(plan.get("description") or "").strip() or tech_description
                is_reference_flow_process = bool(plan.get("is_reference_flow_process"))
                process_reference_flow = str(plan.get("reference_flow_name") or "").strip()
                if is_reference_flow_process or not process_reference_flow:
                    process_reference_flow = target_flow_name

                intended_value: Any = None
                if isinstance(intended_source, dict):
                    if process_id in intended_source:
                        intended_value = intended_source.get(process_id)
                    elif "en" in intended_source or "zh" in intended_source:
                        intended_value = intended_source
                elif intended_source is not None:
                    intended_value = intended_source
                structure = plan.get("structure") if isinstance(plan.get("structure"), dict) else {}
                process_scope = str(structure.get("boundary") or "").strip() or scope
                process_assumptions = _clean_string_list(structure.get("assumptions") or assumptions)
                fallback_intended = _fallback_intended_applications(
                    flow_summary=flow_summary,
                    technical_description=str(process_desc or ""),
                    scope=str(process_scope or ""),
                    assumptions=[str(item).strip() for item in process_assumptions if str(item).strip()],
                    operation=state.get("operation"),
                    process_name=process_name or None,
                )
                intended_applications_ml = _coerce_bilingual_multilang(
                    intended_value,
                    translator=translator,
                    fallback_en=fallback_intended,
                )

                process_info_for_classifier = {
                    "dataSetInformation": {
                        "name": {
                            "baseName": base_name_for_dataset,
                            "treatmentStandardsRoutes": treatment_route,
                            "mixAndLocationTypes": mix_location,
                        },
                        "common:generalComment": process_desc,
                    }
                }

                geo_plan = plan.get("geography") if isinstance(plan.get("geography"), dict) else {}
                geo_candidates = [
                    value.strip()
                    for value in (
                        geo_plan.get("location_code"),
                        geo_plan.get("location"),
                        geo_plan.get("@location"),
                        geo_plan.get("code"),
                        geo_plan.get("location_name"),
                        geo_plan.get("location_name_en"),
                        geo_plan.get("location_name_zh"),
                    )
                    if isinstance(value, str) and value.strip()
                ]
                location_code = _resolve_location_code(
                    candidates=geo_candidates,
                    mix_location=mix_location,
                    llm=llm,
                    process_info=process_info_for_classifier,
                )
                restriction_en = _first_nonempty(
                    geo_plan.get("description_of_restrictions_en"),
                    geo_plan.get("descriptionOfRestrictions"),
                    geo_plan.get("description_of_restrictions"),
                    geo_plan.get("description"),
                )
                restriction_zh = _first_nonempty(
                    geo_plan.get("description_of_restrictions_zh"),
                    geo_plan.get("descriptionOfRestrictionsZh"),
                )
                restriction_entries = _build_multilang_entries(
                    restriction_en,
                    translator=translator,
                    zh_text=restriction_zh,
                )

                proc_uuid = str(uuid4())
                version = "01.01.000"

                classification_path: list[dict[str, Any]] = []
                if llm is not None:
                    try:
                        classifier = ProcessClassifier(llm)
                        classification_path = classifier.run(process_info_for_classifier)
                    except Exception as exc:  # pylint: disable=broad-except
                        LOGGER.warning("process_from_flow.classification_failed", process_id=process_id, error=str(exc))
                if not classification_path:
                    classification_path = [{"@level": "0", "@classId": "C", "#text": "Manufacturing"}]

                matched_entry = exchange_plans.get(process_id) or {}
                exchanges_raw = matched_entry.get("exchanges") or []
                exchange_items: list[ExchangesExchangeItem] = []
                process_reference_items: dict[str, GlobalReferenceTypeVariant1Item] = {}
                reference_internal_id: str | None = None
                next_internal_id = 1
                if isinstance(exchanges_raw, list):
                    exchanges_raw = _dedupe_product_uuid_exchanges(
                        exchanges_raw,
                        reference_direction=reference_direction,
                    )
                for exchange in exchanges_raw:
                    if not isinstance(exchange, dict):
                        continue
                    internal_id = str(next_internal_id)
                    next_internal_id += 1
                    name = str(exchange.get("exchangeName") or "").strip() or "unknown_exchange"
                    direction = str(exchange.get("exchangeDirection") or "").strip()
                    if direction not in {"Input", "Output"}:
                        direction = "Input"
                    if bool(exchange.get("is_reference_flow")):
                        direction = reference_direction
                    exchange_unit = str(exchange.get("unit") or "").strip()
                    selected_uuid = None
                    selected_version = None
                    selected_base_name = None
                    flow_search_block = exchange.get("flow_search") if isinstance(exchange.get("flow_search"), dict) else {}
                    if isinstance(flow_search_block, dict):
                        selected_uuid = flow_search_block.get("selected_uuid")
                        # Try to resolve selected version from candidates list.
                        candidates = flow_search_block.get("candidates")
                        if isinstance(candidates, list) and selected_uuid:
                            for cand in candidates:
                                if isinstance(cand, dict) and cand.get("uuid") == selected_uuid:
                                    selected_version = cand.get("version")
                                    selected_base_name = cand.get("base_name")
                                    break
                    if selected_uuid:
                        reference, reference_info, _ = _build_reference_from_selected_flow(
                            selected_uuid=str(selected_uuid),
                            selected_version=str(selected_version).strip() if selected_version is not None else None,
                            selected_base_name=str(selected_base_name or name),
                            fallback_name=name,
                            crud_client=crud_client,
                            flow_cache=flow_cache,
                            translator=translator,
                            stage="build_process_datasets",
                        )
                    else:
                        reference = _placeholder_flow_reference(name, translator=translator)
                        reference_info = None

                    amount = exchange.get("amount")
                    amount_text = _default_exchange_amount() if amount in (None, "", 0) else str(amount)

                    comment_text = _strip_exchange_comment_tags(exchange.get("generalComment"))
                    evidence_text = _format_evidence_for_comment(_clean_evidence_list(exchange.get("evidence")))
                    if evidence_text:
                        if comment_text:
                            if evidence_text not in comment_text:
                                comment_text = f"{comment_text} {evidence_text}"
                        else:
                            comment_text = evidence_text
                    comment_flow_kind = _exchange_flow_type_for_dedupe(exchange, direction=direction)
                    comment_unit = _resolve_exchange_comment_tag_unit(
                        exchange,
                        reference_info=reference_info,
                        fallback_unit=exchange_unit,
                    )
                    comment_text = _apply_exchange_comment_tags(
                        comment_text,
                        flow_kind=comment_flow_kind,
                        unit=comment_unit,
                    )
                    exchange_item = ExchangesExchangeItem(
                        data_set_internal_id=internal_id,
                        reference_to_flow_data_set=reference,
                        exchange_direction=direction,
                        mean_amount=amount_text,
                        resulting_amount=amount_text,
                        data_derivation_type_status="Estimated",
                    )
                    if comment_text:
                        comment_entries = _build_multilang_entries(comment_text, translator=translator)
                        exchange_item.general_comment = _as_multilang_list(comment_entries or comment_text)
                    if source_reference_index:
                        value_evidence = _dedupe_flows(_clean_evidence_list(exchange.get("value_citations")) + _clean_evidence_list(exchange.get("value_evidence")))
                        if value_evidence:
                            matched_sources = _match_source_references(value_evidence, source_reference_index)
                            matched_items = _coerce_global_reference_items(matched_sources)
                            if matched_items:
                                exchange_item.references_to_data_source = ExchangeItemReferencesToDataSource(reference_to_data_source=matched_items)
                        background_evidence = _dedupe_flows(value_evidence + _clean_evidence_list(exchange.get("evidence")) + _collect_exchange_citations(exchange))
                        if background_evidence:
                            matched_sources = _match_source_references(background_evidence, source_reference_index)
                            matched_items = _coerce_global_reference_items(matched_sources)
                            for item in matched_items:
                                ref_id = _reference_item_id(item)
                                if ref_id:
                                    process_reference_items[ref_id] = item
                    exchange_items.append(exchange_item)

                    if bool(exchange.get("is_reference_flow")):
                        reference_internal_id = internal_id

                if reference_internal_id is None:
                    # Ensure a reference exchange exists even if LLM failed to mark it.
                    reference_internal_id = str(next_internal_id)
                    exchange_items.append(
                        ExchangesExchangeItem(
                            data_set_internal_id=reference_internal_id,
                            reference_to_flow_data_set=_placeholder_flow_reference(
                                process_reference_flow,
                                translator=translator,
                            ),
                            exchange_direction=reference_direction,
                            mean_amount=_default_exchange_amount(),
                            resulting_amount=_default_exchange_amount(),
                            data_derivation_type_status="Estimated",
                            general_comment=_as_multilang_list(
                                _build_multilang_entries(
                                    _apply_exchange_comment_tags(
                                        flow_summary.get("general_comment_en") or "",
                                        flow_kind="product",
                                        unit="unit",
                                    ),
                                    translator=translator,
                                    zh_text=_apply_exchange_comment_tags(
                                        flow_summary.get("general_comment_zh") or "",
                                        flow_kind="product",
                                        unit="unit",
                                    ),
                                )
                            ),
                        )
                    )

                functional_unit = quantitative_ref or f"1 unit of {process_reference_flow}".strip()
                if is_reference_flow_process:
                    if reference_direction == "Input":
                        functional_unit = quantitative_ref or f"1 unit of {target_flow_name} treated"
                    else:
                        functional_unit = quantitative_ref or f"1 unit of {target_flow_name}"

                name_entries = _build_multilang_entries(base_name_for_dataset, translator=translator)
                treatment_entries = _build_multilang_entries(treatment_route, translator=translator)
                mix_entries = _build_multilang_entries(
                    mix_location,
                    translator=translator,
                    zh_text=flow_summary.get("mix_zh"),
                )
                comment_entries = _build_multilang_entries(process_desc, translator=translator)
                functional_unit_zh = None
                if target_flow_name_zh:
                    if reference_direction == "Input":
                        functional_unit_zh = f"处理 1 单位 {target_flow_name_zh}"
                    else:
                        functional_unit_zh = f"1 单位 {target_flow_name_zh}"
                functional_unit_entries = _build_multilang_entries(
                    functional_unit,
                    translator=translator,
                    zh_text=functional_unit_zh,
                )
                tech_text = "; ".join([text for text in [tech_description, process_desc, *assumptions] if text]).strip()
                tech_entries = _build_multilang_entries(tech_text, translator=translator)

                classification_items = _as_classification_items(classification_path)
                classification = DataSetInformationClassificationInformationCommonClassification(common_class=classification_items)
                classification_info = ProcessInformationDataSetInformationClassificationInformation(common_classification=classification)
                dataset_name = ProcessInformationDataSetInformationName(
                    base_name=_as_multilang_list(name_entries or process_name),
                    treatment_standards_routes=_as_multilang_list(treatment_entries or (scope or "Unspecified treatment")),
                    mix_and_location_types=_as_multilang_list(mix_entries or (flow_summary.get("mix_en") or "Unspecified mix/location")),
                )
                data_set_information = ProcessDataSetProcessInformationDataSetInformation(
                    common_uuid=proc_uuid,
                    name=dataset_name,
                    classification_information=classification_info,
                    common_general_comment=_as_multilang_list(comment_entries or process_desc),
                )
                quantitative_reference = ProcessDataSetProcessInformationQuantitativeReference(
                    type="Reference flow(s)",
                    reference_to_reference_flow=reference_internal_id or "1",
                    functional_unit_or_other=_as_multilang_list(functional_unit_entries or functional_unit),
                )
                time_info = ProcessDataSetProcessInformationTime(common_reference_year=int(datetime.now(timezone.utc).strftime("%Y")))
                location_kwargs: dict[str, Any] = {"location": location_code}
                if restriction_entries:
                    location_kwargs["description_of_restrictions"] = _as_multilang_list(restriction_entries)
                location = ProcessInformationGeographyLocationOfOperationSupplyOrProduction(**location_kwargs)
                geography = ProcessDataSetProcessInformationGeography(location_of_operation_supply_or_production=location)
                process_info_kwargs = {
                    "data_set_information": data_set_information,
                    "quantitative_reference": quantitative_reference,
                    "time": time_info,
                    "geography": geography,
                }
                if tech_entries or tech_text:
                    process_info_kwargs["technology"] = ProcessDataSetProcessInformationTechnology(technology_description_and_included_processes=_as_multilang_list(tech_entries or tech_text))
                process_information = ProcessesProcessDataSetProcessInformation(**process_info_kwargs)

                exchanges = ProcessesProcessDataSetExchanges(exchange=exchange_items)
                process_reference_list = list(process_reference_items.values())
                process_reference_items_final = process_reference_list or default_reference_items
                data_sources_kwargs: dict[str, Any] = {"reference_to_data_source": process_reference_items_final}
                modelling_and_validation = ProcessesProcessDataSetModellingAndValidation(
                    lci_method_and_allocation=ProcessDataSetModellingAndValidationLCIMethodAndAllocation(type_of_data_set="Unit process, single operation"),
                    data_sources_treatment_and_representativeness=(ProcessDataSetModellingAndValidationDataSourcesTreatmentAndRepresentativeness(**data_sources_kwargs)),
                    validation=ProcessDataSetModellingAndValidationValidation(review=ModellingAndValidationValidationReview(type="Not reviewed")),
                    compliance_declarations=_compliance_declarations(),
                )
                commissioner_and_goal = ProcessDataSetAdministrativeInformationCommonCommissionerAndGoal(
                    common_reference_to_commissioner=_contact_reference(),
                    common_intended_applications=intended_applications_ml,
                )
                administrative_information = ProcessesProcessDataSetAdministrativeInformation(
                    common_commissioner_and_goal=commissioner_and_goal,
                    data_entry_by=ProcessDataSetAdministrativeInformationDataEntryBy(
                        common_time_stamp=default_timestamp(),
                        common_reference_to_data_set_format=_dataset_format_reference(),
                        common_reference_to_person_or_entity_entering_the_data=_contact_reference(),
                    ),
                    publication_and_ownership=ProcessDataSetAdministrativeInformationPublicationAndOwnership(
                        common_data_set_version=version,
                        common_permanent_data_set_uri=build_portal_uri("process", proc_uuid, version),
                        common_reference_to_ownership_of_data_set=_contact_reference(),
                        common_copyright="false",
                        common_license_type="Free of charge for all users and uses",
                    ),
                )
                process_dataset = ProcessesProcessDataSet(
                    xmlns="http://lca.jrc.it/ILCD/Process",
                    xmlns_common="http://lca.jrc.it/ILCD/Common",
                    xmlns_xsi="http://www.w3.org/2001/XMLSchema-instance",
                    version="1.1",
                    locations="../ILCDLocations.xml",
                    xsi_schema_location="http://lca.jrc.it/ILCD/Process ../../schemas/ILCD_ProcessDataSet.xsd",
                    process_information=process_information,
                    exchanges=exchanges,
                    modelling_and_validation=modelling_and_validation,
                    administrative_information=administrative_information,
                )
                process_model = Processes(process_data_set=process_dataset)

                validated_on_init = False
                try:
                    entity = create_process(process_model, validate=True)
                    validated_on_init = True
                except Exception as exc:  # pylint: disable=broad-except
                    LOGGER.warning("process_from_flow.process_validation_failed", process_id=process_id, error=str(exc))
                    entity = create_process(process_model, validate=False)

                if validated_on_init:
                    errors = entity.last_validation_error()
                    if errors:
                        LOGGER.warning("process_from_flow.process_not_valid", process_id=process_id, error=str(errors))
                else:
                    valid = entity.validate(mode="pydantic")
                    if not valid:
                        errors = entity.last_validation_error()
                        LOGGER.warning("process_from_flow.process_not_valid", process_id=process_id, error=str(errors))
                results.append(entity.model.model_dump(mode="json", by_alias=True, exclude_none=True))
        finally:
            if crud_client:
                crud_client.close()

        step_markers = _update_step_markers(state, "process_datasets")
        checkpoint_state: ProcessFromFlowState = dict(state)
        checkpoint_state["process_datasets"] = results
        checkpoint_state["step_markers"] = step_markers
        _persist_runtime_state(checkpoint_state, reason="after_build_process_datasets")
        return {
            "process_datasets": results,
            "step_markers": step_markers,
        }

    def resolve_placeholders(state: ProcessFromFlowState) -> ProcessFromFlowState:
        if state.get("placeholder_resolution_applied"):
            return {}
        process_datasets = state.get("process_datasets")
        matched_process_exchanges = state.get("matched_process_exchanges")
        if not isinstance(process_datasets, list) or not isinstance(matched_process_exchanges, list):
            precheck = _empty_placeholder_precheck()
            return {
                "placeholder_precheck": precheck,
                "placeholder_resolutions": [],
                "placeholder_report": [],
                "placeholder_resolution_applied": True,
                "step_markers": _update_step_markers(state, "placeholders"),
            }
        updated_datasets = copy.deepcopy(process_datasets)
        updated_matches = copy.deepcopy(matched_process_exchanges)
        placeholders = _collect_placeholder_entries(state, process_datasets=updated_datasets)
        match_index = _index_matched_exchanges(updated_matches)
        precheck = _build_placeholder_precheck(placeholders, match_index)
        LOGGER.info(
            "process_from_flow.placeholder_precheck",
            placeholder_total=int(precheck.get("placeholder_total") or 0),
            matched_placeholder_total=int(precheck.get("matched_placeholder_total") or 0),
            unmatched_placeholder_total=int(precheck.get("unmatched_placeholder_total") or 0),
        )
        checkpoint_state: ProcessFromFlowState = dict(state)
        checkpoint_state["process_datasets"] = updated_datasets
        checkpoint_state["matched_process_exchanges"] = updated_matches
        checkpoint_state["placeholder_precheck"] = precheck
        _persist_runtime_state(checkpoint_state, reason="before_resolve_placeholders")
        if not placeholders:
            return {
                "placeholder_precheck": precheck,
                "placeholder_resolutions": [],
                "placeholder_report": [],
                "placeholder_resolution_applied": True,
                "step_markers": _update_step_markers(state, "placeholders"),
            }
        resolutions: list[dict[str, Any]] = []
        flow_search_client: FlowSearchClient | None = None
        crud_client: DatabaseCrudClient | None = None
        flow_reference_cache: dict[tuple[str, str | None], dict[str, Any]] = {}
        try:
            flow_search_client = FlowSearchClient(settings=settings)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("process_from_flow.placeholder_flow_search_client_failed", error=str(exc))
            flow_search_client = None
        try:
            crud_client = DatabaseCrudClient(settings)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("process_from_flow.placeholder_crud_init_failed", error=str(exc))
            crud_client = None
        try:
            for entry in placeholders:
                matched_exchange = _match_placeholder_exchange(entry, match_index)
                if not matched_exchange:
                    resolutions.append(
                        {
                            **entry,
                            "resolution_status": "unresolved",
                            "resolution_reason": "No matching exchange found in matched_process_exchanges.",
                            "candidate_list": [],
                            "candidate_list_raw": [],
                            "query_payload": {},
                        }
                    )
                    continue

                exchange_name = str(matched_exchange.get("exchangeName") or entry.get("exchange_name") or "").strip()
                comment = _strip_exchange_comment_tags(matched_exchange.get("generalComment"))
                flow_type = _normalize_flow_type(matched_exchange.get("flow_type")) or matched_exchange.get("flow_type")
                direction = str(matched_exchange.get("exchangeDirection") or "").strip()
                unit = str(matched_exchange.get("unit") or "").strip()
                search_hints = matched_exchange.get("search_hints") if isinstance(matched_exchange.get("search_hints"), list) else []
                expected_compartment = _infer_media_suffix(f"{exchange_name} {comment}")

                query_payload = _build_placeholder_one_shot_query_payload(
                    llm=llm,
                    exchange_name=exchange_name or str(entry.get("exchange_name") or "unknown_exchange"),
                    comment=comment,
                    flow_type=flow_type,
                    direction=direction,
                    unit=unit,
                    expected_compartment=expected_compartment,
                    search_hints=search_hints,
                )
                query_exchange_name = str(query_payload.get("exchange_name") or exchange_name or "").strip() or exchange_name
                query_description = str(query_payload.get("description") or "").strip() or None
                query = FlowQuery(exchange_name=query_exchange_name or "unknown_exchange", description=query_description)
                query_text = str(query_payload.get("query_text") or "").strip()

                candidates: list[FlowCandidate] = []
                search_error: str | None = None
                if not query_text:
                    search_error = "Empty one-shot query."
                elif flow_search_client is None:
                    search_error = "FlowSearchClient unavailable for one-shot search."
                else:
                    try:
                        candidates = _one_shot_flow_search_candidates(flow_search_client, query_text)
                    except Exception as exc:  # pylint: disable=broad-except
                        search_error = str(exc)
                        LOGGER.warning(
                            "process_from_flow.placeholder_one_shot_search_failed",
                            exchange=exchange_name,
                            error=search_error,
                        )
                candidates = _dedupe_candidates_by_uuid_version(candidates)
                selection_candidates = candidates[:10]
                filter_info = {
                    "mode": "disabled",
                    "reason": "one_shot_top10_direct_to_selector",
                    "expected_flow_type": flow_type,
                    "expected_compartment": expected_compartment,
                    "candidate_total": len(selection_candidates),
                }

                selector_exchange = {
                    "exchangeName": exchange_name,
                    "generalComment": comment,
                    "flow_type": flow_type,
                    "search_hints": search_hints,
                    "exchangeDirection": direction,
                }
                selected: FlowCandidate | None = None
                selected_reason: str | None = None
                selected_confidence: float | None = None
                selected_strategy: str | None = None
                selected_version: str | None = None
                if selection_candidates:
                    llm_uuid, llm_reason, llm_confidence = _select_placeholder_uuid_with_llm(
                        llm=llm,
                        exchange_context=selector_exchange,
                        query_payload=query_payload,
                        candidates=selection_candidates,
                    )
                    if llm_uuid:
                        selected = next((item for item in selection_candidates if str(item.uuid or "").strip() == llm_uuid), None)
                        if selected is not None and getattr(selected, "uuid", None):
                            selected_reason = llm_reason or "Selected by placeholder UUID selector."
                            selected_confidence = llm_confidence
                            selected_strategy = "llm_uuid_selector"
                        else:
                            selected = None
                            selected_reason = "LLM selected UUID not present in candidate list."
                            selected_confidence = llm_confidence
                            selected_strategy = "llm_uuid_selector"
                    elif llm is not None:
                        selected_reason = llm_reason or "LLM selector returned null."
                        selected_confidence = llm_confidence
                        selected_strategy = "llm_uuid_selector"
                    else:
                        decision = selector.select(query, selector_exchange, selection_candidates)
                        selected = decision.candidate
                        if selected is not None and not getattr(selected, "uuid", None):
                            selected = None
                            selected_reason = "Selected candidate missing UUID; skipping resolution."
                        else:
                            selected_reason = decision.reasoning
                            selected_confidence = decision.score
                            selected_strategy = decision.strategy

                if selected is None:
                    if not selected_reason:
                        if search_error:
                            selected_reason = f"One-shot flow search failed: {search_error}"
                        elif not selection_candidates:
                            selected_reason = "No candidates returned by one-shot flow search."
                        else:
                            selected_reason = "No suitable candidate selected."
                resolution_status = "resolved" if selected is not None else "unresolved"
                candidate_list = _serialize_candidate_list(selection_candidates)
                raw_candidate_list = _serialize_candidate_list(candidates)

                flow_search = matched_exchange.get("flow_search") if isinstance(matched_exchange.get("flow_search"), dict) else {}
                flow_search["secondary_query"] = {"exchange_name": query.exchange_name, "description": query.description}
                flow_search["one_shot_query"] = query_text
                flow_search["one_shot_query_payload"] = {
                    "exchange_name": query_payload.get("exchange_name"),
                    "description": query_payload.get("description"),
                    "cas": query_payload.get("cas"),
                    "classification_hints": query_payload.get("classification_hints"),
                    "flow_type": query_payload.get("flow_type"),
                    "direction": query_payload.get("direction"),
                    "unit": query_payload.get("unit"),
                    "compartment": query_payload.get("compartment"),
                }
                flow_search["resolution_status"] = resolution_status
                flow_search["resolution_reason"] = selected_reason
                flow_search["resolution_confidence"] = selected_confidence
                flow_search["resolution_selector"] = selected_strategy
                flow_search["resolution_candidates"] = candidate_list
                flow_search["resolution_raw_candidates"] = raw_candidate_list
                flow_search["resolution_filters"] = filter_info
                if search_error:
                    flow_search["resolution_search_error"] = search_error
                if selected is not None:
                    flow_search["selected_uuid"] = selected.uuid
                    flow_search["selected_reason"] = selected_reason
                    selected_version = str(selected.version or "").strip() or None
                    if selected_version:
                        flow_search["selected_version"] = selected_version
                matched_exchange["flow_search"] = flow_search

                if selected is not None:
                    process_index = entry.get("process_index")
                    exchange_index = entry.get("exchange_index")
                    if isinstance(process_index, int) and isinstance(exchange_index, int):
                        try:
                            reference, _, resolved_version = _build_reference_from_selected_flow(
                                selected_uuid=str(selected.uuid or ""),
                                selected_version=str(selected.version).strip() if selected.version is not None else None,
                                selected_base_name=str(selected.base_name or exchange_name),
                                fallback_name=exchange_name or str(entry.get("exchange_name") or "unknown_exchange"),
                                crud_client=crud_client,
                                flow_cache=flow_reference_cache,
                                translator=translator,
                                stage="resolve_placeholders",
                            )
                            selected_version = resolved_version or selected_version
                        except Exception as exc:  # pylint: disable=broad-except
                            LOGGER.warning(
                                "process_from_flow.placeholder_reference_build_failed",
                                exchange=exchange_name,
                                flow_uuid=str(selected.uuid or ""),
                                error=str(exc),
                            )
                            reference = _candidate_reference(selected, translator=translator)
                        reference_payload = reference.model_dump(mode="json", by_alias=True, exclude_none=True)
                        try:
                            updated_exchange = updated_datasets[process_index].get("processDataSet", {}).get("exchanges", {}).get("exchange", [])[exchange_index]
                            if isinstance(updated_exchange, dict):
                                updated_exchange["referenceToFlowDataSet"] = reference_payload
                        except Exception as exc:  # pylint: disable=broad-except
                            LOGGER.warning(
                                "process_from_flow.placeholder_update_failed",
                                process_index=process_index,
                                exchange_index=exchange_index,
                                error=str(exc),
                            )
                        if selected_version:
                            flow_search["selected_version"] = selected_version

                resolutions.append(
                    {
                        **entry,
                        "exchange_name": exchange_name or entry.get("exchange_name"),
                        "flow_type": flow_type,
                        "unit": unit or None,
                        "resolution_status": resolution_status,
                        "resolution_reason": selected_reason,
                        "resolution_confidence": selected_confidence,
                        "selected_uuid": selected.uuid if selected is not None else None,
                        "selected_version": selected_version,
                        "candidate_list": candidate_list,
                        "candidate_list_raw": raw_candidate_list,
                        "filters": filter_info,
                        "query_payload": {
                            "exchange_name": query_payload.get("exchange_name"),
                            "description": query_payload.get("description"),
                            "cas": query_payload.get("cas"),
                            "classification_hints": query_payload.get("classification_hints"),
                            "flow_type": query_payload.get("flow_type"),
                            "direction": query_payload.get("direction"),
                            "unit": query_payload.get("unit"),
                            "compartment": query_payload.get("compartment"),
                            "query_text": query_text,
                        },
                        "search_error": search_error,
                    }
                )
        finally:
            if flow_search_client is not None:
                flow_search_client.close()
            if crud_client is not None:
                crud_client.close()
        return {
            "matched_process_exchanges": updated_matches,
            "process_datasets": updated_datasets,
            "placeholder_precheck": precheck,
            "placeholder_resolutions": resolutions,
            "placeholder_report": resolutions,
            "placeholder_resolution_applied": True,
            "step_markers": _update_step_markers(state, "placeholders"),
        }

    def balance_review(state: ProcessFromFlowState) -> ProcessFromFlowState:
        if "balance_review" in state:
            return {}
        auto_balance_revise = bool(state.get("auto_balance_revise")) and not bool(state.get("balance_revise_applied"))
        process_exchanges = state.get("matched_process_exchanges")
        if not isinstance(process_exchanges, list):
            process_exchanges = state.get("process_exchanges")
        if not isinstance(process_exchanges, list):
            process_exchanges = []
        process_plans = {str(item.get("process_id") or item.get("processId") or ""): item for item in (state.get("processes") or []) if isinstance(item, dict)}
        summary = {
            "process_total": 0,
            "process_ok": 0,
            "process_check": 0,
            "process_insufficient": 0,
            "mass_core_check_processes": 0,
            "mass_core_insufficient_processes": 0,
            "unit_mismatch_total": 0,
            "unit_mismatch_processes": 0,
            "density_estimate_total": 0,
            "density_estimate_processes": 0,
            "unit_assumption_total": 0,
            "unit_assumption_processes": 0,
            "mapping_conflict_total": 0,
            "mapping_conflict_processes": 0,
            "balance_excluded_total": 0,
            "balance_excluded_processes": 0,
            "role_missing_total": 0,
            "role_missing_processes": 0,
        }
        if not process_exchanges:
            return {
                "balance_review": [],
                "balance_review_summary": summary,
                "step_markers": _update_step_markers(state, "balance_review"),
            }

        reference_direction = _reference_direction(state.get("operation"))
        flow_cache: dict[str, FlowReferenceInfo | None] = {}
        crud_client: DatabaseCrudClient | None = None
        should_close_crud = False
        needs_crud = False
        for proc in process_exchanges:
            exchanges = proc.get("exchanges") if isinstance(proc, dict) else None
            if not isinstance(exchanges, list):
                continue
            for exchange in exchanges:
                if not isinstance(exchange, dict):
                    continue
                flow_search = exchange.get("flow_search") if isinstance(exchange.get("flow_search"), dict) else {}
                if flow_search.get("selected_uuid"):
                    needs_crud = True
                    break
            if needs_crud:
                break
        if needs_crud:
            try:
                crud_client = DatabaseCrudClient(settings)
                should_close_crud = True
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("process_from_flow.balance_crud_init_failed", error=str(exc))
                crud_client = None

        reviews: list[dict[str, Any]] = []
        try:
            for index, proc in enumerate(process_exchanges):
                if not isinstance(proc, dict):
                    continue
                process_id = str(proc.get("process_id") or proc.get("processId") or "").strip()
                plan = process_plans.get(process_id) if process_id else None
                process_name = ""
                if isinstance(plan, dict):
                    process_name = str(plan.get("name") or "").strip()
                    if not process_name:
                        name_parts = plan.get("name_parts") if isinstance(plan.get("name_parts"), dict) else {}
                        process_name = str(name_parts.get("base_name") or "").strip()
                if not process_name:
                    process_name = f"Process {process_id or (index + 1)}"
                exchanges = proc.get("exchanges") if isinstance(proc.get("exchanges"), list) else []
                balance_state = {
                    "mass": {"inputs": 0.0, "outputs": 0.0, "count": 0, "unit": ""},
                    "energy": {"inputs": 0.0, "outputs": 0.0, "count": 0, "unit": ""},
                }
                core_mass_state = {"inputs": 0.0, "outputs": 0.0, "count": 0, "unit": ""}
                unit_mismatches: list[dict[str, Any]] = []
                density_estimates: list[dict[str, Any]] = []
                unit_assumptions: list[dict[str, Any]] = []
                core_mass_refs: list[dict[str, Any]] = []
                excluded_count = 0
                missing_role_count = 0
                core_exchange_count = 0
                for exchange in exchanges:
                    if not isinstance(exchange, dict):
                        continue
                    material_role = _normalize_material_role(exchange.get("material_role") or exchange.get("materialRole"))
                    if not material_role or material_role == "unknown":
                        missing_role_count += 1
                    balance_exclude = _normalize_balance_exclude(exchange.get("balance_exclude") or exchange.get("balanceExclude"))
                    direction = str(exchange.get("exchangeDirection") or "").strip()
                    if direction not in {"Input", "Output"}:
                        direction = "Input"
                    if bool(exchange.get("is_reference_flow")):
                        direction = reference_direction
                    comment_tags = _parse_exchange_comment_tags(exchange.get("generalComment"))
                    flow_kind = _normalize_flow_type(comment_tags.get(_EXCHANGE_KIND_TAG_NAME))
                    if not flow_kind:
                        flow_kind = _exchange_flow_type_for_dedupe(exchange, direction=direction)
                    is_core_exchange = _is_core_mass_exchange(
                        material_role=material_role,
                        flow_kind=flow_kind,
                        balance_exclude=balance_exclude,
                    )
                    if is_core_exchange:
                        core_exchange_count += 1
                    if balance_exclude is True or (material_role in _BALANCE_EXCLUDE_ROLES):
                        excluded_count += 1
                        continue
                    amount_value = _parse_amount_value(exchange.get("amount"))
                    exchange_unit = str(exchange.get("unit") or "").strip()
                    flow_search = exchange.get("flow_search") if isinstance(exchange.get("flow_search"), dict) else {}
                    selected_uuid = str(flow_search.get("selected_uuid") or "").strip() or None
                    reference_info = None
                    if selected_uuid:
                        if selected_uuid in flow_cache:
                            reference_info = flow_cache[selected_uuid]
                        else:
                            flow_dataset = None
                            if crud_client:
                                try:
                                    flow_dataset = crud_client.select_flow(selected_uuid)
                                except Exception as exc:  # pylint: disable=broad-except
                                    LOGGER.warning(
                                        "process_from_flow.balance_flow_select_failed",
                                        flow_id=selected_uuid,
                                        error=str(exc),
                                    )
                            reference_info = _flow_reference_info_from_dataset(flow_dataset) if flow_dataset else None
                            flow_cache[selected_uuid] = reference_info

                    if selected_uuid and exchange_unit:
                        unit_check = flow_search.get("unit_check") if isinstance(flow_search.get("unit_check"), dict) else None
                        if unit_check is None:
                            unit_check, _, _ = _assess_unit_compatibility(
                                unit=exchange_unit,
                                amount=amount_value,
                                reference_info=reference_info,
                            )
                        if unit_check.get("status") == "mismatch":
                            unit_mismatches.append(
                                {
                                    "exchange_name": str(exchange.get("exchangeName") or "").strip(),
                                    "exchange_unit": unit_check.get("exchange_unit") or exchange_unit or None,
                                    "flow_unit_group": unit_check.get("flow_unit_group"),
                                    "flow_reference_unit": unit_check.get("flow_reference_unit"),
                                    "flow_uuid": selected_uuid,
                                    "reason": unit_check.get("reason"),
                                }
                            )
                    density_used = exchange.get("density_used")
                    if isinstance(density_used, dict):
                        density_estimates.append(
                            {
                                "exchange_name": str(exchange.get("exchangeName") or "").strip(),
                                "density_value": density_used.get("density_value"),
                                "density_unit": density_used.get("density_unit"),
                                "assumptions": density_used.get("assumptions"),
                                "source_type": density_used.get("source_type"),
                            }
                        )

                    balance_unit, assumption_reasons = _resolve_exchange_balance_unit(
                        exchange,
                        reference_info=reference_info,
                        material_role=material_role,
                        flow_kind=flow_kind,
                    )
                    if assumption_reasons:
                        unit_assumptions.append(
                            {
                                "exchange_name": str(exchange.get("exchangeName") or "").strip(),
                                "exchange_direction": direction,
                                "original_unit": exchange_unit or None,
                                "resolved_unit": balance_unit or None,
                                "reasons": assumption_reasons,
                                "flow_uuid": selected_uuid,
                            }
                        )
                    if amount_value is None or not balance_unit:
                        continue
                    dimension, converted, unit_label = _convert_exchange_amount_for_balance(amount_value, balance_unit, reference_info)
                    if dimension and converted is not None:
                        dim_state = balance_state[dimension]
                        if dim_state["unit"] and unit_label and dim_state["unit"] != unit_label:
                            continue
                        if unit_label:
                            dim_state["unit"] = unit_label
                        if direction == "Input":
                            dim_state["inputs"] += converted
                        else:
                            dim_state["outputs"] += converted
                        dim_state["count"] += 1
                        if dimension == "mass" and is_core_exchange:
                            if core_mass_state["unit"] and unit_label and core_mass_state["unit"] != unit_label:
                                continue
                            if unit_label:
                                core_mass_state["unit"] = unit_label
                            if direction == "Input":
                                core_mass_state["inputs"] += converted
                            else:
                                core_mass_state["outputs"] += converted
                            core_mass_state["count"] += 1
                            if selected_uuid:
                                core_mass_refs.append(
                                    {
                                        "uuid": selected_uuid,
                                        "exchange_name": str(exchange.get("exchangeName") or "").strip(),
                                        "direction": direction,
                                        "is_reference_flow": bool(exchange.get("is_reference_flow")),
                                    }
                                )

                mass_summary = _balance_state_summary(balance_state["mass"])
                mass_core_summary = _balance_state_summary(core_mass_state)
                energy_summary = _balance_state_summary(balance_state["energy"])
                status = _merge_balance_status(mass_summary.get("status"), energy_summary.get("status"))
                mapping_conflicts: list[dict[str, Any]] = []
                reference_output_uuids = {
                    str(item.get("uuid") or "").strip() for item in core_mass_refs if item.get("is_reference_flow") and str(item.get("direction") or "").strip() == reference_direction
                }
                if reference_output_uuids:
                    seen_conflicts: set[tuple[str, str]] = set()
                    for item in core_mass_refs:
                        uuid_value = str(item.get("uuid") or "").strip()
                        if not uuid_value:
                            continue
                        if str(item.get("direction") or "").strip() != "Input":
                            continue
                        if item.get("is_reference_flow"):
                            continue
                        if uuid_value not in reference_output_uuids:
                            continue
                        key = (uuid_value, str(item.get("exchange_name") or "").strip())
                        if key in seen_conflicts:
                            continue
                        seen_conflicts.add(key)
                        mapping_conflicts.append(
                            {
                                "flow_uuid": uuid_value,
                                "exchange_name": str(item.get("exchange_name") or "").strip(),
                                "reason": "input_uses_reference_output_flow_uuid",
                            }
                        )
                if core_exchange_count and mass_core_summary.get("status") != "ok":
                    status = "check"
                if unit_mismatches or mapping_conflicts:
                    status = "check"
                note = _build_balance_note(balance_state)
                reviews.append(
                    {
                        "process_id": process_id,
                        "process_name": process_name,
                        "status": status,
                        "mass": mass_summary,
                        "mass_core": mass_core_summary,
                        "energy": energy_summary,
                        "note": note,
                        "unit_mismatches": unit_mismatches,
                        "unit_mismatch_count": len(unit_mismatches),
                        "density_estimates": density_estimates,
                        "density_estimate_count": len(density_estimates),
                        "unit_assumptions": unit_assumptions,
                        "unit_assumption_count": len(unit_assumptions),
                        "mapping_conflicts": mapping_conflicts,
                        "mapping_conflict_count": len(mapping_conflicts),
                        "core_exchange_count": core_exchange_count,
                        "balance_excluded_count": excluded_count,
                        "role_missing_count": missing_role_count,
                        "exchange_count": len(exchanges),
                    }
                )
                summary["process_total"] += 1
                if unit_mismatches:
                    summary["unit_mismatch_total"] += len(unit_mismatches)
                    summary["unit_mismatch_processes"] += 1
                if density_estimates:
                    summary["density_estimate_total"] += len(density_estimates)
                    summary["density_estimate_processes"] += 1
                if unit_assumptions:
                    summary["unit_assumption_total"] += len(unit_assumptions)
                    summary["unit_assumption_processes"] += 1
                if mapping_conflicts:
                    summary["mapping_conflict_total"] += len(mapping_conflicts)
                    summary["mapping_conflict_processes"] += 1
                if excluded_count:
                    summary["balance_excluded_total"] += excluded_count
                    summary["balance_excluded_processes"] += 1
                if missing_role_count:
                    summary["role_missing_total"] += missing_role_count
                    summary["role_missing_processes"] += 1
                if mass_core_summary.get("status") == "check":
                    summary["mass_core_check_processes"] += 1
                elif core_exchange_count and mass_core_summary.get("status") == "insufficient":
                    summary["mass_core_insufficient_processes"] += 1
                if status == "check":
                    summary["process_check"] += 1
                    LOGGER.warning(
                        "process_from_flow.balance_review_flagged",
                        process_id=process_id,
                        process_name=process_name,
                        mass_status=mass_summary.get("status"),
                        mass_ratio=mass_summary.get("ratio"),
                        mass_core_status=mass_core_summary.get("status"),
                        mass_core_ratio=mass_core_summary.get("ratio"),
                        energy_status=energy_summary.get("status"),
                        energy_ratio=energy_summary.get("ratio"),
                        mapping_conflict_count=len(mapping_conflicts),
                        unit_assumption_count=len(unit_assumptions),
                    )
                elif status == "ok":
                    summary["process_ok"] += 1
                else:
                    summary["process_insufficient"] += 1
        finally:
            if should_close_crud and crud_client:
                crud_client.close()

        LOGGER.info("process_from_flow.balance_review_completed", **summary)
        if auto_balance_revise:
            revised_matches, revised_datasets, revise_summary = _apply_balance_auto_revisions(
                state,
                process_exchanges=process_exchanges,
                reviews=reviews,
                settings=settings,
            )
            LOGGER.info(
                "process_from_flow.balance_auto_revise_completed",
                revised_processes=int(revise_summary.get("revised_processes") or 0),
                revised_exchanges=int(revise_summary.get("revised_exchanges") or 0),
                candidate_processes=int(revise_summary.get("candidate_processes") or 0),
            )
            if int(revise_summary.get("revised_exchanges") or 0) > 0:
                checkpoint_state: ProcessFromFlowState = dict(state)
                checkpoint_state["matched_process_exchanges"] = revised_matches
                if isinstance(revised_datasets, list):
                    checkpoint_state["process_datasets"] = revised_datasets
                checkpoint_state["balance_review_initial"] = reviews
                checkpoint_state["balance_review_summary_initial"] = summary
                checkpoint_state["balance_revise_applied"] = True
                checkpoint_state["balance_revise_summary"] = revise_summary
                _persist_runtime_state(checkpoint_state, reason="after_balance_auto_revise")

                rerun_state: ProcessFromFlowState = dict(state)
                rerun_state["matched_process_exchanges"] = revised_matches
                if isinstance(revised_datasets, list):
                    rerun_state["process_datasets"] = revised_datasets
                rerun_state["balance_revise_applied"] = True
                rerun_state.pop("balance_review", None)
                rerun_state.pop("balance_review_summary", None)
                rerun_result = balance_review(rerun_state)
                rerun_result["matched_process_exchanges"] = revised_matches
                if isinstance(revised_datasets, list):
                    rerun_result["process_datasets"] = revised_datasets
                rerun_result["balance_revise_applied"] = True
                rerun_result["balance_revise_summary"] = revise_summary
                rerun_result["balance_review_initial"] = reviews
                rerun_result["balance_review_summary_initial"] = summary
                return rerun_result
            return {
                "balance_review": reviews,
                "balance_review_summary": summary,
                "balance_review_initial": reviews,
                "balance_review_summary_initial": summary,
                "balance_revise_applied": True,
                "balance_revise_summary": revise_summary,
                "step_markers": _update_step_markers(state, "balance_review"),
            }
        return {
            "balance_review": reviews,
            "balance_review_summary": summary,
            "step_markers": _update_step_markers(state, "balance_review"),
        }

    def generate_data_cutoff_principles(state: ProcessFromFlowState) -> ProcessFromFlowState:
        if state.get("data_cutoff_principles_applied") and state.get("data_treatment_principles_applied"):
            return {}
        process_datasets = state.get("process_datasets")
        if not isinstance(process_datasets, list) or not process_datasets:
            return {
                "data_cutoff_principles_applied": True,
                "data_treatment_principles_applied": True,
            }
        process_list = [item for item in (state.get("processes") or []) if isinstance(item, dict)]
        process_ids = [str(item.get("process_id") or "").strip() for item in process_list]
        plan_index = {str(item.get("process_id") or "").strip(): item for item in process_list if isinstance(item, dict)}
        existing_source = state.get("data_cut_off_and_completeness_principles")
        balance_review = state.get("balance_review") if isinstance(state.get("balance_review"), list) else []
        balance_summary = state.get("balance_review_summary") if isinstance(state.get("balance_review_summary"), dict) else {}
        balance_index = {str(item.get("process_id") or "").strip(): item for item in balance_review if isinstance(item, dict) and str(item.get("process_id") or "").strip()}
        updated_datasets = copy.deepcopy(process_datasets)
        principles_map: dict[str, dict[str, str]] = {}
        summary_map: dict[str, dict[str, Any]] = {}
        treatment_map: dict[str, dict[str, str]] = {}
        treatment_summary_map: dict[str, dict[str, Any]] = {}

        for idx, dataset in enumerate(updated_datasets):
            if not isinstance(dataset, dict):
                continue
            process_id = process_ids[idx] if idx < len(process_ids) else f"process_{idx + 1}"
            plan = plan_index.get(process_id) or {}
            process_name = str(plan.get("name") or "").strip()
            summary = _summarize_cutoff_inputs(state, process_id=process_id if process_id else None)
            summary_map[process_id] = summary

            value_source: Any = None
            if isinstance(existing_source, dict):
                if process_id in existing_source:
                    value_source = existing_source.get(process_id)
                elif "en" in existing_source or "zh" in existing_source:
                    value_source = existing_source
            elif existing_source is not None:
                value_source = existing_source

            principles = _normalize_bilingual_text(value_source)
            if not principles and llm is not None:
                payload = {
                    "prompt": DATA_CUTOFF_COMPLETENESS_PROMPT,
                    "context": {
                        "summary": summary,
                        "flow": state.get("flow_summary") or {},
                        "operation": state.get("operation") or "produce",
                        "process": {"process_id": process_id, "process_name": process_name},
                    },
                    "response_format": {"type": "json_object"},
                }
                try:
                    raw = llm.invoke(payload)
                    data = _ensure_dict(raw)
                except Exception as exc:  # pylint: disable=broad-except
                    LOGGER.warning(
                        "process_from_flow.data_cutoff_generation_failed",
                        process_id=process_id,
                        error=str(exc),
                    )
                    data = {}
                principles = _normalize_bilingual_text(data.get("data_cut_off_and_completeness_principles") or data.get("dataCutOffAndCompletenessPrinciples") or data)
                if not principles:
                    en_text = _join_texts(data.get("data_cut_off_and_completeness_principles_en") or data.get("dataCutOffAndCompletenessPrinciplesEn"))
                    zh_text = _join_texts(data.get("data_cut_off_and_completeness_principles_zh") or data.get("dataCutOffAndCompletenessPrinciplesZh"))
                    if en_text:
                        principles["en"] = en_text
                    if zh_text:
                        principles["zh"] = zh_text
                if not principles:
                    fallback_list = _clean_string_list(data.get("data_cut_off_and_completeness_principles") or data.get("dataCutOffAndCompletenessPrinciples"))
                    if fallback_list:
                        principles = _split_bilingual_values(_dedupe_flows(fallback_list))

            fallback_en_list = _fallback_cutoff_principles(summary)
            fallback_en = _join_texts(fallback_en_list)
            if fallback_en and not principles.get("en"):
                principles["en"] = fallback_en
            if principles.get("en") and not principles.get("zh") and translator:
                translated = translator.translate(principles["en"], "zh")
                if translated:
                    principles["zh"] = translated.strip()
            principles_map[process_id] = principles

            entries = _coerce_bilingual_payload(principles, translator=translator, fallback_en=fallback_en)
            process_block = dataset.get("processDataSet")
            if not isinstance(process_block, dict):
                continue
            modelling = process_block.get("modellingAndValidation")
            if not isinstance(modelling, dict):
                modelling = {}
                process_block["modellingAndValidation"] = modelling
            dsr = modelling.get("dataSourcesTreatmentAndRepresentativeness")
            if not isinstance(dsr, dict):
                dsr = {}
                modelling["dataSourcesTreatmentAndRepresentativeness"] = dsr
            dsr["dataCutOffAndCompletenessPrinciples"] = list(entries)
            balance_entry = balance_index.get(process_id)
            treatment_summary = _balance_entry_snapshot(balance_entry)
            treatment_summary_map[process_id] = treatment_summary
            treatment_fallback_en = _join_texts(
                _fallback_data_treatment_principles(
                    process_id=process_id,
                    balance_entry=balance_entry,
                    balance_summary=balance_summary,
                )
            )
            treatment_entries = _coerce_bilingual_payload(
                {"en": treatment_fallback_en} if treatment_fallback_en else {},
                translator=translator,
                fallback_en=treatment_fallback_en,
            )
            dsr["dataTreatmentAndExtrapolationsPrinciples"] = list(treatment_entries)
            treatment_map[process_id] = _normalize_bilingual_text(treatment_entries)
        return {
            "process_datasets": updated_datasets,
            "data_cut_off_and_completeness_principles": principles_map,
            "data_treatment_and_extrapolations_principles": treatment_map,
            "data_treatment_principles_applied": True,
            "data_treatment_summary": treatment_summary_map,
            "data_cutoff_principles_applied": True,
            "data_cutoff_summary": summary_map,
            "step_markers": _update_step_markers(state, "data_cutoff"),
        }

    graph.add_node("load_flow", load_flow)
    graph.add_node("describe_technology", describe_technology)
    graph.add_node("split_processes", split_processes)
    graph.add_node("generate_exchanges", generate_exchanges)
    graph.add_node("enrich_exchange_amounts", enrich_exchange_amounts)
    graph.add_node("match_flows", match_flows)
    graph.add_node("align_exchange_units", align_exchange_units)
    graph.add_node("density_conversion", density_conversion)
    graph.add_node("build_sources", build_sources)
    graph.add_node("generate_intended_applications", generate_intended_applications)
    graph.add_node("build_process_datasets", build_process_datasets)
    graph.add_node("resolve_placeholders", resolve_placeholders)
    graph.add_node("balance_review", balance_review)
    graph.add_node("generate_data_cutoff_principles", generate_data_cutoff_principles)

    graph.set_entry_point("load_flow")
    graph.add_edge("load_flow", "describe_technology")
    graph.add_conditional_edges(
        "describe_technology",
        lambda state: END if str(state.get("stop_after") or "").strip().lower() in {"tech", "references", "reference", "refs", "papers", "sci"} else "split_processes",
    )
    graph.add_conditional_edges(
        "split_processes",
        lambda state: END if (str(state.get("stop_after") or "").strip().lower() == "processes") else "generate_exchanges",
    )
    graph.add_conditional_edges(
        "generate_exchanges",
        lambda state: "enrich_exchange_amounts",
    )
    graph.add_conditional_edges(
        "enrich_exchange_amounts",
        lambda state: END if (str(state.get("stop_after") or "").strip().lower() == "exchanges") else "match_flows",
    )
    graph.add_conditional_edges(
        "match_flows",
        lambda state: END if (str(state.get("stop_after") or "").strip().lower() == "matches") else "align_exchange_units",
    )
    graph.add_edge("align_exchange_units", "density_conversion")
    graph.add_edge("density_conversion", "build_sources")
    graph.add_conditional_edges(
        "build_sources",
        lambda state: END if (str(state.get("stop_after") or "").strip().lower() == "sources") else "generate_intended_applications",
    )
    graph.add_edge("generate_intended_applications", "build_process_datasets")
    graph.add_edge("build_process_datasets", "resolve_placeholders")
    graph.add_edge("resolve_placeholders", "balance_review")
    graph.add_edge("balance_review", "generate_data_cutoff_principles")
    graph.add_edge("generate_data_cutoff_principles", END)

    return graph.compile()


@dataclass(slots=True)
class ProcessFromFlowService:
    """Facade that builds ILCD process datasets from a reference flow via LangGraph."""

    llm: LanguageModelProtocol | None = None
    settings: Settings | None = None
    flow_search_fn: FlowSearchFn | None = None
    selector: CandidateSelector | None = None
    translator: Translator | None = None
    mcp_client: MCPToolClient | None = None

    def run(
        self,
        *,
        flow_path: str | Path,
        operation: str = "produce",
        initial_state: dict[str, Any] | None = None,
        stop_after: str | None = None,
    ) -> ProcessFromFlowState:
        settings = self.settings or get_settings()
        flow_search_fn = self.flow_search_fn or search_flows
        selector: CandidateSelector
        if self.selector is not None:
            selector = self.selector
        elif self.llm is not None:
            selector = LLMCandidateSelector(self.llm, fallback=NoFallbackCandidateSelector())
        else:
            selector = SimilarityCandidateSelector()

        # Create MCP client if not provided and we want to use scientific references
        mcp_client = self.mcp_client
        should_close_mcp = False
        if mcp_client is None and self.llm is not None:
            # Only create MCP client when LLM is available (scientific references only useful with LLM)
            try:
                mcp_client = MCPToolClient(settings)
                should_close_mcp = True
                LOGGER.info("process_from_flow.mcp_client_created", service="TianGong_KB_Remote")
            except Exception as exc:
                LOGGER.warning("process_from_flow.mcp_client_creation_failed", error=str(exc))
                mcp_client = None

        try:
            app = _build_langgraph(
                llm=self.llm,
                settings=settings,
                flow_search_fn=flow_search_fn,
                selector=selector,
                translator=self.translator,
                mcp_client=mcp_client,
            )
            initial: ProcessFromFlowState = {"flow_path": str(flow_path), "operation": operation}
            if stop_after:
                initial["stop_after"] = stop_after
            if initial_state:
                initial.update({k: v for k, v in initial_state.items() if k not in {"flow_path", "operation"}})
            return app.invoke(initial)
        finally:
            if should_close_mcp and mcp_client:
                mcp_client.close()
                LOGGER.info("process_from_flow.mcp_client_closed")
