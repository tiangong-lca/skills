"""Prompt-packaged extractors for the JSON-LD Stage 1 pipeline."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from time import sleep
from typing import Any, TypedDict
from uuid import uuid4

from tiangong_lca_spec.core.constants import build_dataset_format_reference
from tiangong_lca_spec.core.exceptions import ProcessExtractionError
from tiangong_lca_spec.core.json_utils import parse_json_response
from tiangong_lca_spec.core.logging import get_logger
from tiangong_lca_spec.core.uris import build_local_dataset_uri, build_portal_uri
from tiangong_lca_spec.jsonld.converters import (
    DEFAULT_LICENSE,
    ILCD_COMPLIANCE_SOURCE_UUID,
    ILCD_COMPLIANCE_SOURCE_VERSION,
    ILCD_FLOW_XMLNS,
    ILCD_SOURCE_XMLNS,
    TIANGONG_CONTACT_UUID,
    TIANGONG_CONTACT_VERSION,
)
from tiangong_lca_spec.location import LocationCatalog, extract_location_response, get_location_catalog
from tiangong_lca_spec.process_extraction.extractors import (
    LanguageModelProtocol,
    LocationNormalizer,
    ProcessClassifier,
    ProductFlowClassifier,
)
from tiangong_lca_spec.process_extraction.tidas_mapping import build_tidas_process_dataset
from tiangong_lca_spec.process_extraction.validators import is_placeholder_value
from tiangong_lca_spec.tidas import get_schema_repository
from tiangong_lca_spec.tidas.flow_classification_registry import ensure_valid_product_flow_classification
from tiangong_lca_spec.workflow.artifacts import DEFAULT_DATA_SET_VERSION, flow_compliance_declarations

LOGGER = get_logger(__name__)


TIANGONG_CONTACT_URI = build_local_dataset_uri("contact data set", TIANGONG_CONTACT_UUID, TIANGONG_CONTACT_VERSION)
ILCD_COMPLIANCE_SOURCE_URI = build_local_dataset_uri("source data set", ILCD_COMPLIANCE_SOURCE_UUID, ILCD_COMPLIANCE_SOURCE_VERSION)

MASS_FLOW_PROPERTY_UUID = "93a60a56-a3c8-11da-a746-0800200b9a66"
MASS_FLOW_PROPERTY_VERSION = "03.00.003"
MASS_DISTANCE_FLOW_PROPERTY_UUID = "118f2a40-50ec-457c-aa60-9bc6b6af9931"
MASS_DISTANCE_FLOW_PROPERTY_VERSION = "01.01.000"
FLOW_PROPERTY_VERSION_OVERRIDES: dict[str, str] = {
    "838aaa23-0117-11db-92e3-0800200c9a66": "03.00.000",
    "01846770-4cfe-4a25-8ad9-919d8d378345": "03.00.004",
    "16764bbb-d1ea-4eb4-9911-13f0ecd3dfad": "01.01.000",
    "341fd786-b2ad-4552-a762-5eafcab45dee": "01.00.003",
    "441238a3-ba09-46ec-b35b-c30cfba746d1": "02.00.003",
    "93a60a56-a3c8-11da-a746-0800200c9a66": "03.00.003",
    MASS_DISTANCE_FLOW_PROPERTY_UUID: MASS_DISTANCE_FLOW_PROPERTY_VERSION,
}
FLOW_PROPERTY_STANDARD_MAPPINGS: tuple[dict[str, Any], ...] = (
    {
        "aliases": ("energy", "net calorific value"),
        "target_uuid": "93a60a56-a3c8-11da-a746-0800200c9a66",
        "target_name": "Net calorific value",
        "target_version": "03.00.003",
    },
    {
        "aliases": ("mass transport", "mass"),
        "target_uuid": "93a60a56-a3c8-11da-a746-0800200b9a66",
        "target_name": "Mass",
        "target_version": "03.00.003",
    },
    {
        "aliases": (
            "mass*distance",
            "mass distance",
            "transport work",
            "tkm",
            "t*km",
            "ton kilometer",
            "tonne kilometer",
            "kg*km",
        ),
        "target_uuid": MASS_DISTANCE_FLOW_PROPERTY_UUID,
        "target_name": "mass*distance",
        "target_version": MASS_DISTANCE_FLOW_PROPERTY_VERSION,
    },
)

SOURCE_CATEGORY_CLASS_MAP: dict[str, tuple[str, str]] = {
    "images": ("0", "Images"),
    "image": ("0", "Images"),
    "data set formats": ("1", "Data set formats"),
    "dataset formats": ("1", "Data set formats"),
    "databases": ("2", "Databases"),
    "compliance systems": ("3", "Compliance systems"),
    "compliance system": ("3", "Compliance systems"),
    "statistical classifications": ("4", "Statistical classifications"),
    "publications and communications": ("5", "Publications and communications"),
    "publication": ("5", "Publications and communications"),
    "publications": ("5", "Publications and communications"),
    "other source types": ("6", "Other source types"),
    "other": ("6", "Other source types"),
}

SOURCE_PUBLICATION_TYPE_MAP: dict[str, str] = {
    "images": "Other unpublished and grey literature",
    "data set formats": "Software or database",
    "dataset formats": "Software or database",
    "databases": "Software or database",
    "compliance systems": "Other unpublished and grey literature",
    "statistical classifications": "Other unpublished and grey literature",
    "publications and communications": "Article in periodical",
    "publication": "Article in periodical",
    "publications": "Article in periodical",
}

DEFAULT_SOURCE_CLASS = ("6", "Other source types")
DEFAULT_PUBLICATION_TYPE = "Undefined"

GENERIC_FLOW_CLASS_PATH = [
    {"@level": "0", "@classId": "C", "#text": "Manufacturing"},
    {"@level": "1", "@classId": "27", "#text": "Manufacture of electrical equipment"},
    {"@level": "2", "@classId": "272", "#text": "Manufacture of batteries and accumulators"},
    {"@level": "3", "@classId": "2720", "#text": "Manufacture of batteries and accumulators"},
    {"@level": "4", "@classId": "272000", "#text": "Generic battery products"},
]

LOCATION_CODE_MAP = {
    "china": "CN",
    "people's republic of china": "CN",
    "prc": "CN",
    "united states": "US",
    "united states of america": "US",
    "usa": "US",
    "u.s.": "US",
}

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

NAME_SEGMENT_SPLIT = re.compile(r"[;|]")
NAME_TOKEN_SPLIT = re.compile(r"[,，、]")
YEAR_FRAGMENT = re.compile(r"^(?:\d{4})(?:\s*[-–]\s*(?:\d{2,4}))?$")
CJK_CHAR_PATTERN = re.compile(r"[\u2e80-\u2eff\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\ua000-\ua4cf\uac00-\ud7af\uff00-\uffef]+")
NAME_REMARK_KEYWORDS = ("remark", "note", "todo", "关系", "备注", "说明", "看一下", "待定", "待确认")
MIX_KEYWORDS = (
    "generic",
    "mix",
    "market",
    "global",
    "china",
    "asia",
    "europe",
    "america",
    "world",
    "region",
)


def _tokenize_name_string(raw_name: str | None) -> list[str]:
    if not raw_name:
        return []
    segments: list[str] = []
    for segment in NAME_SEGMENT_SPLIT.split(raw_name):
        for token in NAME_TOKEN_SPLIT.split(segment):
            cleaned = token.strip()
            if cleaned:
                segments.append(cleaned)
    return segments


def _is_year_fragment(fragment: str) -> bool:
    return bool(YEAR_FRAGMENT.fullmatch(fragment.strip()))


def _looks_like_mix_fragment(fragment: str) -> bool:
    lowered = fragment.lower()
    if any(keyword in lowered for keyword in MIX_KEYWORDS):
        return True
    if re.fullmatch(r"[A-Z]{2,3}", fragment.strip()):
        return True
    return False


def _deduplicate_tokens(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        cleaned = token.strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _extract_name_components_from_string(raw_name: str | None) -> tuple[str, list[str], list[str]]:
    tokens = _tokenize_name_string(raw_name)
    if not tokens:
        return "", [], []
    base_name = ""
    start_idx = 0
    for idx, token in enumerate(tokens):
        if _is_year_fragment(token):
            continue
        base_name = token.strip()
        start_idx = idx + 1
        break
    treatment_tokens: list[str] = []
    mix_tokens: list[str] = []
    for token in tokens[start_idx:]:
        if _is_year_fragment(token):
            continue
        if _looks_like_mix_fragment(token):
            mix_tokens.append(token.strip())
        else:
            treatment_tokens.append(token.strip())
    return base_name, _deduplicate_tokens(treatment_tokens), _deduplicate_tokens(mix_tokens)


FLOW_LCI_METHOD_MAP = {
    "PRODUCT_FLOW": "Product flow",
    "PRODUCT FLOW": "Product flow",
    "PRODUCT": "Product flow",
    "WASTE_FLOW": "Waste flow",
    "WASTE FLOW": "Waste flow",
    "WASTE": "Waste flow",
    "ELEMENTARY_FLOW": "Elementary flow",
    "ELEMENTARY FLOW": "Elementary flow",
    "ELEMENTARY": "Elementary flow",
}


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _contact_reference() -> dict[str, Any]:
    return {
        "@refObjectId": TIANGONG_CONTACT_UUID,
        "@type": "contact data set",
        "@uri": TIANGONG_CONTACT_URI,
        "@version": TIANGONG_CONTACT_VERSION,
        "common:shortDescription": [
            _language_entry("Tiangong LCA Data Working Group"),
            _language_entry("天工LCA数据团队", "zh"),
        ],
    }


def _ownership_reference() -> dict[str, Any]:
    return _contact_reference()


def _flow_format_reference() -> dict[str, Any]:
    return build_dataset_format_reference()


def _compliance_reference() -> dict[str, Any]:
    return {
        "@refObjectId": ILCD_COMPLIANCE_SOURCE_UUID,
        "@type": "source data set",
        "@uri": ILCD_COMPLIANCE_SOURCE_URI,
        "@version": ILCD_COMPLIANCE_SOURCE_VERSION,
        "common:shortDescription": _language_entry("ILCD Data Network - Entry-level"),
    }


def _coerce_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "content"):
        response = getattr(response, "content")
    if isinstance(response, str):
        data = parse_json_response(response)
    else:
        data = response
    if not isinstance(data, dict):
        raise ProcessExtractionError("LLM response must be a JSON object.")
    return data


SEMANTIC_PROMPT_TEMPLATE = """
You are the semantic interpreter for OpenLCA JSON-LD processes. Read the `context` JSON object and
return ONLY a JSON dictionary with the following keys:
- base_name: core activity/product without geography or year.
- technical_route: feedstock, route, standards, or treatment text derived from names, technology
  fields, or exchange descriptions.
- mix_type: market/mix wording or high-level geography cue (never invent default text such as "Production mix"
  or "at plant"; copy wording from the source or leave empty if it does not exist).
- reference_year: first valid four-digit year for the dataset (prefer validity range start).
- location_text_hint: plain-text geography phrase (country, plant, market, etc.). Do NOT default to "GLO";
  if the geography is unknown, return an empty string so downstream logic can leave the field blank.
- lci_method_principle: choose exactly one of: {principles}.
- description: 2–3 short sentences summarising scope, reference product, and notable inventory traits.

Process name mapping guidelines (apply exactly to the four ILCD name fields):
- processInformation.dataSetInformation.baseName: concise description of the main activity or
  function (e.g., "Coal mining", "Electricity generation").
- processInformation.dataSetInformation.treatmentStandardsRoutes: provide technical descriptors of the
  good, service, or process. Include treatment steps (pyrolysis, catalytic reforming), referenced
  standards (full identifier when available), product quality or grade, intended use, recognised
  production-route names, principal feedstocks, and whether the product is primary or secondary.
  Separate descriptors with commas and use established industrial terminology. Exclude end-of-pipe
  environmental controls.
- processInformation.dataSetInformation.mixAndLocationTypes: state whether the record is a production
  or consumption mix and note the availability location type (e.g., "to consumer", "at plant").
  Separate descriptors with commas.
- processInformation.dataSetInformation.functionalUnitFlowProperties: list technical qualifiers such
  as constituent content or energy content per unit. Separate values with commas. Non-qualifying
  metadata (CAS numbers, synonyms, formulas) belongs in the associated flow dataset, not here.

Classification, mix, and geography rules:
- Use the ProcessClassifier candidate lists to pick the most specific Tiangong category path. Never invent
  codes or truncate the path; if you cannot justify a deeper level, stop at the last supported candidate.
- Respect the "authoritative-first" policy: prefer explicit codes or descriptions in the JSON-LD payload.
- When the source does not mention mix/location cues, leave the mix and geography hints empty—the Python
  layer will keep the ILCD fields blank instead of defaulting to "GLO" or generic phrases.

Parsing requirements:
1. Split `name` strings by `;`, `|`, or long comma-separated segments to isolate base activity, route,
   and geography tokens (示例: "2019-2020 ; 4-LIB cathode production ; ...").
2. Do NOT guess classification codes or ILCD structure—only interpret semantics.
3. Use evidence from `processDocumentation` (validFrom, geographyDescription, technologyDescription,
   useAdvice) plus `description`, `location`, and exchanges.
4. Always keep outputs concise (≤120 characters per field) and copy wording from the document when it
   is reliable; otherwise paraphrase clearly.
"""


class ParsedProcessSemantics(TypedDict, total=False):
    base_name: str
    technical_route: str
    mix_type: str
    reference_year: int
    location_text_hint: str
    lci_method_principle: str
    description: str


class JSONLDSemanticParser:
    """Focuses the LLM on semantic understanding rather than ILCD rendering."""

    LCIMETHOD_POINTER = "/properties/processDataSet/properties/modellingAndValidation/properties/LCIMethodAndAllocation/properties/LCIMethodPrinciple"

    def __init__(self, llm: LanguageModelProtocol) -> None:
        self._llm = llm
        self._schema_repository = get_schema_repository()
        self._prompt = self._build_prompt()

    def parse(self, payload: dict[str, Any]) -> ParsedProcessSemantics:
        LOGGER.info("jsonld.semantic_parser.invoke")
        delays = (0.0, 0.5, 1.0)
        last_error: Exception | None = None
        data: dict[str, Any] = {}
        for attempt, delay in enumerate(delays, start=1):
            if delay:
                sleep(delay)
            try:
                response = self._llm.invoke(
                    {
                        "prompt": self._prompt,
                        "context": payload,
                        "response_format": {"type": "json_object"},
                    }
                )
                data = _coerce_dict(response)
                if not _clean_text(data.get("base_name")):
                    raise ProcessExtractionError("LLM returned empty base_name")
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                LOGGER.warning("jsonld.semantic_parser.retry", attempt=attempt, error=str(exc))
        else:
            LOGGER.warning("jsonld.semantic_parser.fallback", error=str(last_error) if last_error else "unknown error")
            return ParsedProcessSemantics()

        semantics = ParsedProcessSemantics()
        base_name_text = _clean_text(data.get("base_name"))
        if base_name_text:
            semantics["base_name"] = base_name_text
        technical_route = _clean_text(data.get("technical_route"))
        if technical_route:
            semantics["technical_route"] = technical_route
        mix_type = _clean_text(data.get("mix_type"))
        if mix_type:
            semantics["mix_type"] = mix_type
        location_hint = _clean_text(data.get("location_text_hint"))
        if location_hint:
            semantics["location_text_hint"] = location_hint
        description = _clean_text(data.get("description"))
        if description:
            semantics["description"] = description
        principle = self._normalize_principle(data.get("lci_method_principle"))
        if principle:
            semantics["lci_method_principle"] = principle
        reference_year = _coerce_int(data.get("reference_year"))
        if reference_year:
            semantics["reference_year"] = reference_year
        return semantics

    def _build_prompt(self) -> str:
        schema = self._schema_repository.resolve_with_references("tidas_processes.json", self.LCIMETHOD_POINTER)
        principle_values = tuple(item for item in schema.get("enum", []) if isinstance(item, str))
        principles = ", ".join(principle_values) or "Attributional, Consequential, Consequential with attributional components, Not applicable, Other"
        return SEMANTIC_PROMPT_TEMPLATE.format(principles=principles)

    def _normalize_principle(self, value: Any) -> str:
        schema = self._schema_repository.resolve_with_references("tidas_processes.json", self.LCIMETHOD_POINTER)
        allowed = {item for item in schema.get("enum", []) if isinstance(item, str)}
        candidate = (_clean_text(value) or "Not applicable").strip()
        if candidate in allowed:
            return candidate
        for option in allowed:
            if option.lower() == candidate.lower():
                return option
        return "Not applicable"


REFERENCE_YEAR_RE = re.compile(r"(18|19|20|21)\d{2}")
DEFAULT_PROCESS_CLASS_PATH = [
    {"@level": "0", "@classId": "C", "#text": "Manufacturing"},
    {"@level": "1", "@classId": "27", "#text": "Manufacture of electrical equipment"},
    {"@level": "2", "@classId": "272", "#text": "Manufacture of batteries and accumulators"},
    {"@level": "3", "@classId": "2720", "#text": "Manufacture of batteries and accumulators"},
]
DEFAULT_DATASET_VALID_UNTIL_YEAR = 2025

NAME_RECOVERY_PROMPT = """
One or more ILCD name fields are missing. Use the supplied `processInformation`
and original JSON-LD payload to infer concise English phrases for the missing keys.
Respond with JSON containing any of: `baseName`, `treatmentStandardsRoutes`, `mixAndLocationTypes`.
Avoid placeholders; capture the actual process, route, and mix/location descriptions from the evidence provided.
"""


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(_clean_text(item) for item in value if item is not None).strip(", ")
    if isinstance(value, dict):
        text = value.get("#text") or value.get("text") or value.get("value")
        if isinstance(text, str):
            return text.strip()
    return ""


def _has_text_entry(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return bool(
            _clean_text(
                value.get("#text") or value.get("text") or value.get("@value") or value.get("value") or value.get("description"),
            )
        )
    if isinstance(value, list):
        return any(_has_text_entry(item) for item in value)
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _require_text(value: Any, field: str) -> str:
    text = _clean_text(value)
    if not text:
        raise ProcessExtractionError(f"{field} must be provided by the semantic parser.")
    return text


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        try:
            return int(value.strip())
        except ValueError:  # pragma: no cover - defensive
            return None
    return None


def _language_entry(text: str | None, lang: str = "en") -> dict[str, str]:
    value = _clean_text(text) or "Unspecified"
    return {"@xml:lang": lang, "#text": value}


def _extract_reference_year(payload: dict[str, Any], semantics: ParsedProcessSemantics) -> int | None:
    if "reference_year" in semantics:
        return semantics["reference_year"]
    documentation = payload.get("processDocumentation") or {}
    valid_from_text = _clean_text(documentation.get("validFrom"))
    year = _extract_year(valid_from_text)
    if year:
        return year
    for key in ("timeDescription", "creationDate"):
        candidate = _clean_text(documentation.get(key))
        year = _extract_year(candidate)
        if year:
            return year
    for key in ("validUntil", "lastChange", "version", "name", "description"):
        candidate = _clean_text(payload.get(key))
        year = _extract_year(candidate)
        if year:
            return year
    return None


def _extract_year(value: str) -> int | None:
    if not value:
        return None
    match = REFERENCE_YEAR_RE.search(value)
    if match:
        return int(match.group(0))
    return None


def _map_process_type(process_type: str | None) -> str:
    mapping = {
        "UNIT_PROCESS": "Unit process, single operation",
        "SYSTEM_PROCESS": "System process",
        "UNIT_PROCESS_BLACK_BOX": "Unit process, black box",
    }
    candidate = (process_type or "").upper()
    return mapping.get(candidate, "Unit process, single operation")


def _derive_location_hint(payload: dict[str, Any], semantics: ParsedProcessSemantics) -> str:
    if semantics.get("location_text_hint"):
        return semantics["location_text_hint"] or ""
    location = payload.get("location")
    if isinstance(location, dict):
        for key in ("name", "shortname", "code"):
            text = _clean_text(location.get(key))
            if text:
                return text
    documentation = payload.get("processDocumentation") or {}
    for key in ("geographyDescription", "useAdvice"):
        text = _clean_text(documentation.get(key))
        if text:
            return text
    return _clean_text(payload.get("category")) or "Unspecified region"


def _sanitize_name_component(value: str | None) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    normalized = text.strip(" ,;")
    if is_placeholder_value(normalized):
        return ""
    if len(normalized) <= 1:
        return ""
    if CJK_CHAR_PATTERN.search(normalized):
        return ""
    lowered = normalized.lower()
    if any(keyword in lowered for keyword in NAME_REMARK_KEYWORDS):
        return ""
    return normalized


def _derive_process_name_fields(
    payload: dict[str, Any],
    semantics: ParsedProcessSemantics,
) -> tuple[str | None, str | None, str | None]:
    # Prefer LLM semantic outputs when available
    base_candidate = _sanitize_name_component(semantics.get("base_name"))
    treatment_candidate = _sanitize_name_component(semantics.get("technical_route"))
    mix_candidate = _sanitize_name_component(semantics.get("mix_type") or semantics.get("location_text_hint"))
    if base_candidate:
        return base_candidate, treatment_candidate or base_candidate, mix_candidate or None

    # Fallback: rules with sanitization/guardrails
    raw_name = _clean_text(payload.get("name"))
    base_name, treatment_tokens, mix_tokens = _extract_name_components_from_string(raw_name)
    base_name = _sanitize_name_component(base_name)
    treatment_clean = [_sanitize_name_component(token) for token in treatment_tokens]
    treatment_clean = [token for token in treatment_clean if token]
    mix_clean = [_sanitize_name_component(token) for token in mix_tokens]
    mix_clean = [token for token in mix_clean if token]

    treatment_text = "; ".join(_deduplicate_tokens(treatment_clean)) if treatment_clean else None
    mix_text = "; ".join(_deduplicate_tokens(mix_clean)) if mix_clean else None

    return base_name or None, treatment_text, mix_text


def _merge_description(semantics: ParsedProcessSemantics, payload: dict[str, Any]) -> str:
    parts = []
    if semantics.get("description"):
        parts.append(semantics["description"])
    description = _clean_text(payload.get("description"))
    if description:
        parts.append(description)
    documentation = payload.get("processDocumentation") or {}
    technology = _clean_text(documentation.get("technologyDescription"))
    if technology:
        parts.append(technology)
    return ". ".join(part.strip().rstrip(".") for part in parts if part).strip(". ")


def _flow_display_name(flow: dict[str, Any]) -> str:
    raw_name = flow.get("name")
    if isinstance(raw_name, str) and raw_name.strip():
        return raw_name.strip()
    identifier = flow.get("@id")
    if isinstance(identifier, str) and identifier.strip():
        return identifier.strip()
    return "Unnamed flow"


def _summarize_flow_reference_text(flow: dict[str, Any]) -> str:
    base, treatment_tokens, mix_tokens = _extract_name_components_from_string(_clean_text(flow.get("name")))
    parts: list[str] = []
    if base:
        parts.append(base)
    parts.extend(_deduplicate_tokens(treatment_tokens))
    parts.extend(_deduplicate_tokens(mix_tokens))
    summary = "; ".join(parts).strip()
    return summary or _flow_display_name(flow)


def _build_flow_reference(flow: dict[str, Any]) -> dict[str, Any]:
    ref_uuid = flow.get("@id") or str(uuid4())
    version = _normalize_version(flow.get("version"))
    name = _summarize_flow_reference_text(flow)
    return {
        "@type": "flow data set",
        "@refObjectId": ref_uuid,
        "@uri": f"../flows/{ref_uuid}_{version}.xml",
        "@version": version,
        "common:shortDescription": _language_entry(name),
    }


def _build_flow_property_reference(flow_property: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(flow_property, dict):
        return None
    ref_uuid = flow_property.get("@id")
    if not ref_uuid:
        return None
    version = FLOW_PROPERTY_VERSION_OVERRIDES.get(ref_uuid) or _normalize_version(flow_property.get("version"))
    name = _clean_text(flow_property.get("name")) or "Flow property"
    return {
        "@type": "flow property data set",
        "@refObjectId": ref_uuid,
        "@uri": f"../flowproperties/{ref_uuid}_{version}.xml",
        "@version": version,
        "common:shortDescription": _language_entry(name),
    }


def _normalize_version(value: Any) -> str:
    candidate = _clean_text(value)
    if candidate and re.fullmatch(r"\d{2}\.\d{2}\.\d{3}", candidate):
        return candidate
    return DEFAULT_DATA_SET_VERSION


def _ensure_exchange_direction(is_input: Any) -> str:
    return "Input" if bool(is_input) else "Output"


def _format_functional_unit_text(exchange: dict[str, Any]) -> str | None:
    if not isinstance(exchange, dict):
        return None
    amount = exchange.get("amount")
    unit = (exchange.get("unit") or {}).get("name") if isinstance(exchange.get("unit"), dict) else None
    flow = exchange.get("flow") or {}
    flow_name = _clean_text(flow.get("name"))
    if amount is None and not flow_name:
        return None
    amount_text = f"{amount}" if amount is not None else "1"
    unit_text = unit or (flow.get("refUnit") if isinstance(flow, dict) else "")
    parts = [amount_text]
    if unit_text:
        parts.append(unit_text)
    if flow_name:
        parts.append(flow_name)
    return " ".join(parts).strip()


def _build_exchange_entries(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None, str | None]:
    exchanges_payload = payload.get("exchanges")
    if not isinstance(exchanges_payload, list):
        return [], None, None
    entries: list[dict[str, Any]] = []
    reference_id: str | None = None
    functional_text: str | None = None
    for idx, exchange in enumerate(exchanges_payload, start=1):
        if not isinstance(exchange, dict):
            continue
        flow = exchange.get("flow") or {}
        unit = exchange.get("unit") or {}
        entry: dict[str, Any] = {
            "@dataSetInternalID": str(idx),
            "exchangeDirection": _ensure_exchange_direction(exchange.get("isInput")),
            "meanAmount": str(exchange.get("amount", 0) or 0),
            "unit": _clean_text(unit.get("name")) or _clean_text(unit.get("@id")) or _clean_text(flow.get("refUnit")) or "",
            "referenceToFlowDataSet": _build_flow_reference(flow),
        }
        entry["resultingAmount"] = entry["meanAmount"]
        if entry["unit"]:
            entry["resultingAmountUnit"] = entry["unit"]
        property_reference = _build_flow_property_reference(exchange.get("flowProperty"))
        if property_reference:
            entry["referenceToFlowPropertyDataSet"] = property_reference
        entries.append(entry)
        if exchange.get("isQuantitativeReference") and reference_id is None:
            reference_id = entry["@dataSetInternalID"]
            functional_text = _format_functional_unit_text(exchange)
    if entries and reference_id is None:
        reference_id = entries[0]["@dataSetInternalID"]
        functional_text = functional_text or _format_functional_unit_text(exchanges_payload[0])
    return entries, reference_id, functional_text


def _build_synonym_entries(raw_value: Any) -> list[dict[str, str]]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        candidates = [token.strip() for token in re.split(r"[;,]", raw_value) if token.strip()]
    elif isinstance(raw_value, list):
        candidates = [_clean_text(item) for item in raw_value if _clean_text(item)]
    else:
        candidates = [_clean_text(raw_value)]
    return [_language_entry(candidate) for candidate in candidates if candidate]


def _merge_synonym_entries(*values: Any) -> list[dict[str, str]]:
    grouped: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for value in values:
        for entry in _build_synonym_entries(value):
            lang = entry.get("@xml:lang", "en")
            text = (entry.get("#text") or "").strip()
            if not text:
                continue
            lang_seen = seen.setdefault(lang, set())
            key = text.lower()
            if key in lang_seen:
                continue
            lang_seen.add(key)
            grouped.setdefault(lang, []).append(text)
    merged: list[dict[str, str]] = []
    for lang, items in grouped.items():
        merged.append({"@xml:lang": lang, "#text": "; ".join(items)})
    return merged


def _extract_name_text(value: Any) -> str:
    if isinstance(value, dict):
        text = value.get("#text") or value.get("text") or value.get("value")
        if isinstance(text, str) and text.strip():
            return text.strip()
        for candidate in value.values():
            extracted = _extract_name_text(candidate)
            if extracted:
                return extracted
        return ""
    if isinstance(value, list):
        for entry in value:
            extracted = _extract_name_text(entry)
            if extracted:
                return extracted
        return ""
    if isinstance(value, str):
        return value.strip()
    return ""


def _build_technology_block(payload: dict[str, Any], semantics: ParsedProcessSemantics) -> dict[str, Any] | None:
    documentation = payload.get("processDocumentation") or {}
    technology_text = _clean_text(documentation.get("technologyDescription"))
    if not technology_text:
        return None
    return {"technologyDescriptionAndIncludedProcesses": [_language_entry(technology_text)]}


FLOW_SEMANTIC_PROMPT_TEMPLATE = """
You are the semantic interpreter for OpenLCA JSON-LD flows. Read the `context` JSON object and
return ONLY a JSON dictionary with the following keys:
- base_name: primary substance/product name without geography.
- treatment: quality grade, treatment, or route information.
- mix_type: market/mix wording or delivery context. Never insert defaults such as "Production mix" or
  "at plant" unless they are explicitly stated in the source.
- location_text_hint: geography phrase that best matches the flow (market, country, facility). Do NOT
  default to "GLO"; if geography is unknown, emit an empty string so downstream steps can leave it blank.
- flow_type: choose exactly one of: {types}.
- description: 1–2 concise sentences describing composition or applicability.
- synonyms: array of alternate names (strings only, no punctuation).

Flow name guidelines (map JSON-LD names to ILCD flow name fields exactly as described):
- `basename`: English technical name mirroring authoritative datasets. State the substance/product/
  waste as industry uses it, separate descriptors with commas (never semicolons), avoid
  geography/quantities, spell out acronyms on first use, and include state/grade cues (e.g.,
  "Polypropylene, PP, granulate"; "Sulfur dioxide, gaseous"; "Waste glass cullet, mixed colors").
- `treatment`: comma-separated qualitative qualifiers ordered from intrinsic modifications to
  contextual uses—surface/material treatments (Hot rolled, Sterilised), referenced standards/grades
  (EN 10025 S355, ASTM D4806), key performance attributes (UV-resistant, food-grade), intended uses
  (for wafer production, medical packaging), and production/recycling routes (primary production
  route, secondary feedstock, steam cracking route). Keep each entry concise and technical.
- `mix_location`: Comma-separated string specifying the Mix Type (optional) and Location Type (mandatory).
  1. Mix Type: Identify as "Production mix" if it represents a weighted average of production routes 
  within a geography, or "Consumption mix" if it includes trade contributions (imports/exports) from outside the region.
  2. Location Type: Specify the point of availability. Use "at" to denote a point of departure 
  (leaving a facility/storage) and "to" to denote a point of arrival (including transport/logistics). 
  Constraint: Describe the physical availability, not the intended use (e.g., avoid "for" qualifiers).
- `flow_properties`: comma-separated quantitative properties with explicit bases ("45 % Fe mass/mass";
  "9.6 MJ/kg net calorific value"; "90.5 % methane by volume"; "750 g/L total dissolved solids"),
  using SI or accepted industry units and clarifying the measurement basis when it deviates from mass
  fraction.

Classification and flow-property rules:
- Use the ProductFlowClassifier candidate lists to walk the Tiangong category tree level by level; never
  invent class IDs or jump outside the suggested whitelist.
- When mapping `flowProperties`, preserve every JSON-LD entry and apply the built-in standard mappings:
  aliases containing "energy" or "net calorific value" must reference UUID 93a60a56-a3c8-11da-a746-0800200c9a66
  (Net calorific value), and aliases containing "mass transport" or plain "mass" must reference UUID
  93a60a56-a3c8-11da-a746-0800200b9a66 (Mass). Do not introduce new flow properties beyond those already defined.
"""


class ParsedFlowSemantics(TypedDict, total=False):
    base_name: str
    treatment: str
    mix_type: str
    location_text_hint: str
    flow_type: str
    description: str
    synonyms: list[str]


class JSONLDFlowSemanticParser:
    FLOWTYPE_POINTER = "/properties/flowDataSet/properties/modellingAndValidation/properties/LCIMethod/properties/typeOfDataSet"

    def __init__(self, llm: LanguageModelProtocol) -> None:
        self._llm = llm
        self._schema_repository = get_schema_repository()
        self._prompt = self._build_prompt()

    def parse(self, payload: dict[str, Any]) -> ParsedFlowSemantics:
        LOGGER.info("jsonld.flow_semantic_parser.invoke")
        response = self._llm.invoke(
            {
                "prompt": self._prompt,
                "context": payload,
                "response_format": {"type": "json_object"},
            }
        )
        data = _coerce_dict(response)
        semantics = ParsedFlowSemantics()
        semantics["base_name"] = _require_text(data.get("base_name"), "base_name")
        semantics["treatment"] = _clean_text(data.get("treatment"))
        semantics["mix_type"] = _clean_text(data.get("mix_type"))
        semantics["location_text_hint"] = _clean_text(data.get("location_text_hint"))
        semantics["description"] = _clean_text(data.get("description"))
        semantics["flow_type"] = self._normalize_flow_type(data.get("flow_type"))
        synonyms = data.get("synonyms")
        semantics["synonyms"] = _string_list(synonyms)
        return semantics

    def _build_prompt(self) -> str:
        schema = self._schema_repository.resolve_with_references("tidas_flows.json", self.FLOWTYPE_POINTER)
        type_values = tuple(item for item in schema.get("enum", []) if isinstance(item, str))
        types = ", ".join(type_values) or "Product flow, Waste flow, Elementary flow"
        return FLOW_SEMANTIC_PROMPT_TEMPLATE.format(types=types)

    def _normalize_flow_type(self, value: Any) -> str:
        schema = self._schema_repository.resolve_with_references("tidas_flows.json", self.FLOWTYPE_POINTER)
        allowed = {item for item in schema.get("enum", []) if isinstance(item, str)}
        text = (_clean_text(value) or "").strip()
        if text in allowed:
            return text
        for option in allowed:
            if option.lower() == text.lower():
                return option
        return "Product flow"


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [token.strip() for token in value.split(";")]
        return [token for token in parts if token]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            text = _clean_text(item)
            if text:
                result.append(text)
        return result
    return []


def _extract_location_text(payload: dict[str, Any]) -> str | None:
    location = payload.get("location")
    if isinstance(location, dict):
        for key in ("code", "name", "shortname"):
            text = _clean_text(location.get(key))
            if text:
                return text
    return None


INVALID_LOCATION_CODES = {"ROW", "001"}


def _match_location_code(text: str | None) -> str | None:
    candidate = _clean_text(text)
    if not candidate:
        return None
    lookup = candidate.lower()
    if lookup in LOCATION_CODE_MAP:
        return LOCATION_CODE_MAP[lookup]
    candidate_upper = candidate.upper()
    if candidate_upper in INVALID_LOCATION_CODES:
        return None
    if candidate_upper.isalpha() and len(candidate_upper) in (2, 3):
        return candidate_upper
    return None


def _resolve_flow_location(
    semantics: ParsedFlowSemantics,
    payload: dict[str, Any],
    *,
    catalog: LocationCatalog,
    location_normalizer: LocationNormalizer | None = None,
    flow_context: dict[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    candidate_texts: list[str | None] = [
        semantics.get("location_text_hint"),
        _extract_location_text(payload),
    ]
    for text in candidate_texts:
        if not text:
            continue
        code = catalog.best_guess(text)
        if code:
            return code, {"code": code, "description": catalog.describe(code)}
    raw_hint = candidate_texts[0] or candidate_texts[1] if len(candidate_texts) > 1 else None
    initial_code = catalog.best_guess(raw_hint)
    payload_update: dict[str, Any] = {}
    if location_normalizer and (raw_hint or initial_code):
        candidates = catalog.build_candidate_list(raw_hint)
        context = flow_context or {}
        try:
            geography = location_normalizer.run(
                context if any(key in context for key in ("processInformation", "flowInformation")) else {"flowInformation": context},
                hint=raw_hint,
                candidates=candidates,
                initial_code=initial_code,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("jsonld.flow_location_normalization_failed", error=str(exc))
            geography = None
        code_from_response, payload_update = extract_location_response(geography)
        normalized_code = catalog.coerce_code(code_from_response)
        if normalized_code:
            return normalized_code, payload_update
    if initial_code:
        return initial_code, {"code": initial_code, "description": catalog.describe(initial_code)}
    return None, {}


def _infer_elementary_categories(semantics: ParsedFlowSemantics, payload: dict[str, Any]) -> list[dict[str, Any]]:
    fragments: list[str] = []
    for field in (
        semantics.get("description"),
        semantics.get("base_name"),
        semantics.get("mix_type"),
        _clean_text(payload.get("name")),
        _clean_text(payload.get("category")),
    ):
        if field:
            fragments.append(field.lower())
    synonym_values = semantics.get("synonyms") or _string_list(payload.get("synonyms"))
    if synonym_values:
        fragments.append(" ".join(synonym.lower() for synonym in synonym_values))
    location_hint = semantics.get("location_text_hint") or _extract_location_text(payload)
    if location_hint:
        fragments.append(location_hint.lower())
    combined = " ".join(fragments)
    if any(keyword in combined for keyword in ("resource", "extraction", "mining", "raw material")):
        return list(ELEMENTARY_CATEGORY_RESOURCES)
    if "water" in combined or "wastewater" in combined or "effluent" in combined:
        return list(ELEMENTARY_CATEGORY_WATER)
    if "soil" in combined or "ground" in combined or "land" in combined:
        return list(ELEMENTARY_CATEGORY_SOIL)
    if "air" in combined or "atmos" in combined:
        return list(ELEMENTARY_CATEGORY_AIR)
    return list(ELEMENTARY_CATEGORY_OTHER)


def _map_flow_dataset_type(candidate: str | None, payload_value: Any) -> str:
    if candidate:
        return candidate
    text = (_clean_text(payload_value) or "").upper()
    return FLOW_LCI_METHOD_MAP.get(text, "Product flow")


def _default_flow_property_entry() -> dict[str, Any]:
    reference = _build_flow_property_reference(
        {
            "@id": MASS_FLOW_PROPERTY_UUID,
            "name": "Mass",
            "version": MASS_FLOW_PROPERTY_VERSION,
        }
    )
    if reference is None:
        raise ProcessExtractionError("Unable to build default mass flow property reference.")
    return {
        "@dataSetInternalID": "1",
        "meanValue": "1",
        "referenceToFlowPropertyDataSet": reference,
    }


def _map_standard_flow_property(
    flow_property: dict[str, Any] | None,
    flow_uuid: str,
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    if not isinstance(flow_property, dict):
        return flow_property, None
    property_name = _clean_text(flow_property.get("name"))
    if not property_name:
        return flow_property, None
    normalized = property_name.lower()
    for rule in FLOW_PROPERTY_STANDARD_MAPPINGS:
        if normalized in rule["aliases"]:
            audit_record = {
                "flow_uuid": flow_uuid,
                "original_property_name": property_name,
                "original_property_uuid": _clean_text(flow_property.get("@id")),
                "mapped_property_name": rule["target_name"],
                "mapped_property_uuid": rule["target_uuid"],
            }
            mapped = {
                "@id": rule["target_uuid"],
                "name": rule["target_name"],
                "version": rule["target_version"],
            }
            return mapped, audit_record
    return flow_property, None


def _build_flow_properties_block(
    payload: dict[str, Any],
    *,
    flow_uuid: str,
    audit_log: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    flow_properties_payload = payload.get("flowProperties")
    entries: list[dict[str, Any]] = []
    if isinstance(flow_properties_payload, list):
        for idx, factor in enumerate(flow_properties_payload, start=1):
            if not isinstance(factor, dict):
                continue
            mapped_flow_property, audit_record = _map_standard_flow_property(factor.get("flowProperty"), flow_uuid)
            reference = _build_flow_property_reference(mapped_flow_property)
            if reference is None:
                continue
            if audit_record and audit_log is not None:
                audit_log.append(audit_record)
            mean_value = factor.get("conversionFactor", 1)
            unit_hint = _clean_text(factor.get("unit"))
            if mapped_flow_property and mapped_flow_property.get("@id") == MASS_DISTANCE_FLOW_PROPERTY_UUID:
                # Normalize to kg*km reference; convert common inputs (t*km, g*km)
                normalized = _clean_text(mean_value) or "1"
                try:
                    numeric = float(normalized)
                except Exception:
                    numeric = 1.0
                unit_lower = unit_hint.lower() if isinstance(unit_hint, str) else ""
                if unit_lower in ("t*km", "tkm", "ton*km", "tonne*km", "tonne km", "ton km"):
                    numeric *= 1000.0
                elif unit_lower in ("g*km", "gkm"):
                    numeric /= 1000.0
                mean_value = str(numeric)
            entries.append(
                {
                    "@dataSetInternalID": str(idx),
                    "meanValue": str(mean_value if mean_value not in (None, "") else 1),
                    "referenceToFlowPropertyDataSet": reference,
                }
            )
    if not entries:
        default_entry = _default_flow_property_entry()
        entries.append(default_entry)
    return entries, entries[0]["@dataSetInternalID"]


def _apply_flow_classification(
    classifier: ProductFlowClassifier,
    data_info: dict[str, Any],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    classifier_payload = {"flowInformation": {"dataSetInformation": data_info}}
    try:
        path = classifier.run(classifier_payload, source_payload=payload)
    except ProcessExtractionError:
        path = []
    if not path:
        path = list(GENERIC_FLOW_CLASS_PATH)
    else:
        path = list(ensure_valid_product_flow_classification(tuple(path)))
    classification_container = data_info.setdefault("classificationInformation", {}).setdefault("common:classification", {})
    classification_container["common:class"] = path
    return path


def _build_flow_dataset(
    payload: dict[str, Any],
    semantics: ParsedFlowSemantics,
    classifier: ProductFlowClassifier,
    location_normalizer: LocationNormalizer | None,
    catalog: LocationCatalog,
    audit_log: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    flow_uuid = _clean_text(payload.get("@id")) or str(uuid4())
    dataset_version = DEFAULT_DATA_SET_VERSION
    mix_type_text = _clean_text(semantics.get("mix_type"))
    if mix_type_text:
        mix_type_text = mix_type_text.strip(" ,;:.")
        mix_lower = mix_type_text.lower()
        for suffix in (" at", " to", " from"):
            if mix_lower.endswith(suffix):
                mix_type_text = mix_type_text[: -len(suffix)].rstrip(" ,;:.")
                break
    mix_lower = (mix_type_text or "").lower()
    if not mix_type_text:
        mix_and_location_text = "at plant"
    elif any(token in mix_lower for token in (" at ", " to ", " from ")):
        mix_and_location_text = mix_type_text
    else:
        mix_and_location_text = f"{mix_type_text}, at plant"
    name_block: dict[str, Any] = {
        "baseName": _language_entry(semantics["base_name"]),
        "treatmentStandardsRoutes": _language_entry("Unspecified"),
        "mixAndLocationTypes": _language_entry(mix_and_location_text),
    }
    flow_properties_entry = semantics.get("flow_properties")
    if flow_properties_entry:
        name_block["functionalUnitFlowProperties"] = _language_entry(flow_properties_entry)
    data_info: dict[str, Any] = {
        "common:UUID": flow_uuid,
        "name": name_block,
    }
    synonyms = _merge_synonym_entries(semantics.get("synonyms"), payload.get("synonyms"))
    if synonyms:
        data_info["common:synonyms"] = synonyms
    general_comment = semantics.get("description") or _clean_text(payload.get("description"))
    if general_comment:
        data_info["common:generalComment"] = _language_entry(general_comment)
    dataset_type = _map_flow_dataset_type(semantics.get("flow_type"), payload.get("flowType"))
    is_elementary_flow = dataset_type.lower() == "elementary flow"
    if is_elementary_flow:
        categories = _infer_elementary_categories(semantics, payload)
        data_info.pop("classificationInformation", None)
        data_info["common:elementaryFlowCategorization"] = {"common:category": categories}
    else:
        _apply_flow_classification(classifier, data_info, payload)
    flow_properties, reference_property_id = _build_flow_properties_block(
        payload,
        flow_uuid=flow_uuid,
        audit_log=audit_log,
    )
    quantitative_reference = {"referenceToReferenceFlowProperty": reference_property_id}
    flow_information: dict[str, Any] = {
        "dataSetInformation": data_info,
        "quantitativeReference": quantitative_reference,
    }
    location_code, location_payload = _resolve_flow_location(
        semantics,
        payload,
        catalog=catalog,
        location_normalizer=location_normalizer,
        flow_context={"flowInformation": flow_information},
    )
    if location_code:
        geography: dict[str, Any] = {"locationOfSupply": location_code}
        if location_payload.get("description"):
            geography["description"] = location_payload["description"]
        flow_information["geography"] = geography
    technology_text = semantics.get("treatment") or semantics.get("description")
    if technology_text:
        flow_information["technology"] = {"technologicalApplicability": [_language_entry(technology_text)]}
    modelling = {
        "LCIMethod": {
            "typeOfDataSet": dataset_type,
        },
        "complianceDeclarations": flow_compliance_declarations(),
    }
    administrative = {
        "dataEntryBy": {
            "common:referenceToDataSetFormat": _flow_format_reference(),
            "common:referenceToPersonOrEntityEnteringTheData": _contact_reference(),
            "common:timeStamp": _current_timestamp(),
        },
        "publicationAndOwnership": {
            "common:dataSetVersion": dataset_version,
            "common:permanentDataSetURI": build_portal_uri("flow", flow_uuid, dataset_version),
            "common:licenseType": DEFAULT_LICENSE,
            "common:referenceToOwnershipOfDataSet": _ownership_reference(),
        },
    }
    flow_dataset = {
        **ILCD_FLOW_XMLNS,
        "flowInformation": flow_information,
        "flowProperties": {"flowProperty": flow_properties},
        "modellingAndValidation": modelling,
        "administrativeInformation": administrative,
    }
    return flow_dataset


def _compose_intended_applications(semantics: ParsedProcessSemantics, payload: dict[str, Any]) -> str:
    base_name = semantics.get("base_name") or "process"
    location = semantics.get("location_text_hint") or "global markets"
    reference_year = semantics.get("reference_year") or _extract_reference_year(payload, semantics)
    if reference_year:
        return f"Life cycle modelling for {base_name} in {location} (reference year {reference_year})."
    return f"Life cycle modelling for {base_name} in {location}."


def _build_process_administrative_block(
    payload: dict[str, Any],
    semantics: ParsedProcessSemantics,
    dataset_uuid: str,
) -> dict[str, Any]:
    dataset_version = DEFAULT_DATA_SET_VERSION
    intended_text = _compose_intended_applications(semantics, payload)
    return {
        "dataEntryBy": {
            "common:referenceToDataSetFormat": build_dataset_format_reference(),
            "common:referenceToPersonOrEntityEnteringTheData": _contact_reference(),
            "common:timeStamp": _current_timestamp(),
        },
        "publicationAndOwnership": {
            "common:dataSetVersion": dataset_version,
            "common:permanentDataSetURI": build_portal_uri("process", dataset_uuid, dataset_version),
            "common:licenseType": DEFAULT_LICENSE,
            "common:referenceToOwnershipOfDataSet": _ownership_reference(),
        },
        "common:commissionerAndGoal": {
            "common:referenceToCommissioner": _contact_reference(),
            "common:intendedApplications": [_language_entry(intended_text)],
        },
    }


class JSONLDFlowExtractor:
    """Component-based orchestrator for JSON-LD flow extraction."""

    def __init__(
        self,
        llm: LanguageModelProtocol,
        flow_classifier: ProductFlowClassifier,
        location_normalizer: LocationNormalizer,
    ) -> None:
        self._semantic_parser = JSONLDFlowSemanticParser(llm)
        self._classifier = flow_classifier
        self._location_normalizer = location_normalizer
        self._location_catalog = get_location_catalog()
        self._flow_property_audit_log: list[dict[str, str]] = []

    def run(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        LOGGER.info("jsonld.flow_pipeline.start")
        semantics = self._semantic_parser.parse(payload)
        dataset = _build_flow_dataset(
            payload,
            semantics,
            self._classifier,
            self._location_normalizer,
            self._location_catalog,
            audit_log=self._flow_property_audit_log,
        )
        return [{"flowDataSet": dataset}]

    def drain_flow_property_audit_records(self) -> list[dict[str, str]]:
        records = self._flow_property_audit_log
        self._flow_property_audit_log = []
        return records


SOURCE_SEMANTIC_PROMPT_TEMPLATE = """
You are the semantic interpreter for OpenLCA JSON-LD sources. Read the `context` JSON object and
return ONLY a JSON dictionary with the following keys:
- short_name: concise label for the reference.
- category_hint: plain-text description of the source category (e.g., "publications and communications").
- publication_type: choose exactly one of: {publication_types}.
- citation: bibliographic citation text.
- description: brief summary (≤2 sentences).
- year: four-digit publication year if available, else empty string.
- publisher: name of the publisher or organisation, if mentioned.
"""


class ParsedSourceSemantics(TypedDict, total=False):
    short_name: str
    category_hint: str
    publication_type: str
    citation: str
    description: str
    year: str
    publisher: str


class JSONLDSourceSemanticParser:
    PUBLICATION_POINTER = "/properties/sourceDataSet/properties/sourceInformation/properties/dataSetInformation/properties/publicationType"

    def __init__(self, llm: LanguageModelProtocol) -> None:
        self._llm = llm
        self._schema_repository = get_schema_repository()
        self._prompt = self._build_prompt()

    def parse(self, payload: dict[str, Any]) -> ParsedSourceSemantics:
        LOGGER.info("jsonld.source_semantic_parser.invoke")
        response = self._llm.invoke(
            {
                "prompt": self._prompt,
                "context": payload,
                "response_format": {"type": "json_object"},
            }
        )
        data = _coerce_dict(response)
        semantics = ParsedSourceSemantics()
        semantics["short_name"] = _require_text(data.get("short_name"), "short_name")
        semantics["category_hint"] = _clean_text(data.get("category_hint"))
        semantics["publication_type"] = self._normalize_publication_type(data.get("publication_type"))
        semantics["citation"] = _clean_text(data.get("citation"))
        semantics["description"] = _clean_text(data.get("description"))
        semantics["year"] = _clean_text(data.get("year"))
        semantics["publisher"] = _clean_text(data.get("publisher"))
        return semantics

    def _build_prompt(self) -> str:
        schema = self._schema_repository.resolve_with_references("tidas_sources.json", self.PUBLICATION_POINTER)
        values = tuple(item for item in schema.get("enum", []) if isinstance(item, str))
        publication_types = ", ".join(values) or "Undefined, Article in periodical, Monograph"
        return SOURCE_SEMANTIC_PROMPT_TEMPLATE.format(publication_types=publication_types)

    def _normalize_publication_type(self, value: Any) -> str:
        schema = self._schema_repository.resolve_with_references("tidas_sources.json", self.PUBLICATION_POINTER)
        allowed = {item for item in schema.get("enum", []) if isinstance(item, str)}
        text = (_clean_text(value) or "").strip()
        if text in allowed:
            return text
        for option in allowed:
            if option.lower() == text.lower():
                return option
        return DEFAULT_PUBLICATION_TYPE


def _source_class_entry(category: str | None) -> dict[str, str] | None:
    if not category:
        return None
    key = category.strip().lower()
    if key in SOURCE_CATEGORY_CLASS_MAP:
        class_id, label = SOURCE_CATEGORY_CLASS_MAP[key]
        return {"@level": "0", "@classId": class_id, "#text": label}
    if ":" in key:
        prefix, label = key.split(":", 1)
        entry = SOURCE_CATEGORY_CLASS_MAP.get(prefix.strip())
        if entry:
            class_id, text = entry
            return {"@level": "0", "@classId": class_id, "#text": label.strip() or text}
    return None


def _resolve_source_class(category_hint: str | None, payload_category: Any) -> dict[str, str]:
    for candidate in (category_hint, _clean_text(payload_category)):
        entry = _source_class_entry(candidate)
        if entry:
            return entry
    class_id, label = DEFAULT_SOURCE_CLASS
    return {"@level": "0", "@classId": class_id, "#text": label}


def _resolve_publication_type(semantics: ParsedSourceSemantics, payload: dict[str, Any]) -> str:
    if semantics.get("publication_type"):
        return semantics["publication_type"]
    category_hint = semantics.get("category_hint") or _clean_text(payload.get("category"))
    if category_hint:
        mapped = SOURCE_PUBLICATION_TYPE_MAP.get(category_hint.lower())
        if mapped:
            return mapped
    return SOURCE_PUBLICATION_TYPE_MAP.get((_clean_text(payload.get("category")) or "").lower(), DEFAULT_PUBLICATION_TYPE)


def _build_source_dataset(payload: dict[str, Any], semantics: ParsedSourceSemantics) -> dict[str, Any]:
    source_uuid = _clean_text(payload.get("@id")) or str(uuid4())
    dataset_version = DEFAULT_DATA_SET_VERSION
    short_name = _clean_text(payload.get("name")) or semantics.get("short_name") or "Source"
    category_entry = _resolve_source_class(semantics.get("category_hint"), payload.get("category"))
    publication_type = _resolve_publication_type(semantics, payload)
    citation = _clean_text(payload.get("textReference")) or semantics.get("citation") or short_name
    description_parts: list[str] = []
    if semantics.get("description"):
        description_parts.append(semantics["description"])
    payload_description = _clean_text(payload.get("description"))
    if payload_description:
        description_parts.append(payload_description)
    if semantics.get("publisher"):
        description_parts.append(f"Publisher: {semantics['publisher']}")
    description = ". ".join(part.strip().rstrip(".") for part in description_parts if part)
    data_info = {
        "common:UUID": source_uuid,
        "common:shortName": _language_entry(short_name),
        "classificationInformation": {
            "common:classification": {"common:class": category_entry},
        },
        "sourceCitation": citation,
        "publicationType": publication_type,
        "referenceToContact": _contact_reference(),
    }
    if description:
        data_info["sourceDescriptionOrComment"] = [_language_entry(description)]
    source_information: dict[str, Any] = {"dataSetInformation": data_info}
    administrative = {
        "dataEntryBy": {
            "common:referenceToDataSetFormat": build_dataset_format_reference(),
            "common:referenceToPersonOrEntityEnteringTheData": _contact_reference(),
            "common:timeStamp": _current_timestamp(),
        },
        "publicationAndOwnership": {
            "common:dataSetVersion": dataset_version,
            "common:permanentDataSetURI": build_portal_uri("source", source_uuid, dataset_version),
            "common:licenseType": DEFAULT_LICENSE,
            "common:referenceToOwnershipOfDataSet": _ownership_reference(),
        },
    }
    source_dataset = {
        **ILCD_SOURCE_XMLNS,
        "sourceInformation": source_information,
        "administrativeInformation": administrative,
    }
    return source_dataset


class JSONLDSourceExtractor:
    """Component-based orchestrator for JSON-LD source extraction."""

    def __init__(self, llm: LanguageModelProtocol) -> None:
        self._semantic_parser = JSONLDSourceSemanticParser(llm)

    def run(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        LOGGER.info("jsonld.source_pipeline.start")
        semantics = self._semantic_parser.parse(payload)
        dataset = _build_source_dataset(payload, semantics)
        return [{"sourceDataSet": dataset}]


class JSONLDProcessExtractor:
    """Component-based orchestrator for JSON-LD process extraction."""

    def __init__(
        self,
        llm: LanguageModelProtocol,
        process_classifier: ProcessClassifier,
        location_normalizer: LocationNormalizer,
    ) -> None:
        self._llm = llm
        self._semantic_parser = JSONLDSemanticParser(llm)
        self._classifier = process_classifier
        self._location_normalizer = location_normalizer
        self._location_catalog = get_location_catalog()

    def run(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        LOGGER.info("jsonld.process_pipeline.start")
        semantics = self._semantic_parser.parse(payload)
        dataset = self._build_dataset(payload, semantics)
        normalized = build_tidas_process_dataset(dataset)
        process_id = normalized.get("processInformation", {}).get("dataSetInformation", {}).get("common:UUID") or str(uuid4())
        return [{"processDataSet": normalized, "process_id": process_id}]

    def _build_dataset(self, payload: dict[str, Any], semantics: ParsedProcessSemantics) -> dict[str, Any]:
        exchanges, reference_id, functional_unit = _build_exchange_entries(payload)
        process_information, location_hint = self._build_process_information(
            payload,
            semantics,
            reference_id,
            functional_unit,
        )
        dataset_uuid = process_information.get("dataSetInformation", {}).get("common:UUID") or str(uuid4())
        try:
            classification_path = self._classifier.run(process_information)
        except ProcessExtractionError:
            LOGGER.warning("jsonld.process_pipeline.classification_fallback")
            classification_path = list(DEFAULT_PROCESS_CLASS_PATH)
        self._apply_classification(process_information, classification_path)
        self._recover_missing_name_fields(process_information, payload)
        self._apply_location_normalization(process_information, location_hint)
        modelling = self._build_modelling_block(payload, semantics)
        administrative = _build_process_administrative_block(payload, semantics, dataset_uuid)
        dataset = {
            "processInformation": process_information,
            "modellingAndValidation": modelling,
            "administrativeInformation": administrative,
            "exchanges": {"exchange": exchanges},
        }
        return dataset

    def _build_process_information(
        self,
        payload: dict[str, Any],
        semantics: ParsedProcessSemantics,
        reference_id: str | None,
        functional_unit: str | None,
    ) -> tuple[dict[str, Any], str | None]:
        dataset_uuid = _clean_text(payload.get("@id")) or str(uuid4())
        documentation = payload.get("processDocumentation") or {}
        technology_block = _build_technology_block(payload, semantics)
        general_comment = _merge_description(semantics, payload)
        location_hint = _derive_location_hint(payload, semantics)

        base_name_text, treatment_text, mix_text = _derive_process_name_fields(payload, semantics)
        name_block: dict[str, Any] = {}
        if base_name_text:
            name_block["baseName"] = _language_entry(base_name_text)
        if treatment_text:
            name_block["treatmentStandardsRoutes"] = _language_entry(treatment_text)
        elif base_name_text:
            name_block["treatmentStandardsRoutes"] = _language_entry(base_name_text)
        if mix_text:
            name_block["mixAndLocationTypes"] = _language_entry(mix_text)
        elif location_hint:
            name_block["mixAndLocationTypes"] = _language_entry(location_hint)
        if functional_unit:
            name_block["functionalUnitFlowProperties"] = _language_entry(functional_unit)

        data_info: dict[str, Any] = {
            "common:UUID": dataset_uuid,
            "name": name_block,
            "classificationInformation": {"common:classification": {}},
        }
        # Preserve identifierOfSubDataSet only when it exists in the source payload.
        source_identifier = _clean_text(payload.get("identifierOfSubDataSet"))
        if source_identifier:
            data_info["identifierOfSubDataSet"] = source_identifier
        description_entry = general_comment or _clean_text(documentation.get("inventoryMethodDescription"))
        if description_entry:
            data_info["common:generalComment"] = _language_entry(description_entry)
        synonyms = _build_synonym_entries(payload.get("synonyms"))
        if synonyms:
            data_info["common:synonyms"] = synonyms

        quantitative_reference: dict[str, Any] = {
            "@type": "Reference flow(s)",
            "referenceToReferenceFlow": reference_id or "1",
            "functionalUnitOrOther": _language_entry(functional_unit or f"1 unit of {semantics['base_name']}"),
        }

        reference_year = _extract_reference_year(payload, semantics)
        time_block: dict[str, Any] = {}
        if reference_year:
            time_block["common:referenceYear"] = reference_year
        time_block["common:dataSetValidUntil"] = DEFAULT_DATASET_VALID_UNTIL_YEAR

        geography_block = {
            "locationOfOperationSupplyOrProduction": {
                "description": location_hint,
            }
        }

        process_information: dict[str, Any] = {
            "dataSetInformation": data_info,
            "quantitativeReference": quantitative_reference,
            "time": time_block,
            "geography": geography_block,
        }
        if technology_block:
            process_information["technology"] = technology_block
        return process_information, location_hint

    def _apply_classification(self, process_information: dict[str, Any], path: list[dict[str, Any]]) -> None:
        container = process_information.setdefault("dataSetInformation", {}).setdefault("classificationInformation", {}).setdefault("common:classification", {})
        container["common:class"] = path

    def _apply_location_normalization(self, process_information: dict[str, Any], location_hint: str | None) -> None:
        catalog = self._location_catalog
        initial_code = catalog.best_guess(location_hint)
        candidates = catalog.build_candidate_list(location_hint)
        try:
            geography_update = self._location_normalizer.run(
                {"processInformation": process_information},
                hint=location_hint,
                candidates=candidates,
                initial_code=initial_code,
            )
        except Exception as exc:  # noqa: BLE001 - keep pipeline resilient
            LOGGER.warning("jsonld.location_normalization_failed", error=str(exc))
            geography_update = None
        code_from_response, payload = extract_location_response(geography_update)
        final_code = catalog.coerce_code(code_from_response) or initial_code
        geography = process_information.setdefault("geography", {})
        if isinstance(payload, dict):
            geography.update(payload)
        if final_code:
            geography["code"] = final_code
            geography.setdefault("description", catalog.describe(final_code))
            location_block = geography.setdefault("locationOfOperationSupplyOrProduction", {})
            if isinstance(location_block, dict):
                location_block["@location"] = final_code
                if not _clean_text(location_block.get("name")):
                    location_block["name"] = geography.get("description") or location_hint or catalog.describe(final_code)

    def _recover_missing_name_fields(self, process_information: dict[str, Any], payload: dict[str, Any]) -> None:
        data_info = process_information.get("dataSetInformation")
        if not isinstance(data_info, dict):
            return
        name_block = data_info.get("name")
        if not isinstance(name_block, dict):
            return
        required = ("baseName", "treatmentStandardsRoutes", "mixAndLocationTypes")
        missing = [key for key in required if not _has_text_entry(name_block.get(key))]
        if not missing:
            return
        context = {
            "missingFields": missing,
            "processInformation": process_information,
            "sourcePayload": payload,
        }
        try:
            response = self._llm.invoke(
                {
                    "prompt": NAME_RECOVERY_PROMPT,
                    "context": context,
                    "response_format": {"type": "json_object"},
                }
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("jsonld.name_recovery_failed", error=str(exc))
            return
        recovery = response if isinstance(response, dict) else {}
        for key in missing:
            value = _clean_text(recovery.get(key))
            if value:
                name_block[key] = _language_entry(value)

    def _build_modelling_block(self, payload: dict[str, Any], semantics: ParsedProcessSemantics) -> dict[str, Any]:
        return {
            "LCIMethodAndAllocation": {
                "typeOfDataSet": _map_process_type(payload.get("processType")),
                "LCIMethodPrinciple": semantics.get("lci_method_principle") or "Not applicable",
            }
        }
