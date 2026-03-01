"""Thin wrappers to publish flows and processes via Database_CRUD_Tool."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

from tiangong_lca_spec.core.config import Settings, get_settings
from tiangong_lca_spec.core.exceptions import SpecCodingError
from tiangong_lca_spec.core.json_utils import parse_json_response
from tiangong_lca_spec.core.logging import get_logger
from tiangong_lca_spec.core.mcp_client import MCPToolClient
from tiangong_lca_spec.core.uris import build_portal_uri
from tiangong_lca_spec.product_flow_creation import (
    FlowDedupService,
    ProductFlowCreateRequest,
    ProductFlowCreationService,
)
from tiangong_lca_spec.tidas.flow_property_registry import (
    FlowPropertyRegistry,
    get_default_registry,
)

LOGGER = get_logger(__name__)

DATABASE_TOOL_NAME = "Database_CRUD_Tool"
_PROJECT_ROOT_OVERRIDE = (os.getenv("PAB_PROJECT_ROOT") or "").strip()
if _PROJECT_ROOT_OVERRIDE:
    PROJECT_ROOT = Path(_PROJECT_ROOT_OVERRIDE).expanduser()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

_PRODUCT_CATEGORY_SCRIPT_OVERRIDE = (os.getenv("PAB_PRODUCT_CATEGORY_SCRIPT") or "").strip()
if _PRODUCT_CATEGORY_SCRIPT_OVERRIDE:
    _product_category_script_path = Path(_PRODUCT_CATEGORY_SCRIPT_OVERRIDE).expanduser()
    if not _product_category_script_path.is_absolute():
        _product_category_script_path = PROJECT_ROOT / _product_category_script_path
    PRODUCT_CATEGORY_SCRIPT = _product_category_script_path
else:
    PRODUCT_CATEGORY_SCRIPT = PROJECT_ROOT / "scripts" / "md" / "list_product_flow_category_children.py"
FLOW_TEXT_PROMPT = (
    "You generate bilingual ILCD flow text fields for Tiangong LCA.\n"
    "Use the provided exchange, flow search hints, and candidate metadata to fill missing values. "
    "Return strict JSON with keys:\n"
    "- base_name_en, base_name_zh\n"
    "- treatment_en, treatment_zh\n"
    "- mix_en, mix_zh\n"
    "- synonyms_en (list of short terms), synonyms_zh (list of short terms)\n"
    "- comment_en, comment_zh\n\n"
    "Rules:\n"
    "- base_name: short noun phrase, no units/amounts.\n"
    "- treatment/mix: use short phrases describing treatment/route and mix/location.\n"
    "- comment: 1 concise sentence.\n"
    "- Avoid semicolons; use commas if needed.\n"
    "- Output must be valid JSON; do not add extra keys."
)
FLOW_PROPERTY_SELECTOR_PROMPT = (
    "You select the most appropriate ILCD flow property for creating a new placeholder flow.\n"
    "Choose ONLY from the provided candidates. Never invent UUIDs.\n"
    "Base your choice on exchange name, aliases, general comment, unit, direction, and any hints.\n\n"
    "Return strict JSON with keys:\n"
    '- `selected_uuid`: string or null (must be one of candidate UUIDs when not null)\n'
    '- `confidence`: number from 0 to 1\n'
    '- `reason`: short explanation\n'
    "- `is_uncertain`: boolean\n"
    '- `alternatives`: array of objects with keys `uuid` and `why` (optional, keep short)\n\n'
    "Rules:\n"
    "- If multiple candidates are plausible and evidence is weak, set `is_uncertain=true`.\n"
    "- If no candidate is reasonable, return `selected_uuid=null` and `is_uncertain=true`.\n"
    "- Never output a UUID outside the candidate list."
)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        text = value.get("#text")
        if isinstance(text, str):
            return text.strip()
    return str(value).strip()


def _parse_flowsearch_hints(comment: str | None) -> dict[str, list[str] | str]:
    """Parse 'FlowSearch hints:' into a dict of list values."""
    if not comment:
        return {}
    text = comment.strip()
    prefix = "FlowSearch hints:"
    if text.startswith(prefix):
        text = text[len(prefix) :].strip()
    segments = [segment.strip() for segment in text.split("|") if segment.strip()]
    output: dict[str, list[str] | str] = {}
    for segment in segments:
        key, _, value = segment.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if not value or value == "NA":
            output[key] = []
            continue
        parts = [item.strip() for item in value.split(";") if item.strip()]
        output[key] = parts or [value]
    return output


def _normalize_hint_values(hints: Mapping[str, list[str] | str]) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for key, value in hints.items():
        if isinstance(value, str):
            text = _coerce_text(value)
            if text:
                normalized[key] = [text]
            continue
        if isinstance(value, Iterable):
            entries = [_coerce_text(item) for item in value if _coerce_text(item)]
            if entries:
                normalized[key] = entries
    return normalized


def _replace_semicolons(text: str) -> str:
    """Avoid semicolons in flow text fields by replacing with commas."""
    return text.replace("；", "，").replace(";", ",")


def _normalize_text(value: Any) -> str:
    text = _coerce_text(value)
    if not text:
        return ""
    return _replace_semicolons(text)


def _normalize_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return []
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            items.extend(_normalize_text_list(item))
        return items
    if isinstance(value, str):
        raw = value.replace("；", ";")
        parts: list[str] = []
        for chunk in raw.split(";"):
            for piece in chunk.split(","):
                text = piece.strip()
                if text:
                    parts.append(text)
        return parts
    text = _coerce_text(value)
    return [text] if text else []


def _normalize_unit_token(value: str | None) -> str:
    if not value:
        return ""
    text = str(value).strip().lower()
    text = text.replace(" ", "")
    text = text.replace("^", "")
    text = text.replace("³", "3")
    text = text.replace("·", "*")
    text = text.replace("×", "*")
    return text


_PROPERTY_UNIT_DIMENSIONS: dict[str, str] = {
    "kg": "mass",
    "g": "mass",
    "mg": "mass",
    "t": "mass",
    "lb": "mass",
    "lbav": "mass",
    "oz": "mass",
    "mj": "energy",
    "gj": "energy",
    "j": "energy",
    "kwh": "energy",
    "mwh": "energy",
    "tce": "energy",
    "toe": "energy",
    "kcal": "energy",
    "btu": "energy",
    "m3": "volume",
    "l": "volume",
    "liter": "volume",
    "litre": "volume",
    "dm3": "volume",
    "cm3": "volume",
    "ml": "volume",
    "m2": "area",
    "m": "length",
    "km": "length",
    "piece": "items",
    "pcs": "items",
    "pc": "items",
    "ea": "items",
}


def _unit_dimension(unit: str | None) -> str | None:
    return _PROPERTY_UNIT_DIMENSIONS.get(_normalize_unit_token(unit))


def _extract_hint_value(hints: Mapping[str, list[str] | str], keys: Sequence[str]) -> str:
    for key in keys:
        value = hints.get(key)
        if isinstance(value, list) and value:
            text = _normalize_text(value[0])
            if text:
                return text
        if isinstance(value, str):
            text = _normalize_text(value)
            if text:
                return text
    return ""


def _collect_classification_entries(exchange: Mapping[str, Any]) -> list[str]:
    results: list[str] = []
    classification = exchange.get("classificationInformation") or exchange.get("classification")
    if isinstance(classification, Mapping):
        carrier = classification.get("common:classification") or classification.get("classification")
        if isinstance(carrier, Mapping):
            classes = carrier.get("common:class") or carrier.get("class")
        else:
            classes = carrier
    else:
        classes = classification
    if isinstance(classes, list):
        for entry in classes:
            if isinstance(entry, Mapping):
                label = _coerce_text(entry.get("#text") or entry.get("text"))
                level = _coerce_text(entry.get("@level"))
                if label and level:
                    results.append(f"{level}:{label}")
                elif label:
                    results.append(label)
            else:
                text = _coerce_text(entry)
                if text:
                    results.append(text)
    return results


def _collect_tag_entries(exchange: Mapping[str, Any]) -> list[str]:
    tags: list[str] = []
    for key in ("synonyms", "synonym", "CASNumber", "chemFormula", "formula", "additionalInformation"):
        value = exchange.get(key)
        if isinstance(value, str):
            candidate = _coerce_text(value)
            if candidate:
                tags.append(candidate)
        elif isinstance(value, Mapping):
            candidate = _coerce_text(value.get("#text") or value.get("text"))
            if candidate:
                tags.append(candidate)
        elif isinstance(value, Iterable):
            for item in value:
                candidate = _coerce_text(item)
                if candidate:
                    tags.append(candidate)
    return tags


def _collect_reference_summary(exchange: Mapping[str, Any]) -> dict[str, Any] | None:
    reference = exchange.get("referenceToFlowDataSet")
    if not isinstance(reference, Mapping):
        return None
    summary: dict[str, Any] = {}
    for key in ("@refObjectId", "@version", "@uri"):
        text = _coerce_text(reference.get(key))
        if text:
            summary[key] = text
    if reference.get("unmatched:placeholder"):
        summary["placeholder"] = True
    return summary or None


def _collect_candidate_summary(exchange: Mapping[str, Any]) -> dict[str, Any] | None:
    detail = exchange.get("matchingDetail")
    if not isinstance(detail, Mapping):
        return None
    candidate = detail.get("selectedCandidate")
    if not isinstance(candidate, Mapping):
        return None
    summary: dict[str, Any] = {}
    for key in ("uuid", "base_name", "flow_properties", "geography", "classification", "general_comment", "selector", "evaluation_reason"):
        value = candidate.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            text = _coerce_text(value)
            if text:
                summary[key] = text
        elif isinstance(value, Mapping):
            sanitized = {k: _coerce_text(v) for k, v in value.items() if isinstance(k, str) and _coerce_text(v)}
            if sanitized:
                summary[key] = sanitized
        elif isinstance(value, Iterable):
            collected = [_coerce_text(item) for item in value if _coerce_text(item)]
            if collected:
                summary[key] = collected
    return summary or None


def _assign_if_value(target: dict[str, Any], key: str, value: str) -> None:
    if value:
        target[key] = value


def _compose_flow_context(exchange: Mapping[str, Any], hints: Mapping[str, list[str] | str]) -> dict[str, Any]:
    normalized_hints = _normalize_hint_values(hints)
    summary: dict[str, Any] = {}
    _assign_if_value(summary, "name", _coerce_text(exchange.get("exchangeName")))
    _assign_if_value(summary, "direction", _coerce_text(exchange.get("exchangeDirection")))
    _assign_if_value(summary, "unit", _resolve_unit(exchange))
    _assign_if_value(
        summary,
        "amount",
        _coerce_text(exchange.get("meanAmount") or exchange.get("resultingAmount") or exchange.get("amount")),
    )
    _assign_if_value(summary, "general_comment", _extract_general_comment(exchange))
    _assign_if_value(summary, "input_group", _coerce_text(exchange.get("inputGroup")))
    _assign_if_value(summary, "output_group", _coerce_text(exchange.get("outputGroup")))
    _assign_if_value(summary, "location", _coerce_text(exchange.get("location")))
    _assign_if_value(summary, "cas_number", _coerce_text(exchange.get("CASNumber")))
    _assign_if_value(summary, "formula", _coerce_text(exchange.get("chemFormula") or exchange.get("formula")))
    summary["classification_path"] = _collect_classification_entries(exchange)
    summary["tags"] = _collect_tag_entries(exchange)
    reference = _collect_reference_summary(exchange)
    if reference:
        summary["reference"] = reference
    candidate = _collect_candidate_summary(exchange)
    if candidate:
        summary["selected_candidate"] = candidate
    return {
        "exchange": summary,
        "flow_search_hints": normalized_hints,
    }


def _ensure_mapping_response(response: Any) -> Mapping[str, Any]:
    payload = getattr(response, "content", response)
    if isinstance(payload, str):
        parsed = parse_json_response(payload)
    else:
        parsed = payload
    if isinstance(parsed, Mapping):
        return parsed
    raise ValueError("Language model returned non-dict payload")


class LanguageModelProtocol(Protocol):
    """Minimal protocol for language models used during publishing."""

    def invoke(self, input_data: dict[str, Any]) -> Any: ...


class FlowTypeClassifier:
    """Infer flow types using an optional language model with heuristic fallback."""

    ALLOWED_TYPES = {"Product flow", "Elementary flow", "Waste flow"}
    ELEMENTARY_KEYWORDS = ("emission", "to air", "to water", "wastewater", "effluent", "flue gas", "to soil", "released")
    WASTE_KEYWORDS = ("waste", "slag", "scrap", "residue", "ash", "sludge")
    PROMPT = (
        "You classify life cycle assessment exchanges into one of three flow types.\n"
        "- Product flow: exchanges that represent technical products, services, materials, or energy "
        "circulating within the technosphere and available for downstream use or with economic value.\n"
        "- Elementary flow: exchanges that connect the technosphere with the natural environment "
        "(emissions to air/water/soil or extractions of natural resources).\n"
        "- Waste flow: outputs that leave the technosphere as wastes or by-products requiring waste "
        "management or treatment by another activity.\n"
        "Use the provided data (exchange metadata, hints, candidate details) to pick the best match. "
        "Return strict JSON with keys `flow_type` (one of Product flow, Elementary flow, Waste flow) "
        "and `reason` summarising the evidence. Do not invent new categories."
    )

    def __init__(self, llm: LanguageModelProtocol | None = None) -> None:
        self._llm = llm

    def infer(self, exchange: Mapping[str, Any], hints: Mapping[str, list[str] | str]) -> str:
        context = self._build_context(exchange, hints)
        if self._llm is not None:
            try:
                response = self._llm.invoke(
                    {
                        "prompt": self.PROMPT,
                        "context": context,
                        "response_format": {"type": "json_object"},
                    }
                )
                payload = _ensure_mapping_response(response)
                flow_type = self._normalise_flow_type(payload.get("flow_type"))
                if flow_type:
                    return flow_type
                LOGGER.warning(
                    "flow_publish.flow_type_invalid_response",
                    response=payload,
                )
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("flow_publish.flow_type_llm_failed", error=str(exc))
        fallback = self._heuristic_infer(context)
        LOGGER.debug("flow_publish.flow_type_fallback", selected=fallback)
        return fallback

    def _build_context(self, exchange: Mapping[str, Any], hints: Mapping[str, list[str] | str]) -> dict[str, Any]:
        return _compose_flow_context(exchange, hints)

    def _normalise_flow_type(self, raw: Any) -> str | None:
        if not isinstance(raw, str):
            return None
        candidate = raw.strip().lower()
        for allowed in self.ALLOWED_TYPES:
            if candidate == allowed.lower():
                return allowed
        if candidate in {"product", "productflow"}:
            return "Product flow"
        if candidate in {"elementary", "elementaryflow"}:
            return "Elementary flow"
        if candidate in {"waste", "wasteflow"}:
            return "Waste flow"
        return None

    def _heuristic_infer(self, context: Mapping[str, Any]) -> str:
        exchange = context.get("exchange", {})
        hints = context.get("flow_search_hints", {})
        direction = _coerce_text(exchange.get("direction")).lower()
        text_parts = [
            _coerce_text(exchange.get("name")),
            _coerce_text(exchange.get("general_comment")),
            " ".join(exchange.get("classification_path", [])),
            " ".join(exchange.get("tags", [])),
        ]
        hint_fragments: list[str] = []
        if isinstance(hints, Mapping):
            for values in hints.values():
                if isinstance(values, list):
                    hint_fragments.extend([_coerce_text(value) for value in values])
        text_parts.append(" ".join(fragment for fragment in hint_fragments if fragment))
        combined = " ".join(part for part in text_parts if part).lower()

        if any(keyword in combined for keyword in self.ELEMENTARY_KEYWORDS):
            return "Elementary flow"
        if "waste" in combined or any(keyword in combined for keyword in self.WASTE_KEYWORDS):
            return "Waste flow"
        if direction == "input" and any(term in combined for term in (" ambient air", "air", "water extraction", "surface water", "groundwater", "raw water", "ore", "crude")):
            return "Elementary flow"
        if direction == "output" and "slag" in combined:
            return "Waste flow"
        return "Product flow"


class FlowProductCategorySelector:
    """Select the most specific product category path using LLM-guided traversal."""

    STOP_CHOICES = {"stop", "none", "n/a", "na", "null", "skip"}
    PROMPT = (
        "You are helping to classify a life cycle assessment product flow into Tiangong's "
        "product category hierarchy. Each step provides the exchange context, any categories "
        "selected so far, and the direct children available at the current level.\n\n"
        "Return strict JSON with keys:\n"
        "- `choice`: the code of the selected option (must match one of the provided `options.code`) "
        "or `STOP` if none fit.\n"
        "- `reason`: short explanation of the selection.\n\n"
        "Prefer the most specific child that aligns with the exchange name, hints, and candidate data. "
        'If no option is appropriate, respond with `choice: "STOP"`.'
    )

    def __init__(
        self,
        llm: LanguageModelProtocol | None = None,
        *,
        script_path: Path | None = None,
        max_depth: int = 6,
    ) -> None:
        self._llm = llm
        self._script_path = Path(script_path) if script_path else PRODUCT_CATEGORY_SCRIPT
        self._max_depth = max(1, max_depth)
        self._cache: dict[str, list[tuple[str, str]]] = {}
        self._script_available = self._script_path.exists()
        if not self._script_available:
            LOGGER.warning("flow_publish.product_category_script_missing", path=str(self._script_path))

    def select_path(
        self,
        exchange: Mapping[str, Any],
        hints: Mapping[str, list[str] | str],
    ) -> list[tuple[str, str]]:
        context = _compose_flow_context(exchange, hints)
        path: list[tuple[str, str]] = []
        current_code: str | None = None
        for depth in range(self._max_depth):
            options = self._list_children(current_code)
            if not options:
                break
            choice = self._choose_option(context, path, options)
            if not choice:
                break
            match = next((item for item in options if item[0].lower() == choice.lower()), None)
            if not match:
                LOGGER.warning(
                    "flow_publish.product_category_invalid_choice",
                    choice=choice,
                    options=[code for code, _ in options],
                )
                break
            path.append(match)
            current_code = match[0]
        return path

    def _list_children(self, code: str | None) -> list[tuple[str, str]]:
        key = code or "__root__"
        if key in self._cache:
            return self._cache[key]
        if not self._script_available:
            self._cache[key] = []
            return []

        args = [sys.executable, str(self._script_path)]
        if code:
            args.append(code)
        try:
            result = subprocess.run(args, capture_output=True, text=True, check=False)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("flow_publish.product_category_call_failed", code=code, error=str(exc))
            children: list[tuple[str, str]] = []
        else:
            if result.returncode != 0:
                LOGGER.warning(
                    "flow_publish.product_category_script_error",
                    code=code,
                    stderr=result.stderr.strip(),
                )
                children = []
            else:
                children = []
                for raw_line in result.stdout.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    # Hierarchy helper scripts print data rows as "<code>\\t<description>".
                    # Informational lines like "No direct children found for <code>" must be ignored.
                    if "\t" not in line:
                        LOGGER.debug(
                            "flow_publish.product_category_skip_non_data_line",
                            code=code,
                            line=line,
                        )
                        continue
                    cat_code, desc = line.split("\t", 1)
                    cat_code = cat_code.strip()
                    desc = desc.strip()
                    if cat_code:
                        children.append((cat_code, desc))
        self._cache[key] = children
        return children

    def _choose_option(
        self,
        context: Mapping[str, Any],
        path: list[tuple[str, str]],
        options: list[tuple[str, str]],
    ) -> str | None:
        if self._llm is not None:
            try:
                response = self._llm.invoke(
                    {
                        "prompt": self.PROMPT,
                        "context": {
                            "exchange_context": context,
                            "selected_path": self._path_payload(path),
                            "options": self._options_payload(options),
                        },
                        "response_format": {"type": "json_object"},
                    }
                )
                data = _ensure_mapping_response(response)
                choice = self._extract_choice(data, options)
                if choice:
                    return choice
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("flow_publish.product_category_llm_failed", error=str(exc))
        return self._heuristic_option(context, options)

    def _extract_choice(self, data: Mapping[str, Any], options: list[tuple[str, str]]) -> str | None:
        raw_choice = _coerce_text(data.get("choice") or data.get("code") or data.get("class_id"))
        if not raw_choice:
            return None
        candidate = raw_choice.split()[0].strip()
        cleaned = candidate.rstrip(".")
        normalized = cleaned.lower()
        if normalized in self.STOP_CHOICES:
            return None
        for code, _ in options:
            if normalized == code.lower():
                return code
        for code, description in options:
            if normalized and normalized in description.lower():
                return code
        return None

    def _heuristic_option(self, context: Mapping[str, Any], options: list[tuple[str, str]]) -> str | None:
        if not options:
            return None
        exchange = context.get("exchange", {})
        hints = context.get("flow_search_hints", {})
        fragments: list[str] = [
            _coerce_text(exchange.get("name")),
            _coerce_text(exchange.get("general_comment")),
        ]
        fragments.extend(exchange.get("tags", []))
        if isinstance(hints, Mapping):
            for values in hints.values():
                if isinstance(values, list):
                    fragments.extend(values)
        haystack = " ".join(fragment for fragment in fragments if fragment).lower()
        if not haystack:
            return None

        best_code: str | None = None
        best_score = 0.0
        for code, desc in options:
            desc_lower = desc.lower()
            score = 0.0
            if desc_lower and desc_lower in haystack:
                score += 3.0
            if "electric" in desc_lower and "electric" in haystack:
                score += 2.5
            if code.lower() in haystack:
                score += 1.0
            if "transport" in desc_lower and "transport" in haystack:
                score += 1.5
            if score > best_score:
                best_score = score
                best_code = code
        return best_code

    @staticmethod
    def _options_payload(options: list[tuple[str, str]]) -> list[dict[str, str]]:
        return [{"code": code, "description": desc} for code, desc in options]

    @staticmethod
    def _path_payload(path: list[tuple[str, str]]) -> list[dict[str, str]]:
        return [{"code": code, "description": desc} for code, desc in path]


def _infer_flow_type(
    exchange: Mapping[str, Any],
    hints: Mapping[str, list[str] | str],
    *,
    classifier: FlowTypeClassifier | None = None,
    llm: LanguageModelProtocol | None = None,
) -> str:
    engine = classifier or FlowTypeClassifier(llm)
    return engine.infer(exchange, hints)


def _classification_from_path(path: Sequence[tuple[str, str]]) -> dict[str, Any]:
    classes = [
        {
            "@level": str(index),
            "@classId": code,
            "#text": description,
        }
        for index, (code, description) in enumerate(path)
    ]
    if not classes:
        return _default_product_classification()
    return {"common:classification": {"common:class": classes}}


def _default_product_classification() -> dict[str, Any]:
    return {
        "common:classification": {
            "common:class": [
                {
                    "@level": "0",
                    "@classId": "1",
                    "#text": "Ores and minerals; electricity, gas and water",
                }
            ]
        }
    }


def _derive_language_pairs(hints: Mapping[str, list[str] | str], fallback: str) -> tuple[str, str]:
    en_candidates = [item for item in hints.get("en_synonyms", []) or [] if isinstance(item, str)]
    zh_candidates = [item for item in hints.get("zh_synonyms", []) or [] if isinstance(item, str)]
    en = en_candidates[0] if en_candidates else fallback
    zh = zh_candidates[0] if zh_candidates else en
    return en, zh


def _get_nested(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    """Return nested value via keys, or None when any level is missing."""
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _resolve_dataset_root(
    payload: Mapping[str, Any],
    *,
    root_key: str | None,
    dataset_kind: str,
) -> Mapping[str, Any]:
    """Return the ILCD dataset block and validate its structure."""
    if root_key is None:
        target = payload
    else:
        target = payload.get(root_key)
        if target is None:
            raise SpecCodingError(f"{dataset_kind} payload missing '{root_key}'.")
    if not isinstance(target, Mapping):
        location = root_key or "root"
        raise SpecCodingError(f"{dataset_kind} payload must be an object at '{location}'.")
    return target


def _require_uuid(value: Any, dataset_kind: str) -> str:
    uuid_value = _coerce_text(value)
    if not uuid_value:
        raise SpecCodingError(f"Missing common:UUID for {dataset_kind} dataset.")
    return uuid_value


def _build_elementary_classification(hints: Mapping[str, list[str] | str]) -> dict[str, Any]:
    usage = _coerce_text(hints.get("usage_context"))
    usage_lower = usage.lower()
    if "air" in usage_lower or "vent" in usage_lower:
        path = ["Emissions", "Emissions to air", "Emissions to air, unspecified"]
    elif "water" in usage_lower or "effluent" in usage_lower:
        path = ["Emissions", "Emissions to water", "Emissions to water, unspecified"]
    elif "soil" in usage_lower:
        path = ["Emissions", "Emissions to soil", "Emissions to soil, unspecified"]
    else:
        path = ["Emissions", "Emissions to unspecified"]
    categories = []
    for level, label in enumerate(path):
        categories.append({"@level": str(level), "#text": label})
    return {"common:elementaryFlowCategorization": {"common:category": categories}}


def _extract_general_comment(exchange: Mapping[str, Any]) -> str:
    comment = exchange.get("generalComment")
    if isinstance(comment, dict):
        text = comment.get("#text")
        if isinstance(text, str):
            return text.strip()
    if isinstance(comment, str):
        return comment.strip()
    return ""


def _resolve_unit(exchange: Mapping[str, Any]) -> str:
    return _coerce_text(exchange.get("unit"))


def _language_entry(text: str, lang: str = "en") -> dict[str, Any]:
    return {"@xml:lang": lang, "#text": text}


def _bilingual_language_entries(text_en: str, text_zh: str | None = None) -> list[dict[str, Any]]:
    en_text = _normalize_text(text_en) or "Unnamed flow"
    zh_text = _normalize_text(text_zh) or en_text
    return [
        _language_entry(en_text, "en"),
        _language_entry(zh_text, "zh"),
    ]


@dataclass(slots=True, frozen=True)
class FlowPropertyOverride:
    """Override entry used to customise flow property selection."""

    flow_property_uuid: str
    mean_value: str | None = None


@dataclass(slots=True)
class PropertyDecision:
    """Decision result for selecting a flow property during flow publishing."""

    selected_uuid: str | None
    selected_name: str | None = None
    selected_version: str | None = None
    confidence: float = 0.0
    reason: str = ""
    candidate_uuids: tuple[str, ...] = ()
    decision_mode: str = "manual_required"
    validation_passed: bool = False
    validation_errors: tuple[str, ...] = ()
    mean_value: str | None = None
    retry_count: int = 0
    is_uncertain: bool = False


@dataclass
class FlowPublishPlan:
    """Single flow payload ready for publication."""

    uuid: str
    exchange_name: str
    process_name: str
    dataset: Mapping[str, Any]
    exchange_ref: Mapping[str, Any]
    mode: str = "insert"
    flow_property_uuid: str | None = None


class DatabaseCrudClient:
    """Client wrapper over Database_CRUD_Tool for flows/processes CRUD."""

    _FLOW_SELECT_CACHE_VALUES: dict[str, dict[str, Any]] = {}
    _FLOW_SELECT_RECORD_CACHE: dict[str, dict[str, Any]] = {}
    _FLOW_SELECT_CACHE_MISSES: set[str] = set()
    _FLOW_SELECT_RECORD_MISSES: set[str] = set()
    _FLOW_SELECT_CACHE_PATH: Path | None = None
    _FLOW_SELECT_CACHE_LOADED: bool = False
    _FLOW_SELECT_CACHE_DIRTY: bool = False

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        mcp_client: MCPToolClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._mcp = mcp_client or MCPToolClient(self._settings)
        self._server_name = self._settings.flow_search_service_name
        self._ensure_flow_select_cache_loaded()

    def insert_flow(self, dataset: Mapping[str, Any]) -> dict[str, Any]:
        root_key = "flowDataSet" if "flowDataSet" in dataset else None
        flow_root = _resolve_dataset_root(dataset, root_key=root_key, dataset_kind="flow")
        uuid_value = _require_uuid(
            _get_nested(flow_root, ("flowInformation", "dataSetInformation", "common:UUID")),
            "flow",
        )
        json_payload = dataset if root_key else {"flowDataSet": flow_root}
        return self._invoke(
            {
                "operation": "insert",
                "table": "flows",
                "id": uuid_value,
                "jsonOrdered": json_payload,
            }
        )

    def update_flow(self, dataset: Mapping[str, Any]) -> dict[str, Any]:
        root_key = "flowDataSet" if "flowDataSet" in dataset else None
        flow_root = _resolve_dataset_root(dataset, root_key=root_key, dataset_kind="flow")
        uuid_value = _require_uuid(
            _get_nested(flow_root, ("flowInformation", "dataSetInformation", "common:UUID")),
            "flow",
        )
        version_candidate = _coerce_text(
            _get_nested(
                flow_root,
                ("administrativeInformation", "publicationAndOwnership", "common:dataSetVersion"),
            )
        )
        if not version_candidate:
            version_candidate = "01.01.000"
        json_payload = dataset if root_key else {"flowDataSet": flow_root}
        return self._invoke(
            {
                "operation": "update",
                "table": "flows",
                "id": uuid_value,
                "version": version_candidate,
                "jsonOrdered": json_payload,
            }
        )

    def select_flow(self, flow_uuid: str, *, version: str | None = None) -> dict[str, Any] | None:
        uuid_value = _coerce_text(flow_uuid)
        if not uuid_value:
            return None
        version_value = _coerce_text(version) or None

        cached = self._flow_select_cache_get(uuid_value, version_value)
        if cached is not None:
            LOGGER.debug("crud.select_flow_cache_hit", flow_uuid=uuid_value, version=version_value or "*")
            return cached

        if version_value and not self._flow_select_cache_is_miss(uuid_value, version_value):
            raw = self._invoke(
                {
                    "operation": "select",
                    "table": "flows",
                    "id": uuid_value,
                    "version": version_value,
                }
            )
            dataset = self._extract_flow_dataset(raw)
            if dataset is not None:
                self._flow_select_cache_store(uuid_value, dataset, requested_version=version_value)
                return copy.deepcopy(dataset)
            self._flow_select_cache_mark_miss(uuid_value, version_value)

        if self._flow_select_cache_is_miss(uuid_value, None):
            return None

        raw = self._invoke({"operation": "select", "table": "flows", "id": uuid_value})
        dataset = self._extract_flow_dataset(raw)
        if dataset is None:
            self._flow_select_cache_mark_miss(uuid_value, None)
            return None

        self._flow_select_cache_store(uuid_value, dataset, requested_version=version_value)
        return copy.deepcopy(dataset)

    def select_flow_record(self, flow_uuid: str) -> dict[str, Any] | None:
        uuid_value = _coerce_text(flow_uuid)
        if not uuid_value:
            return None
        cached = self._FLOW_SELECT_RECORD_CACHE.get(uuid_value)
        if isinstance(cached, dict):
            LOGGER.debug("crud.select_flow_record_cache_hit", flow_uuid=uuid_value)
            return copy.deepcopy(cached)
        if uuid_value in self._FLOW_SELECT_RECORD_MISSES:
            return None

        raw = self._invoke({"operation": "select", "table": "flows", "id": uuid_value})
        if not isinstance(raw, dict):
            self._FLOW_SELECT_RECORD_MISSES.add(uuid_value)
            self._FLOW_SELECT_CACHE_DIRTY = True
            return None

        data = raw.get("data")
        if isinstance(data, list) and data:
            record = data[0]
            if isinstance(record, dict):
                self._FLOW_SELECT_RECORD_CACHE[uuid_value] = copy.deepcopy(record)
                self._FLOW_SELECT_CACHE_DIRTY = True
                dataset = None
                for key in ("json_ordered", "json"):
                    payload = record.get(key)
                    if isinstance(payload, Mapping) and isinstance(payload.get("flowDataSet"), Mapping):
                        dataset = payload.get("flowDataSet")
                        break
                if isinstance(dataset, Mapping):
                    self._flow_select_cache_store(uuid_value, dataset)
                return copy.deepcopy(record)

        if isinstance(raw.get("flowDataSet"), dict):
            dataset = raw.get("flowDataSet")
            self._flow_select_cache_store(uuid_value, dataset)
            record = {"json": {"flowDataSet": dataset}}
            self._FLOW_SELECT_RECORD_CACHE[uuid_value] = copy.deepcopy(record)
            self._FLOW_SELECT_CACHE_DIRTY = True
            return copy.deepcopy(record)

        self._FLOW_SELECT_RECORD_MISSES.add(uuid_value)
        self._FLOW_SELECT_CACHE_DIRTY = True
        return None

    def insert_process(self, dataset: Mapping[str, Any]) -> dict[str, Any]:
        root_key = "processDataSet" if "processDataSet" in dataset else None
        process_root = _resolve_dataset_root(dataset, root_key=root_key, dataset_kind="process")
        uuid_value = _require_uuid(
            _get_nested(process_root, ("processInformation", "dataSetInformation", "common:UUID")),
            "process",
        )
        json_payload = dataset if root_key else {"processDataSet": process_root}
        return self._invoke(
            {
                "operation": "insert",
                "table": "processes",
                "id": uuid_value,
                "jsonOrdered": json_payload,
            }
        )

    def select_process(self, process_uuid: str, *, version: str | None = None) -> dict[str, Any] | None:
        uuid_value = _coerce_text(process_uuid)
        if not uuid_value:
            return None
        version_value = _coerce_text(version) or None
        if version_value:
            raw = self._invoke(
                {
                    "operation": "select",
                    "table": "processes",
                    "id": uuid_value,
                    "version": version_value,
                }
            )
            dataset = self._extract_process_dataset(raw)
            if dataset is not None:
                return copy.deepcopy(dataset)
        raw = self._invoke({"operation": "select", "table": "processes", "id": uuid_value})
        dataset = self._extract_process_dataset(raw)
        if dataset is None:
            return None
        return copy.deepcopy(dataset)

    def insert_source(self, dataset: Mapping[str, Any]) -> dict[str, Any]:
        root_key = "sourceDataSet" if "sourceDataSet" in dataset else None
        source_root = _resolve_dataset_root(dataset, root_key=root_key, dataset_kind="source")
        uuid_value = _require_uuid(
            _get_nested(source_root, ("sourceInformation", "dataSetInformation", "common:UUID")),
            "source",
        )
        json_payload = dataset if root_key else {"sourceDataSet": source_root}
        return self._invoke(
            {
                "operation": "insert",
                "table": "sources",
                "id": uuid_value,
                "jsonOrdered": json_payload,
            }
        )

    def update_source(self, dataset: Mapping[str, Any]) -> dict[str, Any]:
        root_key = "sourceDataSet" if "sourceDataSet" in dataset else None
        source_root = _resolve_dataset_root(dataset, root_key=root_key, dataset_kind="source")
        uuid_value = _require_uuid(
            _get_nested(source_root, ("sourceInformation", "dataSetInformation", "common:UUID")),
            "source",
        )
        version_candidate = _coerce_text(
            _get_nested(
                source_root,
                ("administrativeInformation", "publicationAndOwnership", "common:dataSetVersion"),
            )
        )
        if not version_candidate:
            version_candidate = "01.01.000"
        json_payload = dataset if root_key else {"sourceDataSet": source_root}
        return self._invoke(
            {
                "operation": "update",
                "table": "sources",
                "id": uuid_value,
                "version": version_candidate,
                "jsonOrdered": json_payload,
            }
        )

    def select_source(self, source_uuid: str, *, version: str | None = None) -> dict[str, Any] | None:
        uuid_value = _coerce_text(source_uuid)
        if not uuid_value:
            return None
        version_value = _coerce_text(version) or None
        if version_value:
            raw = self._invoke(
                {
                    "operation": "select",
                    "table": "sources",
                    "id": uuid_value,
                    "version": version_value,
                }
            )
            dataset = self._extract_source_dataset(raw)
            if dataset is not None:
                return copy.deepcopy(dataset)
        raw = self._invoke({"operation": "select", "table": "sources", "id": uuid_value})
        dataset = self._extract_source_dataset(raw)
        if dataset is None:
            return None
        return copy.deepcopy(dataset)

    def update_process(self, dataset: Mapping[str, Any]) -> dict[str, Any]:
        root_key = "processDataSet" if "processDataSet" in dataset else None
        process_root = _resolve_dataset_root(dataset, root_key=root_key, dataset_kind="process")
        uuid_value = _require_uuid(
            _get_nested(process_root, ("processInformation", "dataSetInformation", "common:UUID")),
            "process",
        )
        version_candidate = _coerce_text(
            _get_nested(
                process_root,
                ("administrativeInformation", "publicationAndOwnership", "common:dataSetVersion"),
            )
        )
        if not version_candidate:
            version_candidate = "01.01.000"
        json_payload = dataset if root_key else {"processDataSet": process_root}
        return self._invoke(
            {
                "operation": "update",
                "table": "processes",
                "id": uuid_value,
                "version": version_candidate,
                "jsonOrdered": json_payload,
            }
        )

    def delete(self, table: str, record_id: str, version: str) -> dict[str, Any]:
        return self._invoke(
            {
                "operation": "delete",
                "table": table,
                "id": record_id,
                "version": version,
            }
        )

    def _invoke(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        LOGGER.debug("crud.invoke", table=payload.get("table"), operation=payload.get("operation"))
        raw = self._mcp.invoke_json_tool(self._server_name, DATABASE_TOOL_NAME, payload)
        if isinstance(raw, str):
            return json.loads(raw)
        if isinstance(raw, dict):
            return raw
        raise SpecCodingError("Unexpected payload returned from Database_CRUD_Tool")

    def close(self) -> None:
        self._flush_flow_select_cache()
        self._mcp.close()

    @classmethod
    def _flow_cache_key(cls, flow_uuid: str, version: str | None) -> str:
        version_key = version or "*"
        return f"{flow_uuid}@{version_key}"

    @staticmethod
    def _extract_flow_dataset(raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        if isinstance(raw.get("flowDataSet"), dict):
            return raw.get("flowDataSet")
        data = raw.get("data")
        if isinstance(data, list) and data:
            record = data[0] if isinstance(data[0], dict) else None
            if isinstance(record, dict):
                for key in ("json_ordered", "json"):
                    payload = record.get(key)
                    if isinstance(payload, dict) and isinstance(payload.get("flowDataSet"), dict):
                        return payload.get("flowDataSet")
        return None

    @staticmethod
    def _extract_process_dataset(raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        if isinstance(raw.get("processDataSet"), dict):
            return raw.get("processDataSet")
        data = raw.get("data")
        if isinstance(data, list) and data:
            record = data[0] if isinstance(data[0], dict) else None
            if isinstance(record, dict):
                for key in ("json_ordered", "json"):
                    payload = record.get(key)
                    if isinstance(payload, dict) and isinstance(payload.get("processDataSet"), dict):
                        return payload.get("processDataSet")
        return None

    @staticmethod
    def _extract_source_dataset(raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        if isinstance(raw.get("sourceDataSet"), dict):
            return raw.get("sourceDataSet")
        data = raw.get("data")
        if isinstance(data, list) and data:
            record = data[0] if isinstance(data[0], dict) else None
            if isinstance(record, dict):
                for key in ("json_ordered", "json"):
                    payload = record.get(key)
                    if isinstance(payload, dict) and isinstance(payload.get("sourceDataSet"), dict):
                        return payload.get("sourceDataSet")
        return None

    @staticmethod
    def _extract_flow_dataset_version(dataset: Mapping[str, Any]) -> str | None:
        admin = dataset.get("administrativeInformation")
        if not isinstance(admin, Mapping):
            return None
        pub = admin.get("publicationAndOwnership")
        if not isinstance(pub, Mapping):
            return None
        version = _coerce_text(pub.get("common:dataSetVersion"))
        return version or None

    @classmethod
    def _resolve_flow_select_cache_path(cls) -> Path | None:
        explicit = os.environ.get("TIANGONG_PFF_FLOW_CACHE_PATH")
        if explicit:
            return Path(explicit)

        run_id = _coerce_text(os.environ.get("TIANGONG_PFF_RUN_ID"))
        if run_id:
            return PROJECT_ROOT / "artifacts" / "process_from_flow" / run_id / "cache" / "flow_select_cache.json"

        state_path = _coerce_text(os.environ.get("TIANGONG_PFF_STATE_PATH"))
        if state_path:
            return Path(state_path).resolve().parent / "flow_select_cache.json"
        return None

    @classmethod
    def _ensure_flow_select_cache_loaded(cls) -> None:
        cache_path = cls._resolve_flow_select_cache_path()
        if cls._FLOW_SELECT_CACHE_LOADED and cls._FLOW_SELECT_CACHE_PATH == cache_path:
            return

        if cls._FLOW_SELECT_CACHE_LOADED and cls._FLOW_SELECT_CACHE_DIRTY:
            cls._flush_flow_select_cache()

        cls._FLOW_SELECT_CACHE_VALUES = {}
        cls._FLOW_SELECT_RECORD_CACHE = {}
        cls._FLOW_SELECT_CACHE_MISSES = set()
        cls._FLOW_SELECT_RECORD_MISSES = set()
        cls._FLOW_SELECT_CACHE_PATH = cache_path
        cls._FLOW_SELECT_CACHE_LOADED = True
        cls._FLOW_SELECT_CACHE_DIRTY = False

        if cache_path is None or not cache_path.exists():
            return
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("crud.select_flow_cache_load_failed", path=str(cache_path), error=str(exc))
            return

        values_raw = payload.get("values") if isinstance(payload, dict) else None
        if isinstance(values_raw, dict):
            for key, value in values_raw.items():
                if isinstance(key, str) and isinstance(value, dict):
                    cls._FLOW_SELECT_CACHE_VALUES[key] = value

        records_raw = payload.get("records") if isinstance(payload, dict) else None
        if isinstance(records_raw, dict):
            for key, value in records_raw.items():
                if isinstance(key, str) and isinstance(value, dict):
                    cls._FLOW_SELECT_RECORD_CACHE[key] = value

        misses_raw = payload.get("misses") if isinstance(payload, dict) else None
        if isinstance(misses_raw, list):
            cls._FLOW_SELECT_CACHE_MISSES = {key for key in misses_raw if isinstance(key, str) and key.strip()}

        record_misses_raw = payload.get("record_misses") if isinstance(payload, dict) else None
        if isinstance(record_misses_raw, list):
            cls._FLOW_SELECT_RECORD_MISSES = {key for key in record_misses_raw if isinstance(key, str) and key.strip()}

        LOGGER.debug(
            "crud.select_flow_cache_loaded",
            path=str(cache_path),
            cached=len(cls._FLOW_SELECT_CACHE_VALUES),
            misses=len(cls._FLOW_SELECT_CACHE_MISSES),
            records=len(cls._FLOW_SELECT_RECORD_CACHE),
            record_misses=len(cls._FLOW_SELECT_RECORD_MISSES),
        )

    @classmethod
    def _flush_flow_select_cache(cls) -> None:
        if not cls._FLOW_SELECT_CACHE_LOADED or not cls._FLOW_SELECT_CACHE_DIRTY:
            return
        cache_path = cls._FLOW_SELECT_CACHE_PATH
        if cache_path is None:
            cls._FLOW_SELECT_CACHE_DIRTY = False
            return

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "values": cls._FLOW_SELECT_CACHE_VALUES,
                "records": cls._FLOW_SELECT_RECORD_CACHE,
                "misses": sorted(cls._FLOW_SELECT_CACHE_MISSES),
                "record_misses": sorted(cls._FLOW_SELECT_RECORD_MISSES),
            }
            temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temp_path.replace(cache_path)
            cls._FLOW_SELECT_CACHE_DIRTY = False
            LOGGER.debug(
                "crud.select_flow_cache_flushed",
                path=str(cache_path),
                cached=len(cls._FLOW_SELECT_CACHE_VALUES),
                misses=len(cls._FLOW_SELECT_CACHE_MISSES),
                records=len(cls._FLOW_SELECT_RECORD_CACHE),
                record_misses=len(cls._FLOW_SELECT_RECORD_MISSES),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("crud.select_flow_cache_flush_failed", path=str(cache_path), error=str(exc))

    @classmethod
    def _flow_select_cache_get(cls, flow_uuid: str, version: str | None) -> dict[str, Any] | None:
        exact_key = cls._flow_cache_key(flow_uuid, version)
        cached = cls._FLOW_SELECT_CACHE_VALUES.get(exact_key)
        if isinstance(cached, dict):
            return copy.deepcopy(cached)

        if version:
            latest_key = cls._flow_cache_key(flow_uuid, None)
            latest = cls._FLOW_SELECT_CACHE_VALUES.get(latest_key)
            if isinstance(latest, dict):
                return copy.deepcopy(latest)
        return None

    @classmethod
    def _flow_select_cache_is_miss(cls, flow_uuid: str, version: str | None) -> bool:
        key = cls._flow_cache_key(flow_uuid, version)
        return key in cls._FLOW_SELECT_CACHE_MISSES

    @classmethod
    def _flow_select_cache_mark_miss(cls, flow_uuid: str, version: str | None) -> None:
        key = cls._flow_cache_key(flow_uuid, version)
        if key not in cls._FLOW_SELECT_CACHE_MISSES:
            cls._FLOW_SELECT_CACHE_MISSES.add(key)
            cls._FLOW_SELECT_CACHE_DIRTY = True

    @classmethod
    def _flow_select_cache_store(
        cls,
        flow_uuid: str,
        dataset: Mapping[str, Any],
        *,
        requested_version: str | None = None,
    ) -> None:
        dataset_copy = copy.deepcopy(dict(dataset))
        keys = {cls._flow_cache_key(flow_uuid, None)}
        if requested_version:
            keys.add(cls._flow_cache_key(flow_uuid, requested_version))
        actual_version = cls._extract_flow_dataset_version(dataset_copy)
        if actual_version:
            keys.add(cls._flow_cache_key(flow_uuid, actual_version))

        changed = False
        for key in keys:
            previous = cls._FLOW_SELECT_CACHE_VALUES.get(key)
            if previous != dataset_copy:
                cls._FLOW_SELECT_CACHE_VALUES[key] = copy.deepcopy(dataset_copy)
                changed = True
            if key in cls._FLOW_SELECT_CACHE_MISSES:
                cls._FLOW_SELECT_CACHE_MISSES.discard(key)
                changed = True
        if flow_uuid in cls._FLOW_SELECT_RECORD_MISSES:
            cls._FLOW_SELECT_RECORD_MISSES.discard(flow_uuid)
            changed = True
        if changed:
            cls._FLOW_SELECT_CACHE_DIRTY = True


class FlowPublisher:
    """Build and optionally publish flow datasets for unmatched or deficient exchanges."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        crud_client: DatabaseCrudClient | None = None,
        dry_run: bool = True,
        flow_property_registry: FlowPropertyRegistry | None = None,
        default_flow_property_uuid: str | None = None,
        flow_property_overrides: Mapping[tuple[str | None, str], FlowPropertyOverride] | None = None,
        llm: LanguageModelProtocol | None = None,
        flow_type_classifier: FlowTypeClassifier | None = None,
        product_category_selector: FlowProductCategorySelector | None = None,
        strict_flow_property_check: bool = False,
        flow_property_candidate_limit: int = 12,
    ) -> None:
        self._settings = settings or get_settings()
        self._crud: DatabaseCrudClient | None = crud_client
        self._dry_run = dry_run
        self._registry = flow_property_registry or get_default_registry()
        self._default_flow_property_uuid = self._resolve_default_property(default_flow_property_uuid)
        self._overrides = dict(flow_property_overrides or {})
        self._llm = llm
        self._flow_type_classifier = flow_type_classifier or FlowTypeClassifier(llm)
        self._product_category_selector = product_category_selector or FlowProductCategorySelector(llm)
        self._flow_creation = ProductFlowCreationService()
        self._flow_dedup: FlowDedupService | None = FlowDedupService(self._crud) if self._crud is not None else None
        self._prepared: list[FlowPublishPlan] = []
        self._strict_flow_property_check = bool(strict_flow_property_check)
        self._flow_property_candidate_limit = max(4, min(int(flow_property_candidate_limit), 20))
        self._flow_property_min_confidence = max(0.0, min(float(os.getenv("TIANGONG_PFF_FLOW_PROPERTY_MIN_CONFIDENCE", "0.6")), 1.0))
        self._flow_property_decisions: list[dict[str, Any]] = []
        self._held_flows: list[dict[str, Any]] = []

    @property
    def flow_property_decisions(self) -> list[dict[str, Any]]:
        """Structured summaries for every flow-property decision in the latest prepare run."""
        return copy.deepcopy(self._flow_property_decisions)

    @property
    def held_flows(self) -> list[dict[str, Any]]:
        """Flows withheld from automatic publication due to unresolved property selection."""
        return copy.deepcopy(self._held_flows)

    def _resolve_default_property(self, requested: str | None) -> str:
        if requested:
            try:
                self._registry.get(requested)
                return requested
            except KeyError as exc:  # pragma: no cover - configuration errors
                raise SpecCodingError(f"Unknown default flow property UUID: {requested}") from exc
        # Prefer Mass when available.
        try:
            return self._registry.get("93a60a56-a3c8-11da-a746-0800200b9a66").uuid
        except KeyError:
            descriptors = self._registry.list()
            if not descriptors:
                raise SpecCodingError("Flow property registry is empty")
            return descriptors[0].uuid

    def prepare_from_alignment(self, alignment: Iterable[Mapping[str, Any]]) -> list[FlowPublishPlan]:
        """Generate publication plans for unmatched exchanges and matched flows missing properties."""
        plans: list[FlowPublishPlan] = []
        self._flow_property_decisions = []
        self._held_flows = []
        for entry in alignment:
            process_name = _coerce_text(entry.get("process_name")) or "Unknown process"
            origin = entry.get("origin_exchanges") or {}
            for exchanges in origin.values():
                exchanges_iter = [exchanges] if isinstance(exchanges, Mapping) else list(exchanges or [])
                for exchange in exchanges_iter:
                    if not isinstance(exchange, Mapping):
                        continue
                    exchange_name = _coerce_text(exchange.get("exchangeName")) or "Unnamed exchange"
                    candidate = self._extract_selected_candidate(exchange)
                    decision = self._resolve_flow_property_decision(process_name, exchange_name, exchange, candidate)
                    if decision.selected_uuid is None:
                        final_action = "fail" if self._strict_flow_property_check else "hold"
                        self._record_flow_property_decision(
                            process_name=process_name,
                            exchange_name=exchange_name,
                            exchange=exchange,
                            decision=decision,
                            final_action=final_action,
                        )
                        hold_entry = {
                            "process_name": process_name,
                            "exchange_name": exchange_name,
                            "unit": _resolve_unit(exchange) or None,
                            "reason": decision.reason or "flow_property_unresolved",
                            "decision_mode": decision.decision_mode,
                            "candidate_uuids": list(decision.candidate_uuids),
                            "validation_errors": list(decision.validation_errors),
                        }
                        self._held_flows.append(hold_entry)
                        LOGGER.warning("flow_publish.flow_property_hold", **hold_entry)
                        if self._strict_flow_property_check:
                            raise SpecCodingError(
                                f"Flow property selection failed for '{exchange_name}' in '{process_name}': "
                                f"{decision.reason or 'manual review required'}"
                            )
                        continue
                    property_uuid = decision.selected_uuid
                    mean_value = decision.mean_value
                    ref = exchange.get("referenceToFlowDataSet")
                    if not isinstance(ref, Mapping):
                        ref = {}
                    if self._is_placeholder_reference(ref):
                        plan = self._build_plan(
                            exchange,
                            process_name,
                            property_uuid,
                            mean_value,
                            candidate=None,
                            mode="insert",
                            # Keep placeholder UUID stable across retries so we do not
                            # mint a fresh flow UUID for the same unresolved exchange.
                            existing_ref=ref,
                        )
                        final_action = "plan_ready" if plan is not None else "skip_build_failed"
                    else:
                        if candidate is None:
                            self._record_flow_property_decision(
                                process_name=process_name,
                                exchange_name=exchange_name,
                                exchange=exchange,
                                decision=decision,
                                final_action="skip_missing_candidate",
                            )
                            continue
                        if self._candidate_has_required_property(candidate, property_uuid):
                            self._record_flow_property_decision(
                                process_name=process_name,
                                exchange_name=exchange_name,
                                exchange=exchange,
                                decision=decision,
                                final_action="skip_candidate_already_has_property",
                            )
                            continue
                        plan = self._build_plan(
                            exchange,
                            process_name,
                            property_uuid,
                            mean_value,
                            candidate=candidate,
                            mode="update",
                            existing_ref=ref,
                        )
                        final_action = "plan_ready" if plan is not None else "skip_build_failed"
                    if plan is not None:
                        plans.append(plan)
                    self._record_flow_property_decision(
                        process_name=process_name,
                        exchange_name=exchange_name,
                        exchange=exchange,
                        decision=decision,
                        final_action=final_action,
                    )
        self._prepared = plans
        LOGGER.info("flow_publish.plans_ready", count=len(plans), held_count=len(self._held_flows))
        return plans

    def set_prepared_plans(self, plans: Iterable[FlowPublishPlan]) -> list[FlowPublishPlan]:
        """Replace the in-memory prepared plans with externally loaded plans."""
        prepared = list(plans)
        self._prepared = prepared
        return prepared

    def publish_prepared(self, plans: Iterable[FlowPublishPlan]) -> list[dict[str, Any]]:
        """Publish an externally provided prepared-plan list."""
        self.set_prepared_plans(plans)
        return self.publish()

    def publish(self) -> list[dict[str, Any]]:
        """Execute inserts or updates for the prepared plans."""
        self._ensure_publish_services()
        if self._crud is None or self._flow_dedup is None:
            raise SpecCodingError("FlowPublisher publish services are unavailable.")
        results: list[dict[str, Any]] = []
        for plan in self._prepared:
            if self._dry_run:
                LOGGER.info(
                    "flow_publish.dry_run",
                    exchange=plan.exchange_name,
                    process=plan.process_name,
                    uuid=plan.uuid,
                    mode=plan.mode,
                )
                continue
            payload = {"flowDataSet": plan.dataset}
            version = (
                _coerce_text(
                    _get_nested(
                        plan.dataset,
                        ("administrativeInformation", "publicationAndOwnership", "common:dataSetVersion"),
                    )
                )
                or "01.01.000"
            )
            decision = self._flow_dedup.decide(
                flow_uuid=plan.uuid,
                version=version,
                preferred_action=plan.mode if plan.mode in {"insert", "update"} else "auto",
            )
            if decision.action == "reuse":
                LOGGER.info(
                    "flow_publish.reuse_existing",
                    exchange=plan.exchange_name,
                    process=plan.process_name,
                    uuid=plan.uuid,
                    reason=decision.reason,
                )
                continue

            actions: list[str] = [decision.action]
            if decision.action == "update":
                actions.append("insert")
            elif decision.action == "insert":
                actions.append("update")

            result: dict[str, Any] | None = None
            last_error: Exception | None = None
            for action in actions:
                try:
                    if action == "update":
                        result = self._crud.update_flow(payload)
                    else:
                        result = self._crud.insert_flow(payload)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    LOGGER.warning(
                        "flow_publish.action_failed",
                        exchange=plan.exchange_name,
                        process=plan.process_name,
                        uuid=plan.uuid,
                        action=action,
                        error=str(exc),
                    )

            if result is None:
                existing = None
                try:
                    existing = self._crud.select_flow(plan.uuid, version=version) or self._crud.select_flow(plan.uuid)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning(
                        "flow_publish.reuse_check_failed",
                        exchange=plan.exchange_name,
                        process=plan.process_name,
                        uuid=plan.uuid,
                        error=str(exc),
                    )
                if isinstance(existing, Mapping):
                    LOGGER.warning(
                        "flow_publish.reuse_existing_after_error",
                        exchange=plan.exchange_name,
                        process=plan.process_name,
                        uuid=plan.uuid,
                    )
                    results.append({"id": plan.uuid, "action": "reuse"})
                    continue
                if isinstance(last_error, Exception):
                    raise SpecCodingError(
                        f"Failed to publish flow '{plan.exchange_name}' ({plan.uuid}) after insert/update attempts."
                    ) from last_error
                raise SpecCodingError(f"Failed to publish flow '{plan.exchange_name}' ({plan.uuid}).")
            results.append(result)
        return results

    def close(self) -> None:
        if self._crud is not None:
            self._crud.close()

    def _ensure_publish_services(self) -> None:
        if self._crud is None:
            self._crud = DatabaseCrudClient(self._settings)
        if self._flow_dedup is None:
            self._flow_dedup = FlowDedupService(self._crud)

    @staticmethod
    def _is_placeholder_reference(reference: Mapping[str, Any]) -> bool:
        return bool(reference.get("unmatched:placeholder"))

    @staticmethod
    def _extract_selected_candidate(exchange: Mapping[str, Any]) -> Mapping[str, Any] | None:
        detail = exchange.get("matchingDetail")
        if not isinstance(detail, Mapping):
            return None
        candidate = detail.get("selectedCandidate")
        if isinstance(candidate, Mapping):
            return candidate
        return None

    def _resolve_flow_property(
        self,
        process_name: str,
        exchange_name: str,
        exchange: Mapping[str, Any],
        candidate: Mapping[str, Any] | None,
    ) -> tuple[str | None, str | None]:
        decision = self._resolve_flow_property_decision(process_name, exchange_name, exchange, candidate)
        return decision.selected_uuid, decision.mean_value

    def _resolve_flow_property_decision(
        self,
        process_name: str,
        exchange_name: str,
        exchange: Mapping[str, Any],
        candidate: Mapping[str, Any] | None,
    ) -> PropertyDecision:
        override = self._overrides.get((process_name, exchange_name)) or self._overrides.get((None, exchange_name))
        if override:
            try:
                descriptor = self._registry.get(override.flow_property_uuid)
            except KeyError as exc:
                raise SpecCodingError(f"Unknown flow property in override: {override.flow_property_uuid}") from exc
            validation_passed, validation_errors = self._validate_property_selection(
                exchange=exchange,
                selected_uuid=descriptor.uuid,
                candidate_uuids=(descriptor.uuid,),
            )
            return self._apply_confidence_gate(PropertyDecision(
                selected_uuid=descriptor.uuid,
                selected_name=descriptor.name,
                selected_version=self._registry.get_version(descriptor.uuid),
                confidence=1.0,
                reason="Manual flow_property_overrides entry matched.",
                candidate_uuids=(descriptor.uuid,),
                decision_mode="override",
                validation_passed=validation_passed,
                validation_errors=validation_errors,
                mean_value=override.mean_value,
            ))

        candidate_property = _coerce_text(candidate.get("flow_properties")) if candidate else ""
        if candidate_property:
            descriptor = self._registry.fuzzy_match(candidate_property)
            if descriptor:
                validation_passed, validation_errors = self._validate_property_selection(
                    exchange=exchange,
                    selected_uuid=descriptor.uuid,
                    candidate_uuids=(descriptor.uuid,),
                )
                return self._apply_confidence_gate(PropertyDecision(
                    selected_uuid=descriptor.uuid if validation_passed else None,
                    selected_name=descriptor.name,
                    selected_version=self._registry.get_version(descriptor.uuid),
                    confidence=0.95 if validation_passed else 0.2,
                    reason=(
                        "Matched selectedCandidate.flow_properties via fuzzy registry lookup."
                        if validation_passed
                        else "Candidate flow property matched text but failed validation."
                    ),
                    candidate_uuids=(descriptor.uuid,),
                    decision_mode="candidate_match",
                    validation_passed=validation_passed,
                    validation_errors=validation_errors,
                    is_uncertain=not validation_passed,
                ))

        hints = _parse_flowsearch_hints(_extract_general_comment(exchange))
        if self._is_ambiguous_unit_selection(exchange):
            return PropertyDecision(
                selected_uuid=None,
                confidence=0.0,
                reason="Reference output unit is item/unit-like without explicit physical dimension evidence; manual review required.",
                candidate_uuids=(),
                decision_mode="manual_required",
                validation_passed=False,
                validation_errors=("unit_dimension_ambiguous",),
                is_uncertain=True,
            )
        ranked_candidates = self._rank_property_candidates(exchange=exchange, hints=hints, candidate=candidate)
        candidate_descriptors = [descriptor for descriptor, _score, _why in ranked_candidates]
        candidate_uuids = tuple(descriptor.uuid for descriptor in candidate_descriptors)
        if not candidate_descriptors:
            return PropertyDecision(
                selected_uuid=None,
                confidence=0.0,
                reason="No compatible flow property candidates could be derived from local resources.",
                candidate_uuids=(),
                decision_mode="manual_required",
                validation_passed=False,
                validation_errors=("candidate_generation_empty",),
                is_uncertain=True,
            )

        top_descriptor, top_score, top_reasons = ranked_candidates[0]
        second_score = ranked_candidates[1][1] if len(ranked_candidates) > 1 else None
        if len(ranked_candidates) == 1:
            return self._apply_confidence_gate(self._decision_from_descriptor(
                exchange=exchange,
                descriptor=top_descriptor,
                candidate_uuids=candidate_uuids,
                confidence=0.92,
                reason=f"Only one unit-compatible candidate available ({top_descriptor.name}).",
                decision_mode="rule_fallback",
            ))

        if self._is_property_rule_strong_match(top_descriptor, top_score, second_score, exchange):
            return self._apply_confidence_gate(self._decision_from_descriptor(
                exchange=exchange,
                descriptor=top_descriptor,
                candidate_uuids=candidate_uuids,
                confidence=min(0.96, 0.65 + max(0.0, top_score) / 20.0),
                reason="Rule-based selection from local candidate ranking. " + "; ".join(top_reasons[:3]),
                decision_mode="rule_fallback",
            ))

        if self._llm is not None:
            llm_payload = self._select_flow_property_with_llm(
                process_name=process_name,
                exchange_name=exchange_name,
                exchange=exchange,
                hints=hints,
                candidates=ranked_candidates,
                retry_errors=(),
            )
            if llm_payload is not None:
                decision = self._decision_from_llm_payload(
                    exchange=exchange,
                    payload=llm_payload,
                    candidates=ranked_candidates,
                    decision_mode="llm_primary",
                )
                if not decision.validation_passed:
                    retry_payload = self._select_flow_property_with_llm(
                        process_name=process_name,
                        exchange_name=exchange_name,
                        exchange=exchange,
                        hints=hints,
                        candidates=ranked_candidates,
                        retry_errors=decision.validation_errors,
                    )
                    if retry_payload is not None:
                        retry_decision = self._decision_from_llm_payload(
                            exchange=exchange,
                            payload=retry_payload,
                            candidates=ranked_candidates,
                            decision_mode="llm_retry",
                            retry_count=1,
                        )
                        if retry_decision.validation_passed:
                            return retry_decision
                        decision = retry_decision
                if decision.validation_passed:
                    return self._apply_confidence_gate(decision)

        if top_score > 0:
            return PropertyDecision(
                selected_uuid=None,
                confidence=min(0.8, 0.5 + max(0.0, top_score) / 30.0),
                reason="Candidates exist but auto-selection is disallowed without high-confidence deterministic/validated decision.",
                candidate_uuids=candidate_uuids,
                decision_mode="manual_required",
                validation_passed=False,
                validation_errors=("auto_selection_blocked",),
                is_uncertain=True,
            )

        return PropertyDecision(
            selected_uuid=None,
            confidence=0.0,
            reason="Candidate set exists but no reliable automatic property decision could be made.",
            candidate_uuids=candidate_uuids,
            decision_mode="manual_required",
            validation_passed=False,
            validation_errors=("selection_uncertain",),
            is_uncertain=True,
        )

    def _is_ambiguous_unit_selection(self, exchange: Mapping[str, Any]) -> bool:
        unit_raw = _resolve_unit(exchange)
        token = _normalize_unit_token(unit_raw)
        dim = _unit_dimension(unit_raw)
        if token in {"unit", "item", "items"}:
            return True
        return dim == "items"

    def _apply_confidence_gate(self, decision: PropertyDecision) -> PropertyDecision:
        if decision.selected_uuid is None:
            return decision
        if decision.confidence >= self._flow_property_min_confidence:
            return decision
        errors = tuple(list(decision.validation_errors) + ["confidence_below_threshold"])
        return PropertyDecision(
            selected_uuid=None,
            selected_name=decision.selected_name,
            selected_version=decision.selected_version,
            confidence=decision.confidence,
            reason=(
                f"Automatic decision confidence {decision.confidence:.3f} below threshold "
                f"{self._flow_property_min_confidence:.3f}; manual review required."
            ),
            candidate_uuids=decision.candidate_uuids,
            decision_mode="manual_required",
            validation_passed=False,
            validation_errors=errors,
            mean_value=decision.mean_value,
            retry_count=decision.retry_count,
            is_uncertain=True,
        )

    def _decision_from_descriptor(
        self,
        *,
        exchange: Mapping[str, Any],
        descriptor: Any,
        candidate_uuids: tuple[str, ...],
        confidence: float,
        reason: str,
        decision_mode: str,
    ) -> PropertyDecision:
        validation_passed, validation_errors = self._validate_property_selection(
            exchange=exchange,
            selected_uuid=descriptor.uuid,
            candidate_uuids=candidate_uuids,
        )
        return PropertyDecision(
            selected_uuid=descriptor.uuid if validation_passed else None,
            selected_name=descriptor.name,
            selected_version=self._registry.get_version(descriptor.uuid),
            confidence=max(0.0, min(float(confidence), 1.0)),
            reason=reason,
            candidate_uuids=candidate_uuids,
            decision_mode=decision_mode,
            validation_passed=validation_passed,
            validation_errors=validation_errors,
            is_uncertain=not validation_passed,
        )

    def _decision_from_llm_payload(
        self,
        *,
        exchange: Mapping[str, Any],
        payload: Mapping[str, Any],
        candidates: Sequence[tuple[Any, float, list[str]]],
        decision_mode: str,
        retry_count: int = 0,
    ) -> PropertyDecision:
        candidate_map = {descriptor.uuid.lower(): descriptor for descriptor, _score, _why in candidates}
        candidate_uuids = tuple(descriptor.uuid for descriptor, _score, _why in candidates)
        selected_uuid = _coerce_text(payload.get("selected_uuid")) or None
        confidence_raw = payload.get("confidence")
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(confidence, 1.0))
        reason = _normalize_text(payload.get("reason")) or "LLM property selection"
        is_uncertain = bool(payload.get("is_uncertain"))

        selected_descriptor = candidate_map.get(selected_uuid.lower()) if selected_uuid else None
        validation_passed, validation_errors = self._validate_property_selection(
            exchange=exchange,
            selected_uuid=selected_uuid,
            candidate_uuids=candidate_uuids,
        )
        if selected_descriptor is None:
            validation_passed = False
            validation_errors = tuple(list(validation_errors) + ["selected_uuid_not_in_candidates"])
            selected_uuid = None
        return PropertyDecision(
            selected_uuid=selected_uuid if validation_passed else None,
            selected_name=selected_descriptor.name if selected_descriptor else None,
            selected_version=self._registry.get_version(selected_descriptor.uuid) if selected_descriptor else None,
            confidence=confidence,
            reason=reason,
            candidate_uuids=candidate_uuids,
            decision_mode=decision_mode,
            validation_passed=validation_passed,
            validation_errors=validation_errors,
            retry_count=retry_count,
            is_uncertain=is_uncertain or not validation_passed,
        )

    def _rank_property_candidates(
        self,
        *,
        exchange: Mapping[str, Any],
        hints: Mapping[str, list[str] | str],
        candidate: Mapping[str, Any] | None,
    ) -> list[tuple[Any, float, list[str]]]:
        unit = _resolve_unit(exchange)
        unit_candidates = self._registry.search_by_unit(unit) if unit else []
        pool = unit_candidates or list(self._registry.list())
        if not pool:
            return []
        text_fragments = self._property_semantic_fragments(exchange=exchange, hints=hints, candidate=candidate)
        scored: list[tuple[Any, float, list[str]]] = []
        for descriptor in pool:
            score, reasons = self._score_property_candidate(
                descriptor=descriptor,
                unit=unit,
                text_fragments=text_fragments,
            )
            scored.append((descriptor, score, reasons))
        scored.sort(key=lambda item: (-item[1], item[0].name.lower(), item[0].uuid.lower()))
        limit = self._flow_property_candidate_limit
        if not unit_candidates:
            limit = min(limit, 8)
        return scored[:limit]

    def _property_semantic_fragments(
        self,
        *,
        exchange: Mapping[str, Any],
        hints: Mapping[str, list[str] | str],
        candidate: Mapping[str, Any] | None,
    ) -> list[str]:
        fragments: list[str] = [
            _coerce_text(exchange.get("exchangeName")),
            _coerce_text(exchange.get("exchangeDirection")),
            _coerce_text(exchange.get("location")),
            _extract_general_comment(exchange),
        ]
        if candidate:
            fragments.extend(
                [
                    _coerce_text(candidate.get("base_name")),
                    _coerce_text(candidate.get("general_comment")),
                    _coerce_text(candidate.get("flow_properties")),
                ]
            )
        normalized_hints = _normalize_hint_values(hints)
        for values in normalized_hints.values():
            fragments.extend(values)
        return [fragment for fragment in fragments if fragment]

    def _score_property_candidate(
        self,
        *,
        descriptor: Any,
        unit: str,
        text_fragments: Sequence[str],
    ) -> tuple[float, list[str]]:
        score = 0.0
        reasons: list[str] = []
        text = " ".join(text_fragments).lower()
        name = descriptor.name.lower()
        unit_dimension = _unit_dimension(unit)
        property_dimension = self._descriptor_dimension(descriptor)

        if unit_dimension and property_dimension:
            if unit_dimension == property_dimension:
                score += 4.0
                reasons.append(f"dimension_match:{unit_dimension}")
            else:
                score -= 8.0
                reasons.append(f"dimension_mismatch:{unit_dimension}->{property_dimension}")

        if name in text:
            score += 6.0
            reasons.append("name_substring_match")

        if "calorific value" in name and unit_dimension == "energy":
            score += 7.0
            reasons.append("energy_calorific_match")
        if "net calorific value" in name and "electric" in text:
            score += 2.5
            reasons.append("electricity_prefers_net_calorific")
        if "gross calorific value" in name and "electric" in text:
            score += 1.5
            reasons.append("electricity_allows_gross_calorific")

        if name == "mass" and unit_dimension == "mass":
            score += 14.0
            reasons.append("generic_mass_for_mass_unit")
        if name == "volume" and unit_dimension == "volume":
            score += 14.0
            reasons.append("generic_volume_for_volume_unit")
        if name == "area" and unit_dimension == "area":
            score += 14.0
            reasons.append("generic_area_for_area_unit")
        if name == "length" and unit_dimension == "length":
            score += 14.0
            reasons.append("generic_length_for_length_unit")

        if "content" in name:
            if any(token in text for token in ("content", "concentration", "fraction", "grade", "composition", "moisture")):
                score += 2.0
                reasons.append("content_signal_present")
            else:
                score -= 7.0
                reasons.append("content_property_without_content_signal")

        # Light lexical overlap for non-generic properties.
        for token in (part for part in re.split(r"[^a-z0-9]+", name) if len(part) >= 4):
            if token in {"value", "gross", "net", "content"}:
                continue
            if token in text:
                score += 1.25
                reasons.append(f"token:{token}")

        return score, reasons

    def _descriptor_dimension(self, descriptor: Any) -> str | None:
        group_name = _coerce_text(getattr(getattr(descriptor, "unit_group", None), "name", ""))
        name_lower = group_name.lower()
        if "energy" in name_lower:
            return "energy"
        if "mass" in name_lower:
            return "mass"
        if "volume" in name_lower:
            return "volume"
        if "area" in name_lower:
            return "area"
        if "length" in name_lower:
            return "length"
        if "item" in name_lower:
            return "items"
        ref_unit = getattr(getattr(descriptor, "unit_group", None), "reference_unit", None)
        ref_unit_name = _coerce_text(getattr(ref_unit, "name", ""))
        return _unit_dimension(ref_unit_name)

    def _is_property_rule_strong_match(
        self,
        descriptor: Any,
        top_score: float,
        second_score: float | None,
        exchange: Mapping[str, Any],
    ) -> bool:
        unit_dimension = _unit_dimension(_resolve_unit(exchange))
        name = descriptor.name.lower()
        if second_score is None:
            return top_score > 0
        margin = top_score - second_score
        if unit_dimension == "mass" and name == "mass" and margin >= 6.0:
            return True
        if unit_dimension == "volume" and name == "volume" and margin >= 4.0:
            return True
        if unit_dimension == "energy" and "calorific value" in name and margin >= 3.0:
            return True
        return top_score >= 14.0 and margin >= 5.0

    def _select_flow_property_with_llm(
        self,
        *,
        process_name: str,
        exchange_name: str,
        exchange: Mapping[str, Any],
        hints: Mapping[str, list[str] | str],
        candidates: Sequence[tuple[Any, float, list[str]]],
        retry_errors: Sequence[str],
    ) -> Mapping[str, Any] | None:
        if self._llm is None or not candidates:
            return None
        candidate_payload = []
        for descriptor, score, reasons in candidates:
            candidate_payload.append(
                {
                    "uuid": descriptor.uuid,
                    "name": descriptor.name,
                    "version": self._registry.get_version(descriptor.uuid),
                    "unit_group": _coerce_text(descriptor.unit_group.name),
                    "reference_unit": _coerce_text(getattr(descriptor.unit_group.reference_unit, "name", "")),
                    "classification": list(descriptor.classification),
                    "heuristic_score": round(score, 3),
                    "heuristic_reasons": reasons[:4],
                }
            )
        context = {
            "process_name": process_name,
            "exchange": {
                "name": exchange_name,
                "aliases": list((_normalize_hint_values(hints).get("en_synonyms") or [])[:6]),
                "comment": _extract_general_comment(exchange),
                "unit": _resolve_unit(exchange),
                "direction": _coerce_text(exchange.get("exchangeDirection")),
                "location": _coerce_text(exchange.get("location")),
                "flow_type": _coerce_text(exchange.get("flowType")),
                "io_kind_tag": _coerce_text(exchange.get("io_kind_tag") or exchange.get("material_role")),
            },
            "candidates": candidate_payload,
        }
        if retry_errors:
            context["validation_feedback"] = list(retry_errors)
        try:
            response = self._llm.invoke(
                {
                    "prompt": FLOW_PROPERTY_SELECTOR_PROMPT,
                    "context": context,
                    "response_format": {"type": "json_object"},
                }
            )
            return _ensure_mapping_response(response)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning(
                "flow_publish.flow_property_llm_failed",
                process=process_name,
                exchange=exchange_name,
                error=str(exc),
                retry=bool(retry_errors),
            )
            return None

    def _validate_property_selection(
        self,
        *,
        exchange: Mapping[str, Any],
        selected_uuid: str | None,
        candidate_uuids: Sequence[str],
    ) -> tuple[bool, tuple[str, ...]]:
        errors: list[str] = []
        if not selected_uuid:
            errors.append("selected_uuid_missing")
            return False, tuple(errors)

        candidate_set = {uuid_value.lower() for uuid_value in candidate_uuids if _coerce_text(uuid_value)}
        if candidate_set and selected_uuid.lower() not in candidate_set:
            errors.append("selected_uuid_not_in_candidates")

        try:
            descriptor = self._registry.get(selected_uuid)
        except KeyError:
            errors.append("selected_uuid_unknown")
            return False, tuple(errors)

        if not _coerce_text(descriptor.uuid):
            errors.append("descriptor_uuid_missing")
        if not _coerce_text(descriptor.name):
            errors.append("descriptor_name_missing")
        if not _coerce_text(self._registry.get_version(descriptor.uuid)):
            errors.append("descriptor_version_missing")

        unit = _resolve_unit(exchange)
        if unit:
            compatible = {uuid_value.lower() for uuid_value in self._registry.compatible_property_uuids_for_unit(unit)}
            if compatible and descriptor.uuid.lower() not in compatible:
                errors.append("unit_incompatible_with_selected_property")

        errors.extend(self._semantic_property_conflicts(exchange=exchange, descriptor=descriptor))
        return (not errors), tuple(errors)

    def _semantic_property_conflicts(self, *, exchange: Mapping[str, Any], descriptor: Any) -> list[str]:
        errors: list[str] = []
        unit_dimension = _unit_dimension(_resolve_unit(exchange))
        name = descriptor.name.lower()
        text = " ".join(
            [
                _coerce_text(exchange.get("exchangeName")),
                _extract_general_comment(exchange),
            ]
        ).lower()

        if unit_dimension == "energy" and ("mass" == name or name.startswith("mass ")):
            errors.append("semantic_conflict:energy_unit_with_mass_property")
        if unit_dimension == "mass" and "calorific value" in name:
            errors.append("semantic_conflict:mass_unit_with_calorific_property")
        if "content" in name and not any(
            token in text for token in ("content", "concentration", "fraction", "grade", "composition", "moisture")
        ):
            errors.append("semantic_conflict:content_property_without_content_signals")
        return errors

    def _record_flow_property_decision(
        self,
        *,
        process_name: str,
        exchange_name: str,
        exchange: Mapping[str, Any],
        decision: PropertyDecision,
        final_action: str,
    ) -> None:
        unit = _resolve_unit(exchange)
        summary: dict[str, Any] = {
            "process_name": process_name,
            "exchange_name": exchange_name,
            "unit": unit or None,
            "direction": _coerce_text(exchange.get("exchangeDirection")) or None,
            "selected_uuid": decision.selected_uuid,
            "selected_name": decision.selected_name,
            "selected_version": decision.selected_version,
            "confidence": round(max(0.0, min(float(decision.confidence), 1.0)), 4),
            "reason": decision.reason or None,
            "candidate_uuids": list(decision.candidate_uuids),
            "decision_mode": decision.decision_mode,
            "validation_passed": bool(decision.validation_passed),
            "validation_errors": list(decision.validation_errors),
            "retry_count": int(decision.retry_count or 0),
            "is_uncertain": bool(decision.is_uncertain),
            "final_action": final_action,
        }
        if decision.mean_value is not None:
            summary["mean_value"] = decision.mean_value
        self._flow_property_decisions.append(summary)
        LOGGER.info("flow_publish.property_decision", **summary)

    def _candidate_has_required_property(
        self,
        candidate: Mapping[str, Any],
        expected_uuid: str,
    ) -> bool:
        candidate_property = _coerce_text(candidate.get("flow_properties"))
        if not candidate_property:
            return False
        descriptor = self._registry.fuzzy_match(candidate_property)
        if descriptor is None:
            return False
        return descriptor.uuid.lower() == expected_uuid.lower()

    def _build_plan(
        self,
        exchange: Mapping[str, Any],
        process_name: str,
        property_uuid: str,
        mean_value: str | None,
        *,
        candidate: Mapping[str, Any] | None,
        mode: str,
        existing_ref: Mapping[str, Any] | None,
    ) -> Optional[FlowPublishPlan]:
        dataset = self._compose_flow_dataset(
            exchange,
            process_name,
            property_uuid,
            mean_value,
            candidate=candidate,
            mode=mode,
            existing_ref=existing_ref,
        )
        if dataset is None:
            return None
        flow_dataset, exchange_ref = dataset
        return FlowPublishPlan(
            uuid=exchange_ref.get("@refObjectId", ""),
            exchange_name=_coerce_text(exchange.get("exchangeName")) or "Unnamed exchange",
            process_name=process_name,
            dataset=flow_dataset,
            exchange_ref=exchange_ref,
            mode=mode,
            flow_property_uuid=property_uuid,
        )

    def _compose_flow_dataset(
        self,
        exchange: Mapping[str, Any],
        process_name: str,
        property_uuid: str,
        mean_value: str | None,
        *,
        candidate: Mapping[str, Any] | None,
        mode: str,
        existing_ref: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        exchange_name = _coerce_text(exchange.get("exchangeName")) or "Unnamed exchange"
        comment = _extract_general_comment(exchange)
        hints = _parse_flowsearch_hints(comment)
        flow_type = _infer_flow_type(exchange, hints, classifier=self._flow_type_classifier)
        if flow_type == "Elementary flow":
            LOGGER.warning(
                "flow_publish.skip_elementary",
                exchange=exchange_name,
                process=process_name,
                reason="Elementary flows must reuse existing records.",
            )
            return None

        text_fields = self._resolve_text_fields(
            exchange=exchange,
            hints=hints,
            candidate=candidate,
            exchange_name=exchange_name,
            process_name=process_name,
            flow_type=flow_type,
        )
        uuid_value = self._resolve_flow_uuid(candidate, existing_ref)
        version = self._resolve_flow_version(candidate, existing_ref, mode)
        en_name, zh_name = self._resolve_language_pairs(candidate, hints, exchange_name)
        en_name = text_fields.get("base_name_en") or en_name
        zh_name = text_fields.get("base_name_zh") or zh_name
        classification = self._resolve_classification(flow_type, hints, candidate, exchange)
        comment_entries = self._resolve_comments(
            comment,
            candidate,
            exchange_name,
            comment_en=text_fields.get("comment_en"),
            comment_zh=text_fields.get("comment_zh"),
        )
        name_block = self._build_name_block(
            candidate,
            hints,
            en_name,
            zh_name,
            treatment_en=text_fields.get("treatment_en"),
            treatment_zh=text_fields.get("treatment_zh"),
            mix_en=text_fields.get("mix_en"),
            mix_zh=text_fields.get("mix_zh"),
        )
        synonyms_block = self._build_synonyms(
            hints,
            en_name,
            zh_name,
            synonyms_en=text_fields.get("synonyms_en"),
            synonyms_zh=text_fields.get("synonyms_zh"),
        )
        class_entries = self._extract_classification_entries(classification)
        if not class_entries:
            class_entries = self._extract_classification_entries(_default_product_classification())

        flow_property = self._registry.get(property_uuid)
        property_version = self._registry.get_version(flow_property.uuid)
        request = ProductFlowCreateRequest(
            class_id=str(class_entries[-1].get("@classId") or ""),
            classification=class_entries,
            base_name_en=en_name,
            base_name_zh=zh_name,
            treatment_en=self._extract_language_text(name_block.get("treatmentStandardsRoutes"), "en") or en_name,
            treatment_zh=self._extract_language_text(name_block.get("treatmentStandardsRoutes"), "zh") or None,
            mix_en=self._extract_language_text(name_block.get("mixAndLocationTypes"), "en") or en_name,
            mix_zh=self._extract_language_text(name_block.get("mixAndLocationTypes"), "zh") or None,
            comment_en=self._extract_language_text(comment_entries, "en") or exchange_name,
            comment_zh=self._extract_language_text(comment_entries, "zh") or None,
            synonyms_en=self._split_synonyms(self._extract_language_text(synonyms_block, "en")),
            synonyms_zh=self._split_synonyms(self._extract_language_text(synonyms_block, "zh")),
            flow_type=flow_type,
            flow_uuid=uuid_value,
            version=version,
            mean_value=mean_value or "1.0",
            flow_property_uuid=flow_property.uuid,
            flow_property_version=property_version,
            flow_property_name_en=flow_property.name,
        )

        try:
            built = self._flow_creation.build(request, allow_validation_fallback=True)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning(
                "flow_publish.flow_validation_failed",
                exchange=exchange_name,
                process=process_name,
                error=str(exc),
            )
            return None

        dataset = dict(built.dataset)
        uuid_value = built.flow_uuid
        version = built.version
        publication = _get_nested(dataset, ("administrativeInformation", "publicationAndOwnership"))
        if isinstance(publication, Mapping):
            publication["common:permanentDataSetURI"] = build_portal_uri("flow", uuid_value, version)

        uri = build_portal_uri("flow", uuid_value, version)
        short_desc_zh = (
            self._extract_language_text(name_block.get("baseName"), "zh")
            or _normalize_text(text_fields.get("base_name_zh"))
            or _normalize_text(zh_name)
        )
        exchange_ref = {
            "@type": "flow data set",
            "@uri": uri,
            "@refObjectId": uuid_value,
            "@version": version,
            "common:shortDescription": _bilingual_language_entries(exchange_name, short_desc_zh),
        }
        return dataset, exchange_ref

    @staticmethod
    def _resolve_flow_uuid(
        candidate: Mapping[str, Any] | None,
        existing_ref: Mapping[str, Any] | None,
    ) -> str:
        candidate_uuid = _coerce_text(candidate.get("uuid")) if candidate else ""
        if candidate_uuid:
            return candidate_uuid
        if existing_ref:
            uuid_value = _coerce_text(existing_ref.get("@refObjectId"))
            if uuid_value:
                return uuid_value
        return str(uuid.uuid4())

    def _resolve_flow_version(
        self,
        candidate: Mapping[str, Any] | None,
        existing_ref: Mapping[str, Any] | None,
        mode: str,
    ) -> str:
        base_version = _coerce_text(candidate.get("version")) if candidate else ""
        if not base_version and existing_ref:
            base_version = _coerce_text(existing_ref.get("@version"))
        if base_version == "00.00.000":
            base_version = "01.01.000"
        if not base_version:
            base_version = "01.01.000"
        if mode == "update":
            return _bump_version(base_version)
        return base_version

    def _resolve_text_fields(
        self,
        *,
        exchange: Mapping[str, Any],
        hints: Mapping[str, list[str] | str],
        candidate: Mapping[str, Any] | None,
        exchange_name: str,
        process_name: str,
        flow_type: str,
    ) -> dict[str, Any]:
        en_name, zh_name = self._resolve_language_pairs(candidate, hints, exchange_name)
        default_comment = _normalize_text(_extract_general_comment(exchange)) or f"Auto-generated for {exchange_name}"
        defaults = {
            "base_name_en": _normalize_text(en_name),
            "base_name_zh": _normalize_text(zh_name) or _normalize_text(en_name),
            "treatment_en": _normalize_text(_coerce_text(candidate.get("treatment_standards_routes")) if candidate else "")
            or _normalize_text(_extract_hint_value(hints, ("treatment", "treatment_standards_routes"))),
            "treatment_zh": "",
            "mix_en": _normalize_text(_coerce_text(candidate.get("mix_and_location_types")) if candidate else "")
            or _normalize_text(_extract_hint_value(hints, ("mix_location", "mix_and_location_types"))),
            "mix_zh": "",
            "synonyms_en": _normalize_text_list(hints.get("en_synonyms") or []),
            "synonyms_zh": _normalize_text_list(hints.get("zh_synonyms") or []),
            "comment_en": default_comment,
            "comment_zh": "",
        }
        if not self._llm:
            return defaults

        context = {
            "exchange": {
                "name": exchange_name,
                "direction": _coerce_text(exchange.get("exchangeDirection")),
                "unit": _coerce_text(exchange.get("unit")),
                "general_comment": _extract_general_comment(exchange),
                "flow_type": flow_type,
                "search_hints": exchange.get("search_hints") if isinstance(exchange.get("search_hints"), list) else [],
            },
            "process": {"name": process_name},
            "flow_search_hints": _normalize_hint_values(hints),
            "selected_candidate": candidate or {},
            "defaults": defaults,
        }
        try:
            response = self._llm.invoke(
                {
                    "prompt": FLOW_TEXT_PROMPT,
                    "context": context,
                    "response_format": {"type": "json_object"},
                }
            )
            payload = _ensure_mapping_response(response)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning(
                "flow_publish.text_llm_failed",
                exchange=exchange_name,
                process=process_name,
                error=str(exc),
            )
            return defaults

        def pick(key: str, fallback: str) -> str:
            value = _normalize_text(payload.get(key))
            return value or fallback

        def pick_list(key: str, fallback: list[str]) -> list[str]:
            values = _normalize_text_list(payload.get(key))
            return values or fallback

        resolved = {
            "base_name_en": pick("base_name_en", defaults["base_name_en"]),
            "base_name_zh": pick("base_name_zh", defaults["base_name_zh"] or defaults["base_name_en"]),
            "treatment_en": pick("treatment_en", defaults["treatment_en"] or defaults["base_name_en"]),
            "treatment_zh": pick("treatment_zh", defaults["treatment_zh"] or defaults["base_name_zh"]),
            "mix_en": pick("mix_en", defaults["mix_en"] or defaults["base_name_en"]),
            "mix_zh": pick("mix_zh", defaults["mix_zh"] or defaults["base_name_zh"]),
            "synonyms_en": pick_list("synonyms_en", defaults["synonyms_en"]),
            "synonyms_zh": pick_list("synonyms_zh", defaults["synonyms_zh"]),
            "comment_en": pick("comment_en", defaults["comment_en"]),
            "comment_zh": pick("comment_zh", defaults["comment_zh"] or defaults["comment_en"]),
        }
        return resolved

    @staticmethod
    def _resolve_language_pairs(
        candidate: Mapping[str, Any] | None,
        hints: Mapping[str, list[str] | str],
        fallback: str,
    ) -> tuple[str, str]:
        candidate_name = _coerce_text(candidate.get("base_name")) if candidate else ""
        base = candidate_name or fallback
        en_name, zh_name = _derive_language_pairs(hints, base)
        return en_name, zh_name

    def _resolve_classification(
        self,
        flow_type: str,
        hints: Mapping[str, list[str] | str],
        candidate: Mapping[str, Any] | None,
        exchange: Mapping[str, Any],
    ) -> dict[str, Any]:
        if flow_type == "Elementary flow":
            return _build_elementary_classification(hints)
        classification_data = candidate.get("classification") if isinstance(candidate, Mapping) else None
        if isinstance(classification_data, list) and classification_data:
            classes: list[dict[str, Any]] = []
            for index, item in enumerate(classification_data):
                if not isinstance(item, Mapping):
                    continue
                level = _coerce_text(item.get("@level")) or str(index)
                class_entry = {
                    "@level": level,
                    "#text": _coerce_text(item.get("#text")) or "",
                }
                class_id = _coerce_text(item.get("@classId"))
                if class_id:
                    class_entry["@classId"] = class_id
                classes.append(class_entry)
            if classes:
                return {"common:classification": {"common:class": classes}}
        if flow_type != "Product flow":
            return _default_product_classification()
        path: list[tuple[str, str]] = []
        try:
            path = self._product_category_selector.select_path(exchange, hints)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("flow_publish.product_category_select_failed", error=str(exc))
            path = []
        if path:
            return _classification_from_path(path)
        LOGGER.debug("flow_publish.product_category_fallback", reason="no_path_selected")
        return _default_product_classification()

    @staticmethod
    def _resolve_comments(
        comment: str,
        candidate: Mapping[str, Any] | None,
        exchange_name: str,
        *,
        comment_en: str | None = None,
        comment_zh: str | None = None,
    ) -> list[dict[str, Any]]:
        comment_en = _normalize_text(comment_en)
        comment_zh = _normalize_text(comment_zh)
        if comment_en or comment_zh:
            if not comment_en:
                comment_en = comment_zh
            if not comment_zh:
                comment_zh = comment_en
            return [
                _language_entry(comment_en or "", "en"),
                _language_entry(comment_zh or "", "zh"),
            ]
        candidate_comment = _coerce_text(candidate.get("general_comment")) if candidate else ""
        if candidate_comment:
            return [_language_entry(candidate_comment)]
        if comment:
            return [_language_entry(comment)]
        return [_language_entry(f"Auto-generated for {exchange_name}")]

    def _build_name_block(
        self,
        candidate: Mapping[str, Any] | None,
        hints: Mapping[str, list[str] | str],
        en_name: str,
        zh_name: str,
        *,
        treatment_en: str | None = None,
        treatment_zh: str | None = None,
        mix_en: str | None = None,
        mix_zh: str | None = None,
    ) -> dict[str, Any]:
        treatment = _coerce_text(candidate.get("treatment_standards_routes")) if candidate else ""
        treatment_values = hints.get("treatmentStandardsRoutes") if isinstance(hints.get("treatmentStandardsRoutes"), list) else []
        mix = _coerce_text(candidate.get("mix_and_location_types")) if candidate else ""
        mix_values = hints.get("mixAndLocationTypes") if isinstance(hints.get("mixAndLocationTypes"), list) else []
        fallback_treatment = _extract_hint_value(hints, ("treatment", "treatment_standards_routes"))
        fallback_mix = _extract_hint_value(hints, ("mix_location", "mix_and_location_types"))
        treatment_en = _normalize_text(treatment_en) or _normalize_text(treatment or (treatment_values[0] if treatment_values else fallback_treatment)) or _normalize_text(en_name)
        treatment_zh = _normalize_text(treatment_zh)
        mix_en = _normalize_text(mix_en) or _normalize_text(mix or (mix_values[0] if mix_values else fallback_mix)) or _normalize_text(en_name)
        mix_zh = _normalize_text(mix_zh)
        name_block = {
            "baseName": [
                _language_entry(_normalize_text(en_name), "en"),
                _language_entry(_normalize_text(zh_name), "zh"),
            ],
            "treatmentStandardsRoutes": [
                _language_entry(treatment_en, "en"),
            ],
            "mixAndLocationTypes": [
                _language_entry(mix_en, "en"),
            ],
        }
        if treatment_zh:
            name_block["treatmentStandardsRoutes"].append(_language_entry(treatment_zh, "zh"))
        if mix_zh:
            name_block["mixAndLocationTypes"].append(_language_entry(mix_zh, "zh"))
        return name_block

    @staticmethod
    def _extract_classification_entries(classification: Mapping[str, Any]) -> list[dict[str, str]]:
        payload = classification.get("common:classification")
        if not isinstance(payload, Mapping):
            return []
        classes = payload.get("common:class")
        if isinstance(classes, Mapping):
            classes = [classes]
        if not isinstance(classes, list):
            return []
        normalized: list[dict[str, str]] = []
        for index, item in enumerate(classes):
            if not isinstance(item, Mapping):
                continue
            level = _coerce_text(item.get("@level")) or str(index)
            text = _coerce_text(item.get("#text"))
            if not text:
                continue
            entry: dict[str, str] = {"@level": level, "#text": text}
            class_id = _coerce_text(item.get("@classId"))
            if class_id:
                entry["@classId"] = class_id
            normalized.append(entry)
        return normalized

    @staticmethod
    def _extract_language_text(entries: Any, lang: str) -> str:
        if isinstance(entries, Mapping):
            entries = [entries]
        if not isinstance(entries, list):
            return ""
        fallback = ""
        target = (lang or "").strip().lower()
        for item in entries:
            if not isinstance(item, Mapping):
                continue
            text = _normalize_text(item.get("#text"))
            if not text:
                continue
            item_lang = _coerce_text(item.get("@xml:lang")).lower()
            if item_lang == target:
                return text
            if not fallback:
                fallback = text
        return fallback

    @staticmethod
    def _split_synonyms(text: str) -> list[str]:
        normalized = _normalize_text(text)
        if not normalized:
            return []
        parts: list[str] = []
        for chunk in normalized.replace("；", ";").split(";"):
            value = chunk.strip()
            if value:
                parts.append(value)
        return parts

    @staticmethod
    def _build_synonyms(
        hints: Mapping[str, list[str] | str],
        en_name: str,
        zh_name: str,
        *,
        synonyms_en: Sequence[str] | str | None = None,
        synonyms_zh: Sequence[str] | str | None = None,
    ) -> list[dict[str, Any]]:
        if synonyms_en is None:
            en_values = hints.get("en_synonyms") or []
        else:
            en_values = _normalize_text_list(synonyms_en)
        if synonyms_zh is None:
            zh_values = hints.get("zh_synonyms") or []
        else:
            zh_values = _normalize_text_list(synonyms_zh)
        if isinstance(en_values, str):
            en_values = [en_values]
        if isinstance(zh_values, str):
            zh_values = [zh_values]
        en_synonyms = "; ".join([_normalize_text(value) for value in (en_values or [en_name]) if _normalize_text(value)])
        zh_synonyms = "; ".join([_normalize_text(value) for value in (zh_values or [zh_name]) if _normalize_text(value)])
        if not en_synonyms:
            en_synonyms = _normalize_text(en_name)
        if not zh_synonyms:
            zh_synonyms = _normalize_text(zh_name)
        return [
            _language_entry(en_synonyms, "en"),
            _language_entry(zh_synonyms, "zh"),
        ]


def _bump_version(version: str) -> str:
    """Increment the patch component of an ILCD version string."""
    parts = version.split(".")
    if len(parts) != 3:
        return version
    major, minor, patch = parts
    try:
        patch_int = int(patch)
    except ValueError:
        return version
    return f"{major}.{minor}.{patch_int + 1:03d}"


class ProcessPublisher:
    """Publish final process datasets once validation passes."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        crud_client: DatabaseCrudClient | None = None,
        dry_run: bool = True,
    ) -> None:
        self._settings = settings or get_settings()
        self._crud = crud_client or DatabaseCrudClient(self._settings)
        self._dry_run = dry_run

    def publish(self, datasets: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for dataset in datasets:
            process_payload: Mapping[str, Any]
            if isinstance(dataset, Mapping):
                candidate = dataset.get("process_data_set") or dataset.get("processDataSet")
                if isinstance(candidate, Mapping):
                    process_payload = candidate
                else:
                    process_payload = dataset
            else:
                raise SpecCodingError("Process dataset must be a mapping.")
            payload = {"processDataSet": process_payload}
            process_info = process_payload.get("processInformation", {})
            name_block = process_info.get("dataSetInformation", {}).get("name", {})
            process_name = _coerce_text(name_block.get("baseName"))
            uuid_value = _coerce_text(process_info.get("dataSetInformation", {}).get("common:UUID"))
            version_value = _coerce_text(
                _get_nested(
                    process_payload,
                    ("administrativeInformation", "publicationAndOwnership", "common:dataSetVersion"),
                )
            ) or "01.01.000"
            if self._dry_run:
                LOGGER.info("process_publish.dry_run", name=process_name)
                continue
            existing = None
            try:
                existing = self._crud.select_process(uuid_value, version=version_value) or self._crud.select_process(uuid_value)
            except Exception as exists_exc:  # noqa: BLE001
                LOGGER.warning(
                    "process_publish.precheck_failed",
                    name=process_name,
                    uuid=uuid_value,
                    error=str(exists_exc),
                )

            actions: list[str] = ["update", "insert"] if isinstance(existing, Mapping) else ["insert", "update"]
            result: dict[str, Any] | None = None
            errors: list[Exception] = []
            for action in actions:
                try:
                    if action == "update":
                        result = self._crud.update_process(payload)
                    else:
                        result = self._crud.insert_process(payload)
                    break
                except SpecCodingError as exc:
                    errors.append(exc)
                    LOGGER.warning(
                        "process_publish.action_failed",
                        name=process_name,
                        uuid=uuid_value,
                        action=action,
                        error=str(exc),
                    )

            if result is None:
                final_existing = existing
                if final_existing is None:
                    try:
                        final_existing = self._crud.select_process(uuid_value, version=version_value) or self._crud.select_process(uuid_value)
                    except Exception as exists_exc:  # noqa: BLE001
                        LOGGER.warning(
                            "process_publish.reuse_check_failed",
                            name=process_name,
                            uuid=uuid_value,
                            error=str(exists_exc),
                        )
                if isinstance(final_existing, Mapping):
                    LOGGER.warning(
                        "process_publish.reuse_existing_after_error",
                        name=process_name,
                        uuid=uuid_value,
                    )
                    results.append({"id": uuid_value, "action": "reuse"})
                    continue
                if errors:
                    raise SpecCodingError(f"Failed to publish process '{process_name or uuid_value}' ({uuid_value})") from errors[-1]
                raise SpecCodingError(f"Failed to publish process '{process_name or uuid_value}' ({uuid_value})")
            results.append(result)
        return results

    def close(self) -> None:
        self._crud.close()
