"""Shared CPC/TIDAS flow classification builder for product and waste flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from tiangong_lca_spec.core.exceptions import ProcessExtractionError
from tiangong_lca_spec.core.json_utils import parse_json_response
from tiangong_lca_spec.core.logging import get_logger
from tiangong_lca_spec.tidas.level_hierarchy import HierarchyEntry, get_product_flow_category_navigator

LOGGER = get_logger(__name__)

FLOW_CLASSIFICATION_LEVEL_PROMPT = (
    "You are selecting level {level} of the Tiangong product flow classification. Use "
    "`context.flow` for the flow summary and `context.candidates` for the allowed options. "
    "Choose exactly one candidate and return JSON with '@level', '@classId', and '#text'. "
    "The '@classId' MUST match one of the provided candidate codes, and '#text' must copy "
    "the candidate description verbatim. Do not include reasoning or extra fields."
)

FLOW_CLASSIFICATION_RETRY_ATTEMPTS = 2


class LanguageModelProtocol(Protocol):
    """Minimal protocol required from language models used in the classifier."""

    def invoke(self, input_data: dict[str, Any]) -> Any: ...


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("#text", "text", "value"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
        parts = [_normalize_text(item) for item in value.values() if isinstance(item, (dict, list, str))]
        return "; ".join(part for part in parts if part)
    if isinstance(value, list):
        parts = [_normalize_text(item) for item in value]
        return "; ".join(part for part in parts if part)
    return str(value).strip()


def _build_flow_summary(
    flow_dataset: dict[str, Any],
    source_payload: dict[str, Any] | None = None,
) -> dict[str, str]:
    flow_info = flow_dataset.get("flowInformation", {})
    if not isinstance(flow_info, dict):
        flow_info = {}
    data_info = flow_info.get("dataSetInformation", {})
    if not isinstance(data_info, dict):
        data_info = {}
    name_block = data_info.get("name", {}) if isinstance(data_info.get("name"), dict) else {}
    summary: dict[str, str] = {}
    summary["baseName"] = _normalize_text(name_block.get("baseName"))
    summary["treatmentStandardsRoutes"] = _normalize_text(name_block.get("treatmentStandardsRoutes"))
    summary["mixAndLocationTypes"] = _normalize_text(name_block.get("mixAndLocationTypes"))
    summary["flowProperties"] = _normalize_text(name_block.get("flowProperties"))
    summary["synonyms"] = _normalize_text(data_info.get("common:synonyms"))
    summary["generalComment"] = _normalize_text(data_info.get("common:generalComment"))
    technology = flow_info.get("technology", {})
    if isinstance(technology, dict):
        summary["technology"] = _normalize_text(technology.get("technologicalApplicability"))
    if source_payload:
        summary["jsonldCategory"] = _normalize_text(source_payload.get("category"))
        summary["jsonldDescription"] = _normalize_text(source_payload.get("description"))
        summary["jsonldSynonyms"] = _normalize_text(source_payload.get("synonyms"))
        summary["jsonldCAS"] = _normalize_text(source_payload.get("cas"))
        summary["jsonldFormula"] = _normalize_text(source_payload.get("formula"))
    summary = {key: value for key, value in summary.items() if value}
    if not summary:
        uuid = data_info.get("common:UUID")
        if isinstance(uuid, str) and uuid.strip():
            summary["flowUUID"] = uuid.strip()
    return summary or {"note": "Insufficient attributes; rely on LLM reasoning."}


def _ensure(response: Any) -> Any:
    if hasattr(response, "content"):
        response = getattr(response, "content")
    if isinstance(response, str):
        return parse_json_response(response)
    return response


def _ensure_dict(response: Any) -> dict[str, Any]:
    data = _ensure(response)
    if not isinstance(data, dict):
        raise ValueError("Expected dictionary output from language model")
    return data


@dataclass
class FlowClassificationBuilder:
    """Level-by-level CPC/TIDAS flow classification builder."""

    llm: LanguageModelProtocol

    def __post_init__(self) -> None:
        self._navigator = get_product_flow_category_navigator()
        self._max_level = self._navigator.max_level

    def run(
        self,
        flow_dataset: dict[str, Any],
        *,
        source_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        LOGGER.info("flow_extraction.classification")
        summary = _build_flow_summary(flow_dataset, source_payload)
        selections: list[dict[str, Any]] = []
        parent_code: str | None = None

        for level in range(0, self._max_level + 1):
            candidates = self._navigator.children(parent_code)
            if not candidates:
                break
            selection = self._select_level(level, candidates, summary)
            if selection is None:
                if level == 0:
                    raise ProcessExtractionError("Unable to determine level-0 classification for flow.")
                LOGGER.warning(
                    "flow_extraction.classification_level_fallback",
                    level=level,
                    parent=parent_code,
                )
                break
            selections.append(selection)
            parent_code = selection["@classId"]
        return selections

    def _select_level(
        self,
        level: int,
        candidates: list[HierarchyEntry],
        summary: dict[str, str],
    ) -> dict[str, Any] | None:
        candidate_codes = {entry.code for entry in candidates}
        descriptions = {entry.code: entry.description for entry in candidates}
        context = {
            "flow": summary,
            "candidates": [
                {
                    "code": entry.code,
                    "description": entry.description,
                    "level": entry.level,
                }
                for entry in candidates
            ],
        }
        payload = {
            "prompt": FLOW_CLASSIFICATION_LEVEL_PROMPT.format(level=level),
            "context": context,
            "response_format": {"type": "json_object"},
        }
        for attempt in range(FLOW_CLASSIFICATION_RETRY_ATTEMPTS):
            response = self.llm.invoke(payload)
            selection = self._parse_selection(response, level, candidate_codes, descriptions)
            if selection:
                return selection
            LOGGER.warning(
                "flow_extraction.classification_retry",
                level=level,
                attempt=attempt + 1,
            )
        return None

    def _parse_selection(
        self,
        response: Any,
        level: int,
        allowed_codes: set[str],
        descriptions: dict[str, str],
    ) -> dict[str, Any] | None:
        try:
            data = _ensure_dict(response)
        except ValueError:
            return None
        raw_code = data.get("@classId") or data.get("classId") or data.get("code")
        class_id = str(raw_code).strip() if raw_code is not None else ""
        if class_id not in allowed_codes:
            return None
        description = data.get("#text") or data.get("description") or descriptions.get(class_id) or ""
        description = description.strip() or descriptions.get(class_id, "")
        return {
            "@level": str(level),
            "@classId": class_id,
            "#text": description,
        }


# Backward-compatible name used by existing callers.
ProductFlowClassifier = FlowClassificationBuilder


__all__ = [
    "FLOW_CLASSIFICATION_LEVEL_PROMPT",
    "FLOW_CLASSIFICATION_RETRY_ATTEMPTS",
    "FlowClassificationBuilder",
    "LanguageModelProtocol",
    "ProductFlowClassifier",
]
