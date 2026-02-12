"""High-level process extraction orchestration built on sequential stages."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, TypedDict

from tiangong_lca_spec.core.config import Settings, get_settings
from tiangong_lca_spec.core.exceptions import (
    ExchangeValidationError,
    ProcessExtractionError,
    SpecCodingError,
)
from tiangong_lca_spec.core.logging import get_logger

from .extractors import (
    LanguageModelProtocol,
    LocationNormalizer,
    ProcessClassifier,
    ProcessListExtractor,
    SectionExtractor,
)
from .hints import enrich_exchange_hints, ensure_flow_hints_dict
from .tidas_mapping import build_tidas_process_dataset
from .validators import is_placeholder_value, validate_exchanges_strict

LOGGER = get_logger(__name__)
_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
_REFERENCE_YEAR_WINDOW = 5000

PROCESS_ID_KEYS = (
    "processId",
    "processID",
    "process_id",
    "processIdentifier",
    "process_identifier",
    "processUUID",
    "processUuid",
    "uuid",
    "UUID",
    "id",
)


class ExtractionState(TypedDict, total=False):
    clean_text: str
    process_blocks: list[dict[str, Any]]
    fallback_reference_year: int
    retry_feedback: str | None


class ProcessExtractionService:
    """Coordinates process extraction by running staged helpers sequentially."""

    def __init__(
        self,
        llm: LanguageModelProtocol,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._process_list_extractor = ProcessListExtractor(llm)
        self._section_extractor = SectionExtractor(llm)
        self._classifier = ProcessClassifier(llm)
        self._location_normalizer = LocationNormalizer(llm)
        self._exchange_retry_attempts = max(1, self._settings.stage2_exchange_retry_attempts)

    def extract(self, clean_text: str) -> list[dict[str, Any]]:
        retry_feedback: str | None = None
        for attempt in range(self._exchange_retry_attempts):
            try:
                return self._run_pipeline(clean_text, retry_feedback)
            except ExchangeValidationError as exc:
                retry_feedback = exc.retry_feedback()
                if attempt == self._exchange_retry_attempts - 1:
                    raise
                LOGGER.warning(
                    "process_extraction.exchange_retry",
                    attempt=attempt + 1,
                    max_attempts=self._exchange_retry_attempts,
                    process=exc.process_name,
                )
        raise ProcessExtractionError("Failed to produce valid exchanges after retries.")

    def _run_pipeline(self, clean_text: str, retry_feedback: str | None) -> list[dict[str, Any]]:
        state: ExtractionState = {"clean_text": clean_text}
        if retry_feedback:
            state["retry_feedback"] = retry_feedback
        fallback_year = _infer_reference_year_from_text(clean_text)
        if fallback_year is not None:
            state["fallback_reference_year"] = fallback_year
        process_candidates = self._list_process_candidates(clean_text, retry_feedback=retry_feedback)
        state["process_candidates"] = process_candidates
        state["process_blocks"] = self._generate_process_blocks(clean_text, process_candidates)
        state = self._classify_process(state)
        state = self._normalize_location(state)
        state = self._finalize(state)
        blocks = state.get("process_blocks") or []
        if not blocks:
            raise ProcessExtractionError("No process blocks generated")
        return blocks

    def _list_process_candidates(
        self,
        clean_text: str,
        *,
        retry_feedback: str | None = None,
    ) -> list[dict[str, Any]]:
        if not clean_text:
            raise ProcessExtractionError("Clean text missing for extraction")
        return self._process_list_extractor.run(clean_text, retry_feedback=retry_feedback)

    def _generate_process_blocks(
        self,
        clean_text: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidates:
            raise ProcessExtractionError("Process list extractor returned no candidates")
        blocks: list[dict[str, Any]] = []
        for candidate in candidates:
            dataset = self._generate_process_dataset(clean_text, candidate)
            block: dict[str, Any] = {
                "processDataSet": dataset,
                "process_id": candidate.get("processId"),
                "process_candidate": candidate,
            }
            blocks.append(block)
        return blocks

    def _generate_process_dataset(
        self,
        clean_text: str,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        retry_feedback: str | None = None
        process_label = candidate.get("name") or candidate.get("processId") or "Unnamed process"
        for attempt in range(self._exchange_retry_attempts):
            response = self._section_extractor.run(
                clean_text,
                focus_process=candidate,
                retry_feedback=retry_feedback,
            )
            dataset = response.get("processDataSet")
            if not isinstance(dataset, dict):
                raise ProcessExtractionError("Process detail extractor must return `processDataSet`.")
            exchanges = _extract_exchanges(dataset)
            _prepare_exchange_metadata(exchanges)
            errors = validate_exchanges_strict(exchanges, geography=None)
            if not errors:
                _serialise_flow_hints(dataset, process_label)
                return dataset
            retry_feedback = _format_detail_retry_feedback(candidate, errors)
        raise ExchangeValidationError(process_label, errors)

    def _classify_process(self, state: ExtractionState) -> ExtractionState:
        blocks = state.get("process_blocks") or []
        if not blocks:
            LOGGER.warning("process_extraction.missing_process_blocks")
            return state

        for block in blocks:
            dataset = block.get("processDataSet")
            if not isinstance(dataset, dict):
                LOGGER.warning("process_extraction.invalid_dataset_block")
                continue
            process_info = dataset.setdefault("processInformation", {})
            if not process_info:
                LOGGER.warning("process_extraction.missing_process_information")
                continue
            classification = self._classifier.run(process_info)
            data_info = process_info.setdefault("dataSetInformation", {})
            classification_info = data_info.setdefault("classificationInformation", {})
            class_entries = list(classification) if isinstance(classification, list) else [classification]
            classification_info["common:classification"] = {"common:class": class_entries}
            classification_info.pop("classification", None)
            block["classification"] = classification

        state["process_blocks"] = blocks
        return state

    def _normalize_location(self, state: ExtractionState) -> ExtractionState:
        blocks = state.get("process_blocks") or []
        if not blocks:
            return state

        for block in blocks:
            dataset = block.get("processDataSet")
            if not isinstance(dataset, dict):
                continue
            process_info = dataset.setdefault("processInformation", {})
            if not process_info:
                continue
            try:
                geography = self._location_normalizer.run(process_info)
            except SpecCodingError as exc:
                LOGGER.warning("process_extraction.location_parse_failed", error=str(exc))
                geography = {}
            if isinstance(geography, str):
                geography = {"description": geography}
            process_info.setdefault("geography", {}).update(geography)
            block["geography"] = geography

        state["process_blocks"] = blocks
        return state

    def _finalize(self, state: ExtractionState) -> ExtractionState:
        blocks = state.get("process_blocks") or []
        if not blocks:
            raise ProcessExtractionError("Process dataset missing at finalize step")

        fallback_year = state.get("fallback_reference_year")
        final_blocks: list[dict[str, Any]] = []
        for block in blocks:
            process_dataset = block.get("processDataSet")
            if not isinstance(process_dataset, dict):
                raise ProcessExtractionError("Process dataset missing in block")

            if fallback_year is not None:
                _apply_reference_year_fallback(process_dataset, fallback_year)
            normalized_dataset = build_tidas_process_dataset(process_dataset)

            process_name = _extract_process_name(normalized_dataset)
            geography_hint = _extract_geography(normalized_dataset)
            exchanges_container = normalized_dataset.get("exchanges") or {}
            exchange_items: list[dict[str, Any]] = []
            if isinstance(exchanges_container, dict):
                raw_items = exchanges_container.get("exchange", [])
                if isinstance(raw_items, list):
                    exchange_items = [item for item in raw_items if isinstance(item, dict)]
                    _prepare_exchange_metadata(exchange_items)
                    for exchange in exchange_items:
                        enrich_exchange_hints(exchange, process_name=process_name, geography=geography_hint)

            issues = validate_exchanges_strict(exchange_items, geography=geography_hint)
            if issues:
                raise ExchangeValidationError(process_name, issues)

            final_block: dict[str, Any] = {
                "processDataSet": normalized_dataset,
            }
            if process_id := block.get("process_id"):
                final_block["process_id"] = process_id
            final_blocks.append(final_block)

        state["process_blocks"] = final_blocks
        return state


def _infer_reference_year_from_text(text: str) -> int | None:
    current_year = datetime.now().year
    if not text:
        return current_year

    window = text[:_REFERENCE_YEAR_WINDOW]
    candidates: list[tuple[int, int]] = []
    for match in _YEAR_PATTERN.finditer(window):
        try:
            year_value = int(match.group(0))
        except ValueError:
            continue
        if 1900 <= year_value <= current_year + 1:
            candidates.append((match.start(), year_value))
    if not candidates:
        return current_year

    for position, year_value in candidates:
        if position <= 300:
            return year_value

    candidates.sort(key=lambda item: (-item[1], item[0]))
    return candidates[0][1]


def _apply_reference_year_fallback(dataset: dict[str, Any], fallback_year: int) -> None:
    if fallback_year is None:
        return

    process_info = dataset.get("processInformation")
    if not isinstance(process_info, dict):
        process_info = {}
        dataset["processInformation"] = process_info

    time_info: dict[str, Any]
    existing_time = process_info.get("time")
    if isinstance(existing_time, dict):
        time_info = existing_time
    else:
        time_info = {}
        process_info["time"] = time_info

    existing_year = _coerce_year(time_info.get("referenceYear"))
    if existing_year is not None:
        time_info["referenceYear"] = existing_year
        return

    existing_year = _coerce_year(time_info.get("common:referenceYear"))
    if existing_year is not None:
        time_info["common:referenceYear"] = existing_year
        return

    valid_fallback = _coerce_year(fallback_year)
    if valid_fallback is not None:
        time_info["referenceYear"] = valid_fallback


def _coerce_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        year_value = int(value)
    elif isinstance(value, str):
        match = _YEAR_PATTERN.search(value)
        if not match:
            return None
        try:
            year_value = int(match.group(0))
        except ValueError:
            return None
    else:
        return None

    current_year = datetime.now().year
    if 1900 <= year_value <= current_year + 1:
        return year_value
    return None


def _extract_process_name(dataset: dict[str, Any]) -> str | None:
    process_info = dataset.get("processInformation")
    if not isinstance(process_info, dict):
        return None
    data_info = process_info.get("dataSetInformation")
    if not isinstance(data_info, dict):
        return None
    name = data_info.get("name")
    if isinstance(name, dict):
        base_name = name.get("baseName")
        if isinstance(base_name, dict):
            return _stringify(base_name.get("#text") or base_name.get("@value") or base_name.get("text"))
        return _stringify(base_name)
    return _stringify(name)


def _extract_geography(dataset: dict[str, Any]) -> str | None:
    process_info = dataset.get("processInformation")
    if not isinstance(process_info, dict):
        return None
    geography = process_info.get("geography")
    location_text = _extract_geography_value(geography)
    if location_text:
        return location_text
    value = _stringify(geography)
    return value or None


def _extract_geography_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        prioritized_keys = (
            "shortName",
            "#text",
            "text",
            "description",
            "locationOfOperation",
            "locationDescription",
            "region",
            "@location",
            "location",
            "code",
        )
        for key in prioritized_keys:
            if key in value:
                result = _extract_geography_value(value[key])
                if result:
                    return result
        supply_block = value.get("locationOfOperationSupplyOrProduction")
        if supply_block is not None:
            result = _extract_geography_value(supply_block)
            if result:
                return result
        for nested in value.values():
            result = _extract_geography_value(nested)
            if result:
                return result
        return None
    if isinstance(value, list):
        for item in value:
            result = _extract_geography_value(item)
            if result:
                return result
    return None


def _extract_process_id(container: dict[str, Any]) -> str | None:
    for key in PROCESS_ID_KEYS:
        value = container.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    data_info = container.get("dataSetInformation")
    if isinstance(data_info, dict):
        for key in ("identifierOfSubDataSet", "common:UUID"):
            value = data_info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    process_info = container.get("processInformation")
    if isinstance(process_info, dict):
        data_info = process_info.get("dataSetInformation")
        if isinstance(data_info, dict):
            for key in ("identifierOfSubDataSet", "common:UUID"):
                value = data_info.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def _derive_dataset_identifier(dataset: dict[str, Any]) -> str | None:
    identifier = _extract_process_id(dataset)
    if identifier:
        return identifier

    process_info = dataset.get("processInformation")
    if isinstance(process_info, dict):
        data_info = process_info.get("dataSetInformation")
        if isinstance(data_info, dict):
            for key in ("identifierOfSubDataSet", "common:UUID"):
                value = data_info.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def _extract_exchanges(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    exchanges_section = dataset.get("exchanges") or {}
    if isinstance(exchanges_section, dict):
        exchanges = exchanges_section.get("exchange")
    else:
        exchanges = None
    if isinstance(exchanges, dict):
        return [exchanges]
    if isinstance(exchanges, list):
        return [item for item in exchanges if isinstance(item, dict)]
    return []


def _prepare_exchange_metadata(exchanges: list[dict[str, Any]]) -> None:
    for exchange in exchanges:
        hints = ensure_flow_hints_dict(exchange)
        name = _stringify(exchange.get("exchangeName"))
        if name and not is_placeholder_value(name):
            continue
        if not isinstance(hints, dict):
            continue
        candidate_name = _stringify(hints.get("basename"))
        if candidate_name and not is_placeholder_value(candidate_name):
            exchange["exchangeName"] = candidate_name


def _serialise_flow_hints(dataset: dict[str, Any], process_name: str | None) -> None:
    exchanges = _extract_exchanges(dataset)
    for exchange in exchanges:
        hints = ensure_flow_hints_dict(exchange)
        if hints is None:
            continue
        enrich_exchange_hints(exchange, process_name=process_name)


def _format_detail_retry_feedback(candidate: dict[str, Any], errors: list[str]) -> str:
    header = f"processId={candidate.get('processId')} name={candidate.get('name')}"
    details = "\n".join(f"- {issue}" for issue in errors)
    return f"{header}\nFix each issue without removing, renaming, or merging the process. " "Keep the same exchanges and regenerate the full `processDataSet`.\n" f"{details}"


def _ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _ensure_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
