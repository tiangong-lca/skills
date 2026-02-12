"""Exchange to flow alignment utilities."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from json import dumps
from typing import Any, Callable, Iterable
from uuid import uuid4

from tiangong_lca_spec.core.config import Settings, get_settings
from tiangong_lca_spec.core.exceptions import FlowAlignmentError, FlowSearchError
from tiangong_lca_spec.core.logging import get_logger
from tiangong_lca_spec.core.models import FlowCandidate, FlowQuery, UnmatchedFlow
from tiangong_lca_spec.core.uris import build_portal_uri
from tiangong_lca_spec.flow_alignment.selector import (
    CandidateSelector,
    LanguageModelProtocol,
    LLMCandidateSelector,
    SelectorDecision,
    SimilarityCandidateSelector,
)
from tiangong_lca_spec.flow_search import search_flows

LOGGER = get_logger(__name__)

FlowSearchCallable = Callable[[FlowQuery], tuple[list[FlowCandidate], list[UnmatchedFlow]]]


class FlowAlignmentService:
    """Aligns exchanges with flow search candidates."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        flow_search_fn: FlowSearchCallable | None = None,
        selector: CandidateSelector | None = None,
        llm: LanguageModelProtocol | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._profile = self._settings.profile
        self._flow_search = flow_search_fn or search_flows
        if selector is not None:
            self._selector = selector
        elif llm is not None:
            self._selector = LLMCandidateSelector(llm)
        else:
            self._selector = SimilarityCandidateSelector()

    def align_exchanges(self, process_dataset: dict[str, Any], paper_md: str | None = None) -> dict[str, Any]:
        try:
            exchanges = list(self._iter_exchanges(process_dataset))
        except KeyError as exc:
            raise FlowAlignmentError("Process dataset missing exchanges") from exc

        process_name = self._extract_process_name(process_dataset)
        LOGGER.info("flow_alignment.start", process=process_name, exchange_count=len(exchanges))

        matched: list[FlowCandidate] = []
        unmatched: list[UnmatchedFlow] = []
        origin_exchanges: dict[str, list[dict[str, Any]]] = {}

        search_jobs: dict[Any, tuple[dict[str, Any], FlowQuery]] = {}
        for exchange in exchanges:
            query = self._build_query(exchange, process_name, paper_md)
            future = self._executor.submit(self._flow_search, query)
            search_jobs[future] = (exchange, query)

        for future in as_completed(search_jobs):
            exchange, query = search_jobs[future]
            exchange_name = self._safe_exchange_name(exchange)
            try:
                matches, misses = future.result()
                updated_exchange = self._consume_search_result(
                    exchange,
                    query,
                    matches,
                    misses,
                    matched,
                    unmatched,
                    process_name,
                )
                origin_exchanges.setdefault(exchange_name, []).append(updated_exchange)
                continue
            except FlowSearchError as exc:
                LOGGER.warning(
                    "flow_alignment.retry_serial",
                    exchange=exchange_name,
                    process=process_name,
                    error=str(exc),
                )
                try:
                    matches, misses = self._flow_search(query)
                    updated_exchange = self._consume_search_result(
                        exchange,
                        query,
                        matches,
                        misses,
                        matched,
                        unmatched,
                        process_name,
                    )
                    origin_exchanges.setdefault(exchange_name, []).append(updated_exchange)
                    continue
                except Exception as serial_exc:  # pylint: disable=broad-except
                    LOGGER.error(
                        "flow_alignment.exchange_failed",
                        exchange=exchange_name,
                        error=str(serial_exc),
                    )
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.error("flow_alignment.exchange_failed", exchange=exchange_name, error=str(exc))
            placeholder = self._apply_placeholder_reference(exchange)
            origin_exchanges.setdefault(exchange_name, []).append(placeholder)
            unmatched.append(self._build_unmatched_flow(exchange, process_name, exchange_name))

        return {
            "process_name": process_name,
            "matched_flows": matched,
            "unmatched_flows": unmatched,
            "origin_exchanges": origin_exchanges,
        }

    def _consume_search_result(
        self,
        exchange: dict[str, Any],
        query: FlowQuery,
        matches: list[FlowCandidate],
        misses: list[UnmatchedFlow],
        matched_acc: list[FlowCandidate],
        unmatched_acc: list[UnmatchedFlow],
        process_name: str | None,
    ) -> dict[str, Any]:
        if misses:
            unmatched_acc.extend(misses)

        decision = self._selector.select(query, exchange, matches)
        candidate = decision.candidate
        if candidate is not None:
            updated_exchange = self._apply_candidate_reference(exchange, decision)
            if candidate.uuid:
                matched_acc.append(candidate)
            else:
                unmatched_acc.append(self._build_unmatched_flow(exchange, process_name, self._safe_exchange_name(exchange)))
            return updated_exchange

        unmatched_acc.append(self._build_unmatched_flow(exchange, process_name, self._safe_exchange_name(exchange)))
        return self._apply_placeholder_reference(exchange)

    def _apply_candidate_reference(self, exchange: dict[str, Any], decision: SelectorDecision) -> dict[str, Any]:
        enriched = dict(exchange)
        candidate = decision.candidate
        if candidate is None:
            return self._apply_placeholder_reference(enriched)
        if candidate.uuid:
            enriched["referenceToFlowDataSet"] = self._candidate_reference(candidate)
        else:
            enriched["referenceToFlowDataSet"] = self._placeholder_reference(self._safe_exchange_name(enriched))
        detail = enriched.get("matchingDetail")
        if not isinstance(detail, dict):
            detail = {}
        formatted_name = self._compose_candidate_short_description(candidate)
        detail["selectedCandidate"] = {
            "uuid": candidate.uuid,
            "base_name": candidate.base_name,
            "treatment_standards_routes": candidate.treatment_standards_routes,
            "mix_and_location_types": candidate.mix_and_location_types,
            "flow_properties": candidate.flow_properties,
            "version": candidate.version,
            "geography": candidate.geography,
            "classification": candidate.classification,
            "general_comment": candidate.general_comment,
            "reasoning": candidate.reasoning,
            "score": decision.score,
            "selector": decision.strategy,
            "evaluation_reason": decision.reasoning,
            "combined_name": formatted_name,
        }
        enriched["matchingDetail"] = detail
        return enriched

    def _apply_placeholder_reference(self, exchange: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(exchange)
        if not self._has_reference(enriched.get("referenceToFlowDataSet")):
            enriched["referenceToFlowDataSet"] = self._placeholder_reference(self._safe_exchange_name(enriched))
        return enriched

    @staticmethod
    def _candidate_reference(candidate: FlowCandidate) -> dict[str, Any]:
        version = candidate.version or "01.01.000"
        uuid = candidate.uuid or str(uuid4())
        uri = build_portal_uri("flow", uuid, version)
        return {
            "@type": "flow data set",
            "@refObjectId": uuid,
            "@version": version,
            "@uri": uri,
            "common:shortDescription": FlowAlignmentService._multilang(FlowAlignmentService._compose_candidate_short_description(candidate)),
        }

    @staticmethod
    def _placeholder_reference(name: str) -> dict[str, Any]:
        identifier = str(uuid4())
        version = "00.00.000"
        uri = build_portal_uri("flow", identifier, version)
        return {
            "@type": "flow data set",
            "@refObjectId": identifier,
            "@version": version,
            "@uri": uri,
            "common:shortDescription": FlowAlignmentService._multilang(name),
            "unmatched:placeholder": True,
        }

    @staticmethod
    def _has_reference(value: Any) -> bool:
        if isinstance(value, dict):
            return "@refObjectId" in value
        if isinstance(value, list):
            return all(isinstance(item, dict) and "@refObjectId" in item for item in value)
        return False

    @staticmethod
    def _multilang(text: str) -> dict[str, Any]:
        label = text or "Unnamed flow"
        return {"@xml:lang": "en", "#text": label}

    @staticmethod
    def _compose_candidate_short_description(candidate: FlowCandidate) -> str:
        components = [
            FlowAlignmentService._candidate_name_segment(candidate.base_name),
            FlowAlignmentService._candidate_name_segment(candidate.treatment_standards_routes),
            FlowAlignmentService._candidate_name_segment(candidate.mix_and_location_types),
            FlowAlignmentService._candidate_name_segment(candidate.flow_properties),
        ]
        raw = "; ".join(components + [""])  # trailing delimiter to match reference implementation
        without_missing = raw.replace("-; ", "")
        if without_missing.endswith("; "):
            without_missing = without_missing[:-2]
        cleaned = without_missing.strip()
        return cleaned or "-"

    @staticmethod
    def _candidate_name_segment(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or "-"
        text = FlowAlignmentService._stringify(value)
        if not text:
            return "-"
        stripped = text.strip()
        return stripped or "-"

    def _build_query(self, exchange: dict[str, Any], process_name: str | None, paper_md: str | None) -> FlowQuery:
        return FlowQuery(
            exchange_name=self._safe_exchange_name(exchange),
            description=self._stringify(exchange.get("generalComment") or exchange.get("comment")),
        )

    def _build_unmatched_flow(
        self,
        exchange: dict[str, Any],
        process_name: str | None,
        exchange_name: str | None,
    ) -> UnmatchedFlow:
        return UnmatchedFlow(
            base_name=exchange_name or self._safe_exchange_name(exchange),
            general_comment=self._stringify(exchange.get("generalComment") or exchange.get("comment")),
            process_name=process_name,
        )

    @property
    def _executor(self) -> ThreadPoolExecutor:
        if not hasattr(self, "__executor"):
            limit = max(1, self._settings.flow_search_max_parallel)
            max_workers = max(1, min(self._profile.concurrency, limit))
            self.__executor = ThreadPoolExecutor(max_workers=max_workers)
        return self.__executor

    def close(self) -> None:
        if hasattr(self, "__executor"):
            self.__executor.shutdown(wait=True)
            delattr(self, "__executor")

    def _iter_exchanges(self, process_dataset: dict[str, Any]) -> Iterable[dict[str, Any]]:
        exchanges_block = process_dataset.get("exchanges") or {}
        exchanges = exchanges_block.get("exchange") or []
        if isinstance(exchanges, list):
            return exchanges
        return [exchanges]

    @staticmethod
    def _extract_process_name(process_dataset: dict[str, Any]) -> str | None:
        process_info = process_dataset.get("processInformation") or process_dataset.get("process_information") or {}
        data_info = process_info.get("dataSetInformation") or process_info.get("data_set_information") or {}
        name_block = data_info.get("name")
        resolved = _resolve_base_name(name_block)
        if resolved:
            return resolved
        return process_dataset.get("process_name") or "unknown_process"

    @staticmethod
    def _safe_exchange_name(exchange: dict[str, Any]) -> str:
        raw_name = (
            exchange.get("exchangeName")
            or exchange.get("name")
            or exchange.get("flowName")
            or FlowAlignmentService._extract_short_description_base(exchange.get("referenceToFlowDataSet"))
            or "unknown_exchange"
        )
        resolved = FlowAlignmentService._stringify(raw_name)
        return resolved or "unknown_exchange"

    @staticmethod
    def _extract_short_description_base(reference: Any) -> str | None:
        if isinstance(reference, dict):
            text = FlowAlignmentService._stringify(reference.get("common:shortDescription"))
            if text:
                parts = [part.strip() for part in text.split(";")]
                return parts[0] if parts else text.strip()
        return None

    @staticmethod
    def _stringify(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, dict):
            text = value.get("#text") or value.get("text") or value.get("@value")
            if text is not None:
                return str(text)
            for candidate in value.values():
                candidate_text = FlowAlignmentService._stringify(candidate)
                if candidate_text:
                    return candidate_text
            return dumps(value, ensure_ascii=False)
        if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, str)):
            parts: list[str] = []
            for item in value:
                text = FlowAlignmentService._stringify(item)
                if text:
                    parts.append(text)
            if parts:
                return "; ".join(parts)
            return None
        return str(value)


def align_exchanges(process_dataset: dict[str, Any], paper_md: str | None = None) -> dict[str, Any]:
    """Functional wrapper around FlowAlignmentService."""
    service = FlowAlignmentService()
    try:
        return service.align_exchanges(process_dataset, paper_md)
    finally:
        service.close()


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
