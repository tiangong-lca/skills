"""Candidate selection strategies for flow alignment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Protocol, Sequence

from tiangong_lca_spec.core.json_utils import parse_json_response
from tiangong_lca_spec.core.logging import get_logger
from tiangong_lca_spec.core.models import FlowCandidate, FlowQuery

LOGGER = get_logger(__name__)


class LanguageModelProtocol(Protocol):
    """Minimal protocol for language models used in candidate selection."""

    def invoke(self, input_data: dict[str, Any]) -> Any: ...


@dataclass(slots=True)
class SelectorDecision:
    """Outcome of a candidate selection operation."""

    candidate: FlowCandidate | None
    score: float | None = None
    reasoning: str | None = None
    strategy: str | None = None


class CandidateSelector(Protocol):
    """Protocol implemented by selection strategies."""

    def select(
        self,
        query: FlowQuery,
        exchange: dict[str, Any],
        candidates: Sequence[FlowCandidate],
    ) -> SelectorDecision: ...


class SimilarityCandidateSelector:
    """Pick the candidate with the highest name-similarity score."""

    def select(
        self,
        query: FlowQuery,
        exchange: dict[str, Any],
        candidates: Sequence[FlowCandidate],
    ) -> SelectorDecision:
        if not candidates:
            return SelectorDecision(candidate=None, score=None, reasoning=None, strategy="similarity")
        scored = [(self._score(query.exchange_name, candidate.base_name), candidate) for candidate in candidates[:10]]
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_candidate = scored[0]
        if best_score <= 0.0:
            return SelectorDecision(candidate=None, score=None, reasoning=None, strategy="similarity")
        return SelectorDecision(
            candidate=best_candidate,
            score=best_score,
            reasoning=f"SequenceMatcher score={best_score:.3f}",
            strategy="similarity",
        )

    @staticmethod
    def _score(reference: str | None, candidate_name: str | None) -> float:
        left = (reference or "").strip().lower()
        right = (candidate_name or "").strip().lower()
        if not left and not right:
            return 0.0
        return SequenceMatcher(None, left, right).ratio()


class NoFallbackCandidateSelector:
    """Disable fallback selection when LLM cannot decide."""

    def select(
        self,
        query: FlowQuery,
        exchange: dict[str, Any],
        candidates: Sequence[FlowCandidate],
    ) -> SelectorDecision:
        return SelectorDecision(
            candidate=None,
            score=None,
            reasoning="LLM selection required; no fallback applied.",
            strategy="llm_no_fallback",
        )


class LLMCandidateSelector:
    """Leverage an LLM to pick the best candidate, with similarity fallback."""

    PROMPT = (
        "You are matching an inventory exchange from a life cycle assessment dataset to "
        "the best flow definition in Tiangong's flow catalogue. Your primary task is to find the "
        "most logically consistent match, acknowledging that most flows are complex Product Flows "
        "(whose CAS/Formula may be absent or irrelevant)."
        "\n\n"
        "**Matching Hierarchy (Highest Priority First):**\n"
        "1. **Primary Attributes (Top Priority):** Prioritize strong alignment on **Flow Name** and "
        "**Geography**. Most product flows are uniquely defined by these two attributes.\n"
        "2. **Chemical Check (Conditional Constraint):** If both the Query Flow and the Candidate Flow "
        "possess a **CAS Number / Formula**, an exact match is a mandatory hard constraint. If either "
        "is missing this data, proceed to the next step.\n"
        "3. **Flow Property Rule (Soft Constraint):** If a candidate matches on Primary Attributes but "
        "the **Flow Property** (e.g., mass vs. volume) is different, still select the candidate, but "
        "note in `reason` that a property addition is required. Flow Property mismatch alone MUST NOT "
        "prevent matching.\n"
        "4. **Secondary Attributes:** Consider **Classification**, **Physical State** (if applicable), "
        "and **General Comment** for final tie-breaking.\n"
        "5. **Role Constraints:** Use exchange direction/role hints when provided. "
        "If `exchangeDirection=Input` and `is_reference_flow=false`, avoid selecting candidates marked "
        "as finished products when semi-finished or raw material options exist. If the exchange name "
        "contains 'ingot' or 'billet', prefer candidates with those terms in base_name or treatment "
        "fields. If `reference_flow_name` is provided, avoid selecting a candidate whose base_name "
        "matches that reference flow for non-reference inputs unless the exchange explicitly describes "
        "recycling or internal reuse.\n"
        "\n"
        "Respond with `best_index: null` if no candidate is appropriate. Prefer candidates whose flow "
        "name, geography, classification, and general comments best align with the exchange details. "
        "The exchange may include `flow_type`, `material_role`, `exchangeDirection`, `is_reference_flow`, "
        "`reference_flow_name`, and `search_hints` aliases; use them. "
        "Candidates include `name_parts` (baseName, treatmentStandardsRoutes, mixAndLocationTypes, flowProperties) and "
        "`flow_type`.\n"
        "Return strict JSON with keys:\n"
        "- `best_index`: integer index into the candidates array (0-based) or null.\n"
        "- `confidence`: number between 0 and 1 estimating confidence (optional).\n"
        "- `reason`: short natural-language justification.\n"
        "Do not include extra commentary."
    )

    def __init__(
        self,
        llm: LanguageModelProtocol,
        *,
        fallback: CandidateSelector | None = None,
    ) -> None:
        self._llm = llm
        self._fallback = fallback or SimilarityCandidateSelector()

    def select(
        self,
        query: FlowQuery,
        exchange: dict[str, Any],
        candidates: Sequence[FlowCandidate],
    ) -> SelectorDecision:
        if not candidates:
            return SelectorDecision(candidate=None, score=None, reasoning=None, strategy="llm")
        try:
            payload = {
                "prompt": self.PROMPT,
                "context": self._build_context(query, exchange, candidates),
            }
            raw_response = self._llm.invoke(payload)
            parsed = self._parse_response(raw_response)
            best_index = parsed.get("best_index")
            if best_index is None:
                fallback_decision = self._fallback.select(query, exchange, candidates)
                if fallback_decision.candidate is not None:
                    return SelectorDecision(
                        candidate=fallback_decision.candidate,
                        score=fallback_decision.score,
                        reasoning=fallback_decision.reasoning or parsed.get("reason"),
                        strategy=f"llm_fallback->{fallback_decision.strategy}",
                    )
                return SelectorDecision(
                    candidate=None,
                    score=self._coerce_float(parsed.get("confidence")),
                    reasoning=parsed.get("reason"),
                    strategy="llm",
                )
            if not isinstance(best_index, int) or not 0 <= best_index < len(candidates):
                LOGGER.warning(
                    "flow_alignment.selector.invalid_index",
                    index=best_index,
                    candidate_count=len(candidates),
                )
                return self._fallback.select(query, exchange, candidates)
            candidate = candidates[best_index]
            return SelectorDecision(
                candidate=candidate,
                score=self._coerce_float(parsed.get("confidence")),
                reasoning=parsed.get("reason"),
                strategy="llm",
            )
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("flow_alignment.selector.llm_failed", error=str(exc))
            return self._fallback.select(query, exchange, candidates)

    def _build_context(
        self,
        query: FlowQuery,
        exchange: dict[str, Any],
        candidates: Sequence[FlowCandidate],
    ) -> str:
        summary = {
            "exchange": {
                "exchange_name": query.exchange_name,
                "description": query.description,
                "direction": exchange.get("exchangeDirection") or exchange.get("direction"),
                "is_reference_flow": exchange.get("is_reference_flow"),
                "reference_flow_name": exchange.get("reference_flow_name"),
                "general_comment": self._stringify_comment(exchange),
                "flow_type": exchange.get("flow_type"),
                "material_role": exchange.get("material_role"),
                "search_hints": exchange.get("search_hints") or [],
            },
            "candidates": [
                {
                    "index": idx,
                    "base_name": candidate.base_name,
                    "name_parts": {
                        "base_name": candidate.base_name,
                        "treatment_standards_routes": candidate.treatment_standards_routes,
                        "mix_and_location_types": candidate.mix_and_location_types,
                        "flow_properties": candidate.flow_properties,
                    },
                    "uuid": candidate.uuid,
                    "flow_type": candidate.flow_type,
                    "version": candidate.version,
                    "cas": candidate.cas,
                    "geography": candidate.geography,
                    "classification": candidate.classification,
                    "classification_path": self._classification_path(candidate.classification),
                    "category_path": candidate.category_path,
                    "general_comment": candidate.general_comment,
                    "flow_property_short_descriptions": self._flow_property_short_descriptions(candidate.flow_properties),
                }
                for idx, candidate in enumerate(candidates[:10])
            ],
        }
        return json.dumps(summary, ensure_ascii=False)

    @staticmethod
    def _stringify_comment(exchange: dict[str, Any]) -> str | None:
        comment = exchange.get("generalComment") or exchange.get("comment")
        if comment is None:
            return None
        if isinstance(comment, dict):
            text = comment.get("#text") or comment.get("text") or comment.get("@value")
            if text:
                return str(text)
        return str(comment)

    @staticmethod
    def _flow_property_short_descriptions(raw: Any) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, str):
            parts = [segment.strip() for segment in raw.replace("|", ";").split(";") if segment.strip()]
            if parts:
                return parts
            return [raw.strip()] if raw.strip() else []
        if isinstance(raw, (list, tuple, set)):
            collected: list[str] = []
            for item in raw:
                collected.extend(LLMCandidateSelector._flow_property_short_descriptions(item))
            return collected
        if isinstance(raw, dict):
            text_keys = ("#text", "text", "@value")
            for key in text_keys:
                if key in raw and isinstance(raw[key], str):
                    return LLMCandidateSelector._flow_property_short_descriptions(raw[key])
            collected: list[str] = []
            for value in raw.values():
                collected.extend(LLMCandidateSelector._flow_property_short_descriptions(value))
            return collected
        return [str(raw)]

    @staticmethod
    def _classification_path(classification: Any) -> list[str]:
        if not isinstance(classification, list):
            return []
        path: list[str] = []
        for item in classification:
            if not isinstance(item, dict):
                continue
            text = item.get("#text") or item.get("text")
            if isinstance(text, str) and text.strip():
                path.append(text.strip())
        return path

    @staticmethod
    def _parse_response(raw_response: Any) -> dict[str, Any]:
        if isinstance(raw_response, dict):
            return raw_response
        if not isinstance(raw_response, str):
            raw_response = str(raw_response)
        return parse_json_response(raw_response)

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
