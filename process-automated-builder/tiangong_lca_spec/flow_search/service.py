"""Public service layer for flow search."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterable

from tiangong_lca_spec.core.config import Settings, get_settings
from tiangong_lca_spec.core.logging import get_logger
from tiangong_lca_spec.core.models import FlowCandidate, FlowQuery, UnmatchedFlow

from .client import FlowSearchClient
from .validators import hydrate_candidate

LOGGER = get_logger(__name__)

if TYPE_CHECKING:
    from tiangong_lca_spec.publishing.crud import DatabaseCrudClient

_CAS_PATTERN = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
_FLOW_TYPE_PATTERN = re.compile(r"\bflow_type\s*[:=]\s*([A-Za-z _-]+)")
_COMPARTMENT_PATTERN = re.compile(r"\bcompartment\s*[:=]\s*(air|water|soil)\b", flags=re.IGNORECASE)
_DIRECTION_PATTERN = re.compile(r"\bdirection\s*[:=]\s*([A-Za-z _-]+)", flags=re.IGNORECASE)
_UNIT_PATTERN = re.compile(r"\bunit\s*[:=]\s*([A-Za-z0-9._*/^-]+)", flags=re.IGNORECASE)
_SEARCH_HINTS_PATTERN = re.compile(r"search_hints\s*[:=]\s*([^|]+)", flags=re.IGNORECASE)
_MEDIA_SUFFIX_PATTERN = re.compile(r"\bto\s+(air|water|soil)\b", flags=re.IGNORECASE)
_MEDIA_TAIL_PATTERN = re.compile(r"\s*,?\s*(?:to|emission to)\s+(?:air|water|soil)\b.*$", flags=re.IGNORECASE)
_NON_WORD_PATTERN = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_SPACE_PATTERN = re.compile(r"\s+")
_FOR_QUALIFIER_PATTERN = re.compile(r"\bfor\s+[^,;|()]+$", flags=re.IGNORECASE)

_ELEMENTARY_FALLBACKS: dict[str, dict[str, str | tuple[str, ...]]] = {
    "methane": {"cas": "74-82-8", "aliases": ("CH4", "甲烷")},
    "carbon dioxide": {"cas": "124-38-9", "aliases": ("CO2", "二氧化碳")},
    "nitrous oxide": {"cas": "10024-97-2", "aliases": ("N2O", "氧化亚氮")},
}

_MEDIA_KEYWORDS: dict[str, tuple[str, ...]] = {
    "air": (" to air", "emission to air", "emissions to air", "排放到大气", "排放到空气", "空气", "大气"),
    "water": (" to water", "emission to water", "emissions to water", "排放到水", "水体", "地表水", "地下水"),
    "soil": (" to soil", "emission to soil", "emissions to soil", "排放到土壤", "土壤"),
}

_PRODUCT_SYNONYM_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(r"\bdrinking water\b", flags=re.IGNORECASE),
        ("tap water", "water supply", "potable water"),
    ),
    (
        re.compile(r"\bpotable water\b", flags=re.IGNORECASE),
        ("tap water", "drinking water", "water supply"),
    ),
    (
        re.compile(r"\bmunicipal water\b", flags=re.IGNORECASE),
        ("tap water", "drinking water", "water supply"),
    ),
    (
        re.compile(r"\belectricity\b", flags=re.IGNORECASE),
        ("power, electric",),
    ),
)


@dataclass(slots=True)
class ElementaryQueryContext:
    is_elementary: bool
    core_name: str
    cas: str | None
    compartment: str | None
    source_qualifier: str | None
    aliases: tuple[str, ...]


@dataclass(slots=True)
class QueryRewriteContext:
    expected_flow_type: str | None
    expected_compartment: str | None
    expected_direction: str | None
    expected_unit: str | None
    core_name: str
    aliases: tuple[str, ...]


class FlowSearchService:
    """High-level facade responsible for flow lookup and validation."""

    def __init__(self, settings: Settings | None = None, *, client: FlowSearchClient | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = client or FlowSearchClient(self._settings)
        self._state_code_filter = self._settings.flow_search_state_code
        self._crud: DatabaseCrudClient | None = None
        if self._state_code_filter is not None:
            from tiangong_lca_spec.publishing.crud import DatabaseCrudClient

            self._crud = DatabaseCrudClient(self._settings)
        self._state_code_cache: dict[str, bool] = {}

    def lookup(self, query: FlowQuery) -> tuple[list[FlowCandidate], list[UnmatchedFlow]]:
        LOGGER.info("flow_search.lookup", exchange=query.exchange_name)
        elementary_context = _analyze_elementary_query(query)
        rewrite_context = _analyze_query_rewrite_context(query)

        if elementary_context.is_elementary:
            raw_candidates = self._search_elementary_candidates(query, elementary_context)
        else:
            primary_query = FlowQuery(
                exchange_name=query.exchange_name,
                description=query.description,
            )
            raw_candidates = self._client.search(primary_query)
            base_matches, _ = self._normalize_candidates(query, raw_candidates)
            if _needs_non_elementary_variant_search(base_matches, rewrite_context):
                raw_candidates = self._search_non_elementary_candidates(
                    query,
                    context=rewrite_context,
                    baseline=raw_candidates,
                )

        matches, filtered_out = self._normalize_candidates(query, raw_candidates)
        if elementary_context.is_elementary:
            matches = _rank_elementary_candidates(matches, elementary_context)
        else:
            matches = _rank_non_elementary_candidates(matches, rewrite_context)
        if not matches:
            if query.exchange_name:
                LOGGER.info(
                    "flow_search.retry_name_only",
                    exchange=query.exchange_name,
                )
                if elementary_context.is_elementary:
                    name_only_query = FlowQuery(exchange_name=query.exchange_name)
                    retry_context = _analyze_elementary_query(name_only_query)
                    fallback_raw = self._search_elementary_candidates(name_only_query, retry_context)
                else:
                    name_only_query = FlowQuery(exchange_name=query.exchange_name)
                    fallback_context = _analyze_query_rewrite_context(name_only_query)
                    base_fallback_raw = self._client.search(name_only_query)
                    base_fallback_matches, _ = self._normalize_candidates(query, base_fallback_raw)
                    if _needs_non_elementary_variant_search(base_fallback_matches, fallback_context):
                        fallback_raw = self._search_non_elementary_candidates(
                            name_only_query,
                            context=fallback_context,
                            baseline=base_fallback_raw,
                        )
                        if not fallback_raw:
                            fallback_raw = base_fallback_raw
                    else:
                        fallback_raw = base_fallback_raw
                fallback_matches, fallback_filtered = self._normalize_candidates(query, fallback_raw)
                if elementary_context.is_elementary:
                    fallback_matches = _rank_elementary_candidates(fallback_matches, elementary_context)
                else:
                    fallback_matches = _rank_non_elementary_candidates(fallback_matches, rewrite_context)
                if fallback_matches:
                    matches.extend(fallback_matches)
                if fallback_filtered:
                    filtered_out.extend(fallback_filtered)
        if matches:
            return matches, filtered_out
        unmatched = UnmatchedFlow(
            base_name=query.exchange_name,
            general_comment=query.description,
        )
        return [], filtered_out + [unmatched]

    def _search_elementary_candidates(self, query: FlowQuery, context: ElementaryQueryContext) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        query_texts = _build_elementary_query_texts(query, context)
        for query_text in query_texts:
            try:
                candidates = self._client.search_query_text(query_text)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning(
                    "flow_search.elementary_variant_failed",
                    exchange=query.exchange_name,
                    query_text=query_text,
                    error=str(exc),
                )
                continue
            if candidates:
                LOGGER.info(
                    "flow_search.elementary_variant_hit",
                    exchange=query.exchange_name,
                    query_text=query_text,
                    candidate_count=len(candidates),
                )
            merged.extend(candidates)
        return merged

    def _search_non_elementary_candidates(
        self,
        query: FlowQuery,
        *,
        context: QueryRewriteContext,
        baseline: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        if baseline:
            merged.extend(baseline)
        query_texts = _build_non_elementary_query_texts(query, context)
        for query_text in query_texts:
            try:
                candidates = self._client.search_query_text(query_text)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning(
                    "flow_search.variant_failed",
                    exchange=query.exchange_name,
                    query_text=query_text,
                    error=str(exc),
                )
                continue
            if candidates:
                LOGGER.info(
                    "flow_search.variant_hit",
                    exchange=query.exchange_name,
                    query_text=query_text,
                    candidate_count=len(candidates),
                )
            merged.extend(candidates)
        return merged

    def _normalize_candidates(self, query: FlowQuery, payload: Iterable[dict]) -> tuple[list[FlowCandidate], list[UnmatchedFlow]]:
        _ = query
        candidates: list[FlowCandidate] = []
        for item in payload or []:
            flow_uuid = str(item.get("uuid") or item.get("flow_uuid") or "").strip()
            if flow_uuid and not self._passes_state_code(flow_uuid):
                continue
            candidates.append(hydrate_candidate(item))
        return candidates, []

    def close(self) -> None:
        self._client.close()
        if self._crud:
            self._crud.close()

    def _passes_state_code(self, flow_uuid: str) -> bool:
        if self._state_code_filter is None or not self._crud:
            return True
        cached = self._state_code_cache.get(flow_uuid)
        if cached is not None:
            return cached
        record = None
        try:
            record = self._crud.select_flow_record(flow_uuid)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning(
                "flow_search.state_code_lookup_failed",
                uuid=flow_uuid,
                error=str(exc),
            )
            # Fail-open: keep the searched candidate when state-code lookup fails.
            # This avoids converting potentially valid matched flows into placeholders
            # just because the CRUD sidecar call is temporarily unavailable.
            self._state_code_cache[flow_uuid] = True
            return True
        state_code = record.get("state_code") if isinstance(record, dict) else None
        matches = state_code == self._state_code_filter
        self._state_code_cache[flow_uuid] = matches
        if not matches:
            LOGGER.info(
                "flow_search.state_code_filtered",
                uuid=flow_uuid,
                state_code=state_code,
                required=self._state_code_filter,
            )
        return matches


def _build_elementary_query_texts(query: FlowQuery, context: ElementaryQueryContext) -> list[str]:
    description = str(query.description or "").strip() or None
    query_texts: list[str] = []
    name_candidates: list[str] = []
    for value in (query.exchange_name, context.core_name, *context.aliases):
        text = str(value or "").strip()
        if text and text not in name_candidates:
            name_candidates.append(text)
    if context.source_qualifier and context.core_name:
        qualified_name = f"{context.core_name}, {context.source_qualifier}"
        if qualified_name not in name_candidates:
            name_candidates.append(qualified_name)

    preferred_name = context.core_name or query.exchange_name
    if context.cas:
        query_texts.append(
            _compose_search_query(
                flow_name=preferred_name,
                description=description,
                cas=context.cas,
                compartment=context.compartment,
            )
        )
    for name in name_candidates:
        query_texts.append(
            _compose_search_query(
                flow_name=name,
                description=description,
                compartment=context.compartment,
            )
        )
    if context.cas:
        query_texts.append(
            _compose_search_query(
                flow_name=context.cas,
                description="flow_type: elementary flow",
                cas=context.cas,
                compartment=context.compartment,
            )
        )
    deduped: list[str] = []
    for item in query_texts:
        cleaned = item.strip()
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return deduped[:8]


def _build_non_elementary_query_texts(query: FlowQuery, context: QueryRewriteContext) -> list[str]:
    original_description = str(query.description or "").strip() or None
    constraint_description = _build_constraint_description(context)
    name_candidates = _build_query_name_candidates(query.exchange_name, context)

    query_texts: list[str] = []
    for index, name in enumerate(name_candidates):
        # Keep the original rich context for the first variant only.
        description = original_description if index == 0 else constraint_description
        query_texts.append(
            _compose_non_elementary_search_query(
                flow_name=name,
                description=description,
                expected_flow_type=context.expected_flow_type,
                direction=context.expected_direction,
                unit=context.expected_unit,
                compartment=context.expected_compartment,
            )
        )
        # Add a concise variant that removes noisy context terms.
        if constraint_description:
            query_texts.append(
                _compose_non_elementary_search_query(
                    flow_name=name,
                    description=constraint_description,
                    expected_flow_type=context.expected_flow_type,
                    direction=context.expected_direction,
                    unit=context.expected_unit,
                    compartment=context.expected_compartment,
                )
            )

    deduped: list[str] = []
    for item in query_texts:
        cleaned = item.strip()
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return deduped[:8]


def _build_constraint_description(context: QueryRewriteContext) -> str | None:
    bits: list[str] = []
    if context.expected_flow_type:
        bits.append(f"flow_type={context.expected_flow_type}")
    if context.expected_direction:
        bits.append(f"direction={context.expected_direction}")
    if context.expected_unit:
        bits.append(f"unit={context.expected_unit}")
    if context.expected_compartment:
        bits.append(f"compartment={context.expected_compartment}")
    if not bits:
        return None
    return f"constraints: {'; '.join(bits)}"


def _build_query_name_candidates(exchange_name: str, context: QueryRewriteContext) -> list[str]:
    seeds: list[str] = []

    def add_seed(value: str | None) -> None:
        text = str(value or "").strip()
        if text and text not in seeds:
            seeds.append(text)

    add_seed(exchange_name)
    add_seed(context.core_name)
    for alias in context.aliases:
        add_seed(alias)

    expanded: list[str] = []

    def add_expanded(value: str | None) -> None:
        text = str(value or "").strip()
        if text and text not in expanded:
            expanded.append(text)

    for seed in seeds:
        add_expanded(seed)
        stripped = _strip_for_qualifier(seed)
        add_expanded(stripped)
        for pattern, replacements in _PRODUCT_SYNONYM_RULES:
            if not pattern.search(seed):
                continue
            for replacement in replacements:
                add_expanded(replacement)
                add_expanded(pattern.sub(replacement, seed))

    return expanded


def _compose_non_elementary_search_query(
    *,
    flow_name: str,
    description: str | None = None,
    expected_flow_type: str | None = None,
    direction: str | None = None,
    unit: str | None = None,
    compartment: str | None = None,
) -> str:
    parts: list[str] = []
    if flow_name:
        parts.append(f"flow_name: {flow_name}")
        parts.append(f"exchange: {flow_name}")
    if description:
        parts.append(f"description: {description}")
    if expected_flow_type:
        parts.append(f"flow_type: {expected_flow_type}")
    if direction:
        parts.append(f"direction: {direction}")
    if unit:
        parts.append(f"unit: {unit}")
    if compartment:
        parts.append(f"compartment: {compartment}")
    return " \n".join(parts)


def _compose_search_query(
    *,
    flow_name: str,
    description: str | None = None,
    cas: str | None = None,
    compartment: str | None = None,
) -> str:
    parts: list[str] = []
    if flow_name:
        parts.append(f"flow_name: {flow_name}")
        parts.append(f"exchange: {flow_name}")
    if description:
        parts.append(f"description: {description}")
    parts.append("flow_type: elementary flow")
    if cas:
        parts.append(f"cas: {cas}")
    if compartment:
        parts.append(f"compartment: {compartment}")
    return " \n".join(parts)


def _analyze_elementary_query(query: FlowQuery) -> ElementaryQueryContext:
    exchange_name = str(query.exchange_name or "").strip()
    description = str(query.description or "").strip()
    combined = f"{exchange_name} {description}".strip()
    normalized = combined.lower()

    flow_type_match = _FLOW_TYPE_PATTERN.search(description) or _FLOW_TYPE_PATTERN.search(exchange_name)
    flow_type = _normalize_flow_type(flow_type_match.group(1)) if flow_type_match else None

    compartment = _extract_compartment(description) or _extract_compartment(exchange_name)
    is_elementary = flow_type == "elementary"
    if flow_type is None:
        if compartment:
            is_elementary = True
        if not is_elementary and ("elementary" in normalized or "emission" in normalized or "resource" in normalized):
            is_elementary = True

    core_name = _strip_media_suffix(exchange_name)
    source_qualifier = _extract_source_qualifier(combined)
    cas = _extract_cas(combined)
    aliases = _extract_search_hints(description)

    lower_core = core_name.lower()
    for token, fallback in _ELEMENTARY_FALLBACKS.items():
        if token not in lower_core:
            continue
        fallback_aliases = fallback.get("aliases")
        if isinstance(fallback_aliases, tuple):
            for alias in fallback_aliases:
                alias_text = str(alias).strip()
                if alias_text and alias_text not in aliases:
                    aliases.append(alias_text)
        fallback_cas = fallback.get("cas")
        if not cas and isinstance(fallback_cas, str) and fallback_cas.strip():
            cas = fallback_cas.strip()

    return ElementaryQueryContext(
        is_elementary=is_elementary,
        core_name=core_name,
        cas=cas,
        compartment=compartment,
        source_qualifier=source_qualifier,
        aliases=tuple(aliases),
    )


def _analyze_query_rewrite_context(query: FlowQuery) -> QueryRewriteContext:
    exchange_name = str(query.exchange_name or "").strip()
    description = str(query.description or "").strip()
    flow_type_match = _FLOW_TYPE_PATTERN.search(description) or _FLOW_TYPE_PATTERN.search(exchange_name)
    expected_flow_type = _normalize_flow_type(flow_type_match.group(1)) if flow_type_match else None
    expected_compartment = _extract_compartment(description) or _extract_compartment(exchange_name)
    expected_direction = _extract_direction(description)
    expected_unit = _extract_unit(description)

    aliases: list[str] = []
    for value in _extract_search_hints(description):
        item = str(value or "").strip()
        if item and item not in aliases:
            aliases.append(item)

    core_name = _strip_for_qualifier(_strip_media_suffix(exchange_name))
    if core_name and core_name != exchange_name:
        aliases.append(core_name)

    for pattern, replacements in _PRODUCT_SYNONYM_RULES:
        if not pattern.search(core_name):
            continue
        for replacement in replacements:
            text = str(replacement or "").strip()
            if text and text not in aliases:
                aliases.append(text)

    deduped_aliases: list[str] = []
    for alias in aliases:
        if alias and alias not in deduped_aliases:
            deduped_aliases.append(alias)

    return QueryRewriteContext(
        expected_flow_type=expected_flow_type,
        expected_compartment=expected_compartment,
        expected_direction=expected_direction,
        expected_unit=expected_unit,
        core_name=core_name or exchange_name,
        aliases=tuple(deduped_aliases),
    )


def _extract_source_qualifier(text: str) -> str | None:
    lowered = text.lower()
    if "fossil" in lowered:
        return "fossil"
    if "biogenic" in lowered or "bio " in lowered or lowered.endswith(" bio") or "(bio)" in lowered:
        return "biogenic"
    return None


def _extract_cas(text: str) -> str | None:
    match = _CAS_PATTERN.search(text)
    if not match:
        return None
    return match.group(0)


def _extract_compartment(text: str) -> str | None:
    if not text:
        return None
    explicit = _COMPARTMENT_PATTERN.search(text)
    if explicit:
        return explicit.group(1).lower()
    suffix = _MEDIA_SUFFIX_PATTERN.search(text)
    if suffix:
        return suffix.group(1).lower()
    lowered = text.lower()
    for key, keywords in _MEDIA_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return key
    return None


def _extract_search_hints(description: str) -> list[str]:
    if not description:
        return []
    match = _SEARCH_HINTS_PATTERN.search(description)
    if not match:
        return []
    raw = match.group(1)
    parts = re.split(r"[;,|]", raw)
    hints: list[str] = []
    for item in parts:
        cleaned = str(item).strip()
        if not cleaned or "=" in cleaned:
            continue
        if cleaned not in hints:
            hints.append(cleaned)
    return hints


def _extract_direction(text: str) -> str | None:
    if not text:
        return None
    match = _DIRECTION_PATTERN.search(text)
    if not match:
        return None
    value = str(match.group(1) or "").strip().lower()
    if not value:
        return None
    if value.startswith("in"):
        return "input"
    if value.startswith("out"):
        return "output"
    return value


def _extract_unit(text: str) -> str | None:
    if not text:
        return None
    match = _UNIT_PATTERN.search(text)
    if not match:
        return None
    value = str(match.group(1) or "").strip()
    return value or None


def _strip_media_suffix(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    stripped = _MEDIA_TAIL_PATTERN.sub("", text)
    return stripped.strip(" ,;") or text


def _strip_for_qualifier(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    stripped = _FOR_QUALIFIER_PATTERN.sub("", value).strip(" ,;")
    return stripped or value


def _normalize_flow_type(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if "elementary" in text or "emission" in text or "resource" in text:
        return "elementary"
    if "product" in text:
        return "product"
    if "waste" in text:
        return "waste"
    if "service" in text:
        return "service"
    return None


def _needs_non_elementary_variant_search(candidates: list[FlowCandidate], context: QueryRewriteContext) -> bool:
    if not candidates:
        return True
    for candidate in candidates:
        score = _score_non_elementary_candidate(candidate, context)
        if score is not None and score >= 11.0:
            return False
    return True


def _rank_non_elementary_candidates(candidates: list[FlowCandidate], context: QueryRewriteContext) -> list[FlowCandidate]:
    scored: list[tuple[float, int, FlowCandidate]] = []
    for index, candidate in enumerate(candidates):
        score = _score_non_elementary_candidate(candidate, context)
        if score is None:
            continue
        scored.append((score, index, candidate))

    if not scored:
        return candidates

    best_by_key: dict[tuple[str, str], tuple[float, int, FlowCandidate]] = {}
    for score, index, candidate in scored:
        key = ((candidate.uuid or "").strip(), str(candidate.version or "").strip())
        existing = best_by_key.get(key)
        if existing is None or score > existing[0]:
            best_by_key[key] = (score, index, candidate)

    ranked = sorted(best_by_key.values(), key=lambda item: (-item[0], item[1]))
    ranked_candidates = [item[2] for item in ranked]
    if not ranked_candidates:
        return candidates
    return ranked_candidates


def _score_non_elementary_candidate(candidate: FlowCandidate, context: QueryRewriteContext) -> float | None:
    candidate_type = _normalize_flow_type(candidate.flow_type)
    expected_type = _normalize_flow_type(context.expected_flow_type)
    if expected_type and candidate_type and candidate_type != expected_type:
        return None

    target_name = context.core_name or ""
    aliases = list(context.aliases)
    name_score = _score_name_alignment(
        query_name=target_name,
        aliases=aliases,
        candidate_name=candidate.base_name,
    )
    if name_score <= 0:
        # Keep fail-open behavior for low-signal queries when aliases are absent.
        if not target_name and not aliases:
            name_score = 1.0
        else:
            return None

    score = name_score
    if expected_type and candidate_type and expected_type == candidate_type:
        score += 4.0
    if expected_type == "product" and candidate_type == "product":
        score += 2.0

    expected_compartment = context.expected_compartment
    candidate_compartment = _candidate_compartment(candidate)
    if expected_compartment and candidate_compartment:
        if expected_compartment == candidate_compartment:
            score += 2.5
        elif expected_type == "elementary":
            return None
        else:
            score -= 1.0

    unit_token = _normalize_name(context.expected_unit or "")
    if unit_token and _contains_word(_normalize_name(candidate.flow_properties or ""), unit_token):
        score += 1.5
    return score


def _rank_elementary_candidates(candidates: list[FlowCandidate], context: ElementaryQueryContext) -> list[FlowCandidate]:
    scored: list[tuple[float, int, FlowCandidate]] = []
    for index, candidate in enumerate(candidates):
        score = _score_elementary_candidate(candidate, context)
        if score is None:
            continue
        scored.append((score, index, candidate))
    if not scored:
        return []

    best_by_key: dict[tuple[str, str], tuple[float, int, FlowCandidate]] = {}
    for score, index, candidate in scored:
        key = ((candidate.uuid or "").strip(), str(candidate.version or "").strip())
        existing = best_by_key.get(key)
        if existing is None or score > existing[0]:
            best_by_key[key] = (score, index, candidate)

    ranked = sorted(best_by_key.values(), key=lambda item: (-item[0], item[1]))
    return [item[2] for item in ranked]


def _score_elementary_candidate(candidate: FlowCandidate, context: ElementaryQueryContext) -> float | None:
    candidate_type = _normalize_flow_type(candidate.flow_type)
    if candidate_type and candidate_type != "elementary":
        return None

    query_cas = _normalize_cas(context.cas)
    candidate_cas = _normalize_cas(candidate.cas)
    if query_cas and candidate_cas and query_cas != candidate_cas:
        return None

    query_compartment = context.compartment
    candidate_compartment = _candidate_compartment(candidate)
    if query_compartment and candidate_compartment and query_compartment != candidate_compartment:
        return None

    name_score = _score_name_alignment(
        query_name=context.core_name,
        aliases=list(context.aliases),
        candidate_name=candidate.base_name,
    )
    if name_score <= 0:
        return None

    score = name_score
    if candidate_type == "elementary":
        score += 8.0
    if query_cas and candidate_cas and query_cas == candidate_cas:
        score += 28.0
    if query_compartment and candidate_compartment and query_compartment == candidate_compartment:
        score += 6.0
    if context.source_qualifier:
        candidate_text = " ".join(
            [
                str(candidate.base_name or ""),
                str(candidate.general_comment or ""),
                str(candidate.category_path or ""),
            ]
        ).lower()
        if context.source_qualifier in candidate_text:
            score += 3.0
    return score


def _score_name_alignment(*, query_name: str, aliases: list[str], candidate_name: str) -> float:
    candidate_norm = _normalize_name(candidate_name)
    if not candidate_norm:
        return 0.0

    targets: list[str] = []
    for value in [query_name, *aliases]:
        normalized = _normalize_name(value)
        if normalized and normalized not in targets:
            targets.append(normalized)
    if not targets:
        return 0.0

    best = 0.0
    for target in targets:
        if candidate_norm == target:
            best = max(best, 20.0)
            continue
        if candidate_norm.startswith(f"{target},") or candidate_norm.startswith(f"{target} ") or candidate_norm.startswith(f"{target}("):
            best = max(best, 16.0)
            continue
        if _contains_word(candidate_norm, target):
            token_gap = len(candidate_norm.split()) - len(target.split())
            if token_gap <= 2:
                best = max(best, 11.0)
            else:
                best = max(best, 2.0)
        similarity = SequenceMatcher(None, target, candidate_norm).ratio()
        if similarity >= 0.92:
            best = max(best, 10.0)
        elif similarity >= 0.8:
            best = max(best, 6.0)
        elif similarity >= 0.7:
            best = max(best, 3.0)

    query_norm = _normalize_name(query_name)
    if query_norm and _contains_word(candidate_norm, query_norm) and not candidate_norm.startswith(query_norm):
        if len(candidate_norm.split()) > len(query_norm.split()) + 2:
            best -= 5.0
    return best


def _contains_word(text: str, term: str) -> bool:
    pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _normalize_name(text: str) -> str:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return ""
    cleaned = _NON_WORD_PATTERN.sub(" ", lowered)
    return _SPACE_PATTERN.sub(" ", cleaned).strip()


def _normalize_cas(text: str | None) -> str | None:
    if not text:
        return None
    match = _CAS_PATTERN.search(str(text))
    if not match:
        return None
    return match.group(0)


def _candidate_compartment(candidate: FlowCandidate) -> str | None:
    parts: list[str] = []
    if candidate.category_path:
        parts.append(str(candidate.category_path))
    if candidate.classification:
        for item in candidate.classification:
            if not isinstance(item, dict):
                continue
            text = item.get("#text") or item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    if candidate.general_comment:
        parts.append(str(candidate.general_comment))
    return _extract_compartment(" ".join(parts))


@lru_cache(maxsize=512)
def _cached_search(query: FlowQuery) -> tuple[tuple[FlowCandidate, ...], tuple[UnmatchedFlow, ...]]:
    service = FlowSearchService()
    try:
        matches, unmatched = service.lookup(query)
    finally:
        service.close()
    return tuple(matches), tuple(unmatched)


def search_flows(query: FlowQuery) -> tuple[list[FlowCandidate], list[UnmatchedFlow]]:
    """Cached flow search, ready for pipeline usage."""
    matches, unmatched = _cached_search(query)
    return list(matches), list(unmatched)
