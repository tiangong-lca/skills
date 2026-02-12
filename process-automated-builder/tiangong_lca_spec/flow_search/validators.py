"""Local validation heuristics for flow search results."""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from typing import Any

from tiangong_lca_spec.core.models import FlowCandidate, FlowQuery


def _normalize(text: str | None) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text).lower().strip()


def name_similarity_score(query: FlowQuery, candidate: dict[str, Any]) -> float:
    return SequenceMatcher(
        None,
        _normalize(query.exchange_name),
        _normalize(candidate.get("base_name") or candidate.get("name")),
    ).ratio()


def passes_similarity(query: FlowQuery, candidate: dict[str, Any], threshold: float = 0.65) -> bool:
    score = name_similarity_score(query, candidate)
    if score >= threshold:
        return True
    description = _normalize(getattr(query, "description", None))
    base_name = _normalize(candidate.get("base_name") or candidate.get("name"))
    if description and base_name and base_name in description:
        return True
    return False


def passes_location(candidate: dict[str, Any], expected: str | None = None) -> bool:
    if not expected:
        return True
    geo = candidate.get("geography") or {}
    code = _normalize(geo.get("code"))
    desc = _normalize(geo.get("description"))
    compare = _normalize(expected)
    return compare in {code, desc}


def hydrate_candidate(candidate: dict[str, Any]) -> FlowCandidate:
    return FlowCandidate(
        uuid=candidate.get("uuid") or candidate.get("flow_uuid"),
        base_name=candidate.get("base_name") or candidate.get("name", ""),
        treatment_standards_routes=candidate.get("treatment_standards_routes"),
        mix_and_location_types=candidate.get("mix_and_location_types"),
        flow_properties=candidate.get("flow_properties"),
        flow_type=candidate.get("flow_type") or candidate.get("flowType"),
        version=candidate.get("version"),
        general_comment=candidate.get("general_comment"),
        geography=candidate.get("geography"),
        classification=candidate.get("classification"),
        cas=candidate.get("cas"),
        category_path=candidate.get("category_path"),
        reasoning=candidate.get("reasoning", ""),
    )
