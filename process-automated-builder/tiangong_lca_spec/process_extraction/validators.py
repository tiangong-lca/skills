"""Strict validators for Stage 2 exchange outputs."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Sequence

FORBIDDEN_VALUES = {
    "",
    "-",
    "na",
    "n/a",
    "unspecified",
    "tbd",
    "glo",
    "global",
    "cn",
}

SHORT_ACRONYM_LIMIT = 3

REQUIRED_HINT_FIELDS: tuple[str, ...] = (
    "basename",
    "treatment",
    "mix_location",
    "source_or_pathway",
    "en_synonyms",
    "state_purity",
    "flow_properties",
    "usage_context",
)

LCIA_HARD_NAME_KEYWORDS: tuple[str, ...] = (
    "potential",
    "impact",
    "depletion",
    "abiotic",
    "photochemical",
    "acidification",
    "eutrophication",
    "respiratory",
    "primary energy",
    "cumulative energy",
    "energy demand",
    "energy depletion",
    "gwp",
    "adp",
    "ep ",
    "ped",
    "ri",
    "cadp",
)

LCIA_RISK_NAME_KEYWORDS: tuple[str, ...] = (
    "life cycle",
    "scenario",
    "system-level",
    "system ",
)

LCIA_UNIT_TOKENS: tuple[str, ...] = (
    "kgco2eq",
    "kgco2e",
    "kgco2equivalent",
    "kgso2eq",
    "kgso2e",
    "kgso2equivalent",
    "kgsbeq",
    "kgantimonyeq",
    "kgpo43eq",
    "kgpo4eq",
    "kgpm25eq",
    "kgcoaleq",
    "kgch4eq",
    "kgnoxeq",
    "kgno2eq",
    "kgnh3eq",
)

ENERGY_IMPACT_TERMS: tuple[str, ...] = (
    "primary energy",
    "cumulative energy",
    "energy depletion",
    "energy demand",
    "ped",
)


def validate_exchanges_strict(
    exchanges: Sequence[dict[str, Any]],
    *,
    geography: str | None = None,
) -> list[str]:
    """Return a list of validation errors for the provided exchanges."""

    errors: list[str] = []
    geography_code = (geography or "").strip()
    geography_upper = geography_code.upper()

    if not exchanges:
        return ["Process must define at least one `exchanges.exchange` entry with quantitative LCI data."]

    for index, exchange in enumerate(exchanges, start=1):
        prefix = f"exchange #{index}"
        name = _coerce_str(exchange.get("exchangeName"))
        if not name:
            errors.append(f"{prefix}: `exchangeName` is required.")
            continue
        if _is_placeholder(name):
            errors.append(f"{prefix}: `exchangeName` uses placeholder value '{name}'.")
        unit_text = _coerce_str(exchange.get("unit"))
        if _has_lcia_signature(name, unit_text):
            errors.append(f"{prefix} ({name}): LCIA indicator detected (unit '{unit_text}'). " "Stage 2 must only emit physical LCI flows, not impact scores.")
            continue
        hints = _extract_flow_hints(exchange)
        if hints is None:
            errors.append(f"{prefix} ({name}): missing `flowHints` object with required fields.")
            continue

        basename = hints.get("basename", "")
        if not basename or _is_placeholder(basename):
            errors.append(f"{prefix} ({name}): `basename` must spell out the full flow name (e.g., 'Liquid nitrogen').")
        elif not _names_consistent(name, basename):
            errors.append(f"{prefix} ({name}): `basename` ('{basename}') must match or be a more formal version of `exchangeName`.")

        for field in REQUIRED_HINT_FIELDS:
            value = hints.get(field)
            if value is None:
                errors.append(f"{prefix} ({name}): missing `{field}`.")
                continue
            if field == "en_synonyms":
                _validate_synonyms(value, prefix, name, errors, basename)
                continue
            value_str = _coerce_str(value)
            if not value_str:
                errors.append(f"{prefix} ({name}): `{field}` must be a non-empty string.")
            elif _is_placeholder(value_str):
                errors.append(f"{prefix} ({name}): `{field}` uses placeholder value '{value_str}'.")
            elif field == "mix_location":
                _validate_mix_location(value_str, prefix, name, errors, geography_upper)
            elif field == "source_or_pathway":
                _validate_source(value_str, prefix, name, errors, geography_upper)

    return errors


def _has_lcia_signature(name: str, unit: str) -> bool:
    if not name or not unit:
        return False
    name_lower = name.lower()
    has_hard = any(keyword in name_lower for keyword in LCIA_HARD_NAME_KEYWORDS)
    has_risk = any(keyword in name_lower for keyword in LCIA_RISK_NAME_KEYWORDS)
    if not (has_hard or has_risk):
        return False
    normalized_unit = _normalize_unit(unit)
    if not normalized_unit:
        return False

    if any(normalized_unit.startswith(token) for token in LCIA_UNIT_TOKENS):
        return True

    # If only a soft/risk keyword is present, require an LCIA-like unit to proceed.
    if not has_hard:
        return False

    if normalized_unit in {"mj", "mjperfunctionalunit", "mjperfu"} or (normalized_unit.startswith("mjper") and "functionalunit" in normalized_unit):
        return any(term in name_lower for term in ENERGY_IMPACT_TERMS)

    return False


def _normalize_unit(unit: str) -> str:
    if not unit:
        return ""
    normalized = unicodedata.normalize("NFKD", unit)
    lowered = normalized.lower()
    return re.sub(r"[^a-z0-9]", "", lowered)


def _extract_flow_hints(exchange: dict[str, Any]) -> dict[str, Any] | None:
    hints = exchange.get("flowHints") or exchange.get("hints")
    if isinstance(hints, dict):
        return hints
    return None


def _validate_synonyms(
    value: Any,
    prefix: str,
    name: str,
    errors: list[str],
    basename: str,
) -> None:
    if isinstance(value, list):
        synonyms = [_coerce_str(item) for item in value if _coerce_str(item)]
    elif isinstance(value, str):
        if ";" in value:
            parts = value.split(";")
        else:
            parts = value.split(",")
        synonyms = [part.strip() for part in parts if part.strip()]
    else:
        synonyms = []
    if not synonyms:
        errors.append(f"{prefix} ({name}): `en_synonyms` must list at least one synonym.")
        return
    if _is_placeholder(synonyms[0]) or (basename and not _names_consistent(basename, synonyms[0])):
        errors.append(f"{prefix} ({name}): first entry of `en_synonyms` must repeat the full flow name (e.g., '{basename}').")
    for idx, synonym in enumerate(synonyms):
        if idx == 0:
            continue
        if synonym.strip().lower() in {"", "-", "na", "n/a", "tbd", "unspecified"}:
            errors.append(f"{prefix} ({name}): `en_synonyms` contains placeholder '{synonym}'.")


def _validate_mix_location(
    value: str,
    prefix: str,
    name: str,
    errors: list[str],
    geography_upper: str,
) -> None:
    if geography_upper and geography_upper not in {"", "GLO"} and geography_upper not in value.upper():
        errors.append(f"{prefix} ({name}): `mix_location` ('{value}') must reference the geography code ({geography_upper}).")


def _validate_source(
    value: str,
    prefix: str,
    name: str,
    errors: list[str],
    geography_upper: str,
) -> None:
    if geography_upper and geography_upper not in {"", "GLO"} and geography_upper not in value.upper():
        errors.append(f"{prefix} ({name}): `source_or_pathway` ('{value}') must mention the geography ({geography_upper}).")


def _names_consistent(primary: str, secondary: str) -> bool:
    p = primary.strip().lower()
    s = secondary.strip().lower()
    return p == s or p in s or s in p


def _coerce_str(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _is_placeholder(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    if not lowered:
        return True
    if lowered in FORBIDDEN_VALUES:
        return True
    token = stripped.replace("-", "")
    if len(token) <= SHORT_ACRONYM_LIMIT and token.upper() == token and token.isalnum():
        return True
    return False


def is_placeholder_value(text: str) -> bool:
    """Public helper exposing the placeholder check for reuse across stages."""

    if not isinstance(text, str):
        return True
    return _is_placeholder(text)
