"""Minimal helpers to serialise FlowSearch hints into ILCD-compatible comments."""

from __future__ import annotations

from typing import Any

REQUIRED_HINT_FIELDS: tuple[str, ...] = (
    "basename",
    "treatment",
    "mix_location",
    "flow_properties",
    "en_synonyms",
    "zh_synonyms",
    "abbreviation",
    "state_purity",
    "source_or_pathway",
    "usage_context",
)

OPTIONAL_HINT_FIELDS: tuple[str, ...] = ("formula_or_CAS",)

HINT_FIELDS: tuple[str, ...] = REQUIRED_HINT_FIELDS + OPTIONAL_HINT_FIELDS


def ensure_flow_hints_dict(exchange: dict[str, Any]) -> dict[str, str] | None:
    """Ensure `exchange["flowHints"]` exists by parsing generalComment if needed."""

    hints = exchange.get("flowHints")
    if isinstance(hints, dict) and hints:
        return hints
    existing_text = _extract_text(exchange.get("generalComment"))
    structured = _extract_structured_hints(exchange, existing_text)
    if any(structured.get(field) for field in REQUIRED_HINT_FIELDS):
        exchange["flowHints"] = structured
        return structured
    return None


def enrich_exchange_hints(
    exchange: dict[str, Any],
    *,
    process_name: str | None = None,  # kept for backward compatibility
    geography: str | None = None,  # kept for backward compatibility
) -> dict[str, str]:
    """Serialise FlowSearch hints into the legacy comment string."""

    existing_text = _extract_text(exchange.get("generalComment"))
    flow_hints_container: dict[str, Any] | None = None
    for key in ("flowHints", "hints"):
        candidate = exchange.get(key)
        if isinstance(candidate, dict):
            flow_hints_container = candidate
            break

    hints = _extract_structured_hints(exchange, existing_text)
    _ensure_geography_tags(hints, geography, flow_hints_container)
    formatted = _format_hints(hints)
    exchange["generalComment"] = {"@xml:lang": "en", "#text": formatted}
    return hints


def _extract_structured_hints(exchange: dict[str, Any], existing_text: str) -> dict[str, str]:
    flow_hints = exchange.get("flowHints") or exchange.get("hints")
    if isinstance(flow_hints, dict):
        return {field: _coerce_hint_value(flow_hints.get(field)) for field in HINT_FIELDS}

    parsed_fields, _ = _parse_existing_fields(existing_text)
    if parsed_fields:
        return {field: parsed_fields.get(field, "") for field in HINT_FIELDS}
    return {field: "" for field in HINT_FIELDS}


def _format_hints(hints: dict[str, str]) -> str:
    parts: list[str] = []
    for field in REQUIRED_HINT_FIELDS:
        parts.append(f"{field}={_safe_value(hints.get(field))}")
        if field == "abbreviation":
            optional_value = _safe_value(hints.get("formula_or_CAS"))
            if optional_value:
                parts.append(f"formula_or_CAS={optional_value}")
    return "FlowSearch hints: " + " | ".join(parts)


def _ensure_geography_tags(
    hints: dict[str, str],
    geography: str | None,
    flow_hints: dict[str, Any] | None,
) -> None:
    code = _normalise_geography_code(geography)
    if not code or code == "GLO":
        return
    for field in ("mix_location", "source_or_pathway"):
        current_value = hints.get(field, "")
        updated_value = _inject_geography_code(current_value, code)
        if updated_value == current_value:
            continue
        hints[field] = updated_value
        if isinstance(flow_hints, dict):
            flow_hints[field] = updated_value


def _inject_geography_code(value: Any, code: str) -> str:
    text = _stringify(value)
    if not text:
        return text
    code_upper = code.upper()
    if code_upper in text.upper():
        return text.strip()
    stripped = text.strip()
    if stripped.endswith(")") and "(" in stripped:
        start = stripped.rfind("(")
        inside = stripped[start + 1 : -1].strip()
        if inside and code_upper not in inside.upper():
            return f"{stripped[: start + 1]}{inside}, {code_upper})"
    return f"{stripped} ({code_upper})"


def _normalise_geography_code(geography: str | None) -> str | None:
    text = _stringify(geography).strip().upper()
    if not text:
        return None
    return text


def _parse_existing_fields(text: str | None) -> tuple[dict[str, str], str]:
    if not text:
        return {}, ""
    stripped = text.strip()
    if not stripped.startswith("FlowSearch hints:"):
        return {}, stripped

    body = stripped[len("FlowSearch hints:") :].strip()
    segments = [segment.strip() for segment in body.split("|")]
    fields: dict[str, str] = {}
    notes = ""
    for segment in segments:
        if "=" not in segment:
            continue
        key, value = segment.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in HINT_FIELDS:
            continue
        if key == "usage_context":
            usage_value, remainder = _separate_notes(value)
            fields[key] = usage_value
            if remainder:
                notes = remainder
        else:
            fields[key] = value

    if not notes:
        tail_marker = "usage_context="
        marker_index = body.find(tail_marker)
        if marker_index != -1:
            tail_text = body[marker_index + len(tail_marker) :].strip()
            _, remainder = _separate_notes(tail_text)
            if remainder:
                notes = remainder
    return fields, notes


def _separate_notes(value: str) -> tuple[str, str]:
    for separator in (". ", "; ", "。", "\n"):
        idx = value.find(separator)
        if idx != -1:
            usage = value[:idx].strip()
            remainder = value[idx + len(separator) :].strip()
            return usage, remainder
    return value.strip(), ""


def _safe_value(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip()


def _coerce_hint_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts = []
        for item in value:
            text = _stringify(item)
            if text:
                parts.append(text)
        return "; ".join(parts)
    if isinstance(value, dict):
        for key in ("#text", "@value", "text"):
            if key in value and value[key]:
                return _stringify(value[key])
        return ""
    return _stringify(value)


def _extract_text(value: Any) -> str:
    if isinstance(value, dict):
        if "#text" in value:
            return _stringify(value["#text"])
        if "_text" in value:
            return _stringify(value["_text"])
        return _stringify(value.get("text"))
    return _stringify(value)


def _stringify(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("#text", "@value", "text", "baseName", "shortName", "name"):
            if key in value:
                return _stringify(value[key])
        return ""
    if value is None:
        return ""
    return str(value).strip()
