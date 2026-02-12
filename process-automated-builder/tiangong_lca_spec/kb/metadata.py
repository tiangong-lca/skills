"""Metadata helpers for knowledge base ingestion."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .config import MetadataFieldDefinition


def build_metadata_entries(
    record: Mapping[str, Any],
    metadata_ids: Mapping[str, str],
    definitions: Iterable[MetadataFieldDefinition],
) -> list[dict[str, str]]:
    """Return the metadata entries to attach for a RIS record."""
    entries: list[dict[str, str]] = []
    for definition in definitions:
        if definition.name not in metadata_ids:
            continue
        value = definition.render_value(record)
        if not value:
            continue
        entries.append({"id": metadata_ids[definition.name], "name": definition.name, "value": value})
    return entries


def format_citation(record: Mapping[str, Any]) -> str:
    """Compose a single citation-style string from RIS fields."""
    authors = _stringify_sequence(record.get("authors"))
    year = _extract_year(record)
    title = _first_nonempty(record.get("title"), record.get("primary_title"))
    journal = _first_nonempty(
        record.get("secondary_title"),
        record.get("alternate_title1"),
        record.get("journal_name"),
        record.get("journal_name_full"),
        record.get("container_title"),
    )
    volume = _coerce_str(record.get("volume"))
    issue = _first_nonempty(record.get("issue"), record.get("number"))
    pages = _format_page_range(record)
    doi = _coerce_str(record.get("doi"))
    url = _stringify_sequence(record.get("urls"), first_only=True)

    segments: list[str] = []
    if authors and year:
        segments.append(f"{authors} ({year})")
    elif authors:
        segments.append(authors)
    elif year:
        segments.append(f"({year})")
    if title:
        segments.append(title)
    journal_parts = [journal] if journal else []
    if volume:
        journal_parts.append(f"vol. {volume}")
    if issue:
        journal_parts.append(f"no. {issue}")
    if pages:
        journal_parts.append(pages)
    if journal_parts:
        segments.append(", ".join(journal_parts))

    citation = ". ".join(segment for segment in segments if segment)
    tail: list[str] = []
    if doi:
        tail.append(f"DOI: {doi}")
    if url:
        tail.append(f"URL: {url}")
    if tail:
        citation = f"{citation}. {' '.join(tail)}" if citation else " ".join(tail)
    return citation.strip()


def _stringify_sequence(value: Any, *, first_only: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if item not in (None, "")]
        if not items:
            return ""
        return items[0] if first_only else "; ".join(items)
    return str(value).strip()


def _first_nonempty(*candidates: Any) -> str:
    for candidate in candidates:
        text = _coerce_str(candidate)
        if text:
            return text
    return ""


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _extract_year(record: Mapping[str, Any]) -> str:
    year = record.get("year")
    if isinstance(year, (int, float)):
        return str(int(year))
    if isinstance(year, str):
        text = year.strip()
        if text:
            return text
    date_value = record.get("date")
    if isinstance(date_value, str) and date_value:
        return date_value.split("/", 1)[0]
    return ""


def _format_page_range(record: Mapping[str, Any]) -> str:
    start = _coerce_str(record.get("start_page") or record.get("pages"))
    end = _coerce_str(record.get("end_page"))
    if start and end:
        if start == end:
            return f"p. {start}"
        return f"pp. {start}-{end}"
    if start:
        return f"p. {start}"
    if end:
        return f"p. {end}"
    return ""
