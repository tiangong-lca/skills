"""Configuration helpers for knowledge base ingestion."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

DEFAULT_TIMEOUT_SECONDS = 60.0


@dataclass(slots=True)
class MetadataFieldDefinition:
    """Describe a metadata field that should exist in the knowledge base dataset."""

    name: str
    type: str = "string"
    source: str | None = None
    value: str | None = None
    default: str | None = None
    join_with: str | None = "; "

    def render_value(self, record: Mapping[str, Any]) -> str | None:
        """Return the value that should be attached for this field."""
        if self.value is not None:
            return self.value

        key = self.source or self.name
        if not key:
            return None
        candidate = record.get(key)
        if candidate is None:
            candidate = self.default
        if candidate is None:
            return None

        if isinstance(candidate, (list, tuple, set)):
            pieces = [str(item).strip() for item in candidate if item not in (None, "")]
            if not pieces:
                return None
            separator = self.join_with if self.join_with is not None else "; "
            return separator.join(pieces)

        if isinstance(candidate, (int, float)):
            return str(candidate)

        text = str(candidate).strip()
        return text or None


DEFAULT_METADATA_FIELDS: tuple[MetadataFieldDefinition, ...] = (
    MetadataFieldDefinition(name="meta", source="meta"),
    MetadataFieldDefinition(name="category", source="category"),
)


@dataclass(slots=True)
class KnowledgeBaseConfig:
    """Runtime configuration for talking to the remote knowledge base."""

    base_url: str
    api_key: str
    dataset_id: str
    request_timeout: float = DEFAULT_TIMEOUT_SECONDS
    metadata_fields: list[MetadataFieldDefinition] = field(default_factory=list)
    pipeline_datasource_type: str = "local_file"
    pipeline_start_node_id: str | None = None
    pipeline_inputs: dict[str, Any] = field(default_factory=dict)
    pipeline_response_mode: str = "blocking"
    pipeline_is_published: bool = True

    def __post_init__(self) -> None:
        if not self.metadata_fields:
            self.metadata_fields = [replace(field) for field in DEFAULT_METADATA_FIELDS]

    @property
    def authorization_header(self) -> dict[str, str]:
        """Return the Authorization header expected by the remote API."""
        return {"Authorization": f"Bearer {self.api_key}"}


def load_kb_config(path: Path) -> KnowledgeBaseConfig:
    """Load the knowledge base configuration from the secrets TOML file."""
    if not path.exists():
        raise SystemExit(f"Secrets file not found: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    section = data.get("kb") or data.get("knowledge_base")
    if not isinstance(section, dict):
        raise SystemExit(f"[kb] section missing in secrets file: {path}")

    base_url = section.get("base_url", "").strip()
    dataset_id = section.get("dataset_id", "").strip()
    api_key = _sanitize_api_key(section.get("api_key"))
    timeout = _coerce_float(section.get("timeout")) or DEFAULT_TIMEOUT_SECONDS

    if not base_url:
        raise SystemExit("Knowledge base base_url missing in secrets configuration.")
    if not dataset_id:
        raise SystemExit("Knowledge base dataset_id missing in secrets configuration.")
    if not api_key:
        raise SystemExit("Knowledge base api_key missing in secrets configuration.")

    metadata_definitions = _load_metadata_definitions(section.get("metadata_fields"))

    pipeline_cfg = section.get("pipeline") or {}
    if pipeline_cfg and not isinstance(pipeline_cfg, dict):
        raise SystemExit("[kb.pipeline] must be a table containing datasource configuration.")

    pipeline_inputs = pipeline_cfg.get("inputs") or {}
    if not isinstance(pipeline_inputs, dict):
        raise SystemExit("[kb.pipeline.inputs] must be a JSON object.")
    pipeline_inputs = dict(pipeline_inputs)

    normalized_base_url = f"{base_url.rstrip('/')}/"

    return KnowledgeBaseConfig(
        base_url=normalized_base_url,
        api_key=api_key,
        dataset_id=dataset_id,
        request_timeout=timeout,
        metadata_fields=metadata_definitions,
        pipeline_datasource_type=pipeline_cfg.get("datasource_type") or "local_file",
        pipeline_start_node_id=_optional_str(pipeline_cfg.get("start_node_id")),
        pipeline_inputs=pipeline_inputs,
        pipeline_response_mode=pipeline_cfg.get("response_mode") or "blocking",
        pipeline_is_published=_coerce_bool(pipeline_cfg.get("is_published"), default=True),
    )


def _load_metadata_definitions(raw: Any) -> list[MetadataFieldDefinition]:
    definitions: list[MetadataFieldDefinition] = []
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            definitions.append(
                MetadataFieldDefinition(
                    name=name,
                    type=str(entry.get("type") or "string"),
                    source=_optional_str(entry.get("source")),
                    value=_optional_str(entry.get("value")),
                    default=_optional_str(entry.get("default")),
                    join_with=_optional_str(entry.get("join_with"), allow_blank=True),
                )
            )
    return definitions or [replace(field) for field in DEFAULT_METADATA_FIELDS]


def _sanitize_api_key(value: Any) -> str:
    if not value:
        return ""
    token = str(value).strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any, *, allow_blank: bool = False) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not allow_blank:
        text = text.strip()
        if not text:
            return None
    return text


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default
