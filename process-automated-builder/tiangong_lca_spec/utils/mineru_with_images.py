from __future__ import annotations

import mimetypes
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

DEFAULT_MINERU_SECTION = "tiangong_mineru_with_image"


@dataclass(slots=True)
class MineruWithImagesConfig:
    url: str
    api_key: str | None
    api_key_header: str
    api_key_prefix: str
    timeout: float
    provider: str | None
    model: str | None
    chunk_type: bool | None
    verify_ssl: bool

    def auth_headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        header = self.api_key_header.strip()
        prefix = self.api_key_prefix.strip()
        if prefix:
            value = f"{prefix} {self.api_key}".strip()
        else:
            value = self.api_key
        return {header: value}


class MineruWithImagesClient:
    def __init__(self, config: MineruWithImagesConfig) -> None:
        self._config = config

    def split_document(
        self,
        file_path: Path,
        *,
        provider: str | None = None,
        model: str | None = None,
        chunk_type: bool | None = None,
        timeout: float | None = None,
    ) -> Any:
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        payload = _build_payload(
            provider if provider is not None else self._config.provider,
            model if model is not None else self._config.model,
        )
        params = _build_params(chunk_type if chunk_type is not None else self._config.chunk_type)
        timeout_seconds = timeout if timeout is not None else self._config.timeout
        headers = self._config.auth_headers()
        with httpx.Client(timeout=timeout_seconds, verify=self._config.verify_ssl, headers=headers) as client:
            with file_path.open("rb") as handle:
                content_type = _guess_content_type(file_path)
                files = {"file": (file_path.name, handle, content_type)}
                response = client.post(
                    self._config.url,
                    params=params,
                    data=payload,
                    files=files,
                )
                response.raise_for_status()
        return _parse_response(response)


def load_mineru_with_images_config(
    path: Path,
    *,
    section_name: str = DEFAULT_MINERU_SECTION,
) -> MineruWithImagesConfig:
    if not path.exists():
        raise SystemExit(f"Secrets file not found: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    section = _resolve_section(data, section_name)
    if section is None:
        raise SystemExit(f"[{section_name}] section missing in secrets file: {path}")

    url = _optional_str(section.get("url"))
    if not url:
        raise SystemExit("Mineru service url missing in secrets configuration.")

    api_key_header = _optional_str(section.get("api_key_header"))
    if not api_key_header:
        raise SystemExit("Mineru api_key_header missing in secrets configuration.")
    api_key_prefix = _optional_str(section.get("api_key_prefix"), allow_blank=True)
    if api_key_prefix is None:
        raise SystemExit("Mineru api_key_prefix missing in secrets configuration.")
    api_key = _sanitize_api_key(section.get("api_key") or section.get("authorization"), api_key_prefix)
    timeout = _coerce_float(section.get("timeout"))
    if timeout is None:
        raise SystemExit("Mineru timeout missing in secrets configuration.")
    provider = _optional_str(section.get("provider"))
    model = _optional_str(section.get("model"))
    chunk_type = _coerce_bool(section.get("chunk_type"), default=None)
    verify_ssl = _coerce_bool(section.get("verify_ssl"), default=None)
    if verify_ssl is None:
        raise SystemExit("Mineru verify_ssl missing in secrets configuration.")

    return MineruWithImagesConfig(
        url=url,
        api_key=api_key,
        api_key_header=api_key_header,
        api_key_prefix=api_key_prefix,
        timeout=timeout,
        provider=provider,
        model=model,
        chunk_type=chunk_type,
        verify_ssl=verify_ssl,
    )


def _resolve_section(data: dict[str, Any], section_name: str) -> dict[str, Any] | None:
    candidates = [
        section_name,
        "mineru_with_images",
        "tiangong_mineru_with_images",
        "mineru_with_image",
        "tiangong_mineru_with_image",
    ]
    seen: set[str] = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        section = data.get(key)
        if isinstance(section, dict):
            return section
    return None


def _build_payload(provider: str | None, model: str | None) -> dict[str, str]:
    payload: dict[str, str] = {}
    if provider:
        payload["provider"] = provider
    if model:
        payload["model"] = model
    return payload


def _build_params(chunk_type: bool | None) -> dict[str, str]:
    if chunk_type is None:
        return {}
    return {"chunk_type": "true" if chunk_type else "false"}


def _guess_content_type(file_path: Path) -> str:
    content_type, _ = mimetypes.guess_type(file_path.name)
    return content_type or "application/octet-stream"


def _parse_response(response: httpx.Response) -> Any:
    if not response.content:
        return None
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type.lower():
        return response.json()
    try:
        return response.json()
    except ValueError:
        return response.text


def _sanitize_api_key(value: Any, prefix: str | None) -> str | None:
    if not value:
        return None
    token = str(value).strip()
    if not token:
        return None
    prefix_text = prefix.strip() if isinstance(prefix, str) else ""
    if prefix_text and token.lower().startswith(f"{prefix_text.lower()} "):
        token = token[len(prefix_text) + 1 :].strip()
    return token or None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any, *, default: bool | None) -> bool | None:
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
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _optional_str(value: Any, *, allow_blank: bool = False) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if allow_blank:
        return text
    return text or None


__all__ = ["MineruWithImagesClient", "MineruWithImagesConfig", "load_mineru_with_images_config"]
