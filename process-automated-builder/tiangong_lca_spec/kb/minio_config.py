"""MinIO configuration helpers for KB artifacts."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from minio import Minio


@dataclass(slots=True)
class MinioConfig:
    """Connection parameters for the MinIO bucket storing KB artifacts."""

    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str
    prefix: str = ""
    secure: bool = True
    session_token: str | None = None

    def normalized_prefix(self) -> str:
        """Return the configured prefix without leading/trailing slashes."""
        text = (self.prefix or "").strip("/")
        return text

    def build_prefix(self, *parts: str | None) -> str:
        """Compose a remote key prefix using the configured root prefix."""
        components = [self.normalized_prefix(), *parts]
        return join_remote_path(*components)


def load_minio_config(path: Path) -> MinioConfig:
    """Load the [minio] block from the provided secrets TOML file."""
    if not path.exists():
        raise SystemExit(f"Secrets file not found: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    section = data.get("minio")
    if not isinstance(section, dict):
        raise SystemExit(f"[minio] section missing in secrets file: {path}")

    endpoint_raw = str(section.get("endpoint") or "").strip()
    access_key = str(section.get("access_key") or "").strip()
    secret_key = str(section.get("secret_key") or "").strip()
    bucket_name = str(section.get("bucket_name") or "").strip()
    prefix = str(section.get("prefix") or "").strip()
    if not endpoint_raw:
        raise SystemExit("MinIO endpoint missing in secrets configuration.")
    if not access_key:
        raise SystemExit("MinIO access_key missing in secrets configuration.")
    if not secret_key:
        raise SystemExit("MinIO secret_key missing in secrets configuration.")
    if not bucket_name:
        raise SystemExit("MinIO bucket_name missing in secrets configuration.")

    endpoint, secure_from_endpoint = _normalize_endpoint(endpoint_raw)
    secure_override = _optional_bool(section.get("secure"))
    secure = secure_override if secure_override is not None else secure_from_endpoint
    session_token = _optional_str(section.get("session_token"))

    return MinioConfig(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket_name=bucket_name,
        prefix=prefix,
        secure=secure,
        session_token=session_token,
    )


def create_minio_client(config: MinioConfig) -> Minio:
    """Instantiate a MinIO client based on the provided configuration."""
    return Minio(
        config.endpoint,
        access_key=config.access_key,
        secret_key=config.secret_key,
        secure=config.secure,
        session_token=config.session_token,
    )


def join_remote_path(*components: str | None) -> str:
    """Join multiple path components using POSIX separators."""
    parts: list[str] = []
    for component in components:
        if not component:
            continue
        text = str(component).strip("/")
        if text:
            parts.append(text)
    return "/".join(parts)


def _normalize_endpoint(value: str) -> tuple[str, bool]:
    parsed = urlparse(value)
    if parsed.scheme:
        endpoint = parsed.netloc or parsed.path
        if not endpoint:
            raise SystemExit(f"Invalid MinIO endpoint: {value}")
        secure = parsed.scheme.lower() != "http"
        return endpoint, secure
    sanitized = value.strip()
    if not sanitized:
        raise SystemExit("MinIO endpoint cannot be blank.")
    return sanitized, True


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
