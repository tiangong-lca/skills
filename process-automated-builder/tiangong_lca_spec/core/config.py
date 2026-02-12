"""Application configuration primitives."""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import SettingsProfile

DEFAULT_SECRETS_PATH = Path(".secrets/secrets.toml")


def _authorization_header(
    api_key: str | None,
    header_name: str | None = None,
    prefix: str | None = None,
) -> dict[str, str]:
    if not api_key:
        return {}
    header = (header_name or "Authorization").strip()
    if not header:
        return {}
    if prefix is None:
        prefix = "Bearer"
    prefix = prefix.strip()
    if prefix:
        value = f"{prefix} {api_key}".strip()
    else:
        value = api_key
    return {header: value}


class Settings(BaseSettings):
    """Central configuration for the spec coding workflow."""

    mcp_base_url: HttpUrl = "https://lcamcp.tiangong.earth/mcp"
    mcp_api_key: str | None = None
    mcp_transport: Literal["streamable_http"] = "streamable_http"
    mcp_connections: dict[str, dict[str, Any]] | None = None
    flow_search_service_name: str = "tiangong_lca_remote"
    flow_search_tool_name: str = "Search_Flows_Tool"
    flow_search_max_parallel: int = 1
    flow_search_state_code: int | None = 100

    request_timeout: float = 30.0
    flow_search_timeout: float | None = None
    max_retries: int = 3
    retry_backoff: float = 0.5
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    workflow_profile: Literal["default", "batch", "debug"] = "default"
    max_concurrency: int = 4

    cache_dir: Path = Path("artifacts/cache")
    artifacts_dir: Path = Path("artifacts")
    flow_hint_catalog_path: Path | None = None
    stage2_exchange_retry_attempts: int = 2

    model_config = SettingsConfigDict(env_prefix="LCA_", env_file=(), extra="ignore")

    @property
    def profile(self) -> SettingsProfile:
        """Expose derived profile information for orchestrator policies."""
        if self.workflow_profile == "batch":
            return SettingsProfile(
                concurrency=self.max_concurrency,
                retry_attempts=self.max_retries + 2,
                cache_results=True,
                profile_name="batch",
            )
        if self.workflow_profile == "debug":
            return SettingsProfile(
                concurrency=1,
                retry_attempts=1,
                cache_results=False,
                profile_name="debug",
            )
        return SettingsProfile(
            concurrency=min(self.max_concurrency, 4),
            retry_attempts=self.max_retries,
            cache_results=True,
            profile_name="default",
        )

    def flow_search_mcp_config(self) -> dict[str, Any]:
        """Return the MCP configuration block for the flow search service."""
        config: dict[str, Any] = {
            "transport": self.mcp_transport,
            "url": str(self.mcp_base_url),
        }
        headers = _authorization_header(self.mcp_api_key)
        if headers:
            config["headers"] = headers
        timeout = self.flow_search_timeout or self.request_timeout
        if timeout and timeout > 0:
            config["timeout"] = float(timeout)
        return config

    def mcp_service_configs(self) -> dict[str, dict[str, Any]]:
        """Return a mapping of MCP service names to their configuration blocks."""
        configs: dict[str, dict[str, Any]] = dict(self.mcp_connections or {})
        flow_service_name = self.flow_search_service_name or "tiangong_lca_remote"
        configs.setdefault(flow_service_name, self.flow_search_mcp_config())
        return configs


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    overrides = _load_settings_overrides()
    settings = Settings(**overrides)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings


def get_mcp_service_configs() -> dict[str, dict[str, Any]]:
    """Convenience helper returning the configured MCP service blocks."""
    return get_settings().mcp_service_configs()


def _load_settings_overrides(secrets_path: Path = DEFAULT_SECRETS_PATH) -> dict[str, Any]:
    """Load configuration overrides from the secrets TOML file."""
    if not secrets_path.exists():
        return {}
    data = _read_toml(secrets_path)
    overrides: dict[str, Any] = {}

    flow_cfg = _extract_section(data, "tiangong_lca_remote", "mcp", "flow_search")
    if flow_cfg:
        overrides.update(
            {
                "mcp_base_url": flow_cfg.get("url", overrides.get("mcp_base_url")),
                "mcp_transport": flow_cfg.get("transport", overrides.get("mcp_transport")),
                "flow_search_service_name": flow_cfg.get("service_name", overrides.get("flow_search_service_name")),
                "flow_search_tool_name": flow_cfg.get("tool_name", overrides.get("flow_search_tool_name")),
            }
        )
        api_key = _sanitize_api_key(flow_cfg.get("api_key") or flow_cfg.get("authorization"))
        if api_key is not None:
            overrides["mcp_api_key"] = api_key
        timeout_value = _coerce_float(flow_cfg.get("timeout"))
        if timeout_value is not None:
            overrides["flow_search_timeout"] = timeout_value

    connections = _extract_mcp_connections(data, overrides)
    if connections:
        overrides["mcp_connections"] = connections

    general_cfg = data.get("lca") or {}
    overrides.update({key: value for key, value in general_cfg.items() if key in Settings.model_fields})
    return {key: value for key, value in overrides.items() if value is not None}


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _extract_section(data: dict[str, Any], *candidates: str) -> dict[str, Any] | None:
    for key in candidates:
        section = data.get(key)
        if isinstance(section, dict):
            return section
    return None


def _sanitize_api_key(value: str | None, prefix: str | None = None) -> str | None:
    if not value:
        return None
    token = value.strip()
    if prefix is None:
        prefix_text = "Bearer"
    elif isinstance(prefix, str):
        prefix_text = prefix.strip()
    else:
        prefix_text = ""
    if prefix_text and token.lower().startswith(f"{prefix_text.lower()} "):
        token = token[len(prefix_text) + 1 :].strip()
    return token or None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None


def _coerce_header_name(value: Any, default: str) -> str:
    if isinstance(value, str):
        name = value.strip()
        if name:
            return name
    return default


def _coerce_prefix(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return default


def _extract_mcp_connections(data: dict[str, Any], defaults: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Collect all MCP service blocks from the secrets file."""
    connections: dict[str, dict[str, Any]] = {}
    skip_keys = {"openai", "kb", "minio", "lca"}
    for section_name, section in data.items():
        if section_name in skip_keys or not isinstance(section, dict):
            continue
        config = _build_mcp_config(section, defaults)
        if not config:
            continue
        service_name = _coerce_service_name(section.get("service_name"), section_name)
        connections[service_name] = config
    return connections


def _build_mcp_config(section: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any] | None:
    transport = section.get("transport") or defaults.get("mcp_transport") or "streamable_http"
    url = section.get("url") or defaults.get("mcp_base_url")
    if not url:
        return None
    config: dict[str, Any] = {
        "transport": transport,
        "url": url,
    }
    header_name = _coerce_header_name(
        section.get("api_key_header"),
        _coerce_header_name(defaults.get("mcp_api_key_header"), "Authorization"),
    )
    prefix = _coerce_prefix(section.get("api_key_prefix"), _coerce_prefix(defaults.get("mcp_api_key_prefix"), "Bearer"))
    api_key = _sanitize_api_key(section.get("api_key") or section.get("authorization"), prefix)
    headers = _authorization_header(api_key, header_name, prefix)
    if headers:
        config["headers"] = headers
    timeout_value = _coerce_float(section.get("timeout"))
    if timeout_value is None:
        timeout_value = _coerce_float(defaults.get("request_timeout"))
    if timeout_value is not None:
        config["timeout"] = float(timeout_value)
    return config


def _coerce_service_name(candidate: Any, default: str) -> str:
    if isinstance(candidate, str):
        name = candidate.strip()
        if name:
            return name
    return default
