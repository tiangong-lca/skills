"""HTTP client for interacting with the Tiangong knowledge base datasets."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Iterable, Mapping

import httpx

from .config import KnowledgeBaseConfig, MetadataFieldDefinition


def _sanitize_filename_for_upload(name: str) -> str:
    """Return the original filename (or a safe fallback when empty)."""
    if not name:
        return "attachment.pdf"
    stripped = name.strip()
    return stripped or "attachment.pdf"


class KnowledgeBaseClient(AbstractContextManager["KnowledgeBaseClient"]):
    """Minimal wrapper around the dataset/document endpoints."""

    def __init__(self, config: KnowledgeBaseConfig) -> None:
        self._config = config
        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=config.request_timeout,
            headers=config.authorization_header,
        )

    # Context manager API -----------------------------------------------------
    def __enter__(self) -> "KnowledgeBaseClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[override]
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    # Dataset operations ------------------------------------------------------
    def fetch_dataset_info(self) -> dict:
        """Return dataset metadata used to determine the indexing behaviour."""
        response = self._client.get(f"datasets/{self._config.dataset_id}")
        response.raise_for_status()
        return response.json()

    def ensure_metadata_fields(self, definitions: Iterable[MetadataFieldDefinition]) -> dict[str, str]:
        """Ensure required metadata fields exist and return their IDs."""
        endpoint = f"datasets/{self._config.dataset_id}/metadata"
        response = self._client.get(endpoint)
        response.raise_for_status()
        payload = response.json()
        existing = {item["name"]: item["id"] for item in payload.get("doc_metadata", []) if isinstance(item, dict)}

        for definition in definitions:
            if definition.name in existing:
                continue
            create_resp = self._client.post(
                endpoint,
                json={"type": definition.type, "name": definition.name},
            )
            create_resp.raise_for_status()
            created = create_resp.json()
            existing[created["name"]] = created["id"]
        return existing

    # Document operations -----------------------------------------------------
    def upload_document(self, file_path: Path, payload: dict) -> dict:
        """Upload a document and return the API response payload."""
        endpoint = f"datasets/{self._config.dataset_id}/document/create-by-file"
        try:
            with file_path.open("rb") as binary:
                upload_name = _sanitize_filename_for_upload(file_path.name)
                files = {"file": (upload_name, binary, "application/pdf")}
                response = self._client.post(endpoint, data={"data": json.dumps(payload)}, files=files)
                response.raise_for_status()
                return response.json()
        except FileNotFoundError as exc:
            raise SystemExit(f"Attachment not found: {file_path}") from exc

    def attach_metadata(self, document_id: str, metadata_entries: list[dict]) -> dict | None:
        """Attach metadata to a document when entries are available."""
        if not metadata_entries:
            return None
        endpoint = f"datasets/{self._config.dataset_id}/documents/metadata"
        response = self._client.post(
            endpoint,
            json={"operation_data": [{"document_id": document_id, "metadata_list": metadata_entries}]},
        )
        response.raise_for_status()
        return response.json()

    # Pipeline operations -----------------------------------------------------
    def upload_pipeline_file(self, file_path: Path) -> dict:
        """Upload a file for use inside a pipeline run."""
        endpoint = "datasets/pipeline/file-upload"
        try:
            with file_path.open("rb") as binary:
                upload_name = _sanitize_filename_for_upload(file_path.name)
                files = {"file": (upload_name, binary, "application/pdf")}
                response = self._client.post(endpoint, files=files)
                response.raise_for_status()
                return response.json()
        except FileNotFoundError as exc:
            raise SystemExit(f"Attachment not found: {file_path}") from exc

    def fetch_pipeline_datasource_plugins(self, *, is_published: bool = True) -> list[dict]:
        """Return the datasource plugin metadata for the dataset pipeline."""
        endpoint = f"datasets/{self._config.dataset_id}/pipeline/datasource-plugins"
        try:
            response = self._client.get(endpoint, params={"is_published": str(is_published).lower()})
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return []
            raise
        payload = response.json()
        items = payload.get("data") if isinstance(payload, dict) else payload
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        return []

    def resolve_pipeline_start_node_id(self, datasource_type: str, *, is_published: bool = True) -> str | None:
        """Find the datasource node ID that matches the configured datasource_type."""
        plugins = self.fetch_pipeline_datasource_plugins(is_published=is_published)
        datasource_type_lower = (datasource_type or "").lower()
        for plugin in plugins:
            node_id = plugin.get("node_id")
            plugin_type = plugin.get("provider_type") or plugin.get("datasource_type") or plugin.get("type") or plugin.get("provider")
            if node_id and plugin_type and str(plugin_type).lower() == datasource_type_lower:
                return str(node_id)
        return None

    def run_pipeline(self, payload: dict) -> dict:
        """Execute the configured pipeline for the dataset."""
        endpoint = f"datasets/{self._config.dataset_id}/pipeline/run"
        response = self._client.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()

    # Retrieval operations ---------------------------------------------------
    def retrieve_chunks(
        self,
        *,
        query: str | None = None,
        retrieval_model: Mapping[str, Any] | None = None,
        top_k: int | None = None,
        score_threshold_enabled: bool | None = None,
        score_threshold: float | None = None,
        metadata_filtering_conditions: Mapping[str, Any] | None = None,
        weights: float | int | None = None,
        extra_payload: Mapping[str, Any] | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> dict:
        """Execute a retrieval query against the dataset and return the raw response."""
        endpoint = f"datasets/{self._config.dataset_id}/retrieve"
        if payload is not None:
            request_payload: dict[str, Any] = dict(payload)
            if query is not None:
                request_payload["query"] = query
        else:
            if query is None or (isinstance(query, str) and not query.strip()):
                raise ValueError("query must be provided when payload is not supplied.")
            request_payload = {"query": query}

        model_payload: dict[str, Any] = {}
        existing_model = request_payload.get("retrieval_model")
        if isinstance(existing_model, Mapping):
            model_payload.update(existing_model)
        if retrieval_model:
            model_payload.update(retrieval_model)
        if top_k is not None:
            model_payload["top_k"] = int(top_k)
        if score_threshold_enabled is not None:
            model_payload["score_threshold_enabled"] = bool(score_threshold_enabled)
        if score_threshold is not None:
            model_payload["score_threshold"] = float(score_threshold)
        if weights is not None:
            model_payload["weights"] = float(weights)
        if metadata_filtering_conditions is not None:
            model_payload["metadata_filtering_conditions"] = dict(metadata_filtering_conditions)
        if model_payload:
            request_payload["retrieval_model"] = model_payload
        if extra_payload:
            request_payload.update(extra_payload)

        response = self._client.post(endpoint, json=request_payload)
        response.raise_for_status()
        return response.json()
