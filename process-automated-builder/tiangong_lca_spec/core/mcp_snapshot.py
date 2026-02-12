"""Helpers for persisting MCP invocation snapshots during a workflow run."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def append_mcp_snapshot(
    *,
    server_name: str,
    tool_name: str,
    arguments: Mapping[str, Any] | None = None,
    status: str,
    payload: Any | None = None,
    error: str | None = None,
) -> None:
    """Append one MCP call snapshot as JSONL, best-effort only."""
    snapshot_dir_value = os.getenv("TIANGONG_PFF_MCP_SNAPSHOT_DIR")
    if not snapshot_dir_value:
        return

    try:
        snapshot_dir = Path(snapshot_dir_value)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        target = snapshot_dir / f"{_sanitize_filename(server_name)}.jsonl"
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "run_id": os.getenv("TIANGONG_PFF_RUN_ID"),
            "server_name": server_name,
            "tool_name": tool_name,
            "status": status,
            "arguments": _json_safe(arguments),
            "payload": _json_safe(payload),
            "error": error,
        }
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Snapshotting must not block the main workflow.
        return


def _sanitize_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        return _to_repr(value)


def _to_repr(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_repr(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_repr(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
