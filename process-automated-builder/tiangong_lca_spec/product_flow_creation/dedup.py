"""Dedup/action decision utilities for product flow publication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

FlowAction = Literal["insert", "update", "reuse"]
ActionPreference = Literal["auto", "insert", "update", "reuse"]


class FlowLookupProtocol(Protocol):
    """Minimal lookup contract used by dedup service."""

    def select_flow(self, flow_uuid: str, *, version: str | None = None) -> dict[str, Any] | None: ...


@dataclass(slots=True, frozen=True)
class FlowDedupDecision:
    action: FlowAction
    exists: bool
    reason: str


class FlowDedupService:
    """Resolve insert/update/reuse action based on remote existence checks."""

    def __init__(self, lookup: FlowLookupProtocol) -> None:
        self._lookup = lookup

    def decide(
        self,
        *,
        flow_uuid: str,
        version: str | None = None,
        preferred_action: ActionPreference = "auto",
    ) -> FlowDedupDecision:
        uuid_value = str(flow_uuid or "").strip()
        if not uuid_value:
            return FlowDedupDecision(action="insert", exists=False, reason="missing_uuid")

        exists = self._exists(uuid_value, version=version)
        preference = preferred_action if preferred_action in {"auto", "insert", "update", "reuse"} else "auto"

        if preference == "reuse":
            if exists:
                return FlowDedupDecision(action="reuse", exists=True, reason="preferred_reuse_exists")
            return FlowDedupDecision(action="insert", exists=False, reason="preferred_reuse_missing")
        if preference == "update":
            if exists:
                return FlowDedupDecision(action="update", exists=True, reason="preferred_update_exists")
            return FlowDedupDecision(action="insert", exists=False, reason="preferred_update_missing")
        if preference == "insert":
            if exists:
                return FlowDedupDecision(action="update", exists=True, reason="preferred_insert_exists_switch_update")
            return FlowDedupDecision(action="insert", exists=False, reason="preferred_insert_missing")

        # auto
        if exists:
            return FlowDedupDecision(action="update", exists=True, reason="auto_exists")
        return FlowDedupDecision(action="insert", exists=False, reason="auto_missing")

    def _exists(self, flow_uuid: str, *, version: str | None = None) -> bool:
        try:
            if version:
                matched = self._lookup.select_flow(flow_uuid, version=version)
                if isinstance(matched, dict):
                    return True
            matched_any = self._lookup.select_flow(flow_uuid)
            return isinstance(matched_any, dict)
        except Exception:  # noqa: BLE001
            return False
