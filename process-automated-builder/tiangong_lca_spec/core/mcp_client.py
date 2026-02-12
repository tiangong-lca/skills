"""Synchronous bridge built on the official python MCP client SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from anyio import ClosedResourceError
from anyio.from_thread import BlockingPortal, start_blocking_portal
from httpx import HTTPStatusError
from mcp import ClientSession, McpError, types
from mcp.client.streamable_http import streamablehttp_client

from tiangong_lca_spec.core.config import Settings, get_settings
from tiangong_lca_spec.core.exceptions import SpecCodingError
from tiangong_lca_spec.core.json_utils import parse_json_response
from tiangong_lca_spec.core.logging import get_logger
from tiangong_lca_spec.core.mcp_snapshot import append_mcp_snapshot

LOGGER = get_logger(__name__)


@dataclass
class _ServerConnection:
    client_cm: Any
    session_cm: Any
    session: ClientSession
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.session_cm is not None:
                self.session_cm.__exit__(None, None, None)
        finally:
            if self.client_cm is not None:
                self.client_cm.__exit__(None, None, None)


class MCPToolClient:
    """Synchronously accessible wrapper around the python-sdk MCP client."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        connections: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._connection_configs = dict(connections) if connections is not None else self._settings.mcp_service_configs()
        self._portal_cm = start_blocking_portal()
        self._portal: BlockingPortal = self._portal_cm.__enter__()
        self._connections: dict[str, _ServerConnection] = {}
        self._closed = False
        LOGGER.debug(
            "mcp_tool_client.initialized",
            servers=list(self._connection_configs.keys()),
        )

    # Public API -----------------------------------------------------------------

    def invoke_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        """Call a remote MCP tool and return payload plus optional attachments."""
        if self._closed:
            raise RuntimeError("Cannot invoke MCP tool on a closed client")
        connection = self._ensure_connection(server_name)
        try:
            payload, attachments = self._call_tool(connection, server_name, tool_name, arguments)
            append_mcp_snapshot(
                server_name=server_name,
                tool_name=tool_name,
                arguments=arguments,
                status="ok",
                payload=payload,
            )
        except ClosedResourceError as exc:
            LOGGER.warning(
                "mcp_tool_client.connection_reset",
                server=server_name,
                error=str(exc),
            )
            self._reset_connection(server_name)
            connection = self._ensure_connection(server_name)
            try:
                payload, attachments = self._call_tool(connection, server_name, tool_name, arguments)
            except Exception as retry_exc:
                append_mcp_snapshot(
                    server_name=server_name,
                    tool_name=tool_name,
                    arguments=arguments,
                    status="error",
                    error=str(retry_exc),
                )
                raise
            append_mcp_snapshot(
                server_name=server_name,
                tool_name=tool_name,
                arguments=arguments,
                status="ok",
                payload=payload,
            )
        except Exception as exc:
            append_mcp_snapshot(
                server_name=server_name,
                tool_name=tool_name,
                arguments=arguments,
                status="error",
                error=str(exc),
            )
            raise
        return payload, attachments

    def invoke_json_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> Any:
        """Invoke a tool and return structured JSON data."""
        payload, _ = self.invoke_tool(server_name, tool_name, arguments)
        if payload is None:
            return None
        if isinstance(payload, (dict, list)):
            return payload
        if isinstance(payload, str):
            raw = payload.strip()
            if not raw:
                return None
            return parse_json_response(raw)
        raise SpecCodingError(f"MCP tool '{tool_name}' on '{server_name}' returned non-JSON payload")

    def close(self) -> None:
        """Close all active MCP sessions."""
        if self._closed:
            return
        self._closed = True
        try:
            for server_name, connection in list(self._connections.items()):
                try:
                    connection.close()
                    LOGGER.debug("mcp_tool_client.session_closed", server=server_name)
                except Exception as exc:  # pragma: no cover - best effort shutdown
                    LOGGER.warning(
                        "mcp_tool_client.close_failed",
                        server=server_name,
                        error=str(exc),
                    )
            self._connections.clear()
        finally:
            self._portal_cm.__exit__(None, None, None)
            LOGGER.debug("mcp_tool_client.closed")

    def __enter__(self) -> "MCPToolClient":
        if self._closed:
            raise RuntimeError("Cannot re-enter a closed MCPToolClient")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # Internal helpers -----------------------------------------------------------

    def _ensure_connection(self, server_name: str) -> _ServerConnection:
        connection = self._connections.get(server_name)
        if connection is not None:
            return connection

        config = self._connection_configs.get(server_name)
        if not config:
            raise SpecCodingError(f"MCP server '{server_name}' is not configured")
        transport = config.get("transport", "streamable_http")
        if transport != "streamable_http":
            raise SpecCodingError(f"Unsupported MCP transport '{transport}' for '{server_name}'")

        url = config.get("url")
        if not url:
            raise SpecCodingError(f"MCP server '{server_name}' is missing a URL")

        headers = config.get("headers")
        timeout = config.get("timeout") or self._settings.request_timeout or 30

        client_async_cm = streamablehttp_client(
            url,
            headers=headers,
            timeout=float(timeout),
        )
        client_cm = self._portal.wrap_async_context_manager(client_async_cm)
        try:
            read_stream, write_stream, _ = client_cm.__enter__()
        except Exception:
            client_cm.__exit__(None, None, None)
            raise

        session_async_cm = ClientSession(read_stream, write_stream)
        session_cm = self._portal.wrap_async_context_manager(session_async_cm)
        try:
            session = session_cm.__enter__()
            self._portal.call(session.initialize)
        except Exception:
            session_cm.__exit__(None, None, None)
            client_cm.__exit__(None, None, None)
            raise

        connection = _ServerConnection(
            client_cm=client_cm,
            session_cm=session_cm,
            session=session,
        )
        self._connections[server_name] = connection
        LOGGER.debug("mcp_tool_client.session_opened", server=server_name)
        return connection

    def _reset_connection(self, server_name: str) -> None:
        connection = self._connections.pop(server_name, None)
        if connection is None:
            return
        try:
            connection.close()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            LOGGER.warning(
                "mcp_tool_client.connection_close_failed",
                server=server_name,
                error=str(exc),
            )

    def _call_tool(
        self,
        connection: _ServerConnection,
        server_name: str,
        tool_name: str,
        arguments: Mapping[str, Any] | None,
    ) -> tuple[Any, Any]:
        args = dict(arguments or {})
        LOGGER.debug(
            "mcp_tool_client.invoke",
            server=server_name,
            tool=tool_name,
            keys=list(args.keys()),
        )
        try:
            result = self._portal.call(connection.session.call_tool, tool_name, args)
        except (McpError, HTTPStatusError) as exc:
            raise SpecCodingError(f"MCP tool '{tool_name}' call failed") from exc

        if result.isError:
            message = self._collect_text(result) or "Unknown MCP tool error"
            raise SpecCodingError(f"MCP tool '{tool_name}' on '{server_name}' reported an error: {message}")

        payload = result.structuredContent
        if payload is None:
            texts = self._collect_text_blocks(result)
            if not texts:
                payload = ""
            elif len(texts) == 1:
                payload = texts[0]
            else:
                payload = texts
        attachments = self._collect_attachments(result)
        return payload, attachments or None

    @staticmethod
    def _collect_text(result: types.CallToolResult) -> str:
        blocks = MCPToolClient._collect_text_blocks(result)
        if not blocks:
            return ""
        if len(blocks) == 1:
            return blocks[0]
        return "\n".join(blocks)

    @staticmethod
    def _collect_text_blocks(result: types.CallToolResult) -> list[str]:
        texts: list[str] = []
        for content in result.content:
            if isinstance(content, types.TextContent):
                texts.append(content.text or "")
        return [item for item in texts if item]

    @staticmethod
    def _collect_attachments(result: types.CallToolResult) -> list[dict[str, Any]]:
        attachments: list[dict[str, Any]] = []
        for content in result.content:
            if isinstance(content, types.TextContent):
                continue
            if hasattr(content, "model_dump"):
                attachments.append(content.model_dump())
            else:
                attachments.append({"type": content.__class__.__name__})
        return attachments


__all__ = ["MCPToolClient"]
