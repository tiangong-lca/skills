"""
Tiangong LCA spec coding orchestration package.

The package exposes modular building blocks for:
- flow search against the MCP service,
- exchange alignment,
- process extraction from literature,
- validation via the TIDAS service,
- an orchestrator that composes the workflow.
"""

from .core.config import Settings, get_settings

__all__ = [
    "Settings",
    "get_settings",
]
