#!/usr/bin/env python
"""Compatibility shim for the legacy process_from_flow workflow entrypoint.

The canonical SI-enhanced orchestration now lives in
`process_from_flow_langgraph.py workflow`.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent
for path in (SCRIPTS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

try:
    from scripts.origin.process_from_flow_langgraph import main as langgraph_main  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from process_from_flow_langgraph import main as langgraph_main  # type: ignore


def main() -> None:
    sys.argv = [sys.argv[0], "workflow", *sys.argv[1:]]
    langgraph_main()


if __name__ == "__main__":  # pragma: no cover
    main()
