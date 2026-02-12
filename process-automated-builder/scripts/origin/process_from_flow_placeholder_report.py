#!/usr/bin/env python
# ruff: noqa: E402
"""Generate a placeholder flow report from process_from_flow state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent
for path in (SCRIPTS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

try:
    from scripts.md._workflow_common import dump_json  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from _workflow_common import dump_json  # type: ignore

from tiangong_lca_spec.process_from_flow.service import (  # type: ignore
    _build_placeholder_report,
)

PROCESS_FROM_FLOW_ARTIFACTS_ROOT = Path("artifacts/process_from_flow")
LATEST_RUN_ID_PATH = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / ".latest_run_id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Run ID under artifacts/process_from_flow.")
    parser.add_argument("--state-path", type=Path, help="Explicit state JSON path to load.")
    parser.add_argument("--output", type=Path, help="Output JSON path (defaults to cache/placeholder_report.json).")
    parser.add_argument("--no-update-state", action="store_true", help="Skip writing the report back to the state file.")
    return parser.parse_args()


def _resolve_state_path(run_id: str | None, state_path: Path | None) -> Path:
    if state_path:
        return state_path
    use_run_id = run_id
    if not use_run_id and LATEST_RUN_ID_PATH.exists():
        use_run_id = LATEST_RUN_ID_PATH.read_text(encoding="utf-8").strip() or None
    if not use_run_id:
        raise SystemExit("Missing --run-id/--state-path and no latest run marker found.")
    path = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / use_run_id / "cache" / "process_from_flow_state.json"
    if not path.exists():
        raise SystemExit(f"State file not found: {path}")
    return path


def _load_state(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"State file must contain an object: {path}")
    return payload


def _resolve_output_path(state_path: Path, output: Path | None) -> Path:
    if output:
        return output
    return state_path.parent / "placeholder_report.json"


def main() -> None:
    args = parse_args()
    state_path = _resolve_state_path(args.run_id, args.state_path)
    state = _load_state(state_path)
    report = _build_placeholder_report(state)
    output_path = _resolve_output_path(state_path, args.output)
    dump_json(report, output_path)

    if not args.no_update_state:
        state["placeholder_report"] = report
        dump_json(state, state_path)

    print(f"Wrote {len(report)} placeholder entries to {output_path}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    main()
