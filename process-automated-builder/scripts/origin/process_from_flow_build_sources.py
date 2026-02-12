#!/usr/bin/env python
# ruff: noqa: E402
"""Build ILCD source datasets from scientific references in process_from_flow state."""

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
    _build_source_datasets_from_references,
    _collect_reference_infos,
    _collect_usage_tag_map,
    _filter_reference_infos_by_usage,
)

PROCESS_FROM_FLOW_ARTIFACTS_ROOT = Path("artifacts/process_from_flow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Run ID under artifacts/process_from_flow.")
    parser.add_argument("--state-path", type=Path, help="Explicit state JSON path to load.")
    parser.add_argument("--output-dir", type=Path, help="Output directory for source datasets.")
    parser.add_argument("--no-update-state", action="store_true", help="Skip writing source datasets back to the state file.")
    return parser.parse_args()


def _resolve_state_path(run_id: str | None, state_path: Path | None) -> Path:
    if state_path:
        return state_path
    if not run_id:
        raise SystemExit("Missing --run-id or --state-path.")
    path = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / run_id / "cache" / "process_from_flow_state.json"
    if not path.exists():
        raise SystemExit(f"State file not found: {path}")
    return path


def _load_state(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"State file must contain an object: {path}")
    return payload


def _extract_source_uuid(source_payload: dict[str, Any]) -> str:
    dataset = source_payload.get("sourceDataSet") if isinstance(source_payload.get("sourceDataSet"), dict) else {}
    info = dataset.get("sourceInformation") if isinstance(dataset.get("sourceInformation"), dict) else {}
    data_info = info.get("dataSetInformation") if isinstance(info.get("dataSetInformation"), dict) else {}
    uuid_value = data_info.get("common:UUID")
    if isinstance(uuid_value, str) and uuid_value.strip():
        return uuid_value.strip()
    raise SystemExit("Generated source payload missing sourceInformation.dataSetInformation.common:UUID")


def _extract_source_version(source_payload: dict[str, Any]) -> str:
    dataset = source_payload.get("sourceDataSet") if isinstance(source_payload.get("sourceDataSet"), dict) else {}
    admin = dataset.get("administrativeInformation") if isinstance(dataset.get("administrativeInformation"), dict) else {}
    pub = admin.get("publicationAndOwnership") if isinstance(admin.get("publicationAndOwnership"), dict) else {}
    version = pub.get("common:dataSetVersion")
    if isinstance(version, str) and version.strip():
        return version.strip()
    return "01.01.000"


def _resolve_output_dir(run_id: str | None, output_dir: Path | None) -> Path:
    if output_dir:
        return output_dir
    if not run_id:
        raise SystemExit("Missing --run-id or --output-dir.")
    return PROCESS_FROM_FLOW_ARTIFACTS_ROOT / run_id / "exports" / "sources"


def main() -> None:
    args = parse_args()
    state_path = _resolve_state_path(args.run_id, args.state_path)
    state = _load_state(state_path)

    scientific_references = state.get("scientific_references")
    if isinstance(scientific_references, dict):
        doi_map, key_map = _collect_usage_tag_map(scientific_references)
        if doi_map or key_map:
            infos = _collect_reference_infos(scientific_references)
            infos = _filter_reference_infos_by_usage(infos, doi_map=doi_map, key_map=key_map)
            source_datasets, source_references = _build_source_datasets_from_references(
                scientific_references,
                reference_infos=infos,
            )
        else:
            source_datasets, source_references = _build_source_datasets_from_references(scientific_references)
    else:
        source_datasets, source_references = _build_source_datasets_from_references(scientific_references)
    if not source_datasets:
        raise SystemExit("No source datasets generated from scientific references.")

    output_dir = _resolve_output_dir(args.run_id, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for payload in source_datasets:
        if not isinstance(payload, dict):
            continue
        uuid_value = _extract_source_uuid(payload)
        version = _extract_source_version(payload)
        filename = f"{uuid_value}_{version}.json"
        target = output_dir / filename
        dump_json(payload, target)
        written.append(target)

    if not args.no_update_state:
        state["source_datasets"] = source_datasets
        state["source_references"] = source_references
        dump_json(state, state_path)

    print(f"Wrote {len(written)} source dataset(s) to {output_dir}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    main()
