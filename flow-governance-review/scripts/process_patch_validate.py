#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

from flow_governance_common import (
    build_flow_indexes,
    deep_diff_paths,
    dump_json,
    dump_jsonl,
    ensure_dir,
    exchange_records,
    extract_process_identity,
    load_rows_from_file,
    path_contains_reference_to_flow,
    process_dataset_from_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate locally patched process rows after flow-reference repairs.")
    parser.add_argument("--original-processes-file", required=True)
    parser.add_argument("--patched-processes-file", required=True)
    parser.add_argument("--scope-flow-files", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tidas-mode", choices=("auto", "required", "skip"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    original_rows = load_rows_from_file(args.original_processes_file)
    patched_rows = load_rows_from_file(args.patched_processes_file)
    flow_rows: list[dict[str, Any]] = []
    for path in args.scope_flow_files:
        flow_rows.extend(load_rows_from_file(path))
    flow_index = build_flow_indexes(flow_rows)
    original_map = {f"{pid}@{version}": row for row in original_rows for pid, version, _name in [extract_process_identity(row)]}

    validator = _load_tidas_validator(args.tidas_mode)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    summary = {"patched_process_count": len(patched_rows), "passed": 0, "failed": 0, "tidas_validation": validator is not None}

    for patched in patched_rows:
        process_id, version, name = extract_process_identity(patched)
        key = f"{process_id}@{version}"
        original = original_map.get(key)
        result = validate_process_patch(original, patched, flow_index, validator)
        result["process_id"] = process_id
        result["process_version"] = version
        result["process_name"] = name
        results.append(result)
        if result["ok"]:
            summary["passed"] += 1
        else:
            summary["failed"] += 1
            failures.append(result)

    dump_json(out_dir / "validation-report.json", {"summary": summary, "results": results})
    dump_jsonl(out_dir / "validation-failures.jsonl", failures)


def _load_tidas_validator(mode: str):
    if mode == "skip":
        return None
    try:
        module = importlib.import_module("tidas_sdk")
    except Exception:
        if mode == "required":
            raise
        return None
    create_process = getattr(module, "create_process", None)
    if create_process is None:
        if mode == "required":
            raise RuntimeError("tidas_sdk.create_process not available.")
        return None
    return create_process


def validate_process_patch(
    original_row: dict[str, Any] | None,
    patched_row: dict[str, Any],
    flow_index: dict[str, Any],
    validator,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if original_row is None:
        issues.append({"type": "missing_original_row", "severity": "error"})
        return {"ok": False, "issues": issues}

    diff_paths = deep_diff_paths(original_row, patched_row)
    illegal_paths = [path for path in diff_paths if not path_contains_reference_to_flow(path)]
    if illegal_paths:
        issues.append(
            {
                "type": "non_reference_changes_detected",
                "severity": "error",
                "paths": [".".join(str(part) for part in path) for path in illegal_paths],
            }
        )

    original_qref = str(
        (process_dataset_from_row(original_row).get("processInformation", {}).get("quantitativeReference", {}) or {}).get(
            "referenceToReferenceFlow", ""
        )
    )
    patched_qref = str(
        (process_dataset_from_row(patched_row).get("processInformation", {}).get("quantitativeReference", {}) or {}).get(
            "referenceToReferenceFlow", ""
        )
    )
    if original_qref != patched_qref:
        issues.append(
            {
                "type": "quantitative_reference_changed",
                "severity": "error",
                "before": original_qref,
                "after": patched_qref,
            }
        )

    original_exchange_count = len(exchange_records(original_row))
    patched_exchange_count = len(exchange_records(patched_row))
    if original_exchange_count != patched_exchange_count:
        issues.append(
            {
                "type": "exchange_count_changed",
                "severity": "error",
                "before": original_exchange_count,
                "after": patched_exchange_count,
            }
        )

    for exchange in exchange_records(patched_row):
        ref = exchange.get("referenceToFlowDataSet") or {}
        flow_uuid = str(ref.get("@refObjectId") or "").strip()
        flow_version = str(ref.get("@version") or "").strip()
        if not flow_uuid or not flow_version:
            issues.append(
                {
                    "type": "missing_flow_reference_after_patch",
                    "severity": "error",
                    "exchange_internal_id": exchange.get("@dataSetInternalID"),
                    "flow_id": flow_uuid,
                    "flow_version": flow_version,
                }
            )
            continue
        if not flow_index["by_uuid_version"].get(f"{flow_uuid}@{flow_version}"):
            issues.append(
                {
                    "type": "patched_reference_not_in_scope_catalog",
                    "severity": "error",
                    "exchange_internal_id": exchange.get("@dataSetInternalID"),
                    "flow_id": flow_uuid,
                    "flow_version": flow_version,
                }
            )

    if validator is not None:
        payload = process_dataset_from_row(patched_row)
        entity = validator(payload)
        if hasattr(entity, "validateEnhanced"):
            validation = entity.validateEnhanced()
        elif hasattr(entity, "validate"):
            validation = entity.validate()
        else:
            validation = None
        success = bool(getattr(validation, "success", False))
        if not success:
            issues.append(
                {
                    "type": "tidas_validation_failed",
                    "severity": "error",
                    "details": str(validation) if validation is not None else "Validator returned no result.",
                }
            )

    return {"ok": not issues, "issues": issues}


if __name__ == "__main__":
    main()
