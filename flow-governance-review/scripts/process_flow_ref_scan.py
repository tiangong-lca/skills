#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from flow_governance_common import (
    alias_lookup,
    build_flow_indexes,
    dump_json,
    dump_jsonl,
    ensure_dir,
    exchange_records,
    extract_process_identity,
    filter_emergy_named_processes,
    load_json,
    load_rows_from_file,
    merge_flow_indexes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan process-to-flow references against a local flow scope.")
    parser.add_argument("--processes-file", required=True)
    parser.add_argument("--scope-flow-files", nargs="+", required=True)
    parser.add_argument("--catalog-flow-files", nargs="*")
    parser.add_argument("--alias-map")
    parser.add_argument(
        "--exclude-emergy",
        action="store_true",
        help="Exclude emergy-named processes before scanning references.",
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    processes = load_rows_from_file(args.processes_file)
    scope_indexes = [build_flow_indexes(load_rows_from_file(path), scope_group=Path(path).stem) for path in args.scope_flow_files]
    scope_index = merge_flow_indexes(*scope_indexes)
    catalog_files = args.catalog_flow_files or args.scope_flow_files
    catalog_indexes = [build_flow_indexes(load_rows_from_file(path), scope_group=Path(path).stem) for path in catalog_files]
    catalog_index = merge_flow_indexes(*catalog_indexes)
    if args.exclude_emergy:
        processes, emergy_excluded_processes = filter_emergy_named_processes(processes, flow_index=catalog_index)
    else:
        emergy_excluded_processes = []
    alias_map = load_json(args.alias_map) if args.alias_map else {}
    findings: list[dict[str, Any]] = []
    summary = {
        "process_count_before_emergy_exclusion": len(processes) + len(emergy_excluded_processes),
        "process_count": len(processes),
        "emergy_excluded_process_count": len(emergy_excluded_processes),
        "exchange_count": 0,
        "issue_counts": {},
        "processes_with_issues": 0,
    }
    issue_process_keys: set[str] = set()

    for process_row in processes:
        process_id, version, name = extract_process_identity(process_row)
        exchanges = exchange_records(process_row)
        summary["exchange_count"] += len(exchanges)
        for exchange in exchanges:
            finding = classify_exchange_ref(
                process_id=process_id,
                process_version=version,
                process_name=name,
                exchange=exchange,
                scope_index=scope_index,
                catalog_index=catalog_index,
                alias_map=alias_map,
            )
            findings.append(finding)
            issue = finding["issue_type"]
            summary["issue_counts"][issue] = summary["issue_counts"].get(issue, 0) + 1
            if issue != "exists_in_target":
                issue_process_keys.add(f"{process_id}@{version}")

    summary["processes_with_issues"] = len(issue_process_keys)
    dump_json(out_dir / "emergy-excluded-processes.json", emergy_excluded_processes)
    dump_json(out_dir / "scan-summary.json", summary)
    dump_json(out_dir / "scan-findings.json", findings)
    dump_jsonl(out_dir / "scan-findings.jsonl", findings)


def classify_exchange_ref(
    *,
    process_id: str,
    process_version: str,
    process_name: str,
    exchange: dict[str, Any],
    scope_index: dict[str, Any],
    catalog_index: dict[str, Any],
    alias_map: dict[str, Any],
) -> dict[str, Any]:
    ref = exchange.get("referenceToFlowDataSet") or {}
    flow_uuid = str(ref.get("@refObjectId") or "").strip()
    flow_version = str(ref.get("@version") or "").strip()
    short_description = _extract_ref_text(ref)
    dataset_internal_id = str(exchange.get("@dataSetInternalID") or "").strip()
    finding = {
        "process_id": process_id,
        "process_version": process_version,
        "process_name": process_name,
        "exchange_internal_id": dataset_internal_id,
        "exchange_direction": exchange.get("exchangeDirection"),
        "reference_flow_id": flow_uuid,
        "reference_flow_version": flow_version,
        "reference_flow_text": short_description,
        "issue_type": "exists_in_target",
        "severity": "info",
        "evidence": {},
    }

    if not flow_uuid:
        finding["issue_type"] = "no_reference"
        finding["severity"] = "error"
        return finding

    alias_target = alias_lookup(alias_map, flow_uuid, flow_version or None)
    if alias_target:
        finding["issue_type"] = "alias_target_available"
        finding["severity"] = "warning"
        finding["evidence"] = {"alias_target": alias_target}

    if not flow_version:
        if scope_index["by_uuid"].get(flow_uuid):
            finding["issue_type"] = "version_missing"
            finding["severity"] = "warning"
            return finding
        if catalog_index["by_uuid"].get(flow_uuid):
            finding["issue_type"] = "exists_outside_target"
            finding["severity"] = "warning"
            return finding
        finding["issue_type"] = "missing_uuid"
        finding["severity"] = "error"
        return finding

    scope_match = scope_index["by_uuid_version"].get(f"{flow_uuid}@{flow_version}")
    if scope_match:
        finding["evidence"] = {"scope_group": scope_match.scope_group, "flow_name": scope_match.name}
        return finding

    if scope_index["by_uuid"].get(flow_uuid):
        finding["issue_type"] = "broken_version"
        finding["severity"] = "error"
        finding["evidence"] = {
            "available_versions_in_target": sorted(record.version for record in scope_index["by_uuid"][flow_uuid]),
        }
        return finding

    catalog_match = catalog_index["by_uuid_version"].get(f"{flow_uuid}@{flow_version}")
    if catalog_match or catalog_index["by_uuid"].get(flow_uuid):
        finding["issue_type"] = "exists_outside_target"
        finding["severity"] = "warning"
        return finding

    finding["issue_type"] = "missing_uuid"
    finding["severity"] = "error"
    return finding


def _extract_ref_text(ref: dict[str, Any]) -> str:
    value = ref.get("common:shortDescription")
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item.get("#text"):
                return str(item["#text"]).strip()
    if isinstance(value, dict) and value.get("#text"):
        return str(value["#text"]).strip()
    return ""


if __name__ == "__main__":
    main()
