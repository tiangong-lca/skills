#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from copy import deepcopy
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
    flow_reference_from_record,
    load_json,
    load_rows_from_file,
    normalize_text,
    patched_flow_reference,
    safe_process_key,
    sync_process_pool_file,
    unified_json_diff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or apply deterministic process-flow reference repairs.")
    parser.add_argument("--processes-file", required=True)
    parser.add_argument("--scope-flow-files", nargs="+", required=True)
    parser.add_argument("--alias-map")
    parser.add_argument("--scan-findings")
    parser.add_argument(
        "--auto-patch-policy",
        choices=("disabled", "alias-only", "alias-or-unique-name"),
        default="alias-only",
        help="Control which deterministic repair heuristics may auto-patch process flow references.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--process-pool-file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    processes = load_rows_from_file(args.processes_file)
    flow_rows: list[dict[str, Any]] = []
    for path in args.scope_flow_files:
        flow_rows.extend(load_rows_from_file(path))
    flow_index = build_flow_indexes(flow_rows)
    alias_map = load_json(args.alias_map) if args.alias_map else {}
    scan_findings = load_json(args.scan_findings) if args.scan_findings else None
    finding_map = _scan_map(scan_findings or [])

    repair_plan: list[dict[str, Any]] = []
    manual_queue: list[dict[str, Any]] = []
    patched_rows: list[dict[str, Any]] = []
    patch_root = ensure_dir(out_dir / "process-patches")

    for process_row in processes:
        process_id, version, name = extract_process_identity(process_row)
        process_key = f"{process_id}@{version}"
        working_row = deepcopy(process_row)
        process_plan: list[dict[str, Any]] = []
        changed = False
        for exchange in exchange_records(working_row):
            exchange_id = str(exchange.get("@dataSetInternalID") or "").strip()
            finding = finding_map.get((process_key, exchange_id))
            action = plan_exchange_repair(
                exchange,
                flow_index,
                alias_map,
                finding,
                auto_patch_policy=args.auto_patch_policy,
            )
            action["process_id"] = process_id
            action["process_version"] = version
            action["process_name"] = name
            process_plan.append(action)
            if action["decision"] == "manual_review":
                manual_queue.append(action)
                continue
            if args.apply and action["decision"] == "auto_patch":
                exchange["referenceToFlowDataSet"] = action["target_reference"]
                changed = True

        repair_plan.extend(process_plan)
        if args.apply and changed:
            patched_rows.append(working_row)
            process_dir = ensure_dir(patch_root / safe_process_key(process_id, version))
            dump_json(process_dir / "before.json", process_row)
            dump_json(process_dir / "after.json", working_row)
            dump_json(process_dir / "evidence.json", [item for item in process_plan if item["decision"] == "auto_patch"])
            (process_dir / "diff.patch").write_text(unified_json_diff(process_row, working_row), encoding="utf-8")

    dump_json(out_dir / "repair-plan.json", repair_plan)
    dump_jsonl(out_dir / "repair-plan.jsonl", repair_plan)
    dump_jsonl(out_dir / "manual-review-queue.jsonl", manual_queue)
    dump_json(
        out_dir / "repair-summary.json",
        {
            "auto_patch_policy": args.auto_patch_policy,
            "process_count": len(processes),
            "repair_item_count": len(repair_plan),
            "decision_counts": {
                "keep_as_is": sum(1 for item in repair_plan if item.get("decision") == "keep_as_is"),
                "auto_patch": sum(1 for item in repair_plan if item.get("decision") == "auto_patch"),
                "manual_review": sum(1 for item in repair_plan if item.get("decision") == "manual_review"),
            },
            "patched_process_count": len(patched_rows) if args.apply else 0,
        },
    )
    if args.apply:
        dump_json(out_dir / "patched-processes.json", patched_rows)
        if args.process_pool_file:
            pool_sync = sync_process_pool_file(args.process_pool_file, patched_rows)
            repair_summary_path = out_dir / "repair-summary.json"
            repair_summary = load_json(repair_summary_path)
            repair_summary["process_pool_sync"] = pool_sync
            dump_json(repair_summary_path, repair_summary)


def _scan_map(scan_findings: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for finding in scan_findings:
        if not isinstance(finding, dict):
            continue
        process_key = f"{finding.get('process_id', '')}@{finding.get('process_version', '')}"
        exchange_id = str(finding.get("exchange_internal_id") or "")
        result[(process_key, exchange_id)] = finding
    return result


def plan_exchange_repair(
    exchange: dict[str, Any],
    flow_index: dict[str, Any],
    alias_map: dict[str, Any],
    finding: dict[str, Any] | None,
    *,
    auto_patch_policy: str,
) -> dict[str, Any]:
    ref = exchange.get("referenceToFlowDataSet") or {}
    flow_uuid = str(ref.get("@refObjectId") or "").strip()
    flow_version = str(ref.get("@version") or "").strip()
    exchange_name = _ref_text(ref) or str(exchange.get("exchangeName") or exchange.get("name") or "").strip()
    base = {
        "exchange_internal_id": str(exchange.get("@dataSetInternalID") or "").strip(),
        "current_flow_id": flow_uuid,
        "current_flow_version": flow_version,
        "exchange_direction": exchange.get("exchangeDirection"),
        "exchange_text": exchange_name,
        "current_issue_type": (finding or {}).get("issue_type"),
        "auto_patch_policy": auto_patch_policy,
        "decision": "manual_review",
        "reason": "no_deterministic_match",
    }

    current_record = flow_index["by_uuid_version"].get(f"{flow_uuid}@{flow_version}") if flow_uuid and flow_version else None
    if current_record:
        return {
            **base,
            "decision": "keep_as_is",
            "reason": "already_in_target",
            "target_flow_id": current_record.id,
            "target_flow_version": current_record.version,
        }

    alias_target = alias_lookup(alias_map, flow_uuid, flow_version or None)
    if alias_target:
        record = flow_index["by_uuid_version"].get(f"{alias_target.get('id')}@{alias_target.get('version')}")
        if record and auto_patch_policy in {"alias-only", "alias-or-unique-name"}:
            return {
                **base,
                "decision": "auto_patch",
                "reason": "direct_alias_map",
                "target_flow_id": record.id,
                "target_flow_version": record.version,
                "target_flow_name": record.name,
                "target_reference": patched_flow_reference(ref, record),
            }
        if record:
            return {
                **base,
                "decision": "manual_review",
                "reason": "alias_target_found_but_policy_disallows_auto_patch",
                "candidate_count": 1,
                "candidate_refs": [{"id": record.id, "version": record.version, "name": record.name}],
            }

    candidates = flow_index["by_name"].get(normalize_text(exchange_name), [])
    if len(candidates) == 1:
        record = candidates[0]
        if auto_patch_policy == "alias-or-unique-name":
            return {
                **base,
                "decision": "auto_patch",
                "reason": "unique_exact_name_match",
                "target_flow_id": record.id,
                "target_flow_version": record.version,
                "target_flow_name": record.name,
                "target_reference": patched_flow_reference(ref, record),
            }
        return {
            **base,
            "decision": "manual_review",
            "reason": "unique_exact_name_match_blocked_by_policy",
            "candidate_count": 1,
            "candidate_refs": [{"id": record.id, "version": record.version, "name": record.name}],
        }
    if len(candidates) > 1:
        return {
            **base,
            "decision": "manual_review",
            "reason": "ambiguous_exact_name_match",
            "candidate_count": len(candidates),
            "candidate_refs": [{"id": item.id, "version": item.version, "name": item.name} for item in candidates],
        }
    return base


def _ref_text(ref: dict[str, Any]) -> str:
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
