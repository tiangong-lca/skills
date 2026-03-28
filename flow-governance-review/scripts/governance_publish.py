#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

from flow_governance_common import (
    build_flow_indexes,
    dataset_payload_from_row,
    dump_json,
    dump_jsonl,
    ensure_dir,
    exchange_records,
    extract_flow_identity,
    extract_process_identity,
    flow_dataset_from_row,
    json_equal,
    load_rows_from_file,
    patched_flow_reference,
    process_dataset_from_row,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROCESS_AUTOMATED_BUILDER_DIR = REPO_ROOT / "process-automated-builder"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and optionally publish reviewed flow/process rows. "
            "Flow rows can publish append-only with a bumped version, and process rows can be rewritten "
            "to reference those published flow versions before publish."
        )
    )
    parser.add_argument("--flow-rows-file", help="Local reviewed flow rows JSON/JSONL file.")
    parser.add_argument(
        "--original-flow-rows-file",
        help="Optional original flow rows JSON/JSONL file used to skip publish for unchanged flows.",
    )
    parser.add_argument("--process-rows-file", help="Local reviewed process rows JSON/JSONL file.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--flow-publish-policy",
        choices=("skip", "append_only_bump", "upsert_current_version"),
        default="append_only_bump",
    )
    parser.add_argument(
        "--process-publish-policy",
        choices=("skip", "append_only_bump", "upsert_current_version"),
        default="append_only_bump",
    )
    parser.add_argument(
        "--no-rewrite-process-flow-refs",
        action="store_true",
        help="Do not rewrite process flow references to the prepared published flow versions.",
    )
    parser.add_argument("--commit", action="store_true", help="Actually publish via Database_CRUD_Tool.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.flow_rows_file and not args.process_rows_file:
        raise SystemExit("Provide at least one of --flow-rows-file or --process-rows-file.")

    out_dir = ensure_dir(args.out_dir)

    input_flow_rows = load_rows_from_file(args.flow_rows_file) if args.flow_rows_file else []
    original_flow_rows = load_rows_from_file(args.original_flow_rows_file) if args.original_flow_rows_file else []
    input_process_rows = load_rows_from_file(args.process_rows_file) if args.process_rows_file else []

    prepared_flow_rows, flow_plans, flow_version_map, skipped_unchanged_flow_rows = _prepare_flow_rows(
        input_flow_rows,
        policy=args.flow_publish_policy,
        original_rows=original_flow_rows,
    )
    prepared_process_rows, process_plans, process_ref_rewrite_evidence = _prepare_process_rows(
        input_process_rows,
        policy=args.process_publish_policy,
        rewrite_refs=not args.no_rewrite_process_flow_refs,
        prepared_flow_rows=prepared_flow_rows,
        flow_version_map=flow_version_map,
    )

    dump_json(out_dir / "prepared-flow-rows.json", prepared_flow_rows)
    dump_json(out_dir / "prepared-process-rows.json", prepared_process_rows)
    dump_json(out_dir / "flow-version-map.json", flow_version_map)
    dump_json(out_dir / "skipped-unchanged-flow-rows.json", skipped_unchanged_flow_rows)
    dump_jsonl(out_dir / "process-flow-ref-rewrite-evidence.jsonl", process_ref_rewrite_evidence)

    flow_reports: list[dict[str, Any]] = []
    process_reports: list[dict[str, Any]] = []
    if args.commit:
        client = _create_crud_client()
        try:
            flow_reports = _publish_flow_plans(flow_plans, client=client, commit=True)
            process_reports = _publish_process_plans(process_plans, client=client, commit=True)
        finally:
            client.close()
    else:
        flow_reports = _publish_flow_plans(flow_plans, client=None, commit=False)
        process_reports = _publish_process_plans(process_plans, client=None, commit=False)

    report = {
        "generated_at_utc": _now_iso(),
        "commit": bool(args.commit),
        "flow_publish_policy": args.flow_publish_policy,
        "process_publish_policy": args.process_publish_policy,
        "rewrite_process_flow_refs": not args.no_rewrite_process_flow_refs,
        "counts": {
            "input_flow_rows": len(input_flow_rows),
            "original_flow_rows": len(original_flow_rows),
            "input_process_rows": len(input_process_rows),
            "prepared_flow_rows": len(prepared_flow_rows),
            "skipped_unchanged_flow_rows": len(skipped_unchanged_flow_rows),
            "prepared_process_rows": len(prepared_process_rows),
            "rewritten_process_flow_refs": len(process_ref_rewrite_evidence),
            "flow_publish_reports": len(flow_reports),
            "process_publish_reports": len(process_reports),
        },
        "files": {
            "prepared_flow_rows": str(out_dir / "prepared-flow-rows.json"),
            "prepared_process_rows": str(out_dir / "prepared-process-rows.json"),
            "flow_version_map": str(out_dir / "flow-version-map.json"),
            "skipped_unchanged_flow_rows": str(out_dir / "skipped-unchanged-flow-rows.json"),
            "process_ref_rewrite_evidence": str(out_dir / "process-flow-ref-rewrite-evidence.jsonl"),
        },
        "flow_reports": flow_reports,
        "process_reports": process_reports,
        "skipped_unchanged_flow_rows": skipped_unchanged_flow_rows,
    }
    dump_json(out_dir / "publish-report.json", report)
    print(str(out_dir / "publish-report.json"))


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _prepare_flow_rows(
    rows: list[dict[str, Any]],
    *,
    policy: str,
    original_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, str]], list[dict[str, Any]]]:
    if policy == "skip":
        return [], [], {}, []
    prepared_rows: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    flow_version_map: dict[str, dict[str, str]] = {}
    skipped_unchanged_rows: list[dict[str, Any]] = []
    original_map = {
        f"{flow_id}@{version}": row
        for row in (original_rows or [])
        for flow_id, version, _name in [extract_flow_identity(row)]
        if flow_id and version
    }
    for row in rows:
        working = copy.deepcopy(row)
        flow_id, current_version, name = extract_flow_identity(working)
        original_row = original_map.get(f"{flow_id}@{current_version}")
        if original_row is not None and json_equal(dataset_payload_from_row(original_row), dataset_payload_from_row(row)):
            skipped_unchanged_rows.append(
                {
                    "entity_type": "flow",
                    "entity_id": flow_id,
                    "entity_name": name,
                    "version": current_version,
                    "reason": "unchanged_vs_original_rows_file",
                }
            )
            continue
        publish_version = current_version
        version_strategy = "keep_current"
        if policy == "append_only_bump":
            publish_version = _bump_ilcd_version(current_version)
            _set_entity_version(working, "flow", publish_version)
            version_strategy = "bump"
            flow_version_map[f"{flow_id}@{current_version}"] = {
                "id": flow_id,
                "source_version": current_version,
                "target_version": publish_version,
            }
        prepared_rows.append(working)
        plans.append(
            {
                "entity_type": "flow",
                "entity_id": flow_id,
                "entity_name": name,
                "original_version": current_version,
                "publish_version": publish_version,
                "version_strategy": version_strategy,
                "publish_policy": policy,
                "row": working,
            }
        )
    return prepared_rows, plans, flow_version_map, skipped_unchanged_rows


def _prepare_process_rows(
    rows: list[dict[str, Any]],
    *,
    policy: str,
    rewrite_refs: bool,
    prepared_flow_rows: list[dict[str, Any]],
    flow_version_map: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if policy == "skip":
        return [], [], []

    prepared_rows = [copy.deepcopy(row) for row in rows]
    rewrite_evidence: list[dict[str, Any]] = []
    if rewrite_refs and prepared_flow_rows and flow_version_map:
        target_index = build_flow_indexes(prepared_flow_rows)["by_uuid_version"]
        for row in prepared_rows:
            process_id, process_version, process_name = extract_process_identity(row)
            for exchange in exchange_records(row):
                current_ref = exchange.get("referenceToFlowDataSet") or {}
                if not isinstance(current_ref, dict):
                    continue
                flow_id = str(current_ref.get("@refObjectId") or "").strip()
                flow_version = str(current_ref.get("@version") or "").strip()
                mapped = flow_version_map.get(f"{flow_id}@{flow_version}")
                if mapped is None:
                    continue
                target_key = f"{mapped['id']}@{mapped['target_version']}"
                target_record = target_index.get(target_key)
                if target_record is None:
                    continue
                exchange["referenceToFlowDataSet"] = patched_flow_reference(current_ref, target_record)
                rewrite_evidence.append(
                    {
                        "process_id": process_id,
                        "process_version_before_publish": process_version,
                        "process_name": process_name,
                        "exchange_internal_id": str(exchange.get("@dataSetInternalID") or "").strip(),
                        "source_flow_id": flow_id,
                        "source_flow_version": flow_version,
                        "target_flow_id": mapped["id"],
                        "target_flow_version": mapped["target_version"],
                        "target_flow_name": target_record.name,
                    }
                )

    plans: list[dict[str, Any]] = []
    for row in prepared_rows:
        process_id, current_version, name = extract_process_identity(row)
        publish_version = current_version
        version_strategy = "keep_current"
        if policy == "append_only_bump":
            publish_version = _bump_ilcd_version(current_version)
            _set_entity_version(row, "process", publish_version)
            version_strategy = "bump"
        plans.append(
            {
                "entity_type": "process",
                "entity_id": process_id,
                "entity_name": name,
                "original_version": current_version,
                "publish_version": publish_version,
                "version_strategy": version_strategy,
                "publish_policy": policy,
                "row": row,
            }
        )
    return prepared_rows, plans, rewrite_evidence


def _publish_flow_plans(
    plans: list[dict[str, Any]],
    *,
    client,
    commit: bool,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for plan in plans:
        row = plan["row"]
        report = {
            "entity_type": "flow",
            "id": plan["entity_id"],
            "name": plan["entity_name"],
            "original_version": plan["original_version"],
            "publish_version": plan["publish_version"],
            "publish_policy": plan["publish_policy"],
            "version_strategy": plan["version_strategy"],
            "status": "prepared" if not commit else "pending",
        }
        if not commit:
            reports.append(report)
            continue
        payload = _publish_payload_from_row(row, "flow")
        flow_id = str(plan["entity_id"] or "").strip()
        publish_version = str(plan["publish_version"] or "").strip()
        try:
            if plan["publish_policy"] == "append_only_bump":
                existing = client.select_flow(flow_id, version=publish_version)
                if isinstance(existing, dict):
                    report["status"] = "skipped_existing"
                else:
                    report["insert"] = client.insert_flow(payload)
                    report["status"] = "inserted"
            else:
                outcome = _publish_with_fallback(
                    prefer_update=True,
                    insert_fn=lambda: client.insert_flow(payload),
                    update_fn=lambda: client.update_flow(payload),
                )
                report.update(outcome)
        except Exception as exc:  # noqa: BLE001
            report["status"] = "failed"
            report["error"] = str(exc)
        reports.append(report)
    return reports


def _publish_process_plans(
    plans: list[dict[str, Any]],
    *,
    client,
    commit: bool,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for plan in plans:
        row = plan["row"]
        report = {
            "entity_type": "process",
            "id": plan["entity_id"],
            "name": plan["entity_name"],
            "original_version": plan["original_version"],
            "publish_version": plan["publish_version"],
            "publish_policy": plan["publish_policy"],
            "version_strategy": plan["version_strategy"],
            "status": "prepared" if not commit else "pending",
        }
        if not commit:
            reports.append(report)
            continue
        payload = _publish_payload_from_row(row, "process")
        process_id = str(plan["entity_id"] or "").strip()
        publish_version = str(plan["publish_version"] or "").strip()
        try:
            if plan["publish_policy"] == "append_only_bump":
                existing = client.select_process(process_id, version=publish_version)
                if isinstance(existing, dict):
                    report["status"] = "skipped_existing"
                else:
                    report["insert"] = client.insert_process(payload)
                    report["status"] = "inserted"
            else:
                outcome = _publish_with_fallback(
                    prefer_update=True,
                    insert_fn=lambda: client.insert_process(payload),
                    update_fn=lambda: client.update_process(payload),
                )
                report.update(outcome)
        except Exception as exc:  # noqa: BLE001
            report["status"] = "failed"
            report["error"] = str(exc)
        reports.append(report)
    return reports


def _publish_with_fallback(
    *,
    prefer_update: bool,
    insert_fn,
    update_fn,
) -> dict[str, Any]:
    attempts: list[dict[str, str]] = []
    ordered_ops = [("update", update_fn), ("insert", insert_fn)] if prefer_update else [("insert", insert_fn), ("update", update_fn)]
    for operation, fn in ordered_ops:
        try:
            result = fn()
            outcome: dict[str, Any] = {
                "status": "updated" if operation == "update" else "inserted",
                operation: result,
            }
            if attempts:
                outcome["fallback_used"] = True
                outcome["write_attempts"] = attempts
            return outcome
        except Exception as exc:  # noqa: BLE001
            attempts.append(
                {
                    "operation": operation,
                    "error": str(exc),
                }
            )
    return {
        "status": "failed",
        "error": attempts[-1]["error"] if attempts else "publish_failed",
        "write_attempts": attempts,
    }


def _publish_payload_from_row(row: dict[str, Any], entity_type: str) -> dict[str, Any]:
    payload = dataset_payload_from_row(row)
    if entity_type == "flow":
        dataset = payload.get("flowDataSet") if isinstance(payload.get("flowDataSet"), dict) else flow_dataset_from_row(row)
        return {"flowDataSet": copy.deepcopy(dataset)}
    if entity_type == "process":
        dataset = payload.get("processDataSet") if isinstance(payload.get("processDataSet"), dict) else process_dataset_from_row(row)
        return {"processDataSet": copy.deepcopy(dataset)}
    raise RuntimeError(f"Unsupported entity_type: {entity_type}")


def _set_entity_version(row: dict[str, Any], entity_type: str, new_version: str) -> None:
    payload = dataset_payload_from_row(row)
    if entity_type == "flow":
        dataset = payload.get("flowDataSet") if isinstance(payload.get("flowDataSet"), dict) else flow_dataset_from_row(row)
    elif entity_type == "process":
        dataset = payload.get("processDataSet") if isinstance(payload.get("processDataSet"), dict) else process_dataset_from_row(row)
    else:
        raise RuntimeError(f"Unsupported entity_type: {entity_type}")

    admin = dataset.get("administrativeInformation")
    if not isinstance(admin, dict):
        dataset["administrativeInformation"] = {}
        admin = dataset["administrativeInformation"]
    publication = admin.get("publicationAndOwnership")
    if not isinstance(publication, dict):
        admin["publicationAndOwnership"] = {}
        publication = admin["publicationAndOwnership"]
    publication["common:dataSetVersion"] = new_version
    row["version"] = new_version


def _bump_ilcd_version(version: str) -> str:
    text = str(version or "").strip()
    parts = text.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return "01.01.001"
    head, middle, tail = parts
    return f"{int(head):0{len(head)}d}.{int(middle):0{len(middle)}d}.{int(tail) + 1:0{len(tail)}d}"


def _create_crud_client():
    if str(PROCESS_AUTOMATED_BUILDER_DIR) not in sys.path:
        sys.path.insert(0, str(PROCESS_AUTOMATED_BUILDER_DIR))
    from tiangong_lca_spec.publishing.crud import DatabaseCrudClient

    return DatabaseCrudClient()


if __name__ == "__main__":
    main()
