#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from flow_governance_common import (
    deep_get,
    dump_json,
    dump_jsonl,
    ensure_dir,
    FLOW_PROCESSING_ARTIFACT_ROOT,
    FLOW_PROCESSING_DATASETS_DIR,
    FLOW_PROCESSING_NAMING_DIR,
    extract_flow_identity,
    extract_process_identity,
    flow_dataset_from_row,
    lang_entries,
    lang_text,
    listify,
    load_json_or_jsonl,
    postgrest_auth_password,
    postgrest_select_page,
    process_row_key,
    process_row_sort_key,
    select_reference_flow_property,
    sync_process_pool_file,
    version_key,
)


DEFAULT_BASE_FLOWS_FILE = (
    FLOW_PROCESSING_DATASETS_DIR / "flows_tidas_sdk_plus_classification_round2_sdk018_all_final_resolved.jsonl"
)
DEFAULT_OVERRIDE_FLOWS_FILE = ""
DEFAULT_PROCESS_POOL_FILE = FLOW_PROCESSING_DATASETS_DIR / "process_pool.jsonl"
DEFAULT_ANALYSIS_DIR = FLOW_PROCESSING_NAMING_DIR / "incomplete-analysis"
DEFAULT_MERGED_OUTPUT_FILE = DEFAULT_ANALYSIS_DIR / "merged-flow-scope.jsonl"
DEFAULT_WITH_PROCESS_OUTPUT_FILE = DEFAULT_ANALYSIS_DIR / "incomplete-with-process-checklist.jsonl"
DEFAULT_ZERO_PROCESS_OUTPUT_FILE = DEFAULT_ANALYSIS_DIR / "incomplete-zero-process.jsonl"
DEFAULT_SUMMARY_OUTPUT_FILE = DEFAULT_ANALYSIS_DIR / "incomplete-summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the final flow pool, optionally apply an override flow file, find flows with incomplete 3-field names, "
            "scan visible referencing processes, sync a shared process pool, and emit "
            "flow completion checklists."
        )
    )
    parser.add_argument("--base-flows-file", default=str(DEFAULT_BASE_FLOWS_FILE))
    parser.add_argument("--override-flows-file", default=str(DEFAULT_OVERRIDE_FLOWS_FILE))
    parser.add_argument("--merged-output-file", default=str(DEFAULT_MERGED_OUTPUT_FILE))
    parser.add_argument("--process-pool-file", default=str(DEFAULT_PROCESS_POOL_FILE))
    parser.add_argument("--with-process-output-file", default=str(DEFAULT_WITH_PROCESS_OUTPUT_FILE))
    parser.add_argument("--zero-process-output-file", default=str(DEFAULT_ZERO_PROCESS_OUTPUT_FILE))
    parser.add_argument("--summary-output-file", default=str(DEFAULT_SUMMARY_OUTPUT_FILE))
    parser.add_argument(
        "--process-scan-mode",
        choices=("auto", "full_scan", "candidate_lookup"),
        default="auto",
        help=(
            "How to find referencing processes. auto uses full_scan when the target flow count "
            "exceeds --full-scan-threshold; otherwise candidate_lookup."
        ),
    )
    parser.add_argument(
        "--full-scan-threshold",
        type=int,
        default=200,
        help="Switch to full visible-process scan in auto mode when target flow count exceeds this threshold.",
    )
    parser.add_argument("--process-page-size", type=int, default=200)
    parser.add_argument(
        "--process-fetch-id-chunk-size",
        type=int,
        default=40,
        help="How many process ids to fetch per full-row chunk when materializing the process pool.",
    )
    parser.add_argument(
        "--flow-id",
        action="append",
        dest="flow_ids",
        default=[],
        help="Optional one or more flow UUID filters for scoped testing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit after filtering, mainly for scoped testing.",
    )
    parser.add_argument("--supabase-url", default=os.getenv("SUPABASE_URL", ""))
    parser.add_argument("--supabase-publishable-key", default=os.getenv("SUPABASE_PUBLISHABLE_KEY", ""))
    parser.add_argument("--access-token", default=os.getenv("SUPABASE_ACCESS_TOKEN", ""))
    parser.add_argument("--email", default=os.getenv("SUPABASE_EMAIL", ""))
    parser.add_argument("--password", default=os.getenv("SUPABASE_PASSWORD", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged_output_file = Path(args.merged_output_file).expanduser().resolve()
    with_process_output_file = Path(args.with_process_output_file).expanduser().resolve()
    zero_process_output_file = Path(args.zero_process_output_file).expanduser().resolve()
    summary_output_file = Path(args.summary_output_file).expanduser().resolve()
    process_pool_file = Path(args.process_pool_file).expanduser().resolve()
    for target in (
        merged_output_file,
        with_process_output_file,
        zero_process_output_file,
        summary_output_file,
        process_pool_file,
    ):
        ensure_dir(target.parent)

    merged_rows, merge_report = merge_final_flow_rows(
        base_rows=load_json_or_jsonl(args.base_flows_file),
        override_rows=load_optional_json_or_jsonl(args.override_flows_file),
    )
    dump_jsonl(merged_output_file, merged_rows)

    overall_three_field_stats = summarize_all_three_field_status(merged_rows)
    target_rows = [row for row in merged_rows if is_incomplete_three_field_flow(row)]
    target_rows = apply_target_flow_filters(target_rows, args.flow_ids, args.limit)

    incomplete_stats = summarize_incomplete_name_fields(target_rows)
    access_token = resolve_access_token(args)
    scan_mode = resolve_process_scan_mode(args, len(target_rows))

    process_scan = scan_referencing_processes(
        args=args,
        access_token=access_token,
        target_rows=target_rows,
        scan_mode=scan_mode,
    )

    process_keys = sorted(process_scan["matched_process_keys"])
    full_process_rows = materialize_process_rows(
        args=args,
        access_token=access_token,
        process_keys=process_keys,
    )
    pool_sync = sync_process_pool_file(process_pool_file, full_process_rows)

    with_process_rows, zero_process_rows = build_completion_outputs(
        target_rows=target_rows,
        process_scan=process_scan,
    )
    dump_jsonl(with_process_output_file, with_process_rows)
    dump_jsonl(zero_process_output_file, zero_process_rows)

    summary = {
        "merged_flow_input": {
            "base_flows_file": str(Path(args.base_flows_file).expanduser().resolve()),
            "override_flows_file": str(Path(args.override_flows_file).expanduser().resolve()),
            "merged_output_file": str(merged_output_file),
            "merge_report": merge_report,
            "merged_flow_count": len(merged_rows),
        },
        "scope": {
            "flow_id_filter": sorted(set(args.flow_ids)),
            "limit": args.limit,
            "scoped_target_flow_count": len(target_rows),
        },
        "three_field_analysis": {
            "definition": {
                "required_name_fields": [
                    "baseName",
                    "treatmentStandardsRoutes",
                    "mixAndLocationTypes",
                ],
                "note": (
                    "A flow is treated as 3-field complete only when all three name blocks contain "
                    "at least one non-empty text entry."
                ),
            },
            "overall_counts": overall_three_field_stats,
            "scoped_incomplete_counts": incomplete_stats,
        },
        "process_scan": {
            "scan_mode": scan_mode,
            "page_size": args.process_page_size,
            "visible_process_rows_scanned": process_scan["visible_process_rows_scanned"],
            "matched_process_count": len(process_keys),
            "target_flow_with_visible_process_count": len(with_process_rows),
            "target_flow_zero_visible_process_count": len(zero_process_rows),
            "visibility_scope_note": (
                "Process evidence is limited to visible state_code in (0,100) rows under current RLS. "
                "Other users' private state_code=0 processes remain invisible."
            ),
        },
        "process_pool": {
            "process_pool_file": str(process_pool_file),
            "sync_report": pool_sync,
        },
        "outputs": {
            "with_process_output_file": str(with_process_output_file),
            "zero_process_output_file": str(zero_process_output_file),
            "summary_output_file": str(summary_output_file),
        },
    }
    dump_json(summary_output_file, summary)
    print(str(summary_output_file))


def load_optional_json_or_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    text = str(path or "").strip()
    if not text:
        return []
    candidate = Path(text).expanduser()
    if not candidate.exists():
        return []
    return load_json_or_jsonl(candidate)


def merge_final_flow_rows(
    *,
    base_rows: list[dict[str, Any]],
    override_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged_by_id: dict[str, dict[str, Any]] = {}
    duplicate_base_ids: Counter[str] = Counter()
    invalid_base_rows = 0
    for row in base_rows:
        if not isinstance(row, dict):
            invalid_base_rows += 1
            continue
        flow_id, _version, _name = extract_flow_identity(row)
        if not flow_id:
            invalid_base_rows += 1
            continue
        if flow_id in merged_by_id:
            duplicate_base_ids[flow_id] += 1
        merged_by_id[flow_id] = row

    replaced = 0
    inserted = 0
    invalid_override_rows = 0
    for row in override_rows:
        if not isinstance(row, dict):
            invalid_override_rows += 1
            continue
        flow_id, _version, _name = extract_flow_identity(row)
        if not flow_id:
            invalid_override_rows += 1
            continue
        if flow_id in merged_by_id:
            replaced += 1
        else:
            inserted += 1
        merged_by_id[flow_id] = row

    merged_rows = sorted(
        merged_by_id.values(),
        key=lambda row: _flow_sort_key(row),
    )
    report = {
        "base_row_count": len(base_rows),
        "override_row_count": len(override_rows),
        "invalid_base_rows": invalid_base_rows,
        "invalid_override_rows": invalid_override_rows,
        "duplicate_base_id_count": len(duplicate_base_ids),
        "replaced_by_id_count": replaced,
        "inserted_new_id_count": inserted,
    }
    return merged_rows, report


def apply_target_flow_filters(
    rows: list[dict[str, Any]],
    flow_ids: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    filtered = rows
    flow_id_filter = {value.strip() for value in flow_ids if str(value or "").strip()}
    if flow_id_filter:
        filtered = [row for row in filtered if extract_flow_identity(row)[0] in flow_id_filter]
    if limit > 0:
        filtered = filtered[:limit]
    return filtered


def summarize_incomplete_name_fields(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    dataset_type_counts: Counter[str] = Counter()
    for row in rows:
        profile = build_flow_profile(row)
        missing_fields = profile["missing_name_fields"]
        counts["incomplete_total"] += 1
        dataset_type_counts[profile["type_of_dataset"] or ""] += 1
        if missing_fields == ["baseName"]:
            counts["missing_base_only"] += 1
        if missing_fields == ["treatmentStandardsRoutes", "mixAndLocationTypes"]:
            counts["missing_treatment_and_location"] += 1
        if missing_fields == ["treatmentStandardsRoutes"]:
            counts["missing_treatment_only"] += 1
        if missing_fields == ["mixAndLocationTypes"]:
            counts["missing_location_only"] += 1
        if "baseName" in missing_fields:
            counts["missing_base_any"] += 1
        if "treatmentStandardsRoutes" in missing_fields:
            counts["missing_treatment_any"] += 1
        if "mixAndLocationTypes" in missing_fields:
            counts["missing_location_any"] += 1
    return {
        **dict(counts),
        "type_of_dataset_counts": dict(dataset_type_counts),
    }


def summarize_all_three_field_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    incomplete_rows = [row for row in rows if is_incomplete_three_field_flow(row)]
    return {
        "total_flow_count": len(rows),
        "three_field_complete_count": len(rows) - len(incomplete_rows),
        "three_field_incomplete_count": len(incomplete_rows),
        "incomplete_breakdown": summarize_incomplete_name_fields(incomplete_rows),
    }


def resolve_access_token(args: argparse.Namespace) -> str:
    if not str(args.supabase_url or "").strip() or not str(args.supabase_publishable_key or "").strip():
        raise SystemExit(
            "SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY are required. "
            "Source the OpenClaw env first, for example: source /home/huimin/.openclaw/.env"
        )
    access_token = str(args.access_token or "").strip()
    if access_token:
        return access_token
    if args.email and args.password:
        return postgrest_auth_password(
            base_url=args.supabase_url,
            apikey=args.supabase_publishable_key,
            email=args.email,
            password=args.password,
        )
    raise SystemExit(
        "Authenticated PostgREST access is required. Set SUPABASE_ACCESS_TOKEN or SUPABASE_EMAIL/SUPABASE_PASSWORD."
    )


def resolve_process_scan_mode(args: argparse.Namespace, target_flow_count: int) -> str:
    if args.process_scan_mode != "auto":
        return args.process_scan_mode
    if target_flow_count > args.full_scan_threshold:
        return "full_scan"
    return "candidate_lookup"


def scan_referencing_processes(
    *,
    args: argparse.Namespace,
    access_token: str,
    target_rows: list[dict[str, Any]],
    scan_mode: str,
) -> dict[str, Any]:
    target_flow_ids = {extract_flow_identity(row)[0] for row in target_rows}
    if not target_flow_ids:
        return {
            "visible_process_rows_scanned": 0,
            "matched_process_keys": set(),
            "matches_by_flow_id": {},
        }

    matches_by_flow_id: dict[str, dict[str, Any]] = {
        flow_id: {
            "process_refs_by_key": {},
        }
        for flow_id in target_flow_ids
    }
    matched_process_keys: set[str] = set()
    visible_process_rows_scanned = 0

    if scan_mode == "full_scan":
        offset = 0
        while True:
            batch = postgrest_select_page(
                base_url=args.supabase_url,
                apikey=args.supabase_publishable_key,
                access_token=access_token,
                table="processes",
                raw_filters={"state_code": "in.(0,100)"},
                columns=(
                    "id,version,state_code,user_id,"
                    "json->processDataSet->processInformation->dataSetInformation->name,"
                    "json->processDataSet->processInformation->quantitativeReference->>referenceToReferenceFlow,"
                    "json->processDataSet->exchanges->exchange"
                ),
                limit=args.process_page_size,
                offset=offset,
                order="id.asc",
            )
            if not batch:
                break
            for row in batch:
                visible_process_rows_scanned += 1
                process_key = process_row_key(row)
                if not process_key:
                    continue
                match_details = build_process_match_details(row, target_flow_ids)
                if not match_details:
                    continue
                matched_process_keys.add(process_key)
                for detail in match_details:
                    record_process_match(matches_by_flow_id, detail)
            if len(batch) < args.process_page_size:
                break
            offset += args.process_page_size
    else:
        for flow_id in sorted(target_flow_ids):
            offset = 0
            while True:
                batch = postgrest_select_page(
                    base_url=args.supabase_url,
                    apikey=args.supabase_publishable_key,
                    access_token=access_token,
                    table="processes",
                    raw_filters={
                        "state_code": "in.(0,100)",
                        "json->processDataSet->exchanges->exchange": "cs." + json_contains_flow_id(flow_id),
                    },
                    columns=(
                        "id,version,state_code,user_id,"
                        "json->processDataSet->processInformation->dataSetInformation->name,"
                        "json->processDataSet->processInformation->quantitativeReference->>referenceToReferenceFlow,"
                        "json->processDataSet->exchanges->exchange"
                    ),
                    limit=args.process_page_size,
                    offset=offset,
                    order="id.asc",
                )
                if not batch:
                    break
                for row in batch:
                    visible_process_rows_scanned += 1
                    process_key = process_row_key(row)
                    if not process_key:
                        continue
                    detail = build_process_match_details(row, {flow_id})
                    if not detail:
                        continue
                    matched_process_keys.add(process_key)
                    for item in detail:
                        record_process_match(matches_by_flow_id, item)
                if len(batch) < args.process_page_size:
                    break
                offset += args.process_page_size

    finalized_matches = finalize_matches_by_flow_id(matches_by_flow_id)
    return {
        "visible_process_rows_scanned": visible_process_rows_scanned,
        "matched_process_keys": matched_process_keys,
        "matches_by_flow_id": finalized_matches,
    }


def finalize_matches_by_flow_id(matches_by_flow_id: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for flow_id, info in matches_by_flow_id.items():
        process_refs = sorted(
            info["process_refs_by_key"].values(),
            key=lambda item: (
                0 if item["references_target_as_quantitative_reference"] else 1,
                item["process_name"],
                item["process_key"],
            ),
        )
        finalized[flow_id] = {
            "process_count": len(process_refs),
            "reference_process_count": sum(
                1 for item in process_refs if item["references_target_as_quantitative_reference"]
            ),
            "exchange_count": sum(item["matched_exchange_count"] for item in process_refs),
            "process_refs": process_refs,
        }
    return finalized


def record_process_match(matches_by_flow_id: dict[str, dict[str, Any]], detail: dict[str, Any]) -> None:
    flow_id = detail["target_flow_id"]
    flow_bucket = matches_by_flow_id[flow_id]
    process_key = detail["process_key"]
    existing = flow_bucket["process_refs_by_key"].get(process_key)
    if not existing or _prefer_process_row(detail, existing):
        flow_bucket["process_refs_by_key"][process_key] = detail


def build_process_match_details(process_row: dict[str, Any], target_flow_ids: set[str]) -> list[dict[str, Any]]:
    process_id, version, fallback_name = extract_process_identity(process_row)
    name = lang_text(process_row.get("name")) or fallback_name
    process_key = process_row_key(process_row)
    if not process_key:
        return []

    exchanges = [item for item in listify(process_row.get("exchange")) if isinstance(item, dict)]
    reference_internal_id = str(process_row.get("referenceToReferenceFlow") or "").strip()
    reference_exchange = None
    for exchange in exchanges:
        if reference_internal_id and str(exchange.get("@dataSetInternalID") or "").strip() == reference_internal_id:
            reference_exchange = exchange
            break

    matches_by_flow_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for exchange in exchanges:
        ref = exchange.get("referenceToFlowDataSet") or {}
        if not isinstance(ref, dict):
            continue
        flow_id = str(ref.get("@refObjectId") or "").strip()
        if flow_id not in target_flow_ids:
            continue
        matches_by_flow_id[flow_id].append(
            {
                "exchange_internal_id": str(exchange.get("@dataSetInternalID") or "").strip(),
                "direction": str(exchange.get("exchangeDirection") or "").strip(),
                "flow_version": str(ref.get("@version") or "").strip(),
                "flow_text": lang_text(ref.get("common:shortDescription")),
                "is_quantitative_reference": bool(
                    reference_internal_id
                    and str(exchange.get("@dataSetInternalID") or "").strip() == reference_internal_id
                ),
            }
        )

    details: list[dict[str, Any]] = []
    reference_flow = {}
    if reference_exchange:
        reference_ref = reference_exchange.get("referenceToFlowDataSet") or {}
        if isinstance(reference_ref, dict):
            reference_flow = {
                "flow_id": str(reference_ref.get("@refObjectId") or "").strip(),
                "flow_version": str(reference_ref.get("@version") or "").strip(),
                "flow_text": lang_text(reference_ref.get("common:shortDescription")),
                "direction": str(reference_exchange.get("exchangeDirection") or "").strip(),
            }
    for flow_id, matched_exchanges in matches_by_flow_id.items():
        details.append(
            {
                "target_flow_id": flow_id,
                "process_key": process_key,
                "process_id": process_id,
                "process_version": version,
                "process_name": name,
                "state_code": process_row.get("state_code"),
                "user_id": str(process_row.get("user_id") or "").strip(),
                "references_target_as_quantitative_reference": any(
                    item["is_quantitative_reference"] for item in matched_exchanges
                ),
                "matched_exchange_count": len(matched_exchanges),
                "matched_exchange_versions": sorted(
                    {
                        item["flow_version"]
                        for item in matched_exchanges
                        if str(item.get("flow_version") or "").strip()
                    },
                    key=version_key,
                ),
                "matched_exchange_directions": sorted(
                    {
                        item["direction"]
                        for item in matched_exchanges
                        if str(item.get("direction") or "").strip()
                    }
                ),
                "matched_exchange_texts": _sorted_texts(
                    item["flow_text"] for item in matched_exchanges if str(item.get("flow_text") or "").strip()
                ),
                "matched_exchanges": matched_exchanges,
                "reference_flow": reference_flow,
            }
        )
    return details


def materialize_process_rows(
    *,
    args: argparse.Namespace,
    access_token: str,
    process_keys: list[str],
) -> list[dict[str, Any]]:
    if not process_keys:
        return []
    wanted_keys = set(process_keys)
    wanted_ids = sorted({key.split("@", 1)[0] for key in process_keys if "@" in key})
    rows_by_key: dict[str, dict[str, Any]] = {}
    for id_chunk in chunked(wanted_ids, args.process_fetch_id_chunk_size):
        offset = 0
        while True:
            batch = postgrest_select_page(
                base_url=args.supabase_url,
                apikey=args.supabase_publishable_key,
                access_token=access_token,
                table="processes",
                raw_filters={
                    "state_code": "in.(0,100)",
                    "id": "in.(" + ",".join(id_chunk) + ")",
                },
                columns="id,version,state_code,user_id,json",
                limit=args.process_page_size,
                offset=offset,
                order="id.asc",
            )
            if not batch:
                break
            for row in batch:
                process_key = process_row_key(row)
                if process_key not in wanted_keys:
                    continue
                existing = rows_by_key.get(process_key)
                if not existing or _prefer_process_row(row, existing):
                    rows_by_key[process_key] = row
            if len(batch) < args.process_page_size:
                break
            offset += args.process_page_size
    materialized = list(rows_by_key.values())
    materialized.sort(key=process_row_sort_key)
    return materialized


def build_completion_outputs(
    *,
    target_rows: list[dict[str, Any]],
    process_scan: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with_process_rows: list[dict[str, Any]] = []
    zero_process_rows: list[dict[str, Any]] = []
    matches_by_flow_id = process_scan["matches_by_flow_id"]
    for row in target_rows:
        profile = build_flow_profile(row)
        flow_id = profile["flow_id"]
        process_info = matches_by_flow_id.get(flow_id, {})
        checklist_entry = {
            "flow": profile,
            "process_ref_stats": {
                "process_count": int(process_info.get("process_count") or 0),
                "reference_process_count": int(process_info.get("reference_process_count") or 0),
                "exchange_count": int(process_info.get("exchange_count") or 0),
            },
            "process_refs": process_info.get("process_refs", []),
            "completion_evidence": build_completion_evidence(process_info.get("process_refs", [])),
        }
        if checklist_entry["process_ref_stats"]["process_count"] > 0:
            with_process_rows.append(checklist_entry)
        else:
            checklist_entry["note"] = (
                "No visible referencing process was found under current RLS. "
                "This does not prove that the flow is truly unused."
            )
            zero_process_rows.append(checklist_entry)
    return with_process_rows, zero_process_rows


def build_completion_evidence(process_refs: list[dict[str, Any]]) -> dict[str, Any]:
    reference_process_names = _sorted_texts(
        item["process_name"] for item in process_refs if str(item.get("process_name") or "").strip()
    )[:20]
    reference_flow_names = _sorted_texts(
        item.get("reference_flow", {}).get("flow_text", "")
        for item in process_refs
        if isinstance(item.get("reference_flow"), dict)
    )[:20]
    matched_exchange_texts = _sorted_texts(
        text
        for item in process_refs
        for text in item.get("matched_exchange_texts", [])
    )[:20]
    return {
        "reference_process_names": reference_process_names,
        "reference_flow_names": reference_flow_names,
        "matched_exchange_texts": matched_exchange_texts,
    }


def build_flow_profile(row: dict[str, Any]) -> dict[str, Any]:
    dataset = flow_dataset_from_row(row)
    info = deep_get(dataset, ["flowInformation", "dataSetInformation"], {})
    name_node = info.get("name") or {}
    flow_id, version, name = extract_flow_identity(row)
    type_of_dataset = str(
        deep_get(dataset, ["modellingAndValidation", "LCIMethod", "typeOfDataSet"])
        or deep_get(dataset, ["modellingAndValidation", "LCIMethodAndAllocation", "typeOfDataSet"])
        or row.get("typeOfDataSet")
        or ""
    ).strip()
    reference_flow_property = select_reference_flow_property(dataset)
    classification_classes = [
        item
        for item in listify(
            deep_get(info, ["classificationInformation", "common:classification", "common:class"], [])
        )
        if isinstance(item, dict)
    ]
    classification_path = [lang_text(item) for item in classification_classes if lang_text(item)]

    presence = {
        "baseName": has_nonempty_lang_entries(name_node.get("baseName")),
        "treatmentStandardsRoutes": has_nonempty_lang_entries(name_node.get("treatmentStandardsRoutes")),
        "mixAndLocationTypes": has_nonempty_lang_entries(name_node.get("mixAndLocationTypes")),
        "flowProperties": has_nonempty_lang_entries(name_node.get("flowProperties")),
    }
    missing_name_fields = [
        field_name
        for field_name in ("baseName", "treatmentStandardsRoutes", "mixAndLocationTypes")
        if not presence[field_name]
    ]

    return {
        "flow_id": flow_id,
        "flow_version": version,
        "flow_key": f"{flow_id}@{version}",
        "flow_name": name,
        "state_code": row.get("state_code"),
        "user_id": str(row.get("user_id") or "").strip(),
        "type_of_dataset": type_of_dataset,
        "name_fields": {
            "baseName": normalize_lang_field(name_node.get("baseName")),
            "treatmentStandardsRoutes": normalize_lang_field(name_node.get("treatmentStandardsRoutes")),
            "mixAndLocationTypes": normalize_lang_field(name_node.get("mixAndLocationTypes")),
            "flowProperties": normalize_lang_field(name_node.get("flowProperties")),
        },
        "name_field_presence": presence,
        "missing_name_fields": missing_name_fields,
        "is_three_field_complete": not missing_name_fields,
        "synonyms": normalize_lang_field(info.get("common:synonyms")),
        "general_comment": normalize_lang_field(info.get("common:generalComment")),
        "classification_path": classification_path,
        "classification_leaf": classification_path[-1] if classification_path else "",
        "cas_number": str(info.get("CASNumber") or "").strip(),
        "reference_flow_property": {
            "flow_property_id": str(reference_flow_property.get("@refObjectId") or "").strip(),
            "flow_property_version": str(reference_flow_property.get("@version") or "").strip(),
            "flow_property_name": lang_text(reference_flow_property.get("common:shortDescription")),
        },
    }


def is_incomplete_three_field_flow(row: dict[str, Any]) -> bool:
    profile = build_flow_profile(row)
    return not profile["is_three_field_complete"]


def has_nonempty_lang_entries(value: Any) -> bool:
    return any(str(item.get("text") or "").strip() for item in lang_entries(value))


def normalize_lang_field(value: Any) -> list[dict[str, str]]:
    return [
        {
            "lang": str(item.get("lang") or "en"),
            "text": str(item.get("text") or "").strip(),
        }
        for item in lang_entries(value)
        if str(item.get("text") or "").strip()
    ]


def chunked(values: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        return [values]
    return [values[index : index + size] for index in range(0, len(values), size)]


def json_contains_flow_id(flow_id: str) -> str:
    return '[{"referenceToFlowDataSet":{"@refObjectId":"%s"}}]' % flow_id


def _flow_sort_key(row: dict[str, Any]) -> tuple[str, tuple[int, ...], str]:
    flow_id, version, name = extract_flow_identity(row)
    return (flow_id, version_key(version), name)


def _state_priority(row: dict[str, Any]) -> int:
    state_code = row.get("state_code")
    if state_code == 0:
        return 2
    if state_code == 100:
        return 1
    return 0


def _prefer_process_row(candidate: dict[str, Any], existing: dict[str, Any]) -> bool:
    candidate_priority = _state_priority(candidate)
    existing_priority = _state_priority(existing)
    if candidate_priority != existing_priority:
        return candidate_priority > existing_priority
    candidate_user_id = str(candidate.get("user_id") or "").strip()
    existing_user_id = str(existing.get("user_id") or "").strip()
    if candidate_user_id != existing_user_id:
        return bool(candidate_user_id) and not bool(existing_user_id)
    return False


def _sorted_texts(values: Any) -> list[str]:
    unique = {str(value or "").strip() for value in values if str(value or "").strip()}
    return sorted(unique)


if __name__ == "__main__":
    main()
