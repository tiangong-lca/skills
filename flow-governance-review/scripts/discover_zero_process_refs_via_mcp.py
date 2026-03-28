#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from analyze_incomplete_flow_names_with_process_pool import (
    build_completion_evidence,
    build_process_match_details,
    finalize_matches_by_flow_id,
    record_process_match,
)
from flow_governance_common import (
    deep_get,
    dump_json,
    dump_jsonl,
    ensure_dir,
    FLOW_PROCESSING_DATASETS_DIR,
    FLOW_PROCESSING_NAMING_DIR,
    lang_text,
    load_json_or_jsonl,
    process_dataset_from_row,
    process_references_flow_id,
    process_row_key,
    process_row_sort_key,
    sync_process_pool_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESS_AUTOMATED_BUILDER_DIR = REPO_ROOT / "process-automated-builder"
if str(PROCESS_AUTOMATED_BUILDER_DIR) not in sys.path:
    sys.path.insert(0, str(PROCESS_AUTOMATED_BUILDER_DIR))

from tiangong_lca_spec.core.config import get_settings  # noqa: E402
from tiangong_lca_spec.core.mcp_client import MCPToolClient  # noqa: E402


DEFAULT_ZERO_PROCESS_FILE = (
    FLOW_PROCESSING_NAMING_DIR / "remaining-after-aggressive" / "remaining-incomplete-zero-process.json"
)
DEFAULT_PROCESS_POOL_FILE = FLOW_PROCESSING_DATASETS_DIR / "process_pool.jsonl"
DEFAULT_OUT_DIR = FLOW_PROCESSING_NAMING_DIR / "remaining-after-aggressive" / "mcp-zero-process-recovery"
DEFAULT_QUERY_TEMPLATE = "flow uuid {flow_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search zero-process flows through MCP Search_Processes_Tool, strictly verify real "
            "referenceToFlowDataSet hits, fetch missing process rows through Database_CRUD_Tool, "
            "and sync the local process pool."
        )
    )
    parser.add_argument("--zero-process-file", default=str(DEFAULT_ZERO_PROCESS_FILE))
    parser.add_argument("--process-pool-file", default=str(DEFAULT_PROCESS_POOL_FILE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--query-template", default=DEFAULT_QUERY_TEMPLATE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent MCP search workers for zero-process flow lookup.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Emit one stderr progress line after this many searched flows.",
    )
    parser.add_argument(
        "--search-tool-name",
        default="Search_Processes_Tool",
        help="MCP search tool used to find referencing processes.",
    )
    parser.add_argument(
        "--no-name-fallback",
        action="store_true",
        help=(
            "By default, flows with no strict hit on 'flow uuid <uuid>' also try the base flow name. "
            "Disable that fallback with this flag."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zero_process_file = Path(args.zero_process_file).expanduser().resolve()
    process_pool_file = Path(args.process_pool_file).expanduser().resolve()
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())

    flow_entries = load_zero_process_entries(zero_process_file)
    if args.limit > 0:
        flow_entries = flow_entries[: args.limit]

    existing_pool_rows = load_json_or_jsonl(process_pool_file) if process_pool_file.exists() else []
    pool_rows_by_key = {
        process_row_key(row): row
        for row in existing_pool_rows
        if process_row_key(row)
    }

    settings = get_settings()
    search_results = search_zero_process_flows(
        settings=settings,
        server_name=settings.flow_search_service_name,
        tool_name=args.search_tool_name,
        flow_entries=flow_entries,
        query_template=args.query_template,
        use_name_fallback=not args.no_name_fallback,
        workers=max(args.workers, 1),
        progress_every=max(args.progress_every, 1),
    )

    discovered_process_keys = sorted(
        {
            process_key
            for result in search_results
            for process_key in result["matched_process_keys"]
        }
    )
    process_rows_by_key: dict[str, dict[str, Any]] = {
        key: pool_rows_by_key[key]
        for key in discovered_process_keys
        if key in pool_rows_by_key
    }
    missing_process_keys = [key for key in discovered_process_keys if key not in process_rows_by_key]

    client = MCPToolClient(settings)
    try:
        fetched_rows, select_failures = fetch_missing_process_rows(
            client=client,
            server_name=settings.flow_search_service_name,
            process_keys=missing_process_keys,
        )
    finally:
        client.close()
    fetched_rows_by_key = {
        process_row_key(row): row
        for row in fetched_rows
        if process_row_key(row)
    }
    process_rows_by_key.update(fetched_rows_by_key)

    pool_sync = sync_process_pool_file(process_pool_file, fetched_rows)
    matched_rows = [process_rows_by_key[key] for key in discovered_process_keys if key in process_rows_by_key]
    matched_rows.sort(key=process_row_sort_key)

    recovered_rows, still_zero_rows = build_recovery_outputs(
        flow_entries=flow_entries,
        search_results=search_results,
        process_rows=matched_rows,
    )

    dump_jsonl(out_dir / "fetched-process-rows.jsonl", fetched_rows)
    dump_jsonl(out_dir / "matched-process-rows.jsonl", matched_rows)
    dump_jsonl(out_dir / "recovered-with-process.jsonl", recovered_rows)
    dump_jsonl(out_dir / "still-zero-after-mcp.jsonl", still_zero_rows)
    dump_jsonl(out_dir / "select-failures.jsonl", select_failures)

    summary = {
        "input_zero_process_file": str(zero_process_file),
        "input_zero_process_count": len(flow_entries),
        "query_template": args.query_template,
        "name_fallback_enabled": not args.no_name_fallback,
        "search_tool_name": args.search_tool_name,
        "workers": max(args.workers, 1),
        "progress_every": max(args.progress_every, 1),
        "matched_flow_count": len(recovered_rows),
        "still_zero_count": len(still_zero_rows),
        "search_error_flow_count": sum(
            1
            for result in search_results
            if any(attempt.get("error") for attempt in result.get("search_attempts", []))
        ),
        "discovered_process_key_count": len(discovered_process_keys),
        "already_in_pool_count": sum(1 for key in discovered_process_keys if key in pool_rows_by_key),
        "missing_process_key_count": len(missing_process_keys),
        "fetched_process_row_count": len(fetched_rows),
        "select_failure_count": len(select_failures),
        "process_pool_sync": pool_sync,
        "files": {
            "fetched_process_rows": str(out_dir / "fetched-process-rows.jsonl"),
            "matched_process_rows": str(out_dir / "matched-process-rows.jsonl"),
            "recovered_with_process": str(out_dir / "recovered-with-process.jsonl"),
            "still_zero_after_mcp": str(out_dir / "still-zero-after-mcp.jsonl"),
            "select_failures": str(out_dir / "select-failures.jsonl"),
        },
    }
    dump_json(out_dir / "summary.json", summary)
    print(str(out_dir / "summary.json"))


def load_zero_process_entries(path: Path) -> list[dict[str, Any]]:
    rows = load_json_or_jsonl(path)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        flow = row.get("flow") if isinstance(row.get("flow"), dict) else row
        if not isinstance(flow, dict):
            continue
        flow_id = str(flow.get("flow_id") or flow.get("id") or "").strip()
        if not flow_id:
            continue
        normalized.append(
            {
                "flow": flow,
                "source_row": row,
            }
        )
    return normalized


def search_zero_process_flows(
    *,
    settings: Any,
    server_name: str,
    tool_name: str,
    flow_entries: list[dict[str, Any]],
    query_template: str,
    use_name_fallback: bool,
    workers: int,
    progress_every: int,
) -> list[dict[str, Any]]:
    if not flow_entries:
        return []

    total = len(flow_entries)
    if workers <= 1:
        results: list[dict[str, Any]] = []
        client = MCPToolClient(settings)
        try:
            for index, entry in enumerate(flow_entries, start=1):
                results.append(
                    search_zero_process_flow_entry(
                        client=client,
                        server_name=server_name,
                        tool_name=tool_name,
                        entry=entry,
                        query_template=query_template,
                        use_name_fallback=use_name_fallback,
                    )
                )
                emit_progress(index, total, progress_every)
        finally:
            client.close()
        return results

    results_by_index: list[dict[str, Any] | None] = [None] * total
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(
                search_zero_process_flow_entry_with_new_client,
                settings,
                server_name,
                tool_name,
                entry,
                query_template,
                use_name_fallback,
            ): index
            for index, entry in enumerate(flow_entries)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results_by_index[index] = future.result()
            completed += 1
            emit_progress(completed, total, progress_every)
    return [result for result in results_by_index if isinstance(result, dict)]


def emit_progress(completed: int, total: int, progress_every: int) -> None:
    if completed <= 0 or total <= 0:
        return
    if completed % progress_every != 0 and completed != total:
        return
    print(
        f"[mcp-zero-process-recovery] searched {completed}/{total} flows",
        file=sys.stderr,
        flush=True,
    )


def search_zero_process_flow_entry_with_new_client(
    settings: Any,
    server_name: str,
    tool_name: str,
    entry: dict[str, Any],
    query_template: str,
    use_name_fallback: bool,
) -> dict[str, Any]:
    client = MCPToolClient(settings)
    try:
        return search_zero_process_flow_entry(
            client=client,
            server_name=server_name,
            tool_name=tool_name,
            entry=entry,
            query_template=query_template,
            use_name_fallback=use_name_fallback,
        )
    finally:
        client.close()


def search_zero_process_flow_entry(
    *,
    client: MCPToolClient,
    server_name: str,
    tool_name: str,
    entry: dict[str, Any],
    query_template: str,
    use_name_fallback: bool,
) -> dict[str, Any]:
    flow = entry["flow"]
    flow_id = str(flow.get("flow_id") or "").strip()
    flow_name = str(flow.get("flow_name") or "").strip()
    query_candidates = [query_template.format(flow_id=flow_id, flow_name=flow_name)]
    if use_name_fallback and flow_name:
        query_candidates.append(flow_name)

    matched_by_key: dict[str, dict[str, Any]] = {}
    attempts: list[dict[str, Any]] = []
    for query_text in query_candidates:
        try:
            raw = client.invoke_json_tool(server_name, tool_name, {"query": query_text})
        except Exception as exc:  # noqa: BLE001
            attempts.append(
                {
                    "query": query_text,
                    "candidate_count": 0,
                    "strict_match_count": 0,
                    "error": str(exc),
                }
            )
            continue
        items = raw.get("data") if isinstance(raw, dict) else raw
        items = items if isinstance(items, list) else []
        strict_match_count = 0
        for item in items:
            projected = project_search_result_to_process_row(item)
            if projected is None:
                continue
            if not process_references_flow_id(projected, flow_id):
                continue
            process_key = process_row_key(projected)
            if not process_key:
                continue
            matched_by_key[process_key] = projected
            strict_match_count += 1
        attempts.append(
            {
                "query": query_text,
                "candidate_count": len(items),
                "strict_match_count": strict_match_count,
            }
        )
        if strict_match_count > 0:
            break

    return {
        "flow": flow,
        "search_attempts": attempts,
        "matched_process_keys": sorted(matched_by_key),
        "matched_process_rows": list(matched_by_key.values()),
    }


def project_search_result_to_process_row(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    process_id = str(item.get("id") or "").strip()
    payload = item.get("json") if isinstance(item.get("json"), dict) else item
    if not process_id or not isinstance(payload, dict):
        return None
    process_dataset = payload.get("processDataSet") if isinstance(payload.get("processDataSet"), dict) else None
    if not isinstance(process_dataset, dict):
        return None
    version = extract_process_version_from_dataset(process_dataset)
    row = {
        "id": process_id,
        "version": version,
        "json": {"processDataSet": process_dataset},
        "state_code": item.get("state_code"),
        "user_id": item.get("user_id"),
        "name": deep_get(process_dataset, ["processInformation", "dataSetInformation", "name"], {}),
        "referenceToReferenceFlow": deep_get(
            process_dataset,
            ["processInformation", "quantitativeReference", "referenceToReferenceFlow"],
            "",
        ),
        "exchange": deep_get(process_dataset, ["exchanges", "exchange"], []),
    }
    return row if process_row_key(row) else None


def extract_process_version_from_dataset(process_dataset: dict[str, Any]) -> str:
    return str(
        deep_get(
            process_dataset,
            ["administrativeInformation", "publicationAndOwnership", "common:dataSetVersion"],
            "",
        )
        or ""
    ).strip()


def fetch_missing_process_rows(
    *,
    client: MCPToolClient,
    server_name: str,
    process_keys: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fetched_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for process_key in process_keys:
        process_id, version = split_process_key(process_key)
        try:
            raw = client.invoke_json_tool(
                server_name,
                "Database_CRUD_Tool",
                {
                    "operation": "select",
                    "table": "processes",
                    "id": process_id,
                    "version": version,
                },
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "process_key": process_key,
                    "process_id": process_id,
                    "version": version,
                    "error": str(exc),
                }
            )
            continue

        selected_rows = raw.get("data") if isinstance(raw, dict) else None
        if not isinstance(selected_rows, list) or not selected_rows:
            failures.append(
                {
                    "process_key": process_key,
                    "process_id": process_id,
                    "version": version,
                    "error": "empty_select_result",
                }
            )
            continue
        selected_row = selected_rows[0]
        if not isinstance(selected_row, dict):
            failures.append(
                {
                    "process_key": process_key,
                    "process_id": process_id,
                    "version": version,
                    "error": "invalid_select_row",
                }
            )
            continue
        fetched_rows.append(selected_row)
    fetched_rows.sort(key=process_row_sort_key)
    return fetched_rows, failures


def split_process_key(process_key: str) -> tuple[str, str]:
    if "@" not in process_key:
        return process_key, ""
    process_id, version = process_key.split("@", 1)
    return process_id, version


def build_recovery_outputs(
    *,
    flow_entries: list[dict[str, Any]],
    search_results: list[dict[str, Any]],
    process_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    flow_ids = {str(entry["flow"].get("flow_id") or "").strip() for entry in flow_entries}
    process_rows_by_key = {process_row_key(row): row for row in process_rows if process_row_key(row)}

    matches_by_flow_id = {
        flow_id: {"process_refs_by_key": {}}
        for flow_id in flow_ids
        if flow_id
    }
    for row in process_rows:
        projected = project_raw_process_row(row)
        for detail in build_process_match_details(projected, flow_ids):
            record_process_match(matches_by_flow_id, detail)
    finalized = finalize_matches_by_flow_id(matches_by_flow_id)

    search_result_by_flow_id = {
        str(item["flow"].get("flow_id") or "").strip(): item
        for item in search_results
    }
    recovered_rows: list[dict[str, Any]] = []
    still_zero_rows: list[dict[str, Any]] = []
    for entry in flow_entries:
        flow = entry["flow"]
        flow_id = str(flow.get("flow_id") or "").strip()
        process_info = finalized.get(flow_id, {})
        search_result = search_result_by_flow_id.get(flow_id, {})
        checklist_entry = {
            "flow": flow,
            "mcp_search": {
                "search_attempts": search_result.get("search_attempts", []),
                "matched_process_key_count": len(search_result.get("matched_process_keys", [])),
                "matched_process_keys": search_result.get("matched_process_keys", []),
            },
            "process_ref_stats": {
                "process_count": int(process_info.get("process_count") or 0),
                "reference_process_count": int(process_info.get("reference_process_count") or 0),
                "exchange_count": int(process_info.get("exchange_count") or 0),
            },
            "process_refs": process_info.get("process_refs", []),
            "completion_evidence": build_completion_evidence(process_info.get("process_refs", [])),
        }
        if checklist_entry["process_ref_stats"]["process_count"] > 0:
            recovered_rows.append(checklist_entry)
        else:
            checklist_entry["note"] = "No MCP-discovered referencing process was verified for this flow."
            still_zero_rows.append(checklist_entry)
    return recovered_rows, still_zero_rows


def project_raw_process_row(row: dict[str, Any]) -> dict[str, Any]:
    dataset = process_dataset_from_row(row)
    return {
        "id": row.get("id"),
        "version": row.get("version") or extract_process_version_from_dataset(dataset),
        "state_code": row.get("state_code"),
        "user_id": row.get("user_id"),
        "json": row.get("json") if isinstance(row.get("json"), dict) else {"processDataSet": dataset},
        "name": deep_get(dataset, ["processInformation", "dataSetInformation", "name"], {}),
        "referenceToReferenceFlow": deep_get(
            dataset,
            ["processInformation", "quantitativeReference", "referenceToReferenceFlow"],
            "",
        ),
        "exchange": deep_get(dataset, ["exchanges", "exchange"], []),
    }


if __name__ == "__main__":
    main()
