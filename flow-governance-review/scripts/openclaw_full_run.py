#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from flow_dedup_candidates import filter_target_flow_rows, json_contains_flow_id
from flow_governance_common import (
    dump_json,
    dump_jsonl,
    ensure_dir,
    extract_flow_identity,
    extract_process_identity,
    load_rows_from_file,
    postgrest_auth_password,
    postgrest_select_page,
    process_references_flow_id,
    process_row_key,
    process_row_sort_key,
    sync_process_pool_file,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROCESS_AUTOMATED_BUILDER_DIR = REPO_ROOT / "process-automated-builder"
DEFAULT_OUT_ROOT = Path.home() / ".openclaw" / "workspace" / "review" / "flow-governance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch live snapshots for one flow UUID and run the standard flow governance pipeline."
    )
    parser.add_argument("--flow-id", required=True, help="Target flow UUID.")
    parser.add_argument("--flow-version", help="Optional exact version filter for the subject flow snapshot.")
    parser.add_argument(
        "--out-dir",
        help="Run root. Defaults to ~/.openclaw/workspace/review/flow-governance/<flow-id>.",
    )
    parser.add_argument(
        "--skip-process-fetch",
        action="store_true",
        help="Skip live process snapshot materialization and run governance without process scan/repair steps.",
    )
    parser.add_argument("--flow-page-size", type=int, default=50)
    parser.add_argument("--process-page-size", type=int, default=100)
    parser.add_argument(
        "--subject-fetch-mode",
        choices=("auto", "mcp", "postgrest"),
        default="auto",
        help="How to materialize the subject flow when only a UUID is provided.",
    )
    parser.add_argument("--user-id", help="Optional override for live state_code=0 candidate-pool fetches.")
    parser.add_argument("--no-user-0", action="store_true")
    parser.add_argument("--no-public-100", action="store_true")
    parser.add_argument(
        "--public-origin-filter",
        choices=("all", "tiangong", "ilcd"),
        default="all",
    )
    parser.add_argument(
        "--auto-patch-policy",
        choices=("disabled", "alias-only", "alias-or-unique-name"),
        default="alias-only",
    )
    parser.add_argument("--enable-review-llm", action="store_true")
    parser.add_argument("--disable-review-llm", action="store_true")
    parser.add_argument("--with-reference-context", action="store_true")
    parser.add_argument("--similarity-threshold", type=float)
    parser.add_argument("--methodology-file")
    parser.add_argument("--methodology-id")
    parser.add_argument("--max-text-review-items", type=int, default=0)
    parser.add_argument("--max-process-contexts", type=int, default=3)
    parser.add_argument("--process-pool-file", help="Optional local process pool file used to reuse and sync process rows.")
    parser.add_argument("--tidas-mode", choices=("auto", "required", "skip"), default="auto")
    parser.add_argument("--supabase-url", default=os.getenv("SUPABASE_URL", ""))
    parser.add_argument("--supabase-publishable-key", default=os.getenv("SUPABASE_PUBLISHABLE_KEY", ""))
    parser.add_argument("--access-token", default=os.getenv("SUPABASE_ACCESS_TOKEN", ""))
    parser.add_argument("--email", default=os.getenv("SUPABASE_EMAIL", ""))
    parser.add_argument("--password", default=os.getenv("SUPABASE_PASSWORD", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.enable_review_llm and args.disable_review_llm:
        raise SystemExit("Use at most one of --enable-review-llm or --disable-review-llm.")

    out_dir = ensure_dir(_default_out_dir(args))
    inputs_dir = ensure_dir(out_dir / "inputs")
    subject_rows, subject_fetch_note = _fetch_subject_flow_rows(args)
    if not subject_rows:
        suffix = f" version {args.flow_version}" if args.flow_version else ""
        raise SystemExit(f"No visible product/waste flow rows found for flow {args.flow_id}{suffix}.")

    subject_rows.sort(key=lambda row: _flow_sort_key(row))
    subject_file = inputs_dir / "live-subject-flows.snapshot.jsonl"
    dump_jsonl(subject_file, subject_rows)

    process_rows: list[dict[str, Any]] = []
    process_file = inputs_dir / "live-processes.snapshot.jsonl"
    process_fetch_enabled = False
    process_fetch_note = "Process fetch disabled by --skip-process-fetch."
    process_fetch_report: dict[str, Any] = {
        "process_pool_enabled": bool(args.process_pool_file),
        "process_pool_file": str(Path(args.process_pool_file).expanduser().resolve()) if args.process_pool_file else "",
        "reused_from_pool_count": 0,
        "fetched_missing_count": 0,
        "remote_metadata_count": 0,
        "pool_sync": None,
        "missing_remote_keys": [],
    }
    if not args.skip_process_fetch:
        process_rows, process_fetch_note, process_fetch_report = _maybe_fetch_process_rows(args)
        process_fetch_enabled = True
    if process_rows:
        process_rows.sort(key=process_row_sort_key)
        dump_jsonl(process_file, process_rows)

    candidate_user_id, candidate_user_note = _resolve_candidate_user_id(args, subject_rows)
    live_fetch_manifest = {
        "flow_id": args.flow_id,
        "flow_version": args.flow_version or "",
        "subject_flow_count": len(subject_rows),
        "subject_flows": [
            {
                "id": flow_id,
                "version": version,
                "name": name,
                "state_code": row.get("state_code"),
                "user_id": row.get("user_id"),
            }
            for row in subject_rows
            for flow_id, version, name in [extract_flow_identity(row)]
        ],
        "process_count": len(process_rows),
        "subject_fetch_mode": args.subject_fetch_mode,
        "subject_fetch_note": subject_fetch_note,
        "process_fetch_enabled": process_fetch_enabled,
        "process_fetch_note": process_fetch_note,
        "process_fetch": process_fetch_report,
        "candidate_user_id": candidate_user_id,
        "candidate_user_note": candidate_user_note,
        "files": {
            "subject_flows_file": str(subject_file),
            "processes_file": str(process_file) if process_file.exists() else "",
        },
    }
    dump_json(out_dir / "live-fetch-manifest.json", live_fetch_manifest)

    governance_cmd = _build_governance_cmd(
        args=args,
        out_dir=out_dir,
        subject_file=subject_file,
        process_rows=process_rows,
        process_file=process_file,
        candidate_user_id=candidate_user_id,
    )
    completed = subprocess.run(governance_cmd, capture_output=True, text=True)
    command_result = {
        "command": governance_cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    dump_json(out_dir / "openclaw-command-result.json", command_result)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    if args.process_pool_file:
        live_fetch_manifest["process_pool_post_governance_sync"] = _sync_process_pool_from_governance_outputs(
            out_dir=out_dir,
            process_pool_file=Path(args.process_pool_file).expanduser().resolve(),
        )
        dump_json(out_dir / "live-fetch-manifest.json", live_fetch_manifest)

    governance_manifest_path = out_dir / "governance-run-manifest.json"
    governance_manifest = json.loads(governance_manifest_path.read_text(encoding="utf-8"))
    handoff_summary = {
        "skill": "flow-governance-review",
        "entrypoint": "openclaw-full-run",
        "flow_id": args.flow_id,
        "flow_version": args.flow_version or "",
        "run_root": str(out_dir),
        "live_fetch_manifest": str(out_dir / "live-fetch-manifest.json"),
        "governance_manifest": str(governance_manifest_path),
        "key_outputs": governance_manifest.get("key_outputs", {}),
        "subject_flow_count": len(subject_rows),
        "process_count": len(process_rows),
        "candidate_user_id": candidate_user_id,
        "process_pool_file": str(Path(args.process_pool_file).expanduser().resolve()) if args.process_pool_file else "",
        "notes": [
            "Subject flow/process snapshots were materialized locally before running the standard governance orchestrator.",
            "The standard skill boundary still applies: residual semantic decisions remain in exported OpenClaw review packs.",
        ],
    }
    dump_json(out_dir / "openclaw-handoff-summary.json", handoff_summary)
    print(str(out_dir / "openclaw-handoff-summary.json"))


def _default_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir:
        return Path(args.out_dir).expanduser().resolve()
    version_slug = f"__{args.flow_version}" if args.flow_version else ""
    return (DEFAULT_OUT_ROOT / f"{args.flow_id}{version_slug}").resolve()


def _can_use_postgrest(args: argparse.Namespace) -> bool:
    return bool(str(args.supabase_url or "").strip() and str(args.supabase_publishable_key or "").strip())


def _resolve_postgrest_access_token(args: argparse.Namespace, *, required: bool) -> str:
    if not _can_use_postgrest(args):
        if required:
            raise RuntimeError("SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY are required for PostgREST fetches.")
        return ""
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
    if required:
        raise RuntimeError(
            "Authenticated PostgREST access is required. Set SUPABASE_ACCESS_TOKEN or SUPABASE_EMAIL/SUPABASE_PASSWORD."
        )
    return ""


def _fetch_subject_flow_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    attempts: list[str] = []
    if args.subject_fetch_mode in {"auto", "mcp"}:
        try:
            rows = _fetch_subject_flow_rows_via_mcp(args)
            rows = filter_target_flow_rows(rows)
            if rows:
                rows, enrich_note = _maybe_enrich_subject_rows_via_postgrest(args, rows)
                note = "Fetched subject flow via MCP Database_CRUD_Tool."
                if enrich_note:
                    note += f" {enrich_note}"
                return rows, note
            attempts.append("MCP returned no visible product/waste flow rows.")
        except Exception as exc:  # noqa: BLE001
            formatted = _format_exception(exc)
            attempts.append(f"MCP fetch failed: {formatted}")
            if args.subject_fetch_mode == "mcp":
                raise SystemExit(f"Failed to fetch subject flow via MCP Database_CRUD_Tool: {formatted}") from exc
    if args.subject_fetch_mode in {"auto", "postgrest"}:
        try:
            access_token = _resolve_postgrest_access_token(args, required=True)
            rows = _fetch_subject_flow_rows_via_postgrest(args, access_token)
            rows = filter_target_flow_rows(rows)
            if rows:
                return rows, "Fetched subject flow via PostgREST."
            attempts.append("PostgREST returned no visible product/waste flow rows.")
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"PostgREST fetch failed: {_format_exception(exc)}")
            if args.subject_fetch_mode == "postgrest":
                raise SystemExit(f"Failed to fetch subject flow via PostgREST: {_format_exception(exc)}") from exc
    suffix = f" version {args.flow_version}" if args.flow_version else ""
    detail = f" Details: {' | '.join(attempts)}" if attempts else ""
    raise SystemExit(f"No visible product/waste flow rows found for flow {args.flow_id}{suffix}.{detail}")


def _fetch_subject_flow_rows_via_postgrest(args: argparse.Namespace, access_token: str) -> list[dict[str, Any]]:
    return _select_all_pages(
        base_url=args.supabase_url,
        apikey=args.supabase_publishable_key,
        access_token=access_token,
        table="flows",
        raw_filters=_subject_flow_filters(args),
        columns="id,version,state_code,user_id,json",
        page_size=max(args.flow_page_size, 1),
    )


def _fetch_subject_flow_rows_via_mcp(args: argparse.Namespace) -> list[dict[str, Any]]:
    client = _create_crud_client()
    try:
        if not args.flow_version:
            record = client.select_flow_record(args.flow_id)
            row = _build_flow_row_from_mcp_record(record)
            if row is not None:
                return [row]
        dataset = client.select_flow(args.flow_id, version=args.flow_version)
        if isinstance(dataset, dict):
            return [_build_flow_row_from_dataset(args.flow_id, dataset, version_hint=args.flow_version)]
        return []
    finally:
        client.close()


def _maybe_enrich_subject_rows_via_postgrest(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    if not rows or not _can_use_postgrest(args):
        return rows, ""
    if all(row.get("state_code") is not None and str(row.get("user_id") or "").strip() for row in rows):
        return rows, ""
    try:
        access_token = _resolve_postgrest_access_token(args, required=False)
    except Exception:
        return rows, ""
    if not access_token:
        return rows, ""
    enriched: list[dict[str, Any]] = []
    enriched_any = False
    for row in rows:
        flow_id, version, _name = extract_flow_identity(row)
        query_rows = _select_all_pages(
            base_url=args.supabase_url,
            apikey=args.supabase_publishable_key,
            access_token=access_token,
            table="flows",
            raw_filters={"id": f"eq.{flow_id}", "version": f"eq.{version}", "state_code": "in.(0,100)"},
            columns="id,version,state_code,user_id,json",
            page_size=1,
        )
        if query_rows:
            enriched.append(query_rows[0])
            enriched_any = True
        else:
            enriched.append(row)
    note = "Enriched subject flow metadata via PostgREST." if enriched_any else ""
    return enriched, note


def _maybe_fetch_process_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    pool_path = Path(args.process_pool_file).expanduser().resolve() if args.process_pool_file else None
    pool_rows = load_rows_from_file(pool_path) if pool_path and pool_path.exists() else []
    pooled_rows = _dedupe_process_rows([row for row in pool_rows if process_references_flow_id(row, args.flow_id)])
    pool_keys = {process_row_key(row) for row in pool_rows if process_row_key(row)}
    report: dict[str, Any] = {
        "process_pool_enabled": pool_path is not None,
        "process_pool_file": str(pool_path) if pool_path else "",
        "pool_row_count": len(pool_rows),
        "reused_from_pool_count": len(pooled_rows),
        "fetched_missing_count": 0,
        "remote_metadata_count": 0,
        "pool_sync": None,
        "missing_remote_keys": [],
    }

    if not _can_use_postgrest(args):
        note = "Process fetch skipped because SUPABASE_URL/SUPABASE_PUBLISHABLE_KEY are not set."
        if pooled_rows:
            note += f" Reused {len(pooled_rows)} process rows from the local process pool."
        return pooled_rows, note, report

    try:
        access_token = _resolve_postgrest_access_token(args, required=False)
    except Exception as exc:  # noqa: BLE001
        note = f"Process fetch skipped because PostgREST auth resolution failed: {exc}"
        if pooled_rows:
            note += f" Reused {len(pooled_rows)} process rows from the local process pool."
        return pooled_rows, note, report
    if not access_token:
        note = "Process fetch skipped because authenticated PostgREST access is unavailable."
        if pooled_rows:
            note += f" Reused {len(pooled_rows)} process rows from the local process pool."
        return pooled_rows, note, report

    metadata_rows = _select_all_pages(
        base_url=args.supabase_url,
        apikey=args.supabase_publishable_key,
        access_token=access_token,
        table="processes",
        raw_filters={
            "state_code": "in.(0,100)",
            "json->processDataSet->exchanges->exchange": "cs." + json_contains_flow_id(args.flow_id),
        },
        columns="id,version,state_code,user_id",
        page_size=max(args.process_page_size, 1),
    )
    remote_keys = sorted(
        {
            f"{row.get('id')}@{row.get('version')}"
            for row in metadata_rows
            if isinstance(row, dict) and str(row.get("id") or "").strip() and str(row.get("version") or "").strip()
        }
    )
    report["remote_metadata_count"] = len(remote_keys)
    missing_keys = [key for key in remote_keys if key not in pool_keys]
    report["missing_remote_keys"] = missing_keys
    fetched_rows, missing_unresolved = _fetch_process_rows_by_keys(args, access_token, missing_keys)
    report["fetched_missing_count"] = len(fetched_rows)
    if missing_unresolved:
        report["missing_remote_keys_unresolved"] = missing_unresolved
    if pool_path and fetched_rows:
        report["pool_sync"] = sync_process_pool_file(pool_path, fetched_rows)

    selected_rows = _dedupe_process_rows([*pooled_rows, *fetched_rows])
    note_parts = []
    if pooled_rows:
        note_parts.append(f"Reused {len(pooled_rows)} process rows from the local process pool.")
    note_parts.append(f"Fetched {len(fetched_rows)} missing referencing process rows via PostgREST.")
    if missing_unresolved:
        note_parts.append(f"{len(missing_unresolved)} remote process keys could not be materialized.")
    return selected_rows, " ".join(note_parts), report


def _build_flow_row_from_mcp_record(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    row: dict[str, Any] = {
        "id": str(record.get("id") or "").strip(),
        "version": str(record.get("version") or "").strip(),
        "state_code": record.get("state_code"),
        "user_id": str(record.get("user_id") or "").strip(),
    }
    if isinstance(record.get("json_ordered"), dict):
        row["json_ordered"] = record["json_ordered"]
    elif isinstance(record.get("json"), dict):
        row["json"] = record["json"]
    elif isinstance(record.get("flowDataSet"), dict):
        row["json"] = {"flowDataSet": record["flowDataSet"]}
    else:
        return None
    flow_id, version, _name = extract_flow_identity(row)
    if flow_id:
        row["id"] = flow_id
    if version:
        row["version"] = version
    return row


def _build_flow_row_from_dataset(flow_id: str, dataset: dict[str, Any], *, version_hint: str | None = None) -> dict[str, Any]:
    version = version_hint or _flow_dataset_version(dataset)
    return {
        "id": str(flow_id or "").strip(),
        "version": version,
        "state_code": None,
        "user_id": "",
        "json": {"flowDataSet": dataset},
    }


def _flow_dataset_version(dataset: dict[str, Any]) -> str:
    publication = (
        dataset.get("administrativeInformation", {})
        .get("publicationAndOwnership", {})
    )
    version = str(publication.get("common:dataSetVersion") or "").strip()
    return version or "01.01.000"


def _create_crud_client():
    if str(PROCESS_AUTOMATED_BUILDER_DIR) not in sys.path:
        sys.path.insert(0, str(PROCESS_AUTOMATED_BUILDER_DIR))
    from tiangong_lca_spec.publishing.crud import DatabaseCrudClient

    return DatabaseCrudClient()


def _format_exception(exc: BaseException) -> str:
    messages: list[str] = []
    _collect_exception_messages(exc, messages)
    cleaned = [message for message in messages if message]
    if cleaned:
        return " | ".join(dict.fromkeys(cleaned))
    return str(exc)


def _collect_exception_messages(exc: BaseException, messages: list[str]) -> None:
    text = str(exc).strip()
    if text:
        messages.append(text)
    for nested in getattr(exc, "exceptions", ()) or ():
        if isinstance(nested, BaseException):
            _collect_exception_messages(nested, messages)
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, BaseException):
        _collect_exception_messages(cause, messages)
    context = getattr(exc, "__context__", None)
    if isinstance(context, BaseException):
        _collect_exception_messages(context, messages)


def _select_all_pages(
    *,
    base_url: str,
    apikey: str,
    access_token: str,
    table: str,
    raw_filters: dict[str, str],
    columns: str,
    page_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        batch = postgrest_select_page(
            base_url=base_url,
            apikey=apikey,
            access_token=access_token,
            table=table,
            raw_filters=raw_filters,
            columns=columns,
            limit=page_size,
            offset=offset,
            order="id.asc",
        )
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def _subject_flow_filters(args: argparse.Namespace) -> dict[str, str]:
    filters = {
        "id": f"eq.{args.flow_id}",
        "state_code": "in.(0,100)",
    }
    if args.flow_version:
        filters["version"] = f"eq.{args.flow_version}"
    return filters


def _resolve_candidate_user_id(args: argparse.Namespace, subject_rows: list[dict[str, Any]]) -> tuple[str, str]:
    if args.no_user_0:
        return "", "state_code=0 candidate pool disabled by --no-user-0."
    explicit_user_id = str(args.user_id or "").strip()
    if explicit_user_id:
        return explicit_user_id, "Using explicit --user-id override."

    derived = sorted(
        {
            str(row.get("user_id") or "").strip()
            for row in subject_rows
            if _state_code_str(row) == "0" and str(row.get("user_id") or "").strip()
        }
    )
    if len(derived) == 1:
        return derived[0], "Derived from the fetched state_code=0 subject flow row."
    if len(derived) > 1:
        return derived[0], "Multiple state_code=0 user_ids were visible; using the first sorted user_id."
    return "", "No visible state_code=0 user_id found; candidate pool will use public rows only."


def _build_governance_cmd(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    subject_file: Path,
    process_rows: list[dict[str, Any]],
    process_file: Path,
    candidate_user_id: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "flow_governance_orchestrator.py"),
        "--subject-flows-file",
        str(subject_file),
        "--out-dir",
        str(out_dir),
        "--auto-patch-policy",
        args.auto_patch_policy,
        "--public-origin-filter",
        args.public_origin_filter,
        "--tidas-mode",
        args.tidas_mode,
    ]
    if process_rows:
        cmd += ["--processes-file", str(process_file)]
    if args.process_pool_file:
        cmd += ["--process-pool-file", str(Path(args.process_pool_file).expanduser().resolve())]
    if args.no_user_0:
        cmd.append("--no-user-0")
    elif candidate_user_id:
        cmd += ["--user-id", candidate_user_id]
    else:
        cmd.append("--no-user-0")
    if args.no_public_100:
        cmd.append("--no-public-100")
    if args.enable_review_llm:
        cmd.append("--enable-review-llm")
    elif args.disable_review_llm:
        cmd.append("--disable-review-llm")
    if args.with_reference_context:
        cmd.append("--with-reference-context")
    if args.similarity_threshold is not None:
        cmd += ["--similarity-threshold", str(args.similarity_threshold)]
    if args.methodology_file:
        cmd += ["--methodology-file", str(Path(args.methodology_file).expanduser().resolve())]
    if args.methodology_id:
        cmd += ["--methodology-id", args.methodology_id]
    if args.max_text_review_items > 0:
        cmd += ["--max-text-review-items", str(args.max_text_review_items)]
    if args.max_process_contexts != 3:
        cmd += ["--max-process-contexts", str(args.max_process_contexts)]
    return cmd


def _flow_sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    flow_id, version, name = extract_flow_identity(row)
    return (flow_id, version, name)


def _process_sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    process_id, version, name = extract_process_identity(row)
    return (process_id, version, name)


def _dedupe_process_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = process_row_key(row)
        if key:
            deduped[key] = row
    return list(deduped.values())


def _fetch_process_rows_by_keys(
    args: argparse.Namespace,
    access_token: str,
    process_keys: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for process_key in process_keys:
        if "@" not in process_key:
            unresolved.append(process_key)
            continue
        process_id, version = process_key.split("@", 1)
        batch = _select_all_pages(
            base_url=args.supabase_url,
            apikey=args.supabase_publishable_key,
            access_token=access_token,
            table="processes",
            raw_filters={
                "id": f"eq.{process_id}",
                "version": f"eq.{version}",
                "state_code": "in.(0,100)",
            },
            columns="id,version,state_code,user_id,json",
            page_size=1,
        )
        if not batch:
            unresolved.append(process_key)
            continue
        rows.append(batch[0])
    return rows, unresolved


def _sync_process_pool_from_governance_outputs(*, out_dir: Path, process_pool_file: Path) -> dict[str, Any]:
    patched_processes_path = out_dir / "repair-apply" / "patched-processes.json"
    if not patched_processes_path.exists():
        return {
            "pool_file": str(process_pool_file),
            "status": "skipped",
            "reason": "repair-apply/patched-processes.json not found.",
        }
    patched_rows = load_rows_from_file(patched_processes_path)
    if not patched_rows:
        return {
            "pool_file": str(process_pool_file),
            "status": "skipped",
            "reason": "repair-apply/patched-processes.json is empty.",
        }
    sync_report = sync_process_pool_file(process_pool_file, patched_rows)
    return {
        "status": "synced",
        "source_file": str(patched_processes_path),
        **sync_report,
    }


def _state_code_str(row: dict[str, Any]) -> str:
    value = row.get("state_code")
    if value is None:
        return ""
    return str(value).strip()


if __name__ == "__main__":
    main()
