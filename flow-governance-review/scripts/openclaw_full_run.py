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
    postgrest_auth_password,
    postgrest_select_page,
)


SCRIPT_DIR = Path(__file__).resolve().parent
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
    access_token = _resolve_access_token(args)

    subject_rows = _fetch_subject_flow_rows(args, access_token)
    if not subject_rows:
        suffix = f" version {args.flow_version}" if args.flow_version else ""
        raise SystemExit(f"No visible product/waste flow rows found for flow {args.flow_id}{suffix}.")

    subject_rows.sort(key=lambda row: _flow_sort_key(row))
    subject_file = inputs_dir / "live-subject-flows.snapshot.jsonl"
    dump_jsonl(subject_file, subject_rows)

    process_rows: list[dict[str, Any]] = []
    process_file = inputs_dir / "live-processes.snapshot.jsonl"
    if not args.skip_process_fetch:
        process_rows = _fetch_process_rows(args, access_token)
        process_rows.sort(key=lambda row: _process_sort_key(row))
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
        "process_fetch_enabled": not args.skip_process_fetch,
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


def _resolve_access_token(args: argparse.Namespace) -> str:
    if not args.supabase_url or not args.supabase_publishable_key:
        raise SystemExit("SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY are required for openclaw-full-run.")
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
        "Authenticated access is required. Set SUPABASE_ACCESS_TOKEN or SUPABASE_EMAIL/SUPABASE_PASSWORD."
    )


def _fetch_subject_flow_rows(args: argparse.Namespace, access_token: str) -> list[dict[str, Any]]:
    rows = _select_all_pages(
        base_url=args.supabase_url,
        apikey=args.supabase_publishable_key,
        access_token=access_token,
        table="flows",
        raw_filters=_subject_flow_filters(args),
        columns="id,version,state_code,user_id,json",
        page_size=max(args.flow_page_size, 1),
    )
    return filter_target_flow_rows(rows)


def _fetch_process_rows(args: argparse.Namespace, access_token: str) -> list[dict[str, Any]]:
    return _select_all_pages(
        base_url=args.supabase_url,
        apikey=args.supabase_publishable_key,
        access_token=access_token,
        table="processes",
        raw_filters={
            "state_code": "in.(0,100)",
            "json->processDataSet->exchanges->exchange": "cs." + json_contains_flow_id(args.flow_id),
        },
        columns="id,version,state_code,user_id,json",
        page_size=max(args.process_page_size, 1),
    )


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


def _state_code_str(row: dict[str, Any]) -> str:
    value = row.get("state_code")
    if value is None:
        return ""
    return str(value).strip()


if __name__ == "__main__":
    main()
