#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError


SCRIPT_DIR = Path(__file__).resolve().parent
FLOW_GOVERNANCE_ROOT = SCRIPT_DIR.parent
LCA_SKILLS_ROOT = FLOW_GOVERNANCE_ROOT.parent
PROCESS_AUTOMATED_BUILDER_ROOT = LCA_SKILLS_ROOT / "process-automated-builder"
FLOW_PROCESSING_ARTIFACT_ROOT = FLOW_GOVERNANCE_ROOT / "assets" / "artifacts" / "flow-processing"
FLOW_PROCESSING_REMEDIATION_DIR = FLOW_PROCESSING_ARTIFACT_ROOT / "remediation"
FLOW_PROCESSING_ROUND1_DIR = FLOW_PROCESSING_REMEDIATION_DIR / "round1"
FLOW_PROCESSING_MCP_SYNC_DIR = FLOW_PROCESSING_REMEDIATION_DIR / "mcp-sync"
if str(PROCESS_AUTOMATED_BUILDER_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESS_AUTOMATED_BUILDER_ROOT))


DEFAULT_INPUT_FILE = (
    FLOW_PROCESSING_ROUND1_DIR / "flows_tidas_sdk_plus_classification_remediated_ready_for_mcp.jsonl"
)
DEFAULT_REMOTE_FAILED_FILE = (
    FLOW_PROCESSING_MCP_SYNC_DIR / "flows_tidas_sdk_plus_classification_remote_validation_failed.jsonl"
)
DEFAULT_SUCCESS_LIST_FILE = (
    FLOW_PROCESSING_MCP_SYNC_DIR / "flows_tidas_sdk_plus_classification_mcp_success_list.json"
)
DEFAULT_REPORT_FILE = (
    FLOW_PROCESSING_MCP_SYNC_DIR / "flows_tidas_sdk_plus_classification_mcp_sync_report.json"
)
DEFAULT_TARGET_USER_ID = "dbcf5d8a-60bb-4dfc-a2b3-e8b4ab9352c0"
SUCCESS_SAVE_EVERY = 25
HTTP_RETRY_ATTEMPTS = 5
HTTP_RETRY_SLEEP_SECONDS = 1.5

_THREAD_LOCAL = threading.local()
_CLIENTS: list[Any] = []
_CLIENTS_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Insert or update remediated flow rows through MCP Database_CRUD_Tool, "
            "and split remote validation failures from successful rows."
        )
    )
    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_INPUT_FILE),
        help="Input JSONL containing remediated flow rows ready for MCP write.",
    )
    parser.add_argument(
        "--out-remote-failed-file",
        default=str(DEFAULT_REMOTE_FAILED_FILE),
        help="Output JSONL containing rows rejected by remote MCP validation/write.",
    )
    parser.add_argument(
        "--out-success-list-file",
        default=str(DEFAULT_SUCCESS_LIST_FILE),
        help="Output JSON file containing successful id/version rows for later SQL use.",
    )
    parser.add_argument(
        "--out-report-file",
        default=str(DEFAULT_REPORT_FILE),
        help="Output JSON summary report path.",
    )
    parser.add_argument(
        "--target-user-id",
        default=DEFAULT_TARGET_USER_ID,
        help="Expected owner user_id for successful inserted/updated rows.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum parallel workers for MCP writes.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for quick validation; 0 means all rows.",
    )
    return parser.parse_args()


def load_jsonl_rows(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            continue
        item = dict(value)
        item["_line_no"] = line_no
        rows.append(item)
    return rows


def dump_json(path: Path | str, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def flow_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("json_ordered") or row.get("jsonOrdered")
    if not isinstance(payload, dict):
        raise ValueError("Row is missing json_ordered/jsonOrdered object.")
    return payload


def flow_version_from_payload(payload: dict[str, Any]) -> str:
    version = (
        (((payload.get("flowDataSet") or {}).get("administrativeInformation") or {})
        .get("publicationAndOwnership") or {})
        .get("common:dataSetVersion")
    )
    if not isinstance(version, str) or not version.strip():
        raise ValueError("Row payload is missing flowDataSet.administrativeInformation.publicationAndOwnership.common:dataSetVersion.")
    return version.strip()


def http_json(method: str, url: str, headers: dict[str, str], body: Any | None = None) -> Any:
    data = None
    request_headers = dict(headers)
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")

    last_error: Exception | None = None
    for attempt in range(1, HTTP_RETRY_ATTEMPTS + 1):
        request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read().decode("utf-8")
                if not raw:
                    return None
                return json.loads(raw)
        except HTTPError as exc:
            status = getattr(exc, "code", None)
            retryable = status in {408, 409, 425, 429, 500, 502, 503, 504}
            if not retryable or attempt >= HTTP_RETRY_ATTEMPTS:
                raise
            last_error = exc
        except (URLError, OSError) as exc:
            if attempt >= HTTP_RETRY_ATTEMPTS:
                raise
            last_error = exc
        time.sleep(HTTP_RETRY_SLEEP_SECONDS * attempt)

    if last_error is not None:
        raise last_error
    raise RuntimeError("http_json exhausted retries without a concrete error")


def login_supabase() -> tuple[str, dict[str, str]]:
    supabase_url = os.environ["SUPABASE_URL"].rstrip("/")
    publishable_key = os.environ["SUPABASE_PUBLISHABLE_KEY"]
    email = os.environ["SUPABASE_EMAIL"]
    password = os.environ["SUPABASE_PASSWORD"]
    payload = http_json(
        "POST",
        f"{supabase_url}/auth/v1/token?grant_type=password",
        {"apikey": publishable_key},
        {"email": email, "password": password},
    )
    access_token = payload["access_token"]
    headers = {
        "apikey": publishable_key,
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }
    return access_token, headers


def visible_exact_rows(rest_headers: dict[str, str], flow_id: str, version: str) -> list[dict[str, Any]]:
    supabase_url = os.environ["SUPABASE_URL"].rstrip("/")
    query = urllib.parse.urlencode(
        {
            "select": "id,version,user_id,state_code",
            "id": f"eq.{flow_id}",
            "version": f"eq.{version}",
        }
    )
    payload = http_json("GET", f"{supabase_url}/rest/v1/flows?{query}", rest_headers)
    return payload if isinstance(payload, list) else []


def target_visible_row(rows: list[dict[str, Any]], target_user_id: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("user_id") or "").strip() == target_user_id:
            return row
    return None


def visible_conflict_reason(stage: str, visible: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    if not visible:
        return reasons
    for item in visible:
        reasons.append(
            {
                "validator": "remote_mcp",
                "stage": stage,
                "path": "",
                "message": "Exact UUID/version is already visible but not writable under the target user.",
                "code": "exact_version_visible_not_owned",
                "visible_user_id": str(item.get("user_id") or ""),
                "visible_state_code": str(item.get("state_code") if item.get("state_code") is not None else ""),
            }
        )
    return reasons


def parse_zod_issues(text: str, *, stage: str) -> list[dict[str, Any]] | None:
    marker = "Errors:"
    if marker not in text:
        return None
    raw = text.split(marker, 1)[1].strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        return None
    try:
        issues = json.loads(message)
    except json.JSONDecodeError:
        return None
    if not isinstance(issues, list):
        return None
    parsed: list[dict[str, Any]] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        path_value = ".".join(str(part) for part in issue.get("path") or [])
        parsed.append(
            {
                "validator": "remote_mcp",
                "stage": stage,
                "path": path_value,
                "message": str(issue.get("message") or text),
                "code": str(issue.get("code") or "remote_validation_error"),
            }
        )
    return parsed or None


def build_error_reasons(stage: str, exc: Exception) -> list[dict[str, Any]]:
    text = str(exc).strip() or exc.__class__.__name__
    parsed = parse_zod_issues(text, stage=stage)
    if parsed:
        return parsed
    return [
        {
            "validator": "remote_mcp",
            "stage": stage,
            "path": "",
            "message": text,
            "code": exc.__class__.__name__,
        }
    ]


def failure_row(source_row: dict[str, Any], reasons: list[dict[str, Any]]) -> dict[str, Any]:
    payload = source_row.get("json_ordered") or source_row.get("jsonOrdered")
    if not isinstance(payload, dict):
        payload = {}
    return {
        "id": source_row.get("id"),
        "user_id": source_row.get("user_id"),
        "json_ordered": payload,
        "reason": reasons,
        "state_code": source_row.get("state_code"),
    }


def get_client() -> Any:
    client = getattr(_THREAD_LOCAL, "crud_client", None)
    if client is None:
        from tiangong_lca_spec.publishing.crud import DatabaseCrudClient

        client = DatabaseCrudClient()
        _THREAD_LOCAL.crud_client = client
        with _CLIENTS_LOCK:
            _CLIENTS.append(client)
    return client


def close_clients() -> None:
    with _CLIENTS_LOCK:
        clients = list(_CLIENTS)
        _CLIENTS.clear()
    for client in clients:
        try:
            client.close()
        except Exception:
            pass


def sync_one(row: dict[str, Any], rest_headers: dict[str, str], target_user_id: str) -> dict[str, Any]:
    flow_id = str(row.get("id") or "").strip()
    line_no = row.get("_line_no")
    try:
        payload = flow_payload(row)
        version = flow_version_from_payload(payload)
        visible_before = visible_exact_rows(rest_headers, flow_id, version)
        own_before = target_visible_row(visible_before, target_user_id)

        if own_before is not None:
            try:
                get_client().update_flow(payload)
                return {
                    "status": "success",
                    "operation": "update_existing",
                    "id": flow_id,
                    "version": version,
                    "line_no": line_no,
                }
            except Exception as exc:  # noqa: BLE001
                return {
                    "status": "failure",
                    "failure_row": failure_row(row, build_error_reasons("update_existing", exc)),
                    "id": flow_id,
                    "version": version,
                    "line_no": line_no,
                }

        if visible_before:
            return {
                "status": "failure",
                "failure_row": failure_row(row, visible_conflict_reason("preflight", visible_before)),
                "id": flow_id,
                "version": version,
                "line_no": line_no,
            }

        try:
            get_client().insert_flow(payload)
            return {
                "status": "success",
                "operation": "insert",
                "id": flow_id,
                "version": version,
                "line_no": line_no,
            }
        except Exception as insert_exc:  # noqa: BLE001
            visible_after = visible_exact_rows(rest_headers, flow_id, version)
            own_after = target_visible_row(visible_after, target_user_id)
            if own_after is not None:
                try:
                    get_client().update_flow(payload)
                    return {
                        "status": "success",
                        "operation": "update_after_insert_error",
                        "id": flow_id,
                        "version": version,
                        "line_no": line_no,
                    }
                except Exception as update_exc:  # noqa: BLE001
                    reasons = build_error_reasons("insert", insert_exc) + build_error_reasons(
                        "update_after_insert_error", update_exc
                    )
                    return {
                        "status": "failure",
                        "failure_row": failure_row(row, reasons),
                        "id": flow_id,
                        "version": version,
                        "line_no": line_no,
                    }
            if visible_after:
                reasons = build_error_reasons("insert", insert_exc) + visible_conflict_reason(
                    "post_insert_error_preflight", visible_after
                )
            else:
                reasons = build_error_reasons("insert", insert_exc)
            return {
                "status": "failure",
                "failure_row": failure_row(row, reasons),
                "id": flow_id,
                "version": version,
                "line_no": line_no,
            }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failure",
            "failure_row": failure_row(row, build_error_reasons("sync_one_unhandled", exc)),
            "id": flow_id,
            "version": "",
            "line_no": line_no,
        }


def write_success_list(path: Path, successes: list[dict[str, Any]]) -> None:
    ordered = sorted(successes, key=lambda item: (int(item.get("line_no") or 0), item.get("id") or ""))
    payload = [
        {
            "id": item["id"],
            "version": item["version"],
            "operation": item["operation"],
        }
        for item in ordered
    ]
    dump_json(path, payload)


def write_report(
    path: Path,
    *,
    total: int,
    completed: int,
    successes: list[dict[str, Any]],
    failure_count: int,
    started_at: float,
) -> None:
    operation_counts: dict[str, int] = {}
    for item in successes:
        operation = str(item.get("operation") or "unknown")
        operation_counts[operation] = operation_counts.get(operation, 0) + 1
    report = {
        "total_rows": total,
        "completed_rows": completed,
        "success_count": len(successes),
        "failure_count": failure_count,
        "operation_counts": operation_counts,
        "started_at_epoch": started_at,
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    dump_json(path, report)


def main() -> int:
    args = parse_args()
    input_file = Path(args.input_file)
    out_remote_failed_file = Path(args.out_remote_failed_file)
    out_success_list_file = Path(args.out_success_list_file)
    out_report_file = Path(args.out_report_file)

    for path in (out_remote_failed_file, out_success_list_file, out_report_file):
        path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl_rows(input_file)
    if not rows:
        raise SystemExit(f"No rows found in {input_file}")
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    access_token, rest_headers = login_supabase()
    os.environ["TIANGONG_LCA_REMOTE_AUTHORIZATION"] = access_token
    os.environ.pop("TIANGONG_LCA_REMOTE_API_KEY", None)

    from tiangong_lca_spec.core.config import get_settings

    get_settings.cache_clear()

    successes: list[dict[str, Any]] = []
    failure_count = 0
    completed = 0
    started_at = time.time()

    out_remote_failed_file.write_text("", encoding="utf-8")

    try:
        with out_remote_failed_file.open("a", encoding="utf-8") as failed_handle:
            with ThreadPoolExecutor(max_workers=max(1, args.max_workers), thread_name_prefix="flow-mcp-sync") as executor:
                future_map: dict[Future[dict[str, Any]], dict[str, Any]] = {
                    executor.submit(sync_one, row, rest_headers, args.target_user_id): row for row in rows
                }
                for future in as_completed(future_map):
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001
                        row = future_map[future]
                        result = {
                            "status": "failure",
                            "failure_row": failure_row(row, build_error_reasons("future_result", exc)),
                            "id": row.get("id"),
                            "version": "",
                            "line_no": row.get("_line_no"),
                        }

                    completed += 1
                    if result["status"] == "success":
                        successes.append(
                            {
                                "id": result["id"],
                                "version": result["version"],
                                "operation": result["operation"],
                                "line_no": result.get("line_no"),
                            }
                        )
                    else:
                        failure_count += 1
                        failed_handle.write(json.dumps(result["failure_row"], ensure_ascii=False) + "\n")
                        failed_handle.flush()

                    if completed % SUCCESS_SAVE_EVERY == 0 or completed == len(rows):
                        write_success_list(out_success_list_file, successes)
                        write_report(
                            out_report_file,
                            total=len(rows),
                            completed=completed,
                            successes=successes,
                            failure_count=failure_count,
                            started_at=started_at,
                        )
                        print(
                            f"[progress] completed={completed}/{len(rows)} success={len(successes)} failure={failure_count}",
                            file=sys.stderr,
                            flush=True,
                        )
    finally:
        close_clients()
        write_success_list(out_success_list_file, successes)
        write_report(
            out_report_file,
            total=len(rows),
            completed=completed,
            successes=successes,
            failure_count=failure_count,
            started_at=started_at,
        )
    print(
        json.dumps(
            {
                "input_file": str(input_file),
                "success_list_file": str(out_success_list_file),
                "remote_failed_file": str(out_remote_failed_file),
                "report_file": str(out_report_file),
                "total_rows": len(rows),
                "success_count": len(successes),
                "failure_count": failure_count,
                "elapsed_seconds": round(time.time() - started_at, 2),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
