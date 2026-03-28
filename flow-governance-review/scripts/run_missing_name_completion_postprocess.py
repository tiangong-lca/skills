#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from flow_governance_common import (
    dump_json,
    dump_jsonl,
    ensure_dir,
    extract_flow_identity,
    FLOW_GOVERNANCE_ROOT,
    FLOW_PROCESSING_NAMING_DIR,
    load_rows_from_file,
    version_key,
)


DEFAULT_PACK_ROOT = FLOW_PROCESSING_NAMING_DIR / "zero-process-completion-pack"
DEFAULT_RUN_SCRIPT = FLOW_GOVERNANCE_ROOT / "scripts" / "run-flow-governance-review.sh"
DEFAULT_ENV_FILE = Path.home() / ".openclaw" / ".env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume/apply/validate/publish the missing-name-field completion batches. "
            "OpenClaw writes decisions per batch; this script handles the deterministic local tail."
        )
    )
    parser.add_argument("--pack-root", default=str(DEFAULT_PACK_ROOT))
    parser.add_argument("--run-script", default=str(DEFAULT_RUN_SCRIPT))
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--batch", action="append", dest="batches", default=[])
    parser.add_argument(
        "--decisions-file-name",
        default="openclaw-decisions.json",
        help="Decision file name expected inside each batch directory.",
    )
    parser.add_argument(
        "--aggregate-subdir",
        default="",
        help="Optional aggregate subdirectory under pack root. Defaults to aggregate or aggregate-<decision-slug>.",
    )
    parser.add_argument("--tidas-mode", choices=("auto", "required", "skip"), default="auto")
    parser.add_argument(
        "--flow-publish-policy",
        choices=("skip", "append_only_bump", "upsert_current_version"),
        default="append_only_bump",
    )
    parser.add_argument("--skip-publish", action="store_true")
    parser.add_argument("--commit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pack_root = Path(args.pack_root).expanduser().resolve()
    decision_slug = _decision_slug(args.decisions_file_name)
    aggregate_subdir = args.aggregate_subdir.strip() or ("aggregate" if args.decisions_file_name == "openclaw-decisions.json" else f"aggregate-{decision_slug}")
    aggregate_dir = ensure_dir(pack_root / aggregate_subdir)
    batch_manifest = json.loads((pack_root / "batch-manifest.json").read_text(encoding="utf-8"))
    selected_batches = select_batches(batch_manifest.get("batches") or [], args.batches)

    batch_status_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    validation_failure_rows: list[dict[str, Any]] = []
    ready_patched_rows: list[dict[str, Any]] = []
    ready_original_rows: list[dict[str, Any]] = []

    for batch in selected_batches:
        status = process_batch(batch=batch, args=args)
        batch_status_rows.append(status)
        if status.get("invalid_decisions_file"):
            invalid_rows.extend(_load_optional_rows(status["invalid_decisions_file"]))
        if status.get("manual_review_unresolved_file"):
            unresolved_rows.extend(_load_optional_rows(status["manual_review_unresolved_file"]))
        if status.get("validation_failures_file"):
            validation_failure_rows.extend(_load_optional_rows(status["validation_failures_file"]))
        if status.get("ready_for_publish") and status.get("patched_rows_file"):
            ready_patched_rows.extend(_load_optional_rows(status["patched_rows_file"]))
            ready_original_rows.extend(_load_optional_rows(status["original_rows_file"]))

    merged_patched_rows = dedupe_flow_rows(ready_patched_rows)
    merged_original_rows = dedupe_flow_rows(ready_original_rows)

    patched_rows_file = aggregate_dir / "patched-flow-rows.jsonl"
    original_rows_file = aggregate_dir / "original-flow-rows.jsonl"
    dump_jsonl(patched_rows_file, merged_patched_rows)
    dump_jsonl(original_rows_file, merged_original_rows)
    dump_jsonl(aggregate_dir / "invalid-decisions.jsonl", invalid_rows)
    dump_jsonl(aggregate_dir / "manual-review-unresolved.jsonl", unresolved_rows)
    dump_jsonl(aggregate_dir / "validation-failures.jsonl", validation_failure_rows)
    dump_json(aggregate_dir / "batch-status.json", {"batches": batch_status_rows})

    pending_batches = [row for row in batch_status_rows if row["status"] == "pending_decisions"]
    blocked_batches = [row for row in batch_status_rows if row["status"] == "blocked"]
    ready_batches = [row for row in batch_status_rows if row["status"] in {"ready", "ready_no_changes"}]
    changed_ready_batches = [row for row in batch_status_rows if row["status"] == "ready"]

    summary: dict[str, Any] = {
        "pack_root": str(pack_root),
        "aggregate_dir": str(aggregate_dir),
        "aggregate_subdir": aggregate_subdir,
        "selected_batch_count": len(selected_batches),
        "decisions_file_name": args.decisions_file_name,
        "pending_batch_count": len(pending_batches),
        "blocked_batch_count": len(blocked_batches),
        "ready_batch_count": len(ready_batches),
        "ready_with_changes_batch_count": len(changed_ready_batches),
        "patched_flow_row_count": len(merged_patched_rows),
        "original_flow_row_count": len(merged_original_rows),
        "invalid_decision_count": len(invalid_rows),
        "manual_review_unresolved_count": len(unresolved_rows),
        "validation_failure_count": len(validation_failure_rows),
        "ready_for_publish": not pending_batches and not blocked_batches and bool(merged_patched_rows),
        "can_commit": not pending_batches and not blocked_batches and bool(merged_patched_rows) and not args.skip_publish,
        "files": {
            "batch_status": str(aggregate_dir / "batch-status.json"),
            "patched_flow_rows": str(patched_rows_file),
            "original_flow_rows": str(original_rows_file),
            "invalid_decisions": str(aggregate_dir / "invalid-decisions.jsonl"),
            "manual_review_unresolved": str(aggregate_dir / "manual-review-unresolved.jsonl"),
            "validation_failures": str(aggregate_dir / "validation-failures.jsonl"),
        },
        "notes": [
            "Only batches with no invalid decisions, no unresolved items, and no validation failures are included in publish scope.",
            "If any batch is pending or blocked, commit publish is skipped.",
            "Commit publish writes through the current MCP/CRUD account; public state promotion remains a separate step if needed.",
        ],
    }

    publish_runs: dict[str, Any] = {}
    if not args.skip_publish and merged_patched_rows:
        if not pending_batches and not blocked_batches:
            dry_run_dir = aggregate_dir / "publish-dry-run"
            run_publish(
                args=args,
                flow_rows_file=patched_rows_file,
                original_rows_file=original_rows_file,
                out_dir=dry_run_dir,
                commit=False,
            )
            publish_runs["dry_run"] = str(dry_run_dir / "publish-report.json")
            if args.commit:
                commit_dir = aggregate_dir / "publish-commit"
                run_publish(
                    args=args,
                    flow_rows_file=patched_rows_file,
                    original_rows_file=original_rows_file,
                    out_dir=commit_dir,
                    commit=True,
                )
                publish_runs["commit"] = str(commit_dir / "publish-report.json")
        else:
            summary["publish_blocked_reason"] = "pending_or_blocked_batches_present"
    elif not merged_patched_rows:
        summary["publish_blocked_reason"] = "no_valid_patched_rows"
    else:
        summary["publish_blocked_reason"] = "publish_skipped_by_flag"

    if publish_runs:
        summary["publish_runs"] = publish_runs

    dump_json(aggregate_dir / "pipeline-status.json", summary)
    print(str(aggregate_dir / "pipeline-status.json"))


def select_batches(all_batches: list[dict[str, Any]], requested_batches: list[str]) -> list[dict[str, Any]]:
    if not requested_batches:
        return all_batches
    wanted = {item.strip() for item in requested_batches if str(item or "").strip()}
    selected = [item for item in all_batches if str(item.get("batch_slug") or "").strip() in wanted]
    missing = sorted(wanted - {str(item.get("batch_slug") or "").strip() for item in selected})
    if missing:
        raise SystemExit(f"Unknown batch slug(s): {', '.join(missing)}")
    return selected


def process_batch(*, batch: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    batch_slug = str(batch.get("batch_slug") or "").strip()
    batch_dir = Path(batch.get("review_pack_file") or "").expanduser().resolve().parent
    review_pack_file = batch_dir / "review-pack.json"
    scope_flows_file = batch_dir / "scope-flows.jsonl"
    decisions_file = batch_dir / args.decisions_file_name
    decision_slug = _decision_slug(args.decisions_file_name)
    apply_dir = ensure_dir(batch_dir / f"applied-{decision_slug}")
    validate_dir = ensure_dir(batch_dir / f"validated-{decision_slug}")
    logs_dir = ensure_dir(batch_dir / f"pipeline-logs-{decision_slug}")

    status: dict[str, Any] = {
        "batch_slug": batch_slug,
        "batch_dir": str(batch_dir),
        "decisions_file_name": args.decisions_file_name,
        "review_item_count": int(batch.get("review_item_count") or 0),
        "review_pack_file": str(review_pack_file),
        "original_rows_file": str(scope_flows_file),
        "decisions_file": str(decisions_file),
        "applied_dir": str(apply_dir),
        "validated_dir": str(validate_dir),
    }
    if not decisions_file.exists():
        status["status"] = "pending_decisions"
        return status

    apply_result = run_flow_governance_command(
        args=args,
        command=[
            "apply-openclaw-text-decisions",
            "--entity-type",
            "flow",
            "--rows-file",
            str(scope_flows_file),
            "--review-pack",
            str(review_pack_file),
            "--decisions-file",
            str(decisions_file),
            "--out-dir",
            str(apply_dir),
        ],
        stdout_file=logs_dir / "apply.stdout.log",
        stderr_file=logs_dir / "apply.stderr.log",
    )
    if not apply_result["ok"]:
        status["status"] = "blocked"
        status["command_error"] = apply_result
        return status

    patched_rows_file = apply_dir / "patched-flows.json"
    validate_result = run_flow_governance_command(
        args=args,
        command=[
            "validate-openclaw-text-decisions",
            "--entity-type",
            "flow",
            "--original-rows-file",
            str(scope_flows_file),
            "--patched-rows-file",
            str(patched_rows_file),
            "--out-dir",
            str(validate_dir),
            "--tidas-mode",
            args.tidas_mode,
        ],
        stdout_file=logs_dir / "validate.stdout.log",
        stderr_file=logs_dir / "validate.stderr.log",
    )
    if not validate_result["ok"]:
        status["status"] = "blocked"
        status["command_error"] = validate_result
        status["patched_rows_file"] = str(patched_rows_file)
        return status

    decision_summary = json.loads((apply_dir / "decision-summary.json").read_text(encoding="utf-8"))
    validation_report = json.loads((validate_dir / "validation-report.json").read_text(encoding="utf-8"))
    invalid_count = int(decision_summary.get("invalid_decisions") or 0)
    unresolved_count = int(decision_summary.get("unresolved_items") or 0)
    patched_count = int(decision_summary.get("patched_entities") or 0)
    validation_failed = int(deep_get(validation_report, ["summary", "failed"]) or 0)

    status.update(
        {
            "decision_summary_file": str(apply_dir / "decision-summary.json"),
            "invalid_decisions_file": str(apply_dir / "invalid-decisions.jsonl"),
            "manual_review_unresolved_file": str(apply_dir / "manual-review-unresolved.jsonl"),
            "patched_rows_file": str(patched_rows_file),
            "validation_report_file": str(validate_dir / "validation-report.json"),
            "validation_failures_file": str(validate_dir / "validation-failures.jsonl"),
            "patched_entities": patched_count,
            "invalid_decisions": invalid_count,
            "unresolved_items": unresolved_count,
            "validation_failed": validation_failed,
        }
    )
    ready = invalid_count == 0 and validation_failed == 0
    status["ready_for_publish"] = ready
    if ready and patched_count > 0:
        status["status"] = "ready"
    elif ready:
        status["status"] = "ready_no_changes"
    else:
        status["status"] = "blocked"
    if unresolved_count > 0:
        status["has_manual_review_residue"] = True
    return status


def run_publish(
    *,
    args: argparse.Namespace,
    flow_rows_file: Path,
    original_rows_file: Path,
    out_dir: Path,
    commit: bool,
) -> None:
    command = [
        "publish-reviewed-data",
        "--flow-rows-file",
        str(flow_rows_file),
        "--original-flow-rows-file",
        str(original_rows_file),
        "--flow-publish-policy",
        args.flow_publish_policy,
        "--process-publish-policy",
        "skip",
        "--out-dir",
        str(out_dir),
    ]
    if commit:
        command.append("--commit")
    logs_dir = ensure_dir(out_dir / "pipeline-logs")
    result = run_flow_governance_command(
        args=args,
        command=command,
        stdout_file=logs_dir / ("publish-commit.stdout.log" if commit else "publish-dry-run.stdout.log"),
        stderr_file=logs_dir / ("publish-commit.stderr.log" if commit else "publish-dry-run.stderr.log"),
    )
    if not result["ok"]:
        raise SystemExit(f"Publish step failed: {result}")


def run_flow_governance_command(
    *,
    args: argparse.Namespace,
    command: list[str],
    stdout_file: Path,
    stderr_file: Path,
) -> dict[str, Any]:
    run_script = Path(args.run_script).expanduser().resolve()
    env_file = Path(args.env_file).expanduser().resolve()
    stdout_file.parent.mkdir(parents=True, exist_ok=True)
    quoted = " ".join(shlex.quote(part) for part in [str(run_script), *command])
    if env_file.exists():
        shell_command = f"set -a && source {shlex.quote(str(env_file))} && set +a && {quoted}"
    else:
        shell_command = quoted
    completed = subprocess.run(
        ["bash", "-lc", shell_command],
        capture_output=True,
        text=True,
    )
    stdout_file.write_text(completed.stdout, encoding="utf-8")
    stderr_file.write_text(completed.stderr, encoding="utf-8")
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout_file": str(stdout_file),
        "stderr_file": str(stderr_file),
    }


def dedupe_flow_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        flow_id, version, _name = extract_flow_identity(row)
        if not flow_id or not version:
            continue
        by_key[f"{flow_id}@{version}"] = row
    merged = list(by_key.values())
    merged.sort(key=lambda row: _flow_sort_key(row))
    return merged


def _flow_sort_key(row: dict[str, Any]) -> tuple[str, tuple[int, ...], str]:
    flow_id, version, name = extract_flow_identity(row)
    return (flow_id, version_key(version), name)


def _load_optional_rows(path: str) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    return load_rows_from_file(target)


def _decision_slug(filename: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "-", str(filename or "").strip())
    return slug or "decisions"


def deep_get(obj: Any, path: list[str], default: Any = None) -> Any:
    cur = obj
    for part in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
        if cur is None:
            return default
    return cur


if __name__ == "__main__":
    main()
