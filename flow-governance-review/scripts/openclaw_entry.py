#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from flow_governance_common import dump_json, ensure_dir, load_rows_from_file


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = Path.home() / ".openclaw" / "workspace" / "review" / "flow-governance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "OpenClaw-facing flow governance entrypoint. Use a flow UUID to fetch the subject flow first, "
            "or pass local flow snapshot files to run governance directly."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--flow-id", help="Target flow UUID. Fetch the subject flow first, then run governance.")
    source_group.add_argument("--subject-flows-file", help="Local subject flow rows JSON/JSONL file.")
    source_group.add_argument("--flows-file", help="Local full-snapshot flow rows JSON/JSONL file.")

    parser.add_argument("--flow-version", help="Optional exact version filter when --flow-id is used.")
    parser.add_argument("--candidate-flows-file", help="Optional local candidate pool snapshot for dedup.")
    parser.add_argument("--processes-file", help="Optional local process rows JSON/JSONL file.")
    parser.add_argument("--process-pool-file", help="Optional local process pool file for process reuse and sync.")
    parser.add_argument("--scope-flow-files", nargs="+", help="Optional explicit downstream scope flow files.")
    parser.add_argument("--out-dir", help="Run root.")
    parser.add_argument("--skip-process-fetch", action="store_true")
    parser.add_argument("--flow-page-size", type=int, default=50)
    parser.add_argument("--process-page-size", type=int, default=100)
    parser.add_argument("--subject-fetch-mode", choices=("auto", "mcp", "postgrest"), default="auto")
    parser.add_argument("--user-id")
    parser.add_argument("--no-user-0", action="store_true")
    parser.add_argument("--no-public-100", action="store_true")
    parser.add_argument("--public-origin-filter", choices=("all", "tiangong", "ilcd"), default="all")
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
    parser.add_argument("--supabase-url")
    parser.add_argument("--supabase-publishable-key")
    parser.add_argument("--access-token")
    parser.add_argument("--email")
    parser.add_argument("--password")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.enable_review_llm and args.disable_review_llm:
        raise SystemExit("Use at most one of --enable-review-llm or --disable-review-llm.")
    out_dir = ensure_dir(_default_out_dir(args))

    if args.flow_id:
        summary_path = _run_flow_id_mode(args, out_dir)
        print(str(summary_path))
        return

    summary_path = _run_local_file_mode(args, out_dir)
    print(str(summary_path))


def _default_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir:
        return Path(args.out_dir).expanduser().resolve()
    if args.flow_id:
        version_slug = f"__{args.flow_version}" if args.flow_version else ""
        return (DEFAULT_OUT_ROOT / f"{args.flow_id}{version_slug}").resolve()
    source = Path(args.subject_flows_file or args.flows_file).expanduser().resolve()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return (DEFAULT_OUT_ROOT / f"local-{source.stem}-{timestamp}").resolve()


def _run_flow_id_mode(args: argparse.Namespace, out_dir: Path) -> Path:
    env = _subprocess_env(args)
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "openclaw_full_run.py"),
        "--flow-id",
        args.flow_id,
        "--out-dir",
        str(out_dir),
        "--subject-fetch-mode",
        args.subject_fetch_mode,
        "--flow-page-size",
        str(args.flow_page_size),
        "--process-page-size",
        str(args.process_page_size),
        "--auto-patch-policy",
        args.auto_patch_policy,
        "--public-origin-filter",
        args.public_origin_filter,
        "--tidas-mode",
        args.tidas_mode,
    ]
    if args.flow_version:
        cmd += ["--flow-version", args.flow_version]
    if args.skip_process_fetch:
        cmd.append("--skip-process-fetch")
    if args.process_pool_file:
        cmd += ["--process-pool-file", str(Path(args.process_pool_file).expanduser().resolve())]
    if args.user_id:
        cmd += ["--user-id", args.user_id]
    if args.no_user_0:
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
    for option_name, option_value in (
        ("--supabase-url", args.supabase_url),
        ("--supabase-publishable-key", args.supabase_publishable_key),
        ("--access-token", args.access_token),
        ("--email", args.email),
        ("--password", args.password),
    ):
        text = str(option_value or "").strip()
        if text:
            cmd += [option_name, text]

    completed = subprocess.run(cmd, capture_output=True, text=True, env=env)
    dump_json(
        out_dir / "openclaw-entry-command-result.json",
        {
            "command": cmd,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        },
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    summary_path = out_dir / "openclaw-handoff-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["entrypoint"] = "openclaw-entry"
    summary["entrypoint_mode"] = "flow_uuid"
    dump_json(summary_path, summary)
    return summary_path


def _run_local_file_mode(args: argparse.Namespace, out_dir: Path) -> Path:
    review_rows_file = Path(args.subject_flows_file or args.flows_file).expanduser().resolve()
    env = _subprocess_env(args)
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "flow_governance_orchestrator.py"),
        "--out-dir",
        str(out_dir),
        "--auto-patch-policy",
        args.auto_patch_policy,
        "--public-origin-filter",
        args.public_origin_filter,
        "--tidas-mode",
        args.tidas_mode,
    ]
    if args.subject_flows_file:
        cmd += ["--subject-flows-file", str(review_rows_file)]
    else:
        cmd += ["--flows-file", str(review_rows_file)]
    candidate_flows_file = args.candidate_flows_file
    if args.subject_flows_file and not candidate_flows_file and not _can_use_live_candidate_pool(args, env):
        candidate_flows_file = str(review_rows_file)
    if candidate_flows_file:
        cmd += ["--candidate-flows-file", str(Path(candidate_flows_file).expanduser().resolve())]
    if args.processes_file:
        cmd += ["--processes-file", str(Path(args.processes_file).expanduser().resolve())]
    if args.process_pool_file:
        cmd += ["--process-pool-file", str(Path(args.process_pool_file).expanduser().resolve())]
    if args.scope_flow_files:
        cmd += ["--scope-flow-files", *[str(Path(path).expanduser().resolve()) for path in args.scope_flow_files]]
    if args.user_id:
        cmd += ["--user-id", args.user_id]
    if args.no_user_0:
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

    completed = subprocess.run(cmd, capture_output=True, text=True, env=env)
    dump_json(
        out_dir / "openclaw-entry-command-result.json",
        {
            "command": cmd,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        },
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    governance_manifest_path = out_dir / "governance-run-manifest.json"
    governance_manifest = json.loads(governance_manifest_path.read_text(encoding="utf-8"))
    process_rows_file = Path(args.processes_file).expanduser().resolve() if args.processes_file else None
    handoff_summary = {
        "skill": "flow-governance-review",
        "entrypoint": "openclaw-entry",
        "entrypoint_mode": "local_snapshot",
        "flow_id": "",
        "flow_version": "",
        "run_root": str(out_dir),
        "source_review_rows_file": str(review_rows_file),
        "governance_manifest": str(governance_manifest_path),
        "key_outputs": governance_manifest.get("key_outputs", {}),
        "subject_flow_count": len(load_rows_from_file(review_rows_file)),
        "process_count": len(load_rows_from_file(process_rows_file)) if process_rows_file else 0,
        "process_pool_file": str(Path(args.process_pool_file).expanduser().resolve()) if args.process_pool_file else "",
        "candidate_user_id": "",
        "notes": [
            "OpenClaw entry accepted a local snapshot directly and ran the standard governance orchestrator.",
            "Residual semantic decisions remain in exported OpenClaw review packs until explicit apply/publish steps are run.",
        ],
    }
    summary_path = out_dir / "openclaw-handoff-summary.json"
    dump_json(summary_path, handoff_summary)
    return summary_path


def _subprocess_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    for key, value in (
        ("SUPABASE_URL", args.supabase_url),
        ("SUPABASE_PUBLISHABLE_KEY", args.supabase_publishable_key),
        ("SUPABASE_ACCESS_TOKEN", args.access_token),
        ("SUPABASE_EMAIL", args.email),
        ("SUPABASE_PASSWORD", args.password),
    ):
        text = str(value or "").strip()
        if text:
            env[key] = text
    return env


def _can_use_live_candidate_pool(args: argparse.Namespace, env: dict[str, str]) -> bool:
    if args.no_user_0 and args.no_public_100:
        return False
    return bool(env.get("SUPABASE_URL", "").strip() and env.get("SUPABASE_PUBLISHABLE_KEY", "").strip())


if __name__ == "__main__":
    main()
