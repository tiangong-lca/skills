#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from flow_governance_common import dump_json, dump_jsonl, ensure_dir, load_rows_from_file


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standard flow governance pipeline from local snapshots.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--subject-flows-file", help="Preferred local draft flow rows JSON/JSONL input.")
    source_group.add_argument("--flows-file", help="Explicit full-snapshot flow rows JSON/JSONL input.")

    parser.add_argument("--candidate-flows-file", help="Optional explicit candidate pool snapshot for dedup.")
    parser.add_argument("--processes-file", help="Optional process snapshot used for ref counts, scan/repair, and flow context.")
    parser.add_argument("--process-pool-file", help="Optional local process pool file to sync patched process rows into.")
    parser.add_argument("--scope-flow-files", nargs="+", help="Optional target flow scope for downstream scan/text-review.")
    parser.add_argument("--old-flow-files", nargs="+", help="Optional pre-governance flow scope for alias-map build.")
    parser.add_argument("--new-flow-files", nargs="+", help="Optional post-governance flow scope for alias-map build.")
    parser.add_argument("--alias-map", help="Optional prebuilt alias map JSON. Skips alias-map build when provided.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--auto-patch-policy",
        choices=("disabled", "alias-only", "alias-or-unique-name"),
        default="alias-only",
    )
    parser.add_argument("--enable-review-llm", action="store_true", help="Enable LLM in review-flows.")
    parser.add_argument("--disable-review-llm", action="store_true", help="Force rule-only review-flows.")
    parser.add_argument("--with-reference-context", action="store_true")
    parser.add_argument("--similarity-threshold", type=float)
    parser.add_argument("--methodology-file")
    parser.add_argument("--methodology-id")
    parser.add_argument("--max-text-review-items", type=int, default=0)
    parser.add_argument("--max-process-contexts", type=int, default=3)
    parser.add_argument("--user-id")
    parser.add_argument("--no-user-0", action="store_true")
    parser.add_argument("--no-public-100", action="store_true")
    parser.add_argument("--live-ref-counts", action="store_true")
    parser.add_argument("--public-origin-filter", choices=("all", "tiangong", "ilcd"))
    parser.add_argument("--tidas-mode", choices=("auto", "required", "skip"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.enable_review_llm and args.disable_review_llm:
        raise SystemExit("Use at most one of --enable-review-llm or --disable-review-llm.")
    if args.candidate_flows_file and not args.subject_flows_file:
        raise SystemExit("--candidate-flows-file requires --subject-flows-file.")

    out_dir = ensure_dir(args.out_dir)
    inputs_dir = ensure_dir(out_dir / "inputs")
    manifest: dict[str, Any] = {"inputs": {}, "steps": []}

    review_rows_file = Path(args.subject_flows_file or args.flows_file).resolve()
    manifest["inputs"]["review_rows_file"] = str(review_rows_file)
    if args.candidate_flows_file:
        manifest["inputs"]["candidate_flows_file"] = str(Path(args.candidate_flows_file).resolve())
    if args.processes_file:
        manifest["inputs"]["processes_file"] = str(Path(args.processes_file).resolve())
    if args.process_pool_file:
        manifest["inputs"]["process_pool_file"] = str(Path(args.process_pool_file).resolve())

    scope_flow_files = _resolve_scope_flow_files(args, review_rows_file)
    manifest["inputs"]["scope_flow_files"] = [str(path) for path in scope_flow_files]
    merged_scope_flow_file = _materialize_merged_rows(scope_flow_files, inputs_dir / "scope-flows.snapshot.jsonl")
    manifest["inputs"]["merged_scope_flow_file"] = str(merged_scope_flow_file)

    review_dir = ensure_dir(out_dir / "review")
    review_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "flow_review_engine.py"),
        "--rows-file",
        str(review_rows_file),
        "--out-dir",
        str(review_dir),
    ]
    if args.enable_review_llm:
        review_cmd.append("--enable-llm")
    elif args.disable_review_llm or not args.enable_review_llm:
        review_cmd.append("--disable-llm")
    if args.with_reference_context:
        review_cmd.append("--with-reference-context")
    if args.similarity_threshold is not None:
        review_cmd += ["--similarity-threshold", str(args.similarity_threshold)]
    if args.methodology_file:
        review_cmd += ["--methodology-file", str(Path(args.methodology_file).resolve())]
    if args.methodology_id:
        review_cmd += ["--methodology-id", args.methodology_id]
    _run_step("review_flows", review_cmd, review_dir, manifest)

    dedup_dir = ensure_dir(out_dir / "dedup")
    dedup_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "flow_dedup_candidates.py"),
        "--out-dir",
        str(dedup_dir),
    ]
    if args.flows_file:
        dedup_cmd += ["--flows-file", str(review_rows_file)]
    else:
        dedup_cmd += ["--subject-flows-file", str(review_rows_file)]
        if args.candidate_flows_file:
            dedup_cmd += ["--candidate-flows-file", str(Path(args.candidate_flows_file).resolve())]
    if args.processes_file:
        dedup_cmd += ["--processes-file", str(Path(args.processes_file).resolve())]
    elif args.live_ref_counts:
        dedup_cmd.append("--live-ref-counts")
    if args.user_id:
        dedup_cmd += ["--user-id", args.user_id]
    if args.no_user_0:
        dedup_cmd.append("--no-user-0")
    if args.no_public_100:
        dedup_cmd.append("--no-public-100")
    if args.public_origin_filter:
        dedup_cmd += ["--public-origin-filter", args.public_origin_filter]
    _run_step("flow_dedup_candidates", dedup_cmd, dedup_dir, manifest)

    dedup_manual_queue_path = dedup_dir / "flow-dedup-manual-review.jsonl"
    if _count_rows(dedup_manual_queue_path) > 0:
        dedup_pack_dir = ensure_dir(out_dir / "dedup-pack")
        dedup_pack_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "openclaw_review_handoff.py"),
            "export-dedup-review-pack",
            "--review-queue-file",
            str(dedup_manual_queue_path),
            "--out-dir",
            str(dedup_pack_dir),
        ]
        _run_step("export_openclaw_dedup_review_pack", dedup_pack_cmd, dedup_pack_dir, manifest)
    else:
        _record_skipped_step(
            manifest,
            "export_openclaw_dedup_review_pack",
            "skipped because dedup/flow-dedup-manual-review.jsonl is empty.",
        )

    effective_alias_map: Path | None = None
    if args.alias_map:
        effective_alias_map = Path(args.alias_map).resolve()
        _record_external_input_step(
            manifest,
            "alias_map",
            {"status": "external_input", "path": str(effective_alias_map)},
        )
    elif args.old_flow_files and args.new_flow_files:
        alias_dir = ensure_dir(out_dir / "alias")
        alias_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "flow_alias_map.py"),
            "--out-dir",
            str(alias_dir),
            "--old-flow-files",
            *[str(Path(path).resolve()) for path in args.old_flow_files],
            "--new-flow-files",
            *[str(Path(path).resolve()) for path in args.new_flow_files],
        ]
        _run_step("build_flow_alias_map", alias_cmd, alias_dir, manifest)
        effective_alias_map = alias_dir / "flow-alias-map.json"
    else:
        _record_skipped_step(
            manifest,
            "build_flow_alias_map",
            "skipped because neither --alias-map nor --old-flow-files/--new-flow-files were provided.",
        )

    if args.processes_file:
        processes_file = Path(args.processes_file).resolve()
        scan_dir = ensure_dir(out_dir / "scan")
        scan_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "process_flow_ref_scan.py"),
            "--processes-file",
            str(processes_file),
            "--scope-flow-files",
            *[str(path) for path in scope_flow_files],
            "--out-dir",
            str(scan_dir),
        ]
        if effective_alias_map:
            scan_cmd += ["--alias-map", str(effective_alias_map)]
        _run_step("scan_process_flow_refs", scan_cmd, scan_dir, manifest)

        repair_dir = ensure_dir(out_dir / "repair")
        repair_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "process_flow_repair.py"),
            "--processes-file",
            str(processes_file),
            "--scope-flow-files",
            *[str(path) for path in scope_flow_files],
            "--scan-findings",
            str(scan_dir / "scan-findings.json"),
            "--auto-patch-policy",
            args.auto_patch_policy,
            "--out-dir",
            str(repair_dir),
        ]
        if effective_alias_map:
            repair_cmd += ["--alias-map", str(effective_alias_map)]
        _run_step("plan_process_flow_repairs", repair_cmd, repair_dir, manifest)

        repair_apply_dir = ensure_dir(out_dir / "repair-apply")
        repair_apply_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "process_flow_repair.py"),
            "--apply",
            "--processes-file",
            str(processes_file),
            "--scope-flow-files",
            *[str(path) for path in scope_flow_files],
            "--scan-findings",
            str(scan_dir / "scan-findings.json"),
            "--auto-patch-policy",
            args.auto_patch_policy,
            "--out-dir",
            str(repair_apply_dir),
        ]
        if args.process_pool_file:
            repair_apply_cmd += ["--process-pool-file", str(Path(args.process_pool_file).resolve())]
        if effective_alias_map:
            repair_apply_cmd += ["--alias-map", str(effective_alias_map)]
        _run_step("apply_process_flow_repairs", repair_apply_cmd, repair_apply_dir, manifest)

        validate_dir = ensure_dir(out_dir / "validate")
        validate_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "process_patch_validate.py"),
            "--original-processes-file",
            str(processes_file),
            "--patched-processes-file",
            str(repair_apply_dir / "patched-processes.json"),
            "--scope-flow-files",
            *[str(path) for path in scope_flow_files],
            "--tidas-mode",
            args.tidas_mode,
            "--out-dir",
            str(validate_dir),
        ]
        _run_step("validate_processes", validate_cmd, validate_dir, manifest)

        manual_queue_path = repair_dir / "manual-review-queue.jsonl"
        if _count_rows(manual_queue_path) > 0:
            ref_pack_dir = ensure_dir(out_dir / "ref-pack")
            ref_pack_cmd = [
                sys.executable,
                str(SCRIPT_DIR / "openclaw_review_handoff.py"),
                "export-ref-review-pack",
                "--processes-file",
                str(processes_file),
                "--review-queue-file",
                str(manual_queue_path),
                "--scope-flow-files",
                *[str(path) for path in scope_flow_files],
                "--out-dir",
                str(ref_pack_dir),
            ]
            _run_step("export_openclaw_ref_review_pack", ref_pack_cmd, ref_pack_dir, manifest)
        else:
            _record_skipped_step(
                manifest,
                "export_openclaw_ref_review_pack",
                "skipped because repair/manual-review-queue.jsonl is empty.",
            )
    else:
        for step_name in (
            "scan_process_flow_refs",
            "plan_process_flow_repairs",
            "apply_process_flow_repairs",
            "validate_processes",
            "export_openclaw_ref_review_pack",
        ):
            _record_skipped_step(manifest, step_name, "skipped because --processes-file was not provided.")

    flow_text_pack_dir = ensure_dir(out_dir / "flow-text-pack")
    flow_text_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "openclaw_review_handoff.py"),
        "export-text-review-pack",
        "--entity-type",
        "flow",
        "--rows-file",
        str(merged_scope_flow_file),
        "--out-dir",
        str(flow_text_pack_dir),
        "--max-process-contexts",
        str(args.max_process_contexts),
    ]
    if args.max_text_review_items > 0:
        flow_text_cmd += ["--max-items", str(args.max_text_review_items)]
    if args.processes_file:
        flow_text_cmd += ["--processes-file", str(Path(args.processes_file).resolve())]
    if args.methodology_file:
        flow_text_cmd += ["--methodology-file", str(Path(args.methodology_file).resolve())]
    if args.methodology_id:
        flow_text_cmd += ["--methodology-id", args.methodology_id]
    _run_step("export_openclaw_flow_text_review_pack", flow_text_cmd, flow_text_pack_dir, manifest)

    flow_classification_pack_dir = ensure_dir(out_dir / "flow-classification-pack")
    flow_classification_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "openclaw_review_handoff.py"),
        "export-classification-review-pack",
        "--rows-file",
        str(merged_scope_flow_file),
        "--review-findings-file",
        str(review_dir / "findings.jsonl"),
        "--out-dir",
        str(flow_classification_pack_dir),
    ]
    if args.methodology_file:
        flow_classification_cmd += ["--methodology-file", str(Path(args.methodology_file).resolve())]
    if args.methodology_id:
        flow_classification_cmd += ["--methodology-id", args.methodology_id]
    _run_step(
        "export_openclaw_flow_classification_review_pack",
        flow_classification_cmd,
        flow_classification_pack_dir,
        manifest,
    )

    manifest["key_outputs"] = {
        "review_summary": str(review_dir / "flow_review_summary.json"),
        "dedup_summary": str(dedup_dir / "flow-dedup-summary.json"),
        "dedup_snapshot_manifest": str(dedup_dir / "flow-snapshot-manifest.json"),
        "dedup_review_pack": str(out_dir / "dedup-pack" / "review-pack.json"),
        "effective_alias_map": str(effective_alias_map) if effective_alias_map else "",
        "flow_text_review_pack": str(flow_text_pack_dir / "review-pack.json"),
        "flow_text_review_context": str(flow_text_pack_dir / "review-pack-context.json"),
        "flow_classification_review_pack": str(flow_classification_pack_dir / "review-pack.json"),
        "flow_classification_review_context": str(flow_classification_pack_dir / "review-pack-context.json"),
    }
    if args.processes_file:
        manifest["key_outputs"].update(
            {
                "scan_summary": str(out_dir / "scan" / "scan-summary.json"),
                "repair_summary": str(out_dir / "repair" / "repair-summary.json"),
                "validation_report": str(out_dir / "validate" / "validation-report.json"),
                "ref_review_pack": str(out_dir / "ref-pack" / "review-pack.json"),
            }
        )
    dump_json(out_dir / "governance-run-manifest.json", manifest)


def _resolve_scope_flow_files(args: argparse.Namespace, default_file: Path) -> list[Path]:
    if args.scope_flow_files:
        return [Path(path).resolve() for path in args.scope_flow_files]
    if args.new_flow_files:
        return [Path(path).resolve() for path in args.new_flow_files]
    return [default_file]


def _materialize_merged_rows(paths: list[Path], target: Path) -> Path:
    if len(paths) == 1:
        return paths[0]
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(load_rows_from_file(path))
    dump_jsonl(target, rows)
    return target


def _run_step(step_name: str, cmd: list[str], out_dir: Path, manifest: dict[str, Any]) -> None:
    completed = subprocess.run(cmd, capture_output=True, text=True)
    step = {
        "step": step_name,
        "status": "completed" if completed.returncode == 0 else "failed",
        "command": cmd,
        "out_dir": str(out_dir),
        "returncode": completed.returncode,
    }
    if completed.stdout.strip():
        step["stdout"] = completed.stdout.strip()
    if completed.stderr.strip():
        step["stderr"] = completed.stderr.strip()
    manifest.setdefault("steps", []).append(step)
    if completed.returncode != 0:
        dump_json(out_dir.parent / "governance-run-manifest.json", manifest)
        raise SystemExit(completed.returncode)


def _record_skipped_step(manifest: dict[str, Any], step_name: str, reason: str) -> None:
    manifest.setdefault("steps", []).append({"step": step_name, "status": "skipped", "reason": reason})


def _record_external_input_step(manifest: dict[str, Any], step_name: str, payload: dict[str, Any]) -> None:
    manifest.setdefault("steps", []).append({"step": step_name, **payload})


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return len(load_rows_from_file(path))


if __name__ == "__main__":
    main()
