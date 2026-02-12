#!/usr/bin/env python
"""Run the process_from_flow workflow with SI integration before Step 1.

Usage:
  uv run python scripts/origin/process_from_flow_workflow.py --flow <path> [options]

This script orchestrates:
  - Step 1a/1b/1c via process_from_flow_langgraph.py --stop-after references
  - reference usability screening (1b optional)
  - SI download + MinerU parsing (1d)
  - reference usage tagging (1e)
  - resume full pipeline so SI affects Step 1/2/3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent
for path in (SCRIPTS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

try:
    from scripts.origin.process_from_flow_langgraph import (  # type: ignore
        build_process_from_flow_run_id,
    )
except ModuleNotFoundError:  # pragma: no cover
    from process_from_flow_langgraph import (  # type: ignore
        build_process_from_flow_run_id,
    )

from tiangong_lca_spec.state_lock import hold_state_file_lock

PROCESS_FROM_FLOW_ARTIFACTS_ROOT = Path("artifacts/process_from_flow")
DEFAULT_SI_SUBDIR = Path("input/si")
MINERU_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".csv", ".tsv", ".xlsx", ".docx"}
WORKFLOW_LOG_SUBDIR = Path("cache/workflow_logs")
WORKFLOW_TIMING_REPORT = Path("cache/workflow_timing_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow", type=Path, required=True, help="Path to the agent-provided reference flow JSON.")
    parser.add_argument("--operation", choices=("produce", "treat"), default="produce", help="Whether the process produces or treats the reference flow.")
    parser.add_argument("--run-id", help="Run identifier under artifacts/process_from_flow/<run_id>.")
    parser.add_argument("--no-translate-zh", action="store_true", help="Skip adding Chinese translations.")
    parser.add_argument(
        "--allow-density-conversion",
        action="store_true",
        help="Allow LLM-based density conversion for mass/volume mismatches.",
    )
    parser.add_argument(
        "--auto-balance-revise",
        action="store_true",
        help=("After the first balance review, auto-revise severe core-mass imbalances " "on non-reference exchanges, then recompute balance review."),
    )
    parser.add_argument("--min-si-hint", default="possible", help="Min si_hint to download (none|possible|likely).")
    parser.add_argument("--si-max-links", type=int, help="Max SI links per DOI.")
    parser.add_argument("--si-timeout", type=float, help="HTTP timeout for SI download.")
    parser.add_argument("--publish", action="store_true", help="Publish generated process datasets after completion.")
    parser.add_argument("--publish-flows", action="store_true", help="Also publish placeholder flow datasets.")
    parser.add_argument("--commit", action="store_true", help="Actually invoke Database_CRUD_Tool (default: dry-run).")
    parser.add_argument(
        "--stop-after",
        choices=("references", "tech", "processes", "exchanges", "matches", "sources", "datasets"),
        help="Stop after a stage (debug only).",
    )
    return parser.parse_args()


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remain = seconds - (minutes * 60)
    return f"{minutes}m{remain:04.1f}s"


def _tail_log(path: Path, *, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _run_python(script: Path, args: list[str], *, log_path: Path | None = None) -> None:
    cmd = [sys.executable, str(script), *args]
    if log_path is None:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{datetime.now(timezone.utc).isoformat()}] CMD: {' '.join(cmd)}\n")
        handle.flush()
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def _run_reference_stage(args: argparse.Namespace, run_id: str, *, log_path: Path | None = None) -> None:
    script = SCRIPT_DIR / "process_from_flow_langgraph.py"
    cmd = [
        "--flow",
        str(args.flow),
        "--operation",
        args.operation,
        "--run-id",
        run_id,
        "--stop-after",
        "references",
    ]
    if args.no_translate_zh:
        cmd.append("--no-translate-zh")
    if args.allow_density_conversion:
        cmd.append("--allow-density-conversion")
    if args.auto_balance_revise:
        cmd.append("--auto-balance-revise")
    _run_python(script, cmd, log_path=log_path)


def _run_usability(args: argparse.Namespace, run_id: str, *, log_path: Path | None = None) -> None:
    script = SCRIPT_DIR / "process_from_flow_reference_usability.py"
    cmd = [
        "--run-id",
        run_id,
    ]
    _run_python(script, cmd, log_path=log_path)


def _run_si_download(args: argparse.Namespace, run_id: str, *, log_path: Path | None = None) -> None:
    script = SCRIPT_DIR / "process_from_flow_download_si.py"
    cmd = [
        "--run-id",
        run_id,
        "--min-si-hint",
        args.min_si_hint,
    ]
    if args.si_max_links is not None:
        cmd.extend(["--max-links", str(args.si_max_links)])
    if args.si_timeout is not None:
        cmd.extend(["--timeout", str(args.si_timeout)])
    _run_python(script, cmd, log_path=log_path)


def _run_usage_tagging(args: argparse.Namespace, run_id: str, *, log_path: Path | None = None) -> None:
    script = SCRIPT_DIR / "process_from_flow_reference_usage_tagging.py"
    cmd = [
        "--run-id",
        run_id,
    ]
    _run_python(script, cmd, log_path=log_path)


def _iter_si_files(run_id: str) -> list[Path]:
    si_root = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / run_id / DEFAULT_SI_SUBDIR
    if not si_root.exists():
        return []
    files: list[Path] = []
    for path in sorted(si_root.rglob("*")):
        if not path.is_file():
            continue
        files.append(path)
    return files


def _run_mineru_for_si(args: argparse.Namespace, run_id: str, *, log_path: Path | None = None) -> None:
    script = SCRIPT_DIR / "mineru_for_process_si.py"
    failures: list[Path] = []
    for path in _iter_si_files(run_id):
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            continue
        if suffix not in MINERU_SUFFIXES:
            print(f"[warn] Skip unsupported SI file: {path}", file=sys.stderr)
            continue
        cmd = [
            str(path),
            "--run-id",
            run_id,
        ]
        try:
            _run_python(script, cmd, log_path=log_path)
        except subprocess.CalledProcessError:
            failures.append(path)
    if failures:
        print(f"[warn] MinerU failed for {len(failures)} SI file(s).", file=sys.stderr)


def _run_main_pipeline(args: argparse.Namespace, run_id: str, *, log_path: Path | None = None) -> None:
    script = SCRIPT_DIR / "process_from_flow_langgraph.py"
    cmd = [
        "--flow",
        str(args.flow),
        "--operation",
        args.operation,
        "--run-id",
        run_id,
        "--resume",
    ]
    if args.no_translate_zh:
        cmd.append("--no-translate-zh")
    if args.allow_density_conversion:
        cmd.append("--allow-density-conversion")
    if args.auto_balance_revise:
        cmd.append("--auto-balance-revise")
    if args.stop_after:
        cmd.extend(["--stop-after", args.stop_after])
    if args.publish:
        cmd.append("--publish")
    if args.publish_flows:
        cmd.append("--publish-flows")
    if args.commit:
        cmd.append("--commit")
    _run_python(script, cmd, log_path=log_path)


def _clear_stop_after(run_id: str) -> None:
    state_path = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / run_id / "cache" / "process_from_flow_state.json"
    if not state_path.exists():
        return
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "stop_after" not in payload:
        return
    payload.pop("stop_after", None)
    with hold_state_file_lock(state_path, reason="workflow.clear_stop_after"):
        state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] Cleared stop_after in {state_path}", file=sys.stderr)


def _write_timing_report(
    *,
    report_path: Path,
    run_id: str,
    flow_path: Path,
    operation: str,
    started_at: datetime,
    stages: list[dict[str, object]],
) -> None:
    total_seconds = float(sum(float(item.get("elapsed_seconds") or 0.0) for item in stages))
    payload = {
        "run_id": run_id,
        "flow_path": str(flow_path),
        "operation": operation,
        "started_at": started_at.isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_elapsed_seconds": round(total_seconds, 3),
        "stages": stages,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.flow.exists():
        raise SystemExit(f"Reference flow file not found: {args.flow}")
    run_id = args.run_id or build_process_from_flow_run_id(args.flow, args.operation)
    run_root = PROCESS_FROM_FLOW_ARTIFACTS_ROOT / run_id
    workflow_log_dir = run_root / WORKFLOW_LOG_SUBDIR
    workflow_log_dir.mkdir(parents=True, exist_ok=True)
    timing_report_path = run_root / WORKFLOW_TIMING_REPORT
    started_at = datetime.now(timezone.utc)

    print(f"[progress] run_id={run_id}", file=sys.stderr)
    print(f"[progress] logs={workflow_log_dir}", file=sys.stderr)

    stage_plan: list[tuple[str, Callable[[Path], None]]] = [
        ("01_references", lambda log: _run_reference_stage(args, run_id, log_path=log)),
        ("02_usability", lambda log: _run_usability(args, run_id, log_path=log)),
        ("03_si_download", lambda log: _run_si_download(args, run_id, log_path=log)),
        ("04_mineru", lambda log: _run_mineru_for_si(args, run_id, log_path=log)),
        ("05_usage_tagging", lambda log: _run_usage_tagging(args, run_id, log_path=log)),
        ("06_clear_stop_after", lambda _log: _clear_stop_after(run_id)),
        ("07_main_pipeline", lambda log: _run_main_pipeline(args, run_id, log_path=log)),
    ]

    stage_records: list[dict[str, object]] = []
    total_start = time.perf_counter()
    total_stages = len(stage_plan)

    for index, (stage_name, runner) in enumerate(stage_plan, start=1):
        stage_log_path = workflow_log_dir / f"{stage_name}.log"
        elapsed_before = time.perf_counter() - total_start
        avg_stage = (
            sum(float(item.get("elapsed_seconds") or 0.0) for item in stage_records) / len(stage_records)
            if stage_records
            else None
        )
        remaining_including_current = total_stages - index + 1
        eta_before = (avg_stage * remaining_including_current) if avg_stage is not None else None
        print(
            (
                f"[progress] stage {index}/{total_stages} start={stage_name} "
                f"elapsed={_format_seconds(elapsed_before)} eta={_format_seconds(eta_before)} "
                f"log={stage_log_path}"
            ),
            file=sys.stderr,
        )

        stage_started_at = datetime.now(timezone.utc)
        stage_start = time.perf_counter()
        try:
            runner(stage_log_path)
        except Exception as exc:  # noqa: BLE001
            stage_elapsed = time.perf_counter() - stage_start
            record = {
                "index": index,
                "name": stage_name,
                "status": "failed",
                "started_at": stage_started_at.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(stage_elapsed, 3),
                "log_path": str(stage_log_path),
                "error": str(exc),
            }
            stage_records.append(record)
            _write_timing_report(
                report_path=timing_report_path,
                run_id=run_id,
                flow_path=args.flow,
                operation=args.operation,
                started_at=started_at,
                stages=stage_records,
            )
            tail = _tail_log(stage_log_path)
            if tail:
                print(f"[progress] stage {stage_name} failed. log tail:\n{tail}", file=sys.stderr)
            raise

        stage_elapsed = time.perf_counter() - stage_start
        record = {
            "index": index,
            "name": stage_name,
            "status": "ok",
            "started_at": stage_started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(stage_elapsed, 3),
            "log_path": str(stage_log_path),
        }
        stage_records.append(record)
        _write_timing_report(
            report_path=timing_report_path,
            run_id=run_id,
            flow_path=args.flow,
            operation=args.operation,
            started_at=started_at,
            stages=stage_records,
        )

        elapsed_after = time.perf_counter() - total_start
        remaining = total_stages - index
        avg_after = sum(float(item.get("elapsed_seconds") or 0.0) for item in stage_records) / len(stage_records)
        eta_after = avg_after * remaining if remaining > 0 else 0.0
        print(
            (
                f"[progress] stage {index}/{total_stages} done={stage_name} "
                f"stage_elapsed={_format_seconds(stage_elapsed)} "
                f"total_elapsed={_format_seconds(elapsed_after)} "
                f"eta={_format_seconds(eta_after)}"
            ),
            file=sys.stderr,
        )

    total_elapsed = time.perf_counter() - total_start
    print(f"[progress] workflow completed run_id={run_id} total={_format_seconds(total_elapsed)}", file=sys.stderr)
    print(f"[progress] timing_report={timing_report_path}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    main()
