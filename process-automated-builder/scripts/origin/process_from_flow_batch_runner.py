#!/usr/bin/env python
"""Batch runner for process_from_flow with persistent state and auto-resume.

Step-1 (P0) objective:
- run multiple flow files concurrently (default worker=3)
- persist scheduler state to JSON
- retry interrupted runs with workflow mode + explicit `--run-id`

This script intentionally focuses on orchestration reliability first.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent.parent
RUN_WRAPPER = SKILL_ROOT / "scripts" / "run-process-automated-builder.mjs"
DEFAULT_WORKERS_ENV_FILE = Path.home() / ".config" / "process-from-flow-batch" / "env"


TaskStatus = Literal["pending", "running", "success", "failed"]
RUN_ID_RE = re.compile(r"run_id=([A-Za-z0-9_\-]+)")
WORKERS_LINE_RE = re.compile(r"^\s*WORKERS\s*=\s*([0-9]+)\s*$", re.MULTILINE)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Task:
    flow_file: str
    status: TaskStatus = "pending"
    attempts: int = 0
    run_id: str | None = None
    process_pid: int | None = None
    started_at: str | None = None
    updated_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    mode: str | None = None
    log_path: str | None = None
    last_error: str | None = None


class BatchRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.flow_dir = args.flow_dir.resolve()
        self.state_path = args.state.resolve()
        self.manifest_path = args.manifest_path.resolve() if args.manifest_path else self.state_path.with_name("batch_manifest.csv")
        self.log_dir = args.log_dir.resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.python_bin = args.python_bin
        self.max_workers = args.workers
        workers_env_file = args.workers_env_file or DEFAULT_WORKERS_ENV_FILE
        self.workers_env_file = workers_env_file.expanduser().resolve()
        self.workers_reload_seconds = max(0.0, float(args.workers_reload_seconds))
        self.max_attempts = args.max_attempts
        self.operation = args.operation
        self.heartbeat_seconds = args.heartbeat_seconds
        self.stall_timeout_seconds = args.stall_timeout_seconds
        self.max_runtime_seconds = args.max_runtime_seconds

        self._last_heartbeat_monotonic = 0.0
        self._last_workers_reload_monotonic = 0.0
        self._workers_env_mtime: float | None = None

        self.tasks: dict[str, Task] = {}
        self.procs: dict[str, subprocess.Popen[str]] = {}
        self._stop = False

    def maybe_reload_workers(self, *, force: bool = False) -> None:
        if self.workers_reload_seconds == 0.0 and not force:
            return

        now_mono = time.monotonic()
        if (
            not force
            and self.workers_reload_seconds > 0.0
            and (now_mono - self._last_workers_reload_monotonic) < self.workers_reload_seconds
        ):
            return
        self._last_workers_reload_monotonic = now_mono

        try:
            mtime = self.workers_env_file.stat().st_mtime
        except FileNotFoundError:
            return
        except Exception as exc:
            print(
                f"[batch] workers_reload_read_error env={self.workers_env_file} err={exc}",
                file=sys.stderr,
            )
            return

        if not force and self._workers_env_mtime is not None and mtime <= self._workers_env_mtime:
            return
        self._workers_env_mtime = mtime

        try:
            text = self.workers_env_file.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            print(
                f"[batch] workers_reload_parse_error env={self.workers_env_file} err={exc}",
                file=sys.stderr,
            )
            return

        match = WORKERS_LINE_RE.search(text)
        if not match:
            return
        new_workers = int(match.group(1))
        if new_workers == self.max_workers:
            return

        prev = self.max_workers
        self.max_workers = new_workers
        print(
            f"[batch] workers_hot_reload old={prev} new={new_workers} env={self.workers_env_file}",
            file=sys.stderr,
        )
        self.save_state()

    def _has_process_exports(self, run_id: str | None) -> bool:
        rid = str(run_id or "").strip()
        if not rid:
            return False
        process_dir = SKILL_ROOT / "artifacts" / "process_from_flow" / rid / "exports" / "processes"
        if not process_dir.exists():
            return False
        try:
            for item in process_dir.iterdir():
                if item.is_file():
                    return True
                if item.is_dir():
                    for nested in item.rglob("*"):
                        if nested.is_file():
                            return True
        except Exception:
            return False
        return False

    def _task_outcome(self, task: Task) -> str:
        if task.status == "success":
            # Workflow mode runs with publish enabled by default; however,
            # interrupted/resume edge cases can end with no process exports.
            if self._has_process_exports(task.run_id):
                return "success_published"
            return "success_no_process_output"
        if task.status == "failed":
            if task.attempts >= self.max_attempts:
                return "failed_after_max_attempts"
            return "failed_retryable"
        return task.status

    def write_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "flow_file",
            "flow_name",
            "status",
            "outcome",
            "attempts",
            "max_attempts",
            "run_id",
            "mode",
            "operation",
            "exit_code",
            "last_error",
            "started_at",
            "finished_at",
            "updated_at",
            "log_path",
        ]
        with self.manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for key in sorted(self.tasks):
                task = self.tasks[key]
                writer.writerow(
                    {
                        "flow_file": task.flow_file,
                        "flow_name": Path(task.flow_file).name,
                        "status": task.status,
                        "outcome": self._task_outcome(task),
                        "attempts": task.attempts,
                        "max_attempts": self.max_attempts,
                        "run_id": task.run_id or "",
                        "mode": task.mode or "",
                        "operation": self.operation,
                        "exit_code": "" if task.exit_code is None else task.exit_code,
                        "last_error": task.last_error or "",
                        "started_at": task.started_at or "",
                        "finished_at": task.finished_at or "",
                        "updated_at": task.updated_at or "",
                        "log_path": task.log_path or "",
                    }
                )

    def _resume_state_exists(self, run_id: str | None) -> bool:
        rid = str(run_id or "").strip()
        if not rid:
            return False
        state_path = SKILL_ROOT / "artifacts" / "process_from_flow" / rid / "cache" / "process_from_flow_state.json"
        return state_path.exists()

    def discover_new_flow_files(self, *, persist: bool = True) -> int:
        discovered = 0
        for fp in sorted(self.flow_dir.glob("*.json")):
            key = str(fp)
            if key in self.tasks:
                continue
            self.tasks[key] = Task(flow_file=key, updated_at=now_iso())
            discovered += 1
        if discovered and persist:
            self.save_state()
        return discovered

    def load_or_init(self) -> None:
        if self.state_path.exists() and not self.args.reset:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            for rec in payload.get("tasks", []):
                task = Task(**rec)
                # crashed process should be resumable
                if task.status == "running":
                    task.status = "failed"
                    task.last_error = "Recovered from interrupted scheduler session"
                    task.process_pid = None
                self.tasks[task.flow_file] = task
        self.discover_new_flow_files(persist=False)

        self.save_state()

    def save_state(self) -> None:
        payload = {
            "updated_at": now_iso(),
            "workers": self.max_workers,
            "max_attempts": self.max_attempts,
            "operation": self.operation,
            "manifest_path": str(self.manifest_path),
            "summary": self.summary(),
            "failed_reason_counts": self.failure_reason_counts(),
            "tasks": [asdict(t) for t in self.tasks.values()],
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.write_manifest()

    def _extract_run_id_from_log(self, log_path: Path) -> str | None:
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        match = RUN_ID_RE.search(text)
        if not match:
            return None
        value = str(match.group(1) or "").strip()
        return value or None

    def _latest_snapshot_idle_seconds(self, run_id: str | None) -> float | None:
        rid = str(run_id or "").strip()
        if not rid:
            return None
        snap_dir = (
            Path(SCRIPT_DIR).parent.parent
            / "artifacts"
            / "process_from_flow"
            / rid
            / "cache"
            / "mcp_snapshots"
        )
        if not snap_dir.exists():
            return None
        latest_mtime: float | None = None
        for path in glob.glob(str(snap_dir / "*.jsonl")):
            try:
                mtime = Path(path).stat().st_mtime
            except FileNotFoundError:
                continue
            if latest_mtime is None or mtime > latest_mtime:
                latest_mtime = mtime
        if latest_mtime is None:
            return None
        return time.time() - latest_mtime

    def start_task(self, key: str, task: Task) -> None:
        flow_path = Path(task.flow_file)
        flow_stem = flow_path.stem
        task.attempts += 1
        task.status = "running"
        task.started_at = now_iso()
        task.updated_at = now_iso()

        log_path = self.log_dir / f"{flow_stem}.attempt{task.attempts}.log"
        task.log_path = str(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if task.run_id and not self._resume_state_exists(task.run_id):
            print(
                f"[batch] resume_state_missing run_id={task.run_id} flow={Path(task.flow_file).name}; fallback=fresh_workflow",
                file=sys.stderr,
            )
            task.run_id = None

        # Use workflow mode for both fresh runs and retries so publish defaults
        # stay consistent and stage orchestration remains identical.
        cmd = [
            "node",
            str(RUN_WRAPPER),
            "legacy",
            "--python-bin",
            self.python_bin,
            "--mode",
            "workflow",
            "--flow-file",
            str(flow_path),
            "--",
            "--operation",
            self.operation,
        ]
        if task.run_id:
            task.mode = "workflow_resume"
            cmd.extend(["--run-id", task.run_id])
        else:
            task.mode = "workflow"

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"[{now_iso()}] CMD: {' '.join(cmd)}\n")
            f.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(SKILL_ROOT),
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )

        task.process_pid = proc.pid
        self.procs[key] = proc
        self.save_state()

    def poll_running(self) -> None:
        done_keys: list[str] = []
        state_changed = False
        for key, proc in list(self.procs.items()):
            rc = proc.poll()
            if rc is None:
                # stall/long-tail detection while process is still running
                task = self.tasks[key]

                # hard cap: total runtime per attempt
                if self.max_runtime_seconds and task.started_at:
                    try:
                        started_dt = datetime.fromisoformat(task.started_at)
                        runtime_for = (datetime.now(timezone.utc) - started_dt).total_seconds()
                        if runtime_for > self.max_runtime_seconds:
                            proc.terminate()
                            task.status = "failed"
                            task.last_error = (
                                f"runtime_exceeded: runtime={runtime_for:.1f}s "
                                f"(timeout={self.max_runtime_seconds:.1f}s)"
                            )
                            task.process_pid = None
                            task.updated_at = now_iso()
                            task.finished_at = now_iso()
                            task.exit_code = 124
                            done_keys.append(key)
                            continue
                    except Exception:
                        pass

                if task.log_path:
                    try:
                        log_path = Path(task.log_path)
                        if not task.run_id:
                            inferred_run_id = self._extract_run_id_from_log(log_path)
                            if inferred_run_id:
                                task.run_id = inferred_run_id
                                task.updated_at = now_iso()
                                state_changed = True
                        log_mtime = log_path.stat().st_mtime
                        idle_for = time.time() - log_mtime

                        # idle timeout: no fresh log output
                        if self.stall_timeout_seconds and idle_for > self.stall_timeout_seconds:
                            # If workflow log is quiet but MCP snapshots are still updating,
                            # treat the task as active and avoid false stall-kill.
                            if not task.run_id:
                                inferred_run_id = self._extract_run_id_from_log(log_path)
                                if inferred_run_id:
                                    task.run_id = inferred_run_id
                            snapshot_idle = self._latest_snapshot_idle_seconds(task.run_id)
                            if snapshot_idle is not None and snapshot_idle <= self.stall_timeout_seconds:
                                continue

                            proc.terminate()
                            task.status = "failed"
                            if snapshot_idle is None:
                                task.last_error = (
                                    f"stalled: no log update for {idle_for:.1f}s "
                                    f"(timeout={self.stall_timeout_seconds:.1f}s; snapshot=missing)"
                                )
                            else:
                                task.last_error = (
                                    f"stalled: no log update for {idle_for:.1f}s "
                                    f"(timeout={self.stall_timeout_seconds:.1f}s; snapshot_idle={snapshot_idle:.1f}s)"
                                )
                            task.process_pid = None
                            task.updated_at = now_iso()
                            task.finished_at = now_iso()
                            task.exit_code = 124
                            done_keys.append(key)
                            continue

                    except FileNotFoundError:
                        pass
                    except Exception:
                        pass
                continue
            task = self.tasks[key]
            task.exit_code = rc
            task.process_pid = None
            task.updated_at = now_iso()
            task.finished_at = now_iso()

            if task.log_path:
                try:
                    text = Path(task.log_path).read_text(encoding="utf-8", errors="replace")
                    for line in text.splitlines():
                        if "run_id=" in line and not task.run_id:
                            # supports tokens like "run_id=..."
                            token = line.split("run_id=", 1)[1].split()[0].strip()
                            task.run_id = token
                except Exception:
                    pass

            if rc == 0:
                task.status = "success"
                task.last_error = None
            else:
                task.status = "failed"
                task.last_error = f"exit_code={rc}"
            done_keys.append(key)

        for key in done_keys:
            self.procs.pop(key, None)
        if done_keys or state_changed:
            self.save_state()

    def pending_or_retry_candidates(self) -> list[str]:
        keys: list[str] = []
        for key, task in self.tasks.items():
            if task.status == "success":
                continue
            if task.status == "running":
                continue
            if task.attempts >= self.max_attempts:
                continue
            keys.append(key)
        return keys

    def summary(self) -> dict[str, int]:
        out = {"pending": 0, "running": 0, "success": 0, "failed": 0}
        for t in self.tasks.values():
            out[t.status] += 1
        return out

    def failure_reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task in self.tasks.values():
            if task.status != "failed":
                continue
            reason = str(task.last_error or "unknown").strip()
            key = reason.split(":", 1)[0] if ":" in reason else reason
            key = key or "unknown"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def emit_heartbeat_if_due(self, *, force: bool = False) -> None:
        now_mono = time.monotonic()
        if not force and (now_mono - self._last_heartbeat_monotonic) < self.heartbeat_seconds:
            return
        self._last_heartbeat_monotonic = now_mono

        stat = self.summary()
        running_lines: list[str] = []
        for key, task in self.tasks.items():
            if task.status != "running":
                continue
            elapsed = "?"
            if task.started_at:
                try:
                    dt = datetime.fromisoformat(task.started_at)
                    elapsed = f"{(datetime.now(timezone.utc) - dt).total_seconds():.1f}s"
                except Exception:
                    pass
            idle = "?"
            if task.log_path and Path(task.log_path).exists():
                idle = f"{time.time() - Path(task.log_path).stat().st_mtime:.1f}s"
            running_lines.append(
                f"{Path(task.flow_file).name} mode={task.mode} attempt={task.attempts} elapsed={elapsed} log_idle={idle}"
            )

        reason_counts = self.failure_reason_counts()
        print(
            f"[heartbeat] workers={self.max_workers}"
            f" pending={stat['pending']} running={stat['running']} success={stat['success']} failed={stat['failed']}"
            f" failed_reasons={reason_counts}",
            file=sys.stderr,
        )
        for line in running_lines:
            print(f"[heartbeat] {line}", file=sys.stderr)

    def run(self) -> int:
        self.load_or_init()
        self.maybe_reload_workers(force=True)

        def handle_stop(signum: int, _frame: object) -> None:
            self._stop = True
            print(f"[batch] received signal={signum}, stopping after current poll...", file=sys.stderr)

        signal.signal(signal.SIGINT, handle_stop)
        signal.signal(signal.SIGTERM, handle_stop)

        while True:
            discovered = self.discover_new_flow_files()
            if discovered:
                print(f"[batch] discovered_new_flows={discovered}", file=sys.stderr)
            self.poll_running()
            self.maybe_reload_workers()
            self.emit_heartbeat_if_due()

            if self._stop:
                self.emit_heartbeat_if_due(force=True)
                self.save_state()
                return 130

            candidates = self.pending_or_retry_candidates()
            free_slots = self.max_workers - len(self.procs)
            if free_slots > 0 and candidates:
                for key in candidates[:free_slots]:
                    self.start_task(key, self.tasks[key])

            if not self.procs:
                # no running process; decide finish condition
                candidates = self.pending_or_retry_candidates()
                if not candidates:
                    if not self.args.watch:
                        break

            time.sleep(self.args.poll_seconds)

        self.emit_heartbeat_if_due(force=True)
        self.save_state()
        final = self.summary()
        final_failed_reasons = self.failure_reason_counts()
        print(f"[batch] final_summary={final} failed_reasons={final_failed_reasons}", file=sys.stderr)
        return 0 if final["failed"] == 0 else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow-dir", type=Path, required=True, help="Directory containing flow JSON files")
    p.add_argument("--state", type=Path, required=True, help="Persistent scheduler state JSON path")
    p.add_argument(
        "--manifest-path",
        type=Path,
        help="Optional CSV path for full flow status manifest (default: sibling of --state named batch_manifest.csv)",
    )
    p.add_argument("--log-dir", type=Path, required=True, help="Directory for per-attempt logs")
    p.add_argument("--workers", type=int, default=3, help="Concurrent worker count")
    p.add_argument(
        "--workers-env-file",
        type=Path,
        default=None,
        help=(
            "Hot-reload source for WORKERS (default: ~/.config/process-from-flow-batch/env). "
            "If the file WORKERS value changes, new scheduling uses the new concurrency."
        ),
    )
    p.add_argument(
        "--workers-reload-seconds",
        type=float,
        default=5.0,
        help="Hot-reload interval for --workers-env-file (0 disables hot reload).",
    )
    p.add_argument("--operation", choices=("produce", "treat"), default="produce")
    p.add_argument("--max-attempts", type=int, default=3, help="Max attempts per flow (resume counts)")
    p.add_argument("--poll-seconds", type=float, default=5.0, help="Polling interval for child processes")
    p.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
        help="Emit scheduler heartbeat every N seconds.",
    )
    p.add_argument(
        "--stall-timeout-seconds",
        type=float,
        default=600.0,
        help="Mark running task stalled if log not updated for N seconds (0 to disable).",
    )
    p.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=0.0,
        help="Hard cap for one attempt total runtime (0 to disable).",
    )
    p.add_argument(
        "--watch",
        action="store_true",
        help="Keep the scheduler process alive and keep polling for newly added flow JSON files.",
    )
    p.add_argument("--python-bin", default=os.environ.get("PAB_PYTHON_BIN", "python3"))
    p.add_argument("--reset", action="store_true", help="Ignore existing state and reinitialize tasks")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runner = BatchRunner(args)
    rc = runner.run()
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
