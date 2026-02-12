"""Cross-process file lock helpers for process_from_flow runtime state writes."""

from __future__ import annotations

import json
import os
import socket
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("State file locking requires fcntl on this platform.") from exc

_STATE_FILE_NAME = "process_from_flow_state.json"
_LOCK_TIMEOUT_ENV = "TIANGONG_PFF_STATE_LOCK_TIMEOUT_SECONDS"
_LOCK_POLL_ENV = "TIANGONG_PFF_STATE_LOCK_POLL_SECONDS"
_DEFAULT_TIMEOUT_SECONDS = 300.0
_DEFAULT_POLL_SECONDS = 0.2


class StateFileLockTimeout(TimeoutError):
    """Raised when the state file lock cannot be acquired before timeout."""


def is_process_state_path(path: Path) -> bool:
    """Return True if the path points to process_from_flow state JSON."""
    return path.name == _STATE_FILE_NAME


def _read_float_env(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _lock_path_for_state(state_path: Path) -> Path:
    return state_path.with_name(f"{state_path.name}.lock")


def _try_acquire_lock(handle: Any) -> bool:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _release_lock(handle: Any) -> None:
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_lock_metadata(handle: Any, *, reason: str) -> None:
    metadata = {
        "owner_pid": os.getpid(),
        "owner_host": socket.gethostname(),
        "reason": reason,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    handle.seek(0)
    handle.truncate(0)
    handle.write(json.dumps(metadata, ensure_ascii=False, indent=2))
    handle.flush()
    os.fsync(handle.fileno())


@contextmanager
def hold_state_file_lock(
    state_path: Path,
    *,
    reason: str,
    timeout_seconds: float | None = None,
    poll_seconds: float | None = None,
    logger: Any | None = None,
) -> Iterator[Path]:
    """Acquire an exclusive lock for a process_from_flow state file."""
    timeout = _read_float_env(_LOCK_TIMEOUT_ENV, _DEFAULT_TIMEOUT_SECONDS) if timeout_seconds is None else float(timeout_seconds)
    poll = _read_float_env(_LOCK_POLL_ENV, _DEFAULT_POLL_SECONDS) if poll_seconds is None else float(poll_seconds)
    timeout = max(timeout, 0.0)
    poll = max(poll, 0.05)

    lock_path = _lock_path_for_state(state_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    with lock_path.open("a+", encoding="utf-8") as handle:
        while True:
            if _try_acquire_lock(handle):
                break
            waited = time.monotonic() - started
            if timeout > 0 and waited >= timeout:
                raise StateFileLockTimeout(
                    f"Timed out after {waited:.2f}s acquiring lock {lock_path} (reason={reason})."
                )
            time.sleep(poll)

        waited = time.monotonic() - started
        try:
            _write_lock_metadata(handle, reason=reason)
        except Exception:
            pass

        if logger is not None:
            logger.info(
                "process_from_flow.state_lock_acquired",
                state_path=str(state_path),
                lock_path=str(lock_path),
                reason=reason,
                waited_seconds=round(waited, 3),
            )

        try:
            yield state_path
        finally:
            _release_lock(handle)
            if logger is not None:
                logger.info(
                    "process_from_flow.state_lock_released",
                    state_path=str(state_path),
                    lock_path=str(lock_path),
                    reason=reason,
                )
