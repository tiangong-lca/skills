#!/usr/bin/env bash
set -euo pipefail

# Safer parallel wrapper for batch flow processing.
# Fixes prior issues where xargs child shells lost env vars and wrote logs to /<file>.log.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SKILL_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
RUNNER="${SCRIPT_DIR}/run-process-automated-builder.sh"

FLOW_DIR=""
OUT_DIR=""
WORKERS=3
OPERATION="produce"
DEFAULT_VENV_PYTHON="${SKILL_DIR}/.venv/bin/python"
if [[ -n "${PAB_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PAB_PYTHON_BIN}"
elif [[ -x "${DEFAULT_VENV_PYTHON}" ]]; then
  PYTHON_BIN="${DEFAULT_VENV_PYTHON}"
else
  PYTHON_BIN="python3"
fi
MAX_ATTEMPTS=3
STALL_TIMEOUT_SECONDS=600
MAX_RUNTIME_SECONDS=0

usage() {
  cat <<'USAGE'
Usage: run-process-automated-builder-parallel.sh --flow-dir <dir> --out-dir <dir> [options]

Options:
  --flow-dir <dir>       Directory containing *.json flow files (required)
  --out-dir <dir>        Output directory for logs/summary/state (required)
  --workers <n>          Concurrent workers (default: 3)
  --operation <mode>     produce|treat (default: produce)
  --python-bin <path>    Python executable for child scripts (default: $PAB_PYTHON_BIN > .venv/bin/python > python3)
  --max-attempts <n>     Max attempts per flow (default: 3)
  --stall-timeout-seconds <n>
                         Kill attempt if log idle exceeds N seconds (default: 600; 0 disables)
  --max-runtime-seconds <n>
                         Hard cap per attempt runtime in seconds (default: 0 disabled)
  -h, --help             Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flow-dir)
      FLOW_DIR="$2"; shift 2 ;;
    --out-dir)
      OUT_DIR="$2"; shift 2 ;;
    --workers)
      WORKERS="$2"; shift 2 ;;
    --operation)
      OPERATION="$2"; shift 2 ;;
    --python-bin)
      PYTHON_BIN="$2"; shift 2 ;;
    --max-attempts)
      MAX_ATTEMPTS="$2"; shift 2 ;;
    --stall-timeout-seconds)
      STALL_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --max-runtime-seconds)
      MAX_RUNTIME_SECONDS="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

[[ -n "${FLOW_DIR}" ]] || { echo "Missing --flow-dir" >&2; exit 2; }
[[ -n "${OUT_DIR}" ]] || { echo "Missing --out-dir" >&2; exit 2; }
[[ -d "${FLOW_DIR}" ]] || { echo "Flow dir not found: ${FLOW_DIR}" >&2; exit 2; }

mkdir -p "${OUT_DIR}"
STATE_PATH="${OUT_DIR}/batch_state.json"
LOG_DIR="${OUT_DIR}/batch_logs"

exec "${PYTHON_BIN}" "${SKILL_DIR}/scripts/origin/process_from_flow_batch_runner.py" \
  --flow-dir "${FLOW_DIR}" \
  --state "${STATE_PATH}" \
  --log-dir "${LOG_DIR}" \
  --workers "${WORKERS}" \
  --operation "${OPERATION}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --stall-timeout-seconds "${STALL_TIMEOUT_SECONDS}" \
  --max-runtime-seconds "${MAX_RUNTIME_SECONDS}" \
  --python-bin "${PYTHON_BIN}"
