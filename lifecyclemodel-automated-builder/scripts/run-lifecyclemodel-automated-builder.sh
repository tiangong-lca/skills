#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SKILL_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

DEFAULT_MANIFEST="${SKILL_DIR}/assets/example-request.json"
DEFAULT_OUT_DIR="${SKILL_DIR}/../artifacts/lifecyclemodel-automated-builder/default-run"

MANIFEST="${DEFAULT_MANIFEST}"
OUT_DIR="${DEFAULT_OUT_DIR}"
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: run-lifecyclemodel-automated-builder.sh [options]

Options:
  --manifest <file>    Manifest JSON path (default: assets/example-request.json)
  --out-dir <dir>      Output directory for local artifacts
  --dry-run            Print the resolved execution plan and exit
  -h, --help           Show this help message
USAGE
}

fail() {
  echo "Error: $*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      [[ $# -ge 2 ]] || fail "--manifest requires a value"
      MANIFEST="$2"
      shift 2
      ;;
    --out-dir)
      [[ $# -ge 2 ]] || fail "--out-dir requires a value"
      OUT_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

[[ -f "${MANIFEST}" ]] || fail "Manifest file not found: ${MANIFEST}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ARGS=(
  --manifest "${MANIFEST}"
  --out-dir "${OUT_DIR}"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
  ARGS+=(--dry-run)
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/lifecyclemodel_automated_builder.py" "${ARGS[@]}"
