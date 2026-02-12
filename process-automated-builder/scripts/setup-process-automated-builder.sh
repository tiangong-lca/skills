#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SKILL_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
VENV_DIR="${SKILL_DIR}/.venv"
PYTHON_BIN="${PAB_BOOTSTRAP_PYTHON:-python3}"
DRY_RUN=0

PACKAGES=(
  "anyio>=4.11.0"
  "beautifulsoup4>=4.14.2"
  "httpx>=0.28.1"
  "jsonschema>=4.25.1"
  "langgraph>=1.0.5"
  "mcp>=1.18.0"
  "minio>=7.2.19"
  "openai>=2.6.0"
  "pydantic-settings>=2.11.0"
  "pydantic>=2.12.3"
  "python-dotenv>=1.1.1"
  "rispy>=0.10.0"
  "structlog>=25.4.0"
  "tenacity>=9.1.2"
  "tidas-sdk>=0.1.5"
  "tidas-tools>=0.0.21"
)

usage() {
  cat <<'USAGE'
Usage: setup-process-automated-builder.sh [options]

Options:
  --python <path>   Python executable for venv creation (default: python3 or $PAB_BOOTSTRAP_PYTHON)
  --venv <path>     Virtual environment directory (default: <skill>/.venv)
  --dry-run         Print planned commands without execution
  -h, --help        Show this help message
USAGE
}

fail() {
  echo "Error: $*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      [[ $# -ge 2 ]] || fail "--python requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv)
      [[ $# -ge 2 ]] || fail "--venv requires a value"
      VENV_DIR="$2"
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

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "python: ${PYTHON_BIN}"
  echo "venv: ${VENV_DIR}"
  echo "packages:"
  printf '  - %s\n' "${PACKAGES[@]}"
  exit 0
fi

if command -v uv >/dev/null 2>&1; then
  uv venv "${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  uv pip install "${PACKAGES[@]}"
else
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip
  pip install "${PACKAGES[@]}"
fi

echo "Environment ready: ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/bin/activate"
