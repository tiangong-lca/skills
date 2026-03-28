#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SKILL_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
WORKSPACE_ROOT="$(cd -- "${SKILL_DIR}/../.." >/dev/null 2>&1 && pwd)"

DEFAULT_CLI_DIR="${WORKSPACE_ROOT}/tiangong-lca-cli"
DEFAULT_BASE_URL="https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1"
DEFAULT_REGION="us-east-1"
DEFAULT_DATA_FILE="${SKILL_DIR}/assets/example-request.json"

CLI_DIR="${TIANGONG_CLI_DIR:-${DEFAULT_CLI_DIR}}"
API_KEY="${TIANGONG_API_KEY:-${TIANGONG_LCA_APIKEY:-}}"
BASE_URL="${TIANGONG_API_BASE_URL:-${SUPABASE_FUNCTIONS_URL:-${DEFAULT_BASE_URL}}}"
REGION="${TIANGONG_REGION:-${SUPABASE_FUNCTION_REGION:-${DEFAULT_REGION}}}"
DATA_FILE="${DEFAULT_DATA_FILE}"
TIMEOUT_SEC=60
COMPACT_JSON=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: run-process-hybrid-search.sh [options]

Options:
  --cli-dir <dir>      Override TIANGONG_CLI_DIR
  --token <key>        Override API key
  --data <file>        JSON body file path (default: assets/example-request.json)
  --base-url <url>     API base URL
  --region <region>    Region header value (default: us-east-1)
  --max-time <sec>     Request timeout in seconds (default: 60)
  --json               Print compact JSON
  --dry-run            Print request details without sending
  -h, --help           Show this help message
USAGE
}

fail() {
  echo "Error: $*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cli-dir)
      [[ $# -ge 2 ]] || fail "--cli-dir requires a value"
      CLI_DIR="$2"
      shift 2
      ;;
    --token|--api-key)
      [[ $# -ge 2 ]] || fail "$1 requires a value"
      API_KEY="$2"
      shift 2
      ;;
    --data|--input)
      [[ $# -ge 2 ]] || fail "$1 requires a value"
      DATA_FILE="$2"
      shift 2
      ;;
    --base-url)
      [[ $# -ge 2 ]] || fail "--base-url requires a value"
      BASE_URL="$2"
      shift 2
      ;;
    --region)
      [[ $# -ge 2 ]] || fail "--region requires a value"
      REGION="$2"
      shift 2
      ;;
    --max-time)
      [[ $# -ge 2 ]] || fail "--max-time requires a value"
      TIMEOUT_SEC="$2"
      shift 2
      ;;
    --json)
      COMPACT_JSON=1
      shift
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

[[ "${TIMEOUT_SEC}" =~ ^[0-9]+$ ]] || fail "--max-time must be a positive integer"
[[ "${TIMEOUT_SEC}" -gt 0 ]] || fail "--max-time must be greater than zero"

CLI_BIN="${CLI_DIR}/bin/tiangong.js"
[[ -f "${CLI_BIN}" ]] || fail "Cannot find TianGong CLI at ${CLI_BIN}. Set TIANGONG_CLI_DIR."
[[ -e "${CLI_DIR}/node_modules/tsx" ]] || fail "TianGong CLI dependencies are missing. Run 'npm install' in ${CLI_DIR}."
[[ -n "${API_KEY}" ]] || fail "Missing API key. Set TIANGONG_API_KEY / TIANGONG_LCA_APIKEY or pass --token."
[[ -f "${DATA_FILE}" ]] || fail "Data file not found: ${DATA_FILE}"

TIMEOUT_MS="$((TIMEOUT_SEC * 1000))"

command=(
  node
  "${CLI_BIN}"
  search
  process
  --input "${DATA_FILE}"
  --api-key "${API_KEY}"
  --base-url "${BASE_URL}"
  --region "${REGION}"
  --timeout-ms "${TIMEOUT_MS}"
)

if [[ "${COMPACT_JSON}" -eq 1 ]]; then
  command+=(--json)
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  command+=(--dry-run)
fi

"${command[@]}"
