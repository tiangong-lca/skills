#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SKILL_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

MODE="workflow"
FLOW_FILE=""
FLOW_JSON=""
FLOW_STDIN=0
PYTHON_BIN="${PAB_PYTHON_BIN:-python3}"
FORWARD_ARGS=()
TEMP_FLOW_FILE=""

usage() {
  cat <<'USAGE'
Usage: run-process-automated-builder.sh [wrapper-options] [-- python-args]

Wrapper options:
  --mode <workflow|langgraph>  Entry script (default: workflow)
  --flow-file <path>           Reference flow JSON file path
  --flow-json <json>           Inline reference flow JSON string
  --flow-stdin                 Read reference flow JSON from stdin
  --python-bin <path>          Python executable (default: python3 or $PAB_PYTHON_BIN)
  -h, --help                   Show this help message

Notes:
  - For new runs, flow input is required.
  - For langgraph resume/publish-only/cleanup-only modes, flow input can be omitted.
  - Any unrecognized arguments are forwarded to the selected Python entry script.
USAGE
}

fail() {
  echo "Error: $*" >&2
  exit 2
}

has_flag() {
  local flag="$1"
  shift || true
  local item
  for item in "$@"; do
    if [[ "${item}" == "${flag}" ]]; then
      return 0
    fi
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || fail "--mode requires a value"
      MODE="$2"
      shift 2
      ;;
    --flow-file)
      [[ $# -ge 2 ]] || fail "--flow-file requires a value"
      FLOW_FILE="$2"
      shift 2
      ;;
    --flow-json)
      [[ $# -ge 2 ]] || fail "--flow-json requires a value"
      FLOW_JSON="$2"
      shift 2
      ;;
    --flow-stdin)
      FLOW_STDIN=1
      shift
      ;;
    --python-bin)
      [[ $# -ge 2 ]] || fail "--python-bin requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      FORWARD_ARGS+=("$@")
      break
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

[[ "${MODE}" == "workflow" || "${MODE}" == "langgraph" ]] || fail "Invalid --mode: ${MODE}"

if [[ -n "${FLOW_FILE}" && -n "${FLOW_JSON}" ]]; then
  fail "--flow-file and --flow-json are mutually exclusive"
fi
if [[ -n "${FLOW_FILE}" && "${FLOW_STDIN}" -eq 1 ]]; then
  fail "--flow-file and --flow-stdin are mutually exclusive"
fi
if [[ -n "${FLOW_JSON}" && "${FLOW_STDIN}" -eq 1 ]]; then
  fail "--flow-json and --flow-stdin are mutually exclusive"
fi

cleanup() {
  if [[ -n "${TEMP_FLOW_FILE}" && -f "${TEMP_FLOW_FILE}" ]]; then
    rm -f "${TEMP_FLOW_FILE}"
  fi
}
trap cleanup EXIT

if [[ -n "${FLOW_JSON}" || "${FLOW_STDIN}" -eq 1 ]]; then
  TEMP_FLOW_FILE="$(mktemp "${TMPDIR:-/tmp}/pab-flow-XXXXXX.json")"
  if [[ -n "${FLOW_JSON}" ]]; then
    printf '%s' "${FLOW_JSON}" > "${TEMP_FLOW_FILE}"
  else
    cat > "${TEMP_FLOW_FILE}"
  fi
  FLOW_FILE="${TEMP_FLOW_FILE}"
fi

if [[ -n "${FLOW_FILE}" && ! -f "${FLOW_FILE}" ]]; then
  fail "Flow file not found: ${FLOW_FILE}"
fi

if has_flag "--flow" "${FORWARD_ARGS[@]}" && [[ -n "${FLOW_FILE}" ]]; then
  fail "Do not pass --flow in forwarded args when using wrapper flow input options"
fi

TARGET_SCRIPT=""
REQUIRE_FLOW=1
FORWARDED_HAS_FLOW=0
if has_flag "--flow" "${FORWARD_ARGS[@]}"; then
  FORWARDED_HAS_FLOW=1
fi

if [[ "${MODE}" == "workflow" ]]; then
  TARGET_SCRIPT="${SKILL_DIR}/scripts/origin/process_from_flow_workflow.py"
else
  TARGET_SCRIPT="${SKILL_DIR}/scripts/origin/process_from_flow_langgraph.py"
  if has_flag "--resume" "${FORWARD_ARGS[@]}" || has_flag "--publish-only" "${FORWARD_ARGS[@]}" || has_flag "--cleanup-only" "${FORWARD_ARGS[@]}"; then
    REQUIRE_FLOW=0
  fi
fi

if [[ "${REQUIRE_FLOW}" -eq 1 && -z "${FLOW_FILE}" && "${FORWARDED_HAS_FLOW}" -eq 0 ]]; then
  fail "Missing flow input. Use --flow-file/--flow-json/--flow-stdin (or forward --flow)."
fi

FLOW_ARG=()
if [[ -n "${FLOW_FILE}" ]]; then
  FLOW_ARG=(--flow "${FLOW_FILE}")
fi

export PYTHONPATH="${SKILL_DIR}:${PYTHONPATH:-}"

exec "${PYTHON_BIN}" "${TARGET_SCRIPT}" "${FLOW_ARG[@]}" "${FORWARD_ARGS[@]}"
