#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
DEFAULT_PYTHON_BIN="${REPO_ROOT}/process-automated-builder/.venv/bin/python"
if [[ -n "${FLOW_GOVERNANCE_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${FLOW_GOVERNANCE_PYTHON_BIN}"
elif [[ -n "${PAB_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PAB_PYTHON_BIN}"
elif [[ -x "${DEFAULT_PYTHON_BIN}" ]]; then
  PYTHON_BIN="${DEFAULT_PYTHON_BIN}"
else
  PYTHON_BIN="python3"
fi

usage() {
  cat <<'USAGE'
Usage: run-flow-governance-review.sh <command> [args...]

Commands:
  openclaw-entry
  openclaw-full-run
  run-governance
  review-flows
  flow-dedup-candidates
  build-flow-alias-map
  scan-process-flow-refs
  plan-process-flow-repairs
  apply-process-flow-repairs
  validate-processes
  export-openclaw-dedup-review-pack
  export-openclaw-ref-review-pack
  apply-openclaw-ref-decisions
  export-openclaw-text-review-pack
  export-openclaw-classification-review-pack
  apply-openclaw-text-decisions
  apply-openclaw-classification-decisions
  validate-openclaw-text-decisions
  validate-openclaw-classification-decisions
  publish-reviewed-data
USAGE
}

command="${1:-}"
if [[ -z "${command}" ]]; then
  usage
  exit 2
fi
shift || true

case "${command}" in
  openclaw-entry)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/openclaw_entry.py" "$@"
    ;;
  openclaw-full-run)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/openclaw_full_run.py" "$@"
    ;;
  run-governance)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/flow_governance_orchestrator.py" "$@"
    ;;
  review-flows)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/flow_review_engine.py" "$@"
    ;;
  flow-dedup-candidates)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/flow_dedup_candidates.py" "$@"
    ;;
  build-flow-alias-map)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/flow_alias_map.py" "$@"
    ;;
  scan-process-flow-refs)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/process_flow_ref_scan.py" "$@"
    ;;
  plan-process-flow-repairs)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/process_flow_repair.py" "$@"
    ;;
  apply-process-flow-repairs)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/process_flow_repair.py" --apply "$@"
    ;;
  validate-processes)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/process_patch_validate.py" "$@"
    ;;
  export-openclaw-dedup-review-pack)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/openclaw_review_handoff.py" export-dedup-review-pack "$@"
    ;;
  export-openclaw-ref-review-pack)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/openclaw_review_handoff.py" export-ref-review-pack "$@"
    ;;
  apply-openclaw-ref-decisions)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/openclaw_review_handoff.py" apply-ref-decisions "$@"
    ;;
  export-openclaw-text-review-pack)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/openclaw_review_handoff.py" export-text-review-pack "$@"
    ;;
  export-openclaw-classification-review-pack)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/openclaw_review_handoff.py" export-classification-review-pack "$@"
    ;;
  apply-openclaw-text-decisions)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/openclaw_review_handoff.py" apply-text-decisions "$@"
    ;;
  apply-openclaw-classification-decisions)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/openclaw_review_handoff.py" apply-classification-decisions "$@"
    ;;
  validate-openclaw-text-decisions)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/openclaw_review_handoff.py" validate-text-decisions "$@"
    ;;
  validate-openclaw-classification-decisions)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/openclaw_review_handoff.py" validate-classification-decisions "$@"
    ;;
  publish-reviewed-data)
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/governance_publish.py" "$@"
    ;;
  *)
    usage
    exit 2
    ;;
esac
