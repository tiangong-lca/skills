#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

usage() {
  cat <<'USAGE'
Usage: run-flow-governance-review.sh <command> [args...]

Commands:
  openclaw-full-run
  run-governance
  review-flows
  flow-dedup-candidates
  build-flow-alias-map
  scan-process-flow-refs
  plan-process-flow-repairs
  apply-process-flow-repairs
  validate-processes
  export-openclaw-ref-review-pack
  apply-openclaw-ref-decisions
  export-openclaw-text-review-pack
  apply-openclaw-text-decisions
  validate-openclaw-text-decisions
USAGE
}

command="${1:-}"
if [[ -z "${command}" ]]; then
  usage
  exit 2
fi
shift || true

case "${command}" in
  openclaw-full-run)
    exec python3 "${SCRIPT_DIR}/openclaw_full_run.py" "$@"
    ;;
  run-governance)
    exec python3 "${SCRIPT_DIR}/flow_governance_orchestrator.py" "$@"
    ;;
  review-flows)
    exec python3 "${SCRIPT_DIR}/flow_review_engine.py" "$@"
    ;;
  flow-dedup-candidates)
    exec python3 "${SCRIPT_DIR}/flow_dedup_candidates.py" "$@"
    ;;
  build-flow-alias-map)
    exec python3 "${SCRIPT_DIR}/flow_alias_map.py" "$@"
    ;;
  scan-process-flow-refs)
    exec python3 "${SCRIPT_DIR}/process_flow_ref_scan.py" "$@"
    ;;
  plan-process-flow-repairs)
    exec python3 "${SCRIPT_DIR}/process_flow_repair.py" "$@"
    ;;
  apply-process-flow-repairs)
    exec python3 "${SCRIPT_DIR}/process_flow_repair.py" --apply "$@"
    ;;
  validate-processes)
    exec python3 "${SCRIPT_DIR}/process_patch_validate.py" "$@"
    ;;
  export-openclaw-ref-review-pack)
    exec python3 "${SCRIPT_DIR}/openclaw_review_handoff.py" export-ref-review-pack "$@"
    ;;
  apply-openclaw-ref-decisions)
    exec python3 "${SCRIPT_DIR}/openclaw_review_handoff.py" apply-ref-decisions "$@"
    ;;
  export-openclaw-text-review-pack)
    exec python3 "${SCRIPT_DIR}/openclaw_review_handoff.py" export-text-review-pack "$@"
    ;;
  apply-openclaw-text-decisions)
    exec python3 "${SCRIPT_DIR}/openclaw_review_handoff.py" apply-text-decisions "$@"
    ;;
  validate-openclaw-text-decisions)
    exec python3 "${SCRIPT_DIR}/openclaw_review_handoff.py" validate-text-decisions "$@"
    ;;
  *)
    usage
    exit 2
    ;;
esac
