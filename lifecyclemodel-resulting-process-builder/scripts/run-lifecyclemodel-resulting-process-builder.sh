#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SKILL_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
WORKSPACE_ROOT="$(cd -- "${SKILL_DIR}/../.." >/dev/null 2>&1 && pwd)"

DEFAULT_CLI_DIR="${WORKSPACE_ROOT}/tiangong-lca-cli"
DEFAULT_INPUT_FILE="${SKILL_DIR}/assets/example-request.json"

CLI_DIR="${TIANGONG_LCA_CLI_DIR:-${DEFAULT_CLI_DIR}}"
TEMP_FILES=()

cleanup() {
  local file
  for file in "${TEMP_FILES[@]:-}"; do
    if [[ -n "${file}" && -e "${file}" ]]; then
      rm -f "${file}"
    fi
  done
}

fail() {
  echo "Error: $*" >&2
  exit 2
}

print_help() {
  cat <<'EOF'
Usage:
  scripts/run-lifecyclemodel-resulting-process-builder.sh build [options]
  scripts/run-lifecyclemodel-resulting-process-builder.sh publish [options]

Build aliases:
  prepare
  project

Wrapper compatibility options for build:
  --request <file>          Alias for the CLI's --input <file>
  --model-file <file>       Synthesize a temporary CLI request from a lifecycle model file
  --projection-role <mode>  primary | all (maps to projection.mode)

Wrapper options:
  --cli-dir <dir>           Override the tiangong-lca-cli repository path

Canonical CLI commands:
  tiangong lifecyclemodel build-resulting-process --input <file>
  tiangong lifecyclemodel publish-resulting-process --run-dir <dir>
EOF
}

trap cleanup EXIT

PARSED_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cli-dir)
      [[ $# -ge 2 ]] || fail "--cli-dir requires a value"
      CLI_DIR="$2"
      shift 2
      ;;
    --cli-dir=*)
      CLI_DIR="${1#--cli-dir=}"
      shift
      ;;
    *)
      PARSED_ARGS+=("$1")
      shift
      ;;
  esac
done

set -- "${PARSED_ARGS[@]}"

CLI_BIN="${CLI_DIR}/bin/tiangong.js"
[[ -f "${CLI_BIN}" ]] || fail "Cannot find TianGong CLI at ${CLI_BIN}. Set TIANGONG_LCA_CLI_DIR or pass --cli-dir."

write_model_request() {
  local model_file="$1"
  local projection_role="$2"
  local request_file
  request_file="$(mktemp "${TMPDIR:-/tmp}/tg-lifecyclemodel-request.XXXXXX")"
  TEMP_FILES+=("${request_file}")

  node - "${request_file}" "${model_file}" "${projection_role}" <<'EOF'
const fs = require('node:fs');
const path = require('node:path');

const [requestFile, modelFileRaw, projectionRole] = process.argv.slice(2);
const mode = projectionRole === 'all' ? 'all-subproducts' : 'primary-only';
const payload = {
  source_model: {
    json_ordered_path: path.resolve(modelFileRaw),
  },
  projection: {
    mode,
  },
  process_sources: {
    allow_remote_lookup: false,
  },
  publish: {
    intent: 'prepare_only',
    prepare_process_payloads: true,
    prepare_relation_payloads: true,
  },
};

fs.writeFileSync(requestFile, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
EOF

  printf '%s\n' "${request_file}"
}

run_build() {
  local projection_role="primary"
  local input_path=""
  local model_file=""
  local show_help=0
  local -a forward_args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --request)
        [[ $# -ge 2 ]] || fail "--request requires a value"
        input_path="$2"
        shift 2
        ;;
      --request=*)
        input_path="${1#--request=}"
        shift
        ;;
      --input)
        [[ $# -ge 2 ]] || fail "--input requires a value"
        input_path="$2"
        shift 2
        ;;
      --input=*)
        input_path="${1#--input=}"
        shift
        ;;
      --model-file)
        [[ $# -ge 2 ]] || fail "--model-file requires a value"
        model_file="$2"
        shift 2
        ;;
      --model-file=*)
        model_file="${1#--model-file=}"
        shift
        ;;
      --projection-role)
        [[ $# -ge 2 ]] || fail "--projection-role requires a value"
        projection_role="$2"
        shift 2
        ;;
      --projection-role=*)
        projection_role="${1#--projection-role=}"
        shift
        ;;
      -h|--help)
        show_help=1
        shift
        ;;
      *)
        forward_args+=("$1")
        shift
        ;;
    esac
  done

  case "${projection_role}" in
    primary|all) ;;
    *)
      fail "--projection-role must be one of: primary, all"
      ;;
  esac

  if [[ "${show_help}" -eq 1 ]]; then
    node "${CLI_BIN}" lifecyclemodel build-resulting-process --help
    return
  fi

  if [[ -n "${input_path}" && -n "${model_file}" ]]; then
    fail "Use either --request/--input or --model-file, not both."
  fi

  if [[ -n "${model_file}" ]]; then
    input_path="$(write_model_request "${model_file}" "${projection_role}")"
  elif [[ -z "${input_path}" ]]; then
    input_path="${DEFAULT_INPUT_FILE}"
  fi

  local -a command=(
    node
    "${CLI_BIN}"
    lifecyclemodel
    build-resulting-process
    --input
    "${input_path}"
  )
  command+=("${forward_args[@]}")
  "${command[@]}"
}

run_publish() {
  local show_help=0
  local -a forward_args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        show_help=1
        shift
        ;;
      *)
        forward_args+=("$1")
        shift
        ;;
    esac
  done

  if [[ "${show_help}" -eq 1 ]]; then
    node "${CLI_BIN}" lifecyclemodel publish-resulting-process --help
    return
  fi

  local -a command=(
    node
    "${CLI_BIN}"
    lifecyclemodel
    publish-resulting-process
  )
  command+=("${forward_args[@]}")
  "${command[@]}"
}

subcommand="${1:-}"
if [[ -z "${subcommand}" ]]; then
  print_help
  exit 0
fi

case "${subcommand}" in
  -h|--help|help)
    print_help
    ;;
  build|prepare|project)
    shift
    run_build "$@"
    ;;
  publish)
    shift
    run_publish "$@"
    ;;
  *)
    fail "Unknown subcommand: ${subcommand}"
    ;;
esac
