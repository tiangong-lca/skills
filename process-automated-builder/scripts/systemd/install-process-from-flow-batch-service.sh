#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SKILL_DIR="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"

SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
ENV_DIR="${HOME}/.config/process-from-flow-batch"
RUNNER_WRAPPER="${ENV_DIR}/run-service.sh"
DEFAULT_VENV_PYTHON="${SKILL_DIR}/.venv/bin/python"

mkdir -p "${SYSTEMD_USER_DIR}" "${ENV_DIR}"

cp "${SCRIPT_DIR}/process-from-flow-batch.service" "${SYSTEMD_USER_DIR}/process-from-flow-batch.service"

PYTHON_BIN="${PAB_PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${DEFAULT_VENV_PYTHON}" ]]; then
    PYTHON_BIN="${DEFAULT_VENV_PYTHON}"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable is not runnable: ${PYTHON_BIN}" >&2
  exit 1
fi

ESCAPED_SKILL_DIR="$(printf '%q' "${SKILL_DIR}")"
ESCAPED_PYTHON_BIN="$(printf '%q' "${PYTHON_BIN}")"
cat > "${RUNNER_WRAPPER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

SKILL_DIR=${ESCAPED_SKILL_DIR}
PYTHON_BIN=${ESCAPED_PYTHON_BIN}

cd "\${SKILL_DIR}"
exec "\${PYTHON_BIN}" "\${SKILL_DIR}/scripts/origin/process_from_flow_batch_runner.py" \\
  --flow-dir "\${FLOW_DIR}" \\
  --state "\${STATE_PATH}" \\
  --log-dir "\${LOG_DIR}" \\
  --workers "\${WORKERS}" \\
  --operation "\${OPERATION}" \\
  --max-attempts "\${MAX_ATTEMPTS}" \\
  --poll-seconds "\${POLL_SECONDS}" \\
  --heartbeat-seconds "\${HEARTBEAT_SECONDS}" \\
  --stall-timeout-seconds "\${STALL_TIMEOUT_SECONDS}" \\
  --watch \\
  --python-bin "\${PYTHON_BIN}"
EOF
chmod +x "${RUNNER_WRAPPER}"

if [[ ! -f "${ENV_DIR}/env" ]]; then
  cp "${SCRIPT_DIR}/process-from-flow-batch.env.example" "${ENV_DIR}/env"
  echo "Created ${ENV_DIR}/env from example. Please edit it before starting service."
else
  echo "Env file already exists: ${ENV_DIR}/env"
fi

echo "Installed service file to ${SYSTEMD_USER_DIR}/process-from-flow-batch.service"
echo "Rendered daemon wrapper to ${RUNNER_WRAPPER}"
echo "Resolved skill dir: ${SKILL_DIR}"
echo "Resolved python bin: ${PYTHON_BIN}"
echo
echo "Next steps:"
echo "  1) Edit: ${ENV_DIR}/env"
echo "  2) systemctl --user daemon-reload"
echo "  3) systemctl --user enable --now process-from-flow-batch.service"
echo "  4) journalctl --user -u process-from-flow-batch.service -f"
