# Operations Playbook (Standalone)

## 1) Bootstrap Runtime
```bash
process-automated-builder/scripts/setup-process-automated-builder.sh
source process-automated-builder/.venv/bin/activate
```

## 2) Run End-to-End (Recommended)
```bash
export TIANGONG_LCA_REMOTE_TRANSPORT="streamable_http"
export TIANGONG_LCA_REMOTE_SERVICE_NAME="TianGong_LCA_Remote"
export TIANGONG_LCA_REMOTE_URL="https://lcamcp.tiangong.earth/mcp"
export TIANGONG_LCA_REMOTE_API_KEY="<your-api-key>"
export OPENAI_API_KEY="<your-openai-api-key>"
export OPENAI_MODEL="gpt-5"

process-automated-builder/scripts/run-process-automated-builder.sh \
  --mode workflow \
  --flow-file /abs/path/to/reference-flow.json \
  -- --operation produce

# canonical direct entrypoint
python3 process-automated-builder/scripts/origin/process_from_flow_langgraph.py \
  workflow \
  --flow /abs/path/to/reference-flow.json \
  --operation produce
```

## 3) Run with Inline JSON (Agent-Friendly)
```bash
process-automated-builder/scripts/run-process-automated-builder.sh \
  --mode workflow \
  --flow-json '{"flowDataSet": {...}}' \
  -- --operation produce
```

## 4) Batch Parallel Run (Safe Wrapper)
```bash
process-automated-builder/scripts/run-process-automated-builder-parallel.sh \
  --flow-dir /abs/path/to/flow-dir \
  --out-dir /abs/path/to/batch-out \
  --workers 3 \
  --operation produce \
  --python-bin process-automated-builder/.venv/bin/python
```

Notes:
- Uses persistent `batch_state.json` and per-attempt logs in `batch_logs/`.
- Avoids prior xargs/env leakage that could write logs to `/<file>.log` and trigger permission-denied false failures.
- Supports auto-resume on interrupted sessions.

## 5) Stage Debugging
```bash
process-automated-builder/scripts/run-process-automated-builder.sh \
  --mode langgraph \
  --flow-file /abs/path/to/reference-flow.json \
  -- --stop-after matches --operation produce
```

## 6) Resume Existing Run
```bash
process-automated-builder/scripts/run-process-automated-builder.sh \
  --mode langgraph \
  -- --resume --run-id <run_id>
```

## 7) Publish Existing Run
```bash
process-automated-builder/scripts/run-process-automated-builder.sh \
  --mode langgraph \
  -- --publish-only --run-id <run_id> --commit
```

Optional debug switches during publish:
- `--skip-flow-auto-build`
- `--skip-process-update`

## 7b) Run flow-auto-build Only
```bash
process-automated-builder/scripts/run-process-automated-builder.sh \
  --mode langgraph \
  -- flow-auto-build --run-id <run_id>
```

## 7c) Run process-update Only
```bash
process-automated-builder/scripts/run-process-automated-builder.sh \
  --mode langgraph \
  -- process-update --run-id <run_id>
```

## 8) Background Persistent Run (systemd user service)
```bash
# install service + default env template
process-automated-builder/scripts/systemd/install-process-from-flow-batch-service.sh

# edit runtime settings
$EDITOR ~/.config/process-from-flow-batch/env

# start and enable auto-restart
systemctl --user daemon-reload
systemctl --user enable --now process-from-flow-batch.service

# monitor
systemctl --user status process-from-flow-batch.service
journalctl --user -u process-from-flow-batch.service -f
```

Notes:
- Service runs batch runner with `--watch` and keeps polling `FLOW_DIR` for newly added `*.json`.
- Service also loads `~/.openclaw/.env` by default for API/MCP credentials.
- Install script renders `~/.config/process-from-flow-batch/run-service.sh` with the current checkout path and detected Python executable; rerun it after moving the repo or changing interpreters.
- Service is configured with `Restart=always`; if runner is externally killed, it relaunches and continues from `STATE_PATH`.
- Default `STALL_TIMEOUT_SECONDS` in env example is set to `1800` to reduce false positives on long stage-7 runs.

## 8a) One-Command Submit to Daemon Queue
```bash
process-automated-builder/scripts/systemd/submit-process-from-flow.sh \
  --flow-file /abs/path/to/reference-flow.json

# or inline
process-automated-builder/scripts/systemd/submit-process-from-flow.sh \
  --flow-json '{"flowDataSet": {...}}'
```

Notes:
- The submit script writes a uniquely named JSON into `FLOW_DIR`.
- By default it also starts/enables `process-from-flow-batch.service`.

## Runtime Notes
- New runs require flow input; no default flow file is used.
- Resume mode can omit `--flow` and read it from cached state.
- `flow-auto-build` and `process-update` subcommands also do not require `--flow`.
- Flow-search MCP configuration is read from `TIANGONG_LCA_REMOTE_*` env vars.
- OpenAI configuration is read from `OPENAI_*` (or `LCA_OPENAI_*`) env vars.
- Literature MCP (`TianGong_KB_Remote`) can be configured by `TIANGONG_KB_REMOTE_*` env vars.
- MinerU OCR client can be configured by `TIANGONG_MINERU_WITH_IMAGE_*` env vars (`TIANGONG_MINERU_WITH_IMAGE_RETURN_TXT` defaults to `true`).
- `--publish` and `--commit` may invoke remote CRUD services; use dry-run first.
- `--publish` / `--publish-only` now execute one sequence: `flow-auto-build -> process-update -> flow publish -> process publish -> source publish`.
- Method-policy auto-repair is enabled by default in flow-auto-build/process-update/publish paths; see `cache/method_policy_autofix_report.json` for deterministic fixes, retry attempts, and any `manual_required` residue.
- LLM cost report is enabled by default in CLI runs; output path is `cache/llm_cost_report.json`.
- OpenClaw-friendly summary is written to `cache/agent_handoff_summary.json`; prefer that before loading full state or logs.
- Disable cost report with `--no-cost-report`; override prices with `--cost-input-price-per-1m` / `--cost-output-price-per-1m` or env `TIANGONG_PFF_COST_INPUT_PRICE_PER_1M` / `TIANGONG_PFF_COST_OUTPUT_PRICE_PER_1M`.

## Parallel Orchestration Rules
- Run-level parallel (recommended):
  - Start multiple runs in parallel with different `run_id`s (or let each run auto-generate one).
- In-run parallel (restricted):
  - Respect barrier order: `01 -> 02 -> 03` serial, `04` fan-out allowed, `05 -> 06 -> 07` serial.
  - Main pipeline only parallelizes `flow_search` requests; writeback remains ordered.
- Single-writer rule:
  - Never run multiple state-writing scripts concurrently for the same `run_id`.
  - This includes `process_from_flow_reference_usability.py`, `process_from_flow_download_si.py`, `process_from_flow_reference_usage_tagging.py`, and `process_from_flow_langgraph.py`.
  - State writes are guarded by `process_from_flow_state.json.lock`; lock timeout can be tuned by `TIANGONG_PFF_STATE_LOCK_TIMEOUT_SECONDS`.

## Flow Search Concurrency
- Tune with `LCA_FLOW_SEARCH_MAX_PARALLEL` (default `1`).
- Effective workers are capped by workflow profile concurrency (`LCA_MAX_CONCURRENCY` / profile).

## Failure Triage
- Missing deps/import errors:
  - Re-run setup script and ensure the venv is active.
- Missing flow errors:
  - Provide `--flow-file`, `--flow-json`, or `--flow-stdin` for new runs.
- Placeholder-heavy outputs:
  - Run through Step 6 and inspect `cache/placeholder_report.json`.
- Long runtime:
  - Check `cache/workflow_timing_report.json`; Step 4 matching is usually dominant.
