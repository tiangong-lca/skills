---
name: process-automated-builder
description: Execute and troubleshoot the `process_from_flow` automation chain with a CLI-first workflow. Use `tiangong process auto-build|resume-build|publish-build|batch-build` as the canonical path, and use legacy Python scripts only for transitional stages that are not yet migrated.
---

# Process Automated Builder

## Scope
- Build ILCD process and source datasets from one reference flow input.
- Run, resume, stop, inspect, and publish the workflow safely.
- Diagnose failures in references/SI processing, matching, unit alignment, placeholder resolution, and post-build reviews.

## Execution Baseline (CLI First)
1. Read `references/workflow-map.md` and `references/operations-playbook.md`.
2. Use `tiangong process auto-build` to create local run artifacts and a deterministic run directory.
3. Continue with `tiangong process resume-build`, `tiangong process publish-build`, or `tiangong process batch-build` as needed.
4. Use legacy Python entrypoints only for stages that have not yet been migrated into CLI modules.

## Execution Split
- CLI-owned local handoff stages (canonical):
  - `tiangong process auto-build`
  - `tiangong process resume-build`
  - `tiangong process publish-build`
  - `tiangong process batch-build`
- Legacy transitional stages (not canonical):
  - `scripts/run-process-automated-builder.sh`
  - `scripts/origin/process_from_flow_langgraph.py`
  - Legacy MCP/OpenAI/KB/TianGong unstructured integrations used by those scripts

## Parallel Execution Contract
- `Run-level parallel`: multiple flow inputs can run concurrently, but each run must use a distinct `run_id`.
- `In-run parallel`: only fan-out inside approved stage internals; stage barriers stay fixed.
- Barrier policy:
  - `01 -> 02 -> 03` strict serial.
  - `04` may fan-out over SI files.
  - `05 -> 06 -> 07` strict serial convergence.
- Single-writer rule:
  - Never let multiple agents write the same `artifacts/process_from_flow/<run_id>/cache/process_from_flow_state.json`.
  - Within one `run_id`, only one active writer process is allowed at a time.
  - Enforcement is code-level: writers acquire `process_from_flow_state.json.lock` before state writes.
- Flow-search parallelism:
  - `07_main_pipeline` now parallelizes only `flow_search` requests, then applies selector and state updates in original exchange order.
  - Tune with env `LCA_FLOW_SEARCH_MAX_PARALLEL` (bounded by profile concurrency).

## Canonical CLI Commands
```bash
tiangong process auto-build --help
tiangong process resume-build --help
tiangong process publish-build --help
tiangong process batch-build --help
```

## Legacy Transitional Commands
```bash
scripts/setup-process-automated-builder.sh
source .venv/bin/activate
export TIANGONG_LCA_REMOTE_TRANSPORT="streamable_http"
export TIANGONG_LCA_REMOTE_SERVICE_NAME="TianGong_LCA_Remote"
export TIANGONG_LCA_REMOTE_URL="https://lcamcp.tiangong.earth/mcp"
export TIANGONG_LCA_REMOTE_API_KEY="<your-api-key>"
export OPENAI_API_KEY="<your-openai-api-key>"
export OPENAI_MODEL="gpt-5"

scripts/run-process-automated-builder.sh --mode workflow --flow-file /abs/path/reference-flow.json -- --operation produce
scripts/run-process-automated-builder.sh --mode langgraph --flow-file /abs/path/reference-flow.json -- --stop-after matches --operation produce
scripts/run-process-automated-builder.sh --mode langgraph -- --resume --run-id <run_id>
scripts/run-process-automated-builder.sh --mode langgraph -- --publish-only --run-id <run_id> --commit
scripts/run-process-automated-builder.sh --mode langgraph -- flow-auto-build --run-id <run_id>
scripts/run-process-automated-builder.sh --mode langgraph -- process-update --run-id <run_id>
python3 scripts/origin/process_from_flow_langgraph.py workflow --flow /abs/path/reference-flow.json --operation produce
```

## Bundled Python Scripts
- Wrapper and setup: `scripts/run-process-automated-builder.sh`, `scripts/setup-process-automated-builder.sh`
- Main chain: `scripts/origin/process_from_flow_langgraph.py`
- Compatibility shim: `scripts/origin/process_from_flow_workflow.py` forwards to `process_from_flow_langgraph.py workflow`
- SI and references: `scripts/origin/process_from_flow_download_si.py`, `scripts/origin/mineru_for_process_si.py`, `scripts/origin/process_from_flow_reference_usability.py`, `scripts/origin/process_from_flow_reference_usage_tagging.py`
- Maintenance: `scripts/origin/process_from_flow_build_sources.py`, `scripts/origin/process_from_flow_placeholder_report.py`
- Shared helper copied for LangGraph CLI import path: `scripts/md/_workflow_common.py`

## Runtime Requirements
- Canonical path: use `tiangong` CLI env contract only (`TIANGONG_LCA_*`) as documented in `tiangong-lca-cli`.
- Legacy path: if you execute standalone Python scripts, they still require legacy runtime/env contracts:
  - Flow search MCP: `TIANGONG_LCA_REMOTE_*`
  - LLM: `OPENAI_*` / `LCA_OPENAI_*`
  - KB MCP: `TIANGONG_KB_REMOTE_*`
  - TianGong unstructured service: `TIANGONG_MINERU_WITH_IMAGE_*` (endpoint detail: `/mineru_with_images`)

## Fast Troubleshooting
- Missing `process_datasets` or `source_datasets`: verify `stop_after` did not stop before dataset stages.
- Too many placeholders: run through Step 6 (`resolve_placeholders`) and inspect `cache/placeholder_report.json`.
- Unit mismatch failures: inspect Step 4b `flow_search.unit_check`; density conversion only applies to product/waste mass<->volume mismatches.
- Slow runs: inspect `cache/workflow_timing_report.json`; Step 4 matching is usually the longest stage.
- OpenClaw handoff: inspect `cache/agent_handoff_summary.json` first instead of loading the full state/log set.

## Load References On Demand
- `references/process-from-flow-workflow.md`: complete migrated workflow spec (core flow, orchestration flow, state, outputs, publishing, stop rules).
- `references/workflow-map.md`: CLI-first execution map with legacy transitional boundary.
- `references/operations-playbook.md`: CLI-first runbook plus legacy fallback commands.
