---
name: process-automated-builder
description: Execute and troubleshoot the end-to-end `process_from_flow` automation pipeline that derives ILCD `process_datasets` and `source_datasets` from a reference flow dataset, including literature retrieval, route/process splitting, exchange generation, flow matching, placeholder resolution, balance review, and publish/resume orchestration. Use when running or debugging `scripts/origin/process_from_flow_workflow.py` or `scripts/origin/process_from_flow_langgraph.py`.
---

# Process Automated Builder

## Scope
- Build ILCD process and source datasets from one reference flow input.
- Run, resume, stop, inspect, and publish the workflow safely.
- Diagnose failures in references/SI processing, matching, unit alignment, placeholder resolution, and post-build reviews.

## Execution Baseline
1. Read `references/workflow-map.md` and `references/operations-playbook.md`.
2. Bootstrap the standalone Python environment.
3. Run the wrapper with agent-provided flow input (`--flow-file`, `--flow-json`, or `--flow-stdin`).
4. Inspect run artifacts and continue with `--resume` or `--publish-only` when needed.

## Commands
```bash
scripts/setup-process-automated-builder.sh
source .venv/bin/activate

scripts/run-process-automated-builder.sh --mode workflow --flow-file /abs/path/reference-flow.json -- --operation produce
scripts/run-process-automated-builder.sh --mode langgraph --flow-file /abs/path/reference-flow.json -- --stop-after matches --operation produce
scripts/run-process-automated-builder.sh --mode langgraph -- --resume --run-id <run_id>
```

## Bundled Python Scripts
- Wrapper and setup: `scripts/run-process-automated-builder.sh`, `scripts/setup-process-automated-builder.sh`
- Main chain: `scripts/origin/process_from_flow_workflow.py`, `scripts/origin/process_from_flow_langgraph.py`
- SI and references: `scripts/origin/process_from_flow_download_si.py`, `scripts/origin/mineru_for_process_si.py`, `scripts/origin/process_from_flow_reference_usability.py`, `scripts/origin/process_from_flow_reference_usage_tagging.py`
- Maintenance: `scripts/origin/process_from_flow_build_sources.py`, `scripts/origin/process_from_flow_placeholder_report.py`
- Shared helper copied for LangGraph CLI import path: `scripts/md/_workflow_common.py`

## Runtime Requirements
- Use bundled runtime package `tiangong_lca_spec/` shipped with this skill.
- Install Python dependencies via `scripts/setup-process-automated-builder.sh`.
- Keep `.secrets/secrets.toml` available when running with LLM/MCP integrations.

## Fast Troubleshooting
- Missing `process_datasets` or `source_datasets`: verify `stop_after` did not stop before dataset stages.
- Too many placeholders: run through Step 6 (`resolve_placeholders`) and inspect `cache/placeholder_report.json`.
- Unit mismatch failures: inspect Step 4b `flow_search.unit_check`; density conversion only applies to product/waste mass<->volume mismatches.
- Slow runs: inspect `cache/workflow_timing_report.json`; Step 4 matching is usually the longest stage.

## Load References On Demand
- `references/process-from-flow-workflow.md`: complete migrated workflow spec (core flow, orchestration flow, state, outputs, publishing, stop rules).
- `references/workflow-map.md`: standalone skill execution map (input/output contracts and run control).
- `references/operations-playbook.md`: operational commands for setup, run, resume, and publish.
