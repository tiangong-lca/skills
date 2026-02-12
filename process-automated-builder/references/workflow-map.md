# Workflow Map (Standalone Skill Runtime)

## Goal
- Accept an agent-provided reference flow JSON.
- Execute end-to-end `process_from_flow` generation.
- Produce ILCD `process_datasets` and `source_datasets`.

## Input Contract (Agent -> Skill)
- Required for new runs:
  - `flow` JSON payload (ILCD flowDataSet wrapper) as file path, inline JSON, or stdin.
- Optional controls:
  - `operation`: `produce` or `treat`.
  - `run_id`, `stop_after`, publish flags, density conversion flags.

## Execution Layers
1. Wrapper layer
   - `scripts/run-process-automated-builder.sh`
   - Normalizes flow input (`--flow-file`, `--flow-json`, `--flow-stdin`).
   - Dispatches to `workflow` mode or `langgraph` mode.
2. Orchestration layer
   - `scripts/origin/process_from_flow_workflow.py`
   - Stages: references -> usability -> SI download -> MinerU -> usage tagging -> resume main pipeline.
3. Core graph layer
   - `scripts/origin/process_from_flow_langgraph.py`
   - Invokes `tiangong_lca_spec.process_from_flow.ProcessFromFlowService`.

## Output Contract (Skill -> Agent)
- Run root: `artifacts/process_from_flow/<run_id>/`
- Core outputs:
  - `exports/processes/*.json`
  - `exports/sources/*.json` (if generated)
  - `cache/process_from_flow_state.json`
- Diagnostics:
  - `cache/workflow_logs/*.log`
  - `cache/workflow_timing_report.json`
  - `cache/placeholder_report.json` (if generated)

## Control Flow
1. New run: pass flow input, generate run_id, execute full chain.
2. Debug run: use `--stop-after <stage>`, inspect state/logs, then resume.
3. Resume run: use `--mode langgraph --resume --run-id <id>`; flow path is read from cached state when omitted.
4. Publish run: use `--publish-only` (optionally `--publish-flows` and `--commit`).
