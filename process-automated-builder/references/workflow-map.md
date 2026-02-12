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
- Runtime env for flow-search MCP:
  - `TIANGONG_LCA_REMOTE_TRANSPORT`
  - `TIANGONG_LCA_REMOTE_SERVICE_NAME`
  - `TIANGONG_LCA_REMOTE_URL`
  - `TIANGONG_LCA_REMOTE_API_KEY`
- Runtime env for LLM:
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL` (optional)
  - `OPENAI_BASE_URL` (optional)
- Runtime env for KB MCP (literature retrieval path):
  - `TIANGONG_KB_REMOTE_TRANSPORT`
  - `TIANGONG_KB_REMOTE_SERVICE_NAME`
  - `TIANGONG_KB_REMOTE_URL`
  - `TIANGONG_KB_REMOTE_API_KEY`
- Runtime env for MinerU OCR (SI parsing path):
  - `TIANGONG_MINERU_WITH_IMAGE_URL`
  - `TIANGONG_MINERU_WITH_IMAGE_API_KEY` (optional)
  - `TIANGONG_MINERU_WITH_IMAGE_TIMEOUT` (optional)

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

## Parallel Modes
- `Run-level parallel`:
  - Run multiple flows concurrently with different `run_id`s.
  - Safe default for agent orchestration throughput.
- `In-run parallel`:
  - Allow parallel fan-out only inside approved stage internals.
  - Current implementation parallelizes `flow_search` RPC calls in `match_flows`, but preserves ordered result fill.

## Barriers and Single-Writer
- Stage barriers:
  - `01 -> 02 -> 03` serial.
  - `04` can fan-out by SI file.
  - `05 -> 06 -> 07` serial convergence.
- Single-writer rule:
  - For one `run_id`, exactly one writer may update `cache/process_from_flow_state.json`.
  - Do not launch usability / SI download / usage tagging / main pipeline as concurrent writers on the same run.
  - This is now enforced with a file lock at `cache/process_from_flow_state.json.lock`.

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
