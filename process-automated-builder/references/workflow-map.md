# Workflow Map (CLI First, Transitional Legacy Runtime)

## Goal
- Accept an agent-provided reference flow JSON.
- Execute the Node wrapper -> CLI-owned local `process_from_flow` handoff stages and keep legacy orchestration available only as a transition path.
- Produce ILCD `process_datasets` and `source_datasets`.

## Input Contract (Agent -> Skill)
- Required for canonical new runs:
  - `node scripts/run-process-automated-builder.mjs auto-build --input <request.json>`
  - or compatibility flow input flags (`--flow-file`, `--flow-json`, `--flow-stdin`) that synthesize a temporary CLI request
- Required for legacy new runs:
  - `flow` JSON payload (ILCD flowDataSet wrapper) as file path, inline JSON, or stdin.
- Optional controls:
  - `operation`: `produce` or `treat`.
  - `run_id`, `stop_after`, publish flags, density conversion flags.
- Legacy runtime env for flow-search MCP:
  - `TIANGONG_LCA_REMOTE_TRANSPORT`
  - `TIANGONG_LCA_REMOTE_SERVICE_NAME`
  - `TIANGONG_LCA_REMOTE_URL`
  - `TIANGONG_LCA_REMOTE_API_KEY`
- Legacy runtime env for LLM:
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL` (optional)
  - `OPENAI_BASE_URL` (optional)
- Legacy runtime env for KB MCP (literature retrieval path):
  - `TIANGONG_KB_REMOTE_TRANSPORT`
  - `TIANGONG_KB_REMOTE_SERVICE_NAME`
  - `TIANGONG_KB_REMOTE_URL`
  - `TIANGONG_KB_REMOTE_API_KEY`
- Legacy runtime env for TianGong unstructured service (SI parsing path):
  - `TIANGONG_MINERU_WITH_IMAGE_URL`
  - `TIANGONG_MINERU_WITH_IMAGE_API_KEY` (optional)
  - `TIANGONG_MINERU_WITH_IMAGE_TIMEOUT` (optional)
  - `TIANGONG_MINERU_WITH_IMAGE_RETURN_TXT` (optional, default: `true`, endpoint detail: `/mineru_with_images`)

## Execution Layers
1. Node wrapper -> CLI handoff layer (canonical)
   - `node scripts/run-process-automated-builder.mjs auto-build`
   - `node scripts/run-process-automated-builder.mjs resume-build`
   - `node scripts/run-process-automated-builder.mjs publish-build`
   - `node scripts/run-process-automated-builder.mjs batch-build`
2. Direct CLI layer
   - `tiangong process auto-build`
   - `tiangong process resume-build`
   - `tiangong process publish-build`
   - `tiangong process batch-build`
3. Legacy wrapper/orchestration layer (transitional only)
   - `node scripts/run-process-automated-builder.mjs legacy ...`
   - `scripts/run-process-automated-builder.sh` compatibility shim
4. Legacy orchestration layer (transitional only)
   - `scripts/origin/process_from_flow_langgraph.py workflow`
   - Stages: references -> usability -> SI download -> TianGong unstructured service parsing -> usage tagging -> resume main pipeline.
5. Core graph layer
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
  - `cache/agent_handoff_summary.json`
  - `cache/placeholder_report.json` (if generated)
  - `cache/flow_auto_build_manifest.jsonl`
  - `cache/process_update_report.json`
  - `cache/flow_publish_results.jsonl`
  - `cache/flow_publish_failures.jsonl`
  - `cache/publish_summary.json`

## Method Policy Reference
- Guardrail file: `references/ilcd_method_guardrails.md`
- Purpose: enforce FU/reference-flow basis consistency and comparability assumptions for dataset/database building.
- Integration: prompt-time policy grounding in Step 2 process split and Step 3 exchanges generation.
- Publish-time behavior (default): auto-fix deterministic violations first (FU/reference alignment, quantitative-reference amount sync, missing `@version` backfill), then continue publish.
- Rebuild behavior (default): if unresolved semantic conflicts remain (e.g., held flow-property decisions or remaining placeholder refs), auto-run one extra `flow-auto-build -> process-update` repair pass before publish.
- Manual fallback: only unresolved items after automatic repair are marked in `cache/method_policy_autofix_report.json` under `manual_required`.

## Control Flow
1. Canonical new run: use `node scripts/run-process-automated-builder.mjs auto-build ...`.
2. Canonical resume/publish: use `node scripts/run-process-automated-builder.mjs resume-build ...` or `publish-build ...`.
3. Canonical batch: use `node scripts/run-process-automated-builder.mjs batch-build ...`.
4. Legacy debug/resume/publish: use `node scripts/run-process-automated-builder.mjs legacy ...` only when a stage is not yet migrated.

## Preflight Chain Continuity Gate (P0)
- Added between `enrich_exchange_amounts` and `match_flows`.
- Builds a `chain_contract` from normalized processes (`from_pid`, `to_pid`, `reference_flow_name`).
- Validates that each upstream `reference_flow_name` appears in downstream main inputs (input exchanges), using label-insensitive and case/whitespace-normalized matching.
- On failure, writes structured errors in `chain_preflight.errors` (e.g., `code=missing_main_input_link`) and ends the graph early, blocking downstream matching/publish path.
