---
name: lifecyclemodel-automated-builder
description: Build and validate native TianGong TIDAS lifecycle model `json_ordered` datasets from existing process records, using AI to choose candidate processes from account-owned processes and public `state_code=100` data. Use when you need read-only discovery, local model assembly, local validation, or a later MCP insert that only sends `jsonOrdered`.
---

# Lifecycle Model Automated Builder

## Scope
- Discover candidate processes from the current account and public `state_code=100` records.
- Read a small number of existing lifecycle models via MCP as structural reference only.
- Use AI to decide which processes are worth combining into lifecycle models.
- Assemble native `json_ordered` lifecycle model datasets locally.
- Validate locally before any remote mutation.
- Default to read-only remote access.

## Read First
1. `references/workflow.md`
2. `references/source-analysis.md`
3. `references/model-contract.md`

## Guardrails
- Treat `TIANGONG_LCA_REMOTE_API_KEY` as a runtime secret. Read it from env only.
- The skill produces `json_ordered` only. It does not emit `json_tg`, `rule_verification`, or generated resulting-process artifacts.
- Remote operations allowed by default:
  - MCP search tools
  - MCP CRUD `select`
  - local schema validation
- Remote mutation is optional and gated:
  - only `Database_CRUD_Tool insert lifecyclemodels` with `jsonOrdered`
  - never `delete`

## Workflow
1. Run `scripts/run-lifecyclemodel-automated-builder.sh --dry-run` with a manifest.
2. Build a local execution plan that covers:
   - account process discovery
   - public `state_code=100` discovery
   - optional reference lifecyclemodel discovery
   - AI selection
   - native lifecycle model assembly
   - strict validation
3. During assembly, preserve TianGong native model conventions from `tiangong-lca-next`:
   - `lifeCycleModelInformation.quantitativeReference.referenceToReferenceProcess`
   - `technology.processes.processInstance[*].referenceToProcess`
   - `technology.processes.processInstance[*].connections.outputExchange`
   - computed `@multiplicationFactor`
   - a valid `referenceToResultingProcess` reference inside `json_ordered`
4. Validate with strict `tidas-sdk` plus `tidas-tools` classification checks.
5. If remote insert is later approved, send only `jsonOrdered`; downstream MCP logic owns any platform-specific derivation.

## Commands
```bash
export TIANGONG_LCA_REMOTE_TRANSPORT="streamable_http"
export TIANGONG_LCA_REMOTE_SERVICE_NAME="TianGong_LCA_Remote"
export TIANGONG_LCA_REMOTE_URL="https://lcamcp.tiangong.earth/mcp"
export TIANGONG_LCA_REMOTE_API_KEY="<runtime-only-secret>"
export OPENAI_API_KEY="<your-openai-api-key>"
export OPENAI_MODEL="gpt-5"

lifecyclemodel-automated-builder/scripts/run-lifecyclemodel-automated-builder.sh \
  --manifest lifecyclemodel-automated-builder/assets/example-request.json \
  --dry-run

lifecyclemodel-automated-builder/scripts/run-lifecyclemodel-automated-builder.sh \
  --manifest /abs/path/request.json \
  --out-dir /abs/path/out

lifecyclemodel-automated-builder/scripts/run-lifecyclemodel-automated-builder.sh \
  --manifest lifecyclemodel-automated-builder/assets/example-local-runs.json \
  --out-dir /abs/path/local-run-test

lifecyclemodel-automated-builder/scripts/run-lifecyclemodel-automated-builder.sh \
  --manifest lifecyclemodel-automated-builder/assets/example-reference-and-reuse.json \
  --out-dir /abs/path/reference-reuse-test

TIANGONG_LCA_REMOTE_API_KEY="<runtime-only-secret>" \
lifecyclemodel-automated-builder/scripts/run-lifecyclemodel-automated-builder.sh \
  --manifest lifecyclemodel-automated-builder/assets/example-local-publish.json \
  --out-dir /abs/path/local-publish-test
```

## Fast Triage
- Missing API key: export runtime env; never write credentials into files.
- Validation failures on required model fields: inspect `references/model-contract.md`.
- Topology disagreements: inspect `references/source-analysis.md` for native lifecycle model conventions.
- If the user asks to upload, keep the mutation path limited to `jsonOrdered` insert unless explicitly expanded later.

## Bundled Resources
- `scripts/run-lifecyclemodel-automated-builder.sh`: thin CLI wrapper.
- `scripts/lifecyclemodel_automated_builder.py`: read-only planner, reference-model reader, local native model builder, validator, and optional MCP insert gate.
- `assets/example-request.json`: example batch manifest.
- `assets/example-local-runs.json`: local `process-automated-builder` run test manifest.
- `assets/example-reference-and-reuse.json`: local build plus MCP reference-model discovery manifest.
- `assets/example-local-publish.json`: single-run MCP insert test manifest.
- `references/workflow.md`: workflow and AI selection policy.
- `references/source-analysis.md`: extracted conventions from `tiangong-lca-next`, `tidas-sdk`, `tidas-tools`, and current MCP.
- `references/model-contract.md`: native `json_ordered` fields required before validation or publish.
