---
name: lifecyclemodel-automated-builder
description: Assemble native TianGong TIDAS lifecyclemodel `json_ordered` artifacts from existing local process-build runs through the unified TianGong CLI. Use when you already have `process-automated-builder` outputs and need the local lifecyclemodel auto-build slice without Python, MCP, or remote writes.
---

# Lifecycle Model Automated Builder

Use this skill when the source of truth is a set of existing local `process-automated-builder` run directories and the next step is to assemble a native lifecyclemodel artifact locally.

## Read First
1. `references/workflow.md`
2. `references/model-contract.md`
3. `references/source-analysis.md`

## Guardrails
- The canonical runtime path is `skill -> Node wrapper -> tiangong CLI`.
- The current canonical slices are:
  - `tiangong lifecyclemodel auto-build`
  - `tiangong lifecyclemodel validate-build`
  - `tiangong lifecyclemodel publish-build`
- The canonical lifecyclemodel builder path remains local-first:
  - no Python workflow
  - no MCP transport
  - no remote lifecyclemodel CRUD
  - no reference-model discovery against KB / LLM services
- The skill produces native `json_ordered` only. It does not emit `json_tg`, `rule_verification`, or resulting-process artifacts.
- `validate-build` and `publish-build` now exist as dedicated CLI follow-up commands; do not reintroduce those stages inside the skill.
- Only `local_runs[]` is executable today. Discovery hints may be recorded as deferred notes, but they are not executed inside this skill.

## Workflow
1. Prepare a manifest whose core input is `local_runs[]`.
2. Run `node scripts/run-lifecyclemodel-automated-builder.mjs build --input <manifest> --out-dir <dir>`.
3. During assembly, preserve TianGong native model conventions from `tiangong-lca-next`:
   - `lifeCycleModelInformation.quantitativeReference.referenceToReferenceProcess`
   - `technology.processes.processInstance[*].referenceToProcess`
   - `technology.processes.processInstance[*].connections.outputExchange`
   - computed `@multiplicationFactor`
   - a valid `referenceToResultingProcess` reference inside `json_ordered`
4. Review the local outputs:
   - `run-plan.json`
   - `resolved-manifest.json`
   - `selection/selection-brief.md`
   - `discovery/reference-model-summary.json`
   - `models/**/tidas_bundle/lifecyclemodels/*.json`
   - `models/**/summary.json`
   - `models/**/connections.json`
   - `models/**/process-catalog.json`
   - `reports/lifecyclemodel-auto-build-report.json`
5. If the workflow later needs validation or publish handoff, call the dedicated CLI follow-up commands instead of rebuilding those paths inside the skill:
   - `node scripts/run-lifecyclemodel-automated-builder.mjs validate --run-dir <dir>`
   - `node scripts/run-lifecyclemodel-automated-builder.mjs publish --run-dir <dir>`
6. If someone asks for remote discovery or AI-assisted model selection, add it as a native `tiangong lifecyclemodel ...` capability first.

## Commands
```bash
node lifecyclemodel-automated-builder/scripts/run-lifecyclemodel-automated-builder.mjs build \
  --input lifecyclemodel-automated-builder/assets/example-request.json \
  --out-dir /abs/path/local-run-test \
  --dry-run

node lifecyclemodel-automated-builder/scripts/run-lifecyclemodel-automated-builder.mjs build \
  --input /abs/path/request.json \
  --out-dir /abs/path/out

node lifecyclemodel-automated-builder/scripts/run-lifecyclemodel-automated-builder.mjs build \
  --input lifecyclemodel-automated-builder/assets/example-local-runs.json \
  --out-dir /abs/path/local-run-test

node lifecyclemodel-automated-builder/scripts/run-lifecyclemodel-automated-builder.mjs validate \
  --run-dir /abs/path/local-run-test

node lifecyclemodel-automated-builder/scripts/run-lifecyclemodel-automated-builder.mjs publish \
  --run-dir /abs/path/local-run-test
```

## Fast Triage
- Local CLI override issues: set `TIANGONG_LCA_CLI_DIR` or pass `--cli-dir` only when you intentionally need an unpublished working tree.
- Missing `local_runs[]`: the current canonical slice only accepts local process-build runs.
- Validation/publish follow-up: use the dedicated CLI subcommands against one existing auto-build run; they stay local-only and do not perform remote writes.
- Validation failures on required model fields: inspect `references/model-contract.md`.
- Topology disagreements: inspect `references/source-analysis.md` for native lifecycle model conventions.
- If the user asks for remote discovery or writes, explain that the canonical path intentionally stops at local auto-build plus local validate/publish handoff for now.

## Bundled Resources
- `scripts/run-lifecyclemodel-automated-builder.mjs`: native Node wrapper that delegates to `tiangong lifecyclemodel ...`.
- `assets/example-request.json`: minimal current-slice manifest using `local_runs[]`.
- `assets/example-local-runs.json`: multi-run local assembly manifest example.
- `references/workflow.md`: current CLI-backed workflow and deferred slices.
- `references/source-analysis.md`: extracted conventions from `tiangong-lca-next`, `tidas-sdk`, and `tidas-tools`.
- `references/model-contract.md`: native `json_ordered` fields required before validation or publish.
