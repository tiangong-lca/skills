---
name: lifecyclemodel-resulting-process-builder
description: Build deterministic local resulting-process datasets from an existing lifecycle model, plus lifecyclemodel/resulting-process relation metadata and publish handoff artifacts. Use when a lifecycle model already exists and you need a formal aggregated resulting `processDataSet` without routing through `process-automated-builder`.
---

# Lifecycle Model Resulting Process Builder

Use this skill when the source of truth is already a lifecycle model `json_ordered` file and the next step is to deterministically compute the aggregated resulting process plus relation handoff artifacts, not to synthesize a process from external flow evidence.

## Run Workflow

1. By default the wrapper runs the published CLI through `npx -y @tiangong-lca/cli@latest`. Use `TIANGONG_LCA_CLI_DIR` or `--cli-dir` only for local dev/CI overrides.
2. Use `node scripts/run-lifecyclemodel-resulting-process-builder.mjs build ...` to delegate to `tiangong lifecyclemodel build-resulting-process`.
3. Use `node scripts/run-lifecyclemodel-resulting-process-builder.mjs publish ...` to delegate to `tiangong lifecyclemodel publish-resulting-process`.
4. Confirm the local artifacts in the run directory before any later `tiangong publish run` step.

The active runtime path is `skill -> Node wrapper -> tiangong CLI`. Python and MCP are no longer part of the normal execution path for this skill.

## What The Implementation Does

- validates `assets/request.schema.json`
- loads a lifecycle model from `source_model.json_ordered` or `source_model.json_ordered_path`
- resolves referenced process datasets from local process exports
- extracts process instances and graph edges from model topology
- aggregates exchanges across included processes and cancels internal linked flows
- derives:
  - `process-projection-bundle.json`
  - `projection-report.json`
  - `publish-bundle.json` via `publish`
  - `publish-intent.json` via `publish`
- emits relation payloads containing:
  - `generated_from_lifecyclemodel_id`
  - `generated_from_lifecyclemodel_version`
  - `projection_role`
  - `projection_signature`
- supports `primary-only`
- accepts `all-subproducts` requests conservatively and reports when only a primary aggregated process can be emitted
- keeps all work local; no remote write path is executed here

## Inputs

Always provide:

- `source_model`
- `projection`
- `publish`

Provide `process_sources` when local process resolution is not discoverable from the model path.

The source model may be provided as:

- `source_model.id`
- `source_model.json_ordered_path`
- `source_model.json_ordered`

Referenced process datasets may be provided via:

- `process_sources.process_catalog_path`
- `process_sources.run_dirs[]`
- `process_sources.process_json_dirs[]`
- `process_sources.process_json_files[]`
- auto-detected sibling directories such as `processes/` or `*-processes/` when using `--model-file`

Canonical request files should use `process_sources.allow_remote_lookup`, but the normal skill flow is still local-first and should keep it `false` unless deterministic remote process lookup is explicitly needed. When `process_sources.allow_remote_lookup=true`, supply `TIANGONG_LCA_API_BASE_URL` and `TIANGONG_LCA_API_KEY` or pass `--base-url` and `--api-key` through to the CLI.

## Outputs

- `request.normalized.json`
- `source-model.normalized.json`
- `source-model.summary.json`
- `projection-report.json`
- `process-projection-bundle.json`
- `publish-bundle.json` from `publish`
- `publish-intent.json` from `publish`

## Commands

```bash
node scripts/run-lifecyclemodel-resulting-process-builder.mjs build \
  --request assets/example-request.json \
  --out-dir /abs/path/run-001

node scripts/run-lifecyclemodel-resulting-process-builder.mjs build \
  --model-file assets/example-model.json \
  --projection-role primary \
  --out-dir /abs/path/run-001

node scripts/run-lifecyclemodel-resulting-process-builder.mjs publish \
  --run-dir /abs/path/run-001 \
  --publish-processes \
  --publish-relations

# Force a local CLI working tree during dev/CI
TIANGONG_LCA_CLI_DIR=/path/to/tiangong-lca-cli \
  node scripts/run-lifecyclemodel-resulting-process-builder.mjs build --json
```

Use `--model-file` only when local process sources can be inferred from the model location. Use `--request` to pin `process_sources.*` explicitly. The wrapper keeps `--request` and `--model-file` as compatibility flags, but the underlying CLI contract is `--input <request.json>` for build and `--run-dir <dir>` for publish.

## Separation Rule

- use `process-automated-builder` for flow-to-process synthesis
- use this skill for lifecyclemodel-to-resulting-process aggregation/build

They are different pipelines and should stay separate.

## Load References On Demand

- `references/projection-workflow.md`: intake to publish-handoff stages
- `references/projection-contract.md`: bundle shape and relation semantics
- `references/builder-invocation-contract.md`: caller/callee contract
- `references/integration-notes.md`: current cross-project architecture notes
