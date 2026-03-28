---
name: lifecyclemodel-resulting-process-builder
description: Build deterministic local resulting-process datasets from an existing lifecycle model, plus lifecyclemodel/resulting-process relation metadata and publish handoff artifacts. Use when a lifecycle model already exists and you need a formal aggregated resulting `processDataSet` without routing through `process-automated-builder`.
---

# Lifecycle Model Resulting Process Builder

Use this skill when the source of truth is already a lifecycle model `json_ordered` file and the next step is to deterministically compute the aggregated resulting process plus relation handoff artifacts, not to synthesize a process from external flow evidence.

## What The Implementation Does

- validates `assets/request.schema.json`
- loads a lifecycle model from `source_model.json_ordered` or `source_model.json_ordered_path`
- resolves referenced process datasets from local process exports or optional MCP lookup
- extracts process instances and graph edges from model topology
- aggregates exchanges across included processes and cancels internal linked flows
- derives:
  - `process-projection-bundle.json`
  - `projection-report.json`
  - `publish-bundle.json` via `publish`
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

Provide `process_sources` when local process resolution is not discoverable from the model path or when remote MCP lookup should be constrained.

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
- optional MCP lookup driven by `TIANGONG_LCA_REMOTE_*`

## Outputs

- `request.normalized.json`
- `source-model.normalized.json`
- `source-model.summary.json`
- `projection-report.json`
- `process-projection-bundle.json`
- `publish-bundle.json` from `publish`

## Commands

```bash
python3 scripts/lifecyclemodel_resulting_process_builder.py build \
  --request assets/example-request.json \
  --out-dir /abs/path/run-001

python3 scripts/lifecyclemodel_resulting_process_builder.py build \
  --model-file assets/example-model.json \
  --out-dir /abs/path/run-001

python3 scripts/lifecyclemodel_resulting_process_builder.py publish \
  --run-dir /abs/path/run-001 \
  --publish-processes \
  --publish-relations
```

Use `--model-file` only when the builder can infer local process sources from the model location or when remote MCP lookup is available. Use `--request` to pin `process_sources.*` explicitly.

## Separation Rule

- use `process-automated-builder` for flow-to-process synthesis
- use this skill for lifecyclemodel-to-resulting-process aggregation/build

They are different pipelines and should stay separate.

## Load References On Demand

- `references/projection-workflow.md`: intake to publish-handoff stages
- `references/projection-contract.md`: bundle shape and relation semantics
- `references/builder-invocation-contract.md`: caller/callee contract
- `references/integration-notes.md`: current cross-project architecture notes
