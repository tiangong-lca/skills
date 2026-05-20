---
name: lifecyclemodel-recursive-orchestrator
description: Plan and orchestrate recursive LCA assembly across `process-automated-builder`, `lifecyclemodel-automated-builder`, and `lifecyclemodel-resulting-process-builder`, including reuse decisions, submodel builds, resulting-process construction, graph manifests, and lineage manifests. Use when a product system request spans multiple model/process layers and needs a reproducible local orchestration run before any publish handoff.
---

# Lifecycle Model Recursive Orchestrator

Use this skill when one request needs more than a single builder call. The orchestrator validates a request, resolves each node to reuse/build/cutoff, invokes the native `tiangong lifecyclemodel orchestrate` actions, and writes a full local run directory.

## What The Implementation Does

- validates `assets/request.schema.json`
- normalizes `root`, optional extra `nodes`, and dependency `edges`
- chooses per-node resolution in this order:
  - reuse existing resulting process
  - reuse existing process
  - reuse existing lifecycle model
  - build with `lifecyclemodel-automated-builder`
  - build with `process-automated-builder`
  - unresolved / cutoff
- records invocation results for native CLI builder slices
- emits:
  - `assembly-plan.json`
  - `graph-manifest.json`
  - `lineage-manifest.json`
  - `boundary-report.json`
  - `publish-bundle.json` via `publish`

## Request Shape

Required top-level fields:

- `goal`
- `root`
- `orchestration`
- `publish`

Optional top-level fields:

- `candidate_sources`
- `nodes`
- `edges`
- `notes`

The `root` block may directly carry:

- candidate lists
- `process_builder`
- `submodel_builder`
- `projector`

Supported `process_builder` fields are intentionally narrow:

- `flow_file`
- `flow_json`
- `run_id`

Removed legacy fields such as `mode=langgraph` and `python_bin` are not part of the supported request surface anymore.

Use extra `nodes` only for additional dependencies or subsystems beyond the root node.

## Downstream Builders

- `process_builder` reuses the same native slice as `tiangong process auto-build`
- `submodel_builder` reuses the same native slice as `tiangong lifecyclemodel auto-build`
- `projector` reuses the same native slice as `tiangong lifecyclemodel build-resulting-process`

The wrapper does not call other skills directly. It delegates to `tiangong lifecyclemodel orchestrate`, and the CLI orchestrator owns request normalization, node resolution, invocation ordering, and final manifests.

Current limitation:

- candidate discovery must already be resolved into the request or into local artifacts upstream
- the submodel builder path only covers the CLI-backed `auto-build` slice
- `validate-build` and `publish-build` for lifecyclemodels now exist in the CLI, but this orchestrator does not invoke them yet

## Typical Workflow

1. Read `assets/request.schema.json` and prepare a manifest.
2. Run `plan` and inspect the generated `assembly-plan.json`.
3. Run `execute` to invoke native CLI builder slices and persist invocation logs.
4. Run `publish` only to prepare a local handoff bundle. Remote writes remain outside this skill and should later go through `tiangong publish run`.

## Commands

```bash
node scripts/run-lifecyclemodel-recursive-orchestrator.mjs plan \
  --request assets/example-request.json \
  --out-dir /abs/path/run-001 \
  --json

node scripts/run-lifecyclemodel-recursive-orchestrator.mjs execute \
  --request assets/example-request.json \
  --out-dir /abs/path/run-001 \
  --allow-process-build \
  --allow-submodel-build \
  --json

node scripts/run-lifecyclemodel-recursive-orchestrator.mjs publish \
  --run-dir /abs/path/run-001 \
  --publish-lifecyclemodels \
  --publish-resulting-process-relations \
  --json
```

## Load References On Demand

- `references/workflow.md`: stage-level orchestration flow and policy knobs
- `references/data-model.md`: graph/lineage fields and version contract
- `references/invocation-contracts.md`: exact handoff contract to each downstream skill
- `references/minimal-relation-fields.md`: minimum relation tuple to preserve
