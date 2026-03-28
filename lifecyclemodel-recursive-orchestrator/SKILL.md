---
name: lifecyclemodel-recursive-orchestrator
description: Plan and orchestrate recursive LCA assembly across `process-automated-builder`, `lifecyclemodel-automated-builder`, and `lifecyclemodel-resulting-process-builder`, including reuse decisions, submodel builds, resulting-process construction, graph manifests, and lineage manifests. Use when a product system request spans multiple model/process layers and needs a reproducible local orchestration run before any publish handoff.
---

# Lifecycle Model Recursive Orchestrator

Use this skill when one request needs more than a single builder call. The orchestrator validates a request, resolves each node to reuse/build/cutoff, runs downstream wrappers when needed, and writes a full local run directory.

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
- records invocation results for downstream wrappers
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

Use extra `nodes` only for additional dependencies or subsystems beyond the root node.

## Downstream Builders

- `process_builder` calls `process-automated-builder/scripts/run-process-automated-builder.sh`
- `submodel_builder` calls `lifecyclemodel-automated-builder/scripts/run-lifecyclemodel-automated-builder.sh`
- `projector` calls `lifecyclemodel-resulting-process-builder/scripts/run-lifecyclemodel-resulting-process-builder.sh`

The orchestrator does not reimplement those skills. It owns request normalization, node resolution, invocation ordering, and final manifests.

## Typical Workflow

1. Read `assets/request.schema.json` and prepare a manifest.
2. Run `plan` and inspect the generated `assembly-plan.json`.
3. Run `execute` to invoke downstream wrappers and persist invocation logs.
4. Run `publish` only to prepare a local handoff bundle. Remote writes remain outside this skill.

## Commands

```bash
python3 scripts/lifecyclemodel_recursive_orchestrator.py plan \
  --request assets/example-request.json \
  --out-dir /abs/path/run-001

python3 scripts/lifecyclemodel_recursive_orchestrator.py execute \
  --request assets/example-request.json \
  --out-dir /abs/path/run-001

python3 scripts/lifecyclemodel_recursive_orchestrator.py publish \
  --run-dir /abs/path/run-001 \
  --publish-lifecyclemodels \
  --publish-resulting-process-relations

scripts/run-lifecyclemodel-recursive-orchestrator.sh \
  execute --request assets/example-request.json --out-dir /abs/path/run-001
```

## Load References On Demand

- `references/workflow.md`: stage-level orchestration flow and policy knobs
- `references/data-model.md`: graph/lineage fields and version contract
- `references/invocation-contracts.md`: exact handoff contract to each downstream skill
- `references/integration-notes.md`: current MCP / Next alignment notes
- `references/minimal-relation-fields.md`: minimum relation tuple to preserve
- `references/next-mcp-alignment.md`: short-term contract alignment guidance
