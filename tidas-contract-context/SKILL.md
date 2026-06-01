---
name: tidas-contract-context
description: Fetch SDK-backed TIDAS schema, methodology YAML, runtime ruleset, and AI context-pack artifacts through the tiangong-lca CLI. Use before AI authoring or repairing TIDAS process, flow, source, contact, unitgroup, flowproperty, lifecyclemodel, or lciamethod data.
---

# TIDAS Contract Context

Use this skill when an agent needs the authoritative TIDAS contract for AI data import, conversion repair, or source-document authoring.

## Boundaries

- This skill is a thin wrapper over `tiangong-lca dataset context-pack` and `tiangong-lca dataset contract get`.
- Do not copy schema, methodology YAML, or ruleset files into this skill.
- Do not generate or mutate TIDAS rows here. Authoring and conversion should use CLI or Foundry workflows after the context pack exists.

## Required Inputs

- Target TIDAS type: `process`, `flow`, `source`, `contact`, `unitgroup`, `flowproperty`, `lifecyclemodel`, or `lciamethod`.
- Output directory under the current task workspace.
- Include list when narrower context is needed: `schema`, `methodology`, `ruleset`.

## Commands

Create an AI-ready context pack:

```bash
tiangong-lca dataset context-pack \
  --type process \
  --profile ai-import \
  --include schema,methodology,ruleset \
  --out-dir .foundry/workspaces/<task-id>/context/process \
  --json
```

Create a raw contract artifact pack:

```bash
tiangong-lca dataset contract get \
  --type flow \
  --include schema,methodology,ruleset \
  --out-dir .foundry/workspaces/<task-id>/contract/flow \
  --json
```

## Verification

After the command completes, verify these artifacts before using the context:

- `outputs/contract-report.json`
- `outputs/contract-manifest.json`
- `outputs/schema.json` when `schema` was requested
- `outputs/methodology.yaml` when `methodology` is available for the target type
- `outputs/runtime-ruleset.json` when rules exist for the target type
- `outputs/ai-context.json` and `outputs/ai-context.md` for `context-pack`

The manifest hash values are the durable proof of which contract version the AI saw.
