---
name: tidas-data-import
description: Orchestrate SDK-backed TIDAS context, tidas-tools package conversion, and source-document authoring through the tiangong-lca CLI. Use for Foundry import tasks that start from packaged LCA datasets or PDF/Excel/source files.
---

# TIDAS Data Import

Use this skill when Foundry needs to create or prepare TIDAS data from external inputs.

## Boundaries

- This skill orchestrates CLI commands only.
- Do not embed schema, methodology YAML, ruleset, converter logic, or prompt templates in this skill.
- Fetch the target contract context before AI generation, conversion repair, or review.
- Treat CLI reports as the durable artifact contract for downstream Foundry routing.

## Lane 1: Packaged LCA Dataset

Use this lane for zipped or directory-based packages that `tidas-tools` can detect or convert.

```bash
tiangong-lca dataset context-pack \
  --type process \
  --profile ai-import \
  --include schema,methodology,ruleset \
  --out-dir .foundry/workspaces/<task-id>/context/process \
  --json

tiangong-lca dataset import-lca convert \
  --input /abs/path/source-package \
  --output-dir .foundry/workspaces/<task-id>/conversion \
  --from-format auto \
  --target tidas \
  --json
```

Required verification artifacts:

- `context/process/outputs/contract-report.json`
- `context/process/outputs/ai-context.md`
- `conversion/outputs/import-lca-report.json`
- `conversion/conversion-report.json`
- `conversion/tidas/` when conversion succeeds
- entity-level curation queue artifacts after downstream validation and QA: `curation-queue/outputs/curation-queue-manifest.json`, `curation-queue/outputs/curation-queue-tasks.jsonl`, `curation-queue/outputs/curation-queue-locks.json`, and `curation-queue/outputs/curation-queue-blockers.jsonl`

## Lane 2: Source Document Authoring

Use this lane for PDF, Excel, web exports, screenshots, or free text that must be turned into TIDAS candidate data.

```bash
tiangong-lca dataset author \
  --input /abs/path/source.pdf \
  --output-dir .foundry/workspaces/<task-id>/authoring \
  --target-types process,flow \
  --language zh-CN \
  --json
```

Required verification artifacts:

- `authoring/outputs/source-extract.json`
- `authoring/context/<type>/outputs/ai-context.md`
- `authoring/outputs/authoring-report.json`

## Downstream Gates

After either lane creates candidate TIDAS data, run the existing dataset gates that match the target type:

```bash
tiangong-lca dataset validate --help
tiangong-lca qa process --help
tiangong-lca qa flow --help
node scripts/foundry.mjs dataset-curation-queue-build --help
node scripts/foundry.mjs dataset-curation-gate --help
node scripts/foundry.mjs dataset-curation-cleanup --help
```

For structured external dataset imports, build the queue after schema validation and deterministic QA and before AI repair or publish preparation:

```bash
npm run dataset:curation-queue:build -- \
  --processes .foundry/workspaces/<task-id>/rows/processes.normalized.jsonl \
  --flows .foundry/workspaces/<task-id>/rows/flows.normalized.jsonl \
  --support .foundry/workspaces/<task-id>/rows/sources.normalized.jsonl \
  --out-dir .foundry/workspaces/<task-id>/curation-queue
```

The queue is the execution contract for support, flow, and process work. Do not select arbitrary clean-looking rows or write final semantic fields from task-local scripts; use the queue task input, closure, blocker, and run-plan artifacts.

When running `dataset-curation-gate`, pass `--queue-dir .foundry/workspaces/<task-id>/curation-queue` so AI authoring packages include the entity queue task, dependency closure, referenced flow rows, and batch support rows in addition to schema, YAML, profile context, schema blockers, and QA findings.

For process rows, treat `tiangong-lca qa process` as deterministic QA output only. It classifies structural issues, reference-flow/exchange evidence, and material-balance observations; Foundry owns profile policy, AI authoring packages, curated patches/build plans, import-only trace cleanup, profile waivers, and the final prewrite decision.

Block publish preparation when source evidence, schema validation, required fields, duplicate checks, runtime rulesets, or Foundry process curation actions are unresolved.
