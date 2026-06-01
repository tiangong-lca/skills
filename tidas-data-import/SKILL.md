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
tiangong-lca dataset review --help
tiangong-lca dataset bilingual --help
```

Block publish preparation when source evidence, schema validation, required fields, bilingual quality, duplicate checks, or runtime rulesets are unresolved.
