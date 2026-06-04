---
name: tidas-data-import
description: Orchestrate SDK-backed TIDAS context, tidas-tools package conversion, and source-document authoring through the tiangong-lca CLI. Use for Foundry import tasks that start from packaged LCA datasets or PDF/Excel/source files.
---

# TIDAS Data Import

Use this skill when Foundry needs to create or prepare TIDAS data from external inputs.

## Boundaries

- This skill orchestrates CLI commands only.
- Do not embed schema, methodology YAML, ruleset, converter logic, or prompt templates in this skill.
- Do not vendor external source-evidence research skills into this repository.
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

### Source Evidence Runtime Skills

For source-document authoring that needs SCI paper or academic journal evidence, resolve the latest external research skill at runtime:

```bash
npx skills use https://github.com/tiangong-ai/skills \
  --skill tiangong-kb-sci-search \
  --full-depth
```

`tiangong-kb-sci-search` searches only the `sci` channel. Do not use it as a report, patent, general web, or all-source retrieval wrapper. If the task needs those other source classes, keep them as separate evidence channels.

When a Foundry task uses an external runtime skill, write a resolution record before retrieval:

```text
.foundry/workspaces/<task-id>/runtime-skills/runtime-skill-resolution.json
```

The record should include the `npx skills` command, source repo, resolved `refs/heads/main` commit from `git ls-remote https://github.com/tiangong-ai/skills.git refs/heads/main`, skill name, evidence channel, timestamp, and output artifact paths. Retrieved papers are evidence candidates only; field values still need evidence dossier capture, limitations/conflicts, validation, curation, dry-run, and verification gates.

## Downstream Gates

After either lane creates candidate TIDAS data, run the existing dataset gates that match the target type:

```bash
tiangong-lca dataset validate --help
tiangong-lca qa process --help
tiangong-lca qa flow --help
node scripts/foundry.mjs dataset-curation-queue-build --help
node scripts/foundry.mjs dataset-curation-gate --help
node scripts/foundry.mjs dataset-authoring-task-build --help
node scripts/foundry.mjs dataset-patch-apply --help
node scripts/foundry.mjs dataset-curation-cleanup --help
node scripts/foundry.mjs dataset-post-authoring-finalize --help
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

When curation gate has AI-required action items, build explicit authoring tasks before writing any patch. Prefer batch mode from the gate report so all entity patches can be combined and applied once:

```bash
npm run dataset:authoring-task:build -- \
  --curation-gate-report .foundry/workspaces/<task-id>/curation-gate/dataset-curation-gate-report.json \
  --out-dir .foundry/workspaces/<task-id>/authoring-tasks
```

Use single-package mode for targeted rework:

```bash
npm run dataset:authoring-task:build -- \
  --authoring-package .foundry/workspaces/<task-id>/curation-gate/ai-authoring-packages/<type>-<uuid>.authoring-package.json \
  --out-dir .foundry/workspaces/<task-id>/authoring-tasks/<type>-<uuid>
```

Codex/skill must read `ai-authoring-task.md` and the full authoring package for each task, then write only structured patch JSON. The final patch file must not retain `template_status=requires_ai_completion` or any `__AI_FILL_*` placeholder. In batch mode, write per-task `ai-patches.json` files first, then collect them into the manifest's batch patch target:

```bash
npm run dataset:authoring-patch:collect -- \
  --task-manifest .foundry/workspaces/<task-id>/authoring-tasks/authoring-task-manifest.json
```

Do not hand-edit row JSONL files and do not call database writes from the authoring task. If collect blocks, repair the per-task patch files before apply.

Apply AI output deterministically:

```bash
npm run dataset:patch:apply -- \
  --input .foundry/workspaces/<task-id>/rows/<type>.jsonl \
  --patch .foundry/workspaces/<task-id>/authoring-tasks/ai-patches.batch.json \
  --out .foundry/workspaces/<task-id>/rows/<type>.patched.jsonl \
  --out-dir .foundry/workspaces/<task-id>/patch-apply \
  --authoring-package-dir .foundry/workspaces/<task-id>/curation-gate/ai-authoring-packages \
  --require-authoring-package \
  --require-action-item-closure
```

If patch apply blocks on missing evidence, failed `test`, bad path, package mismatch, or unclosed action items, return to the authoring task. After a completed apply, use Foundry's process finalize wrapper for process rows:

```bash
npm run dataset:post-authoring-finalize -- \
  --type process \
  --rows-file .foundry/workspaces/<task-id>/rows/processes.patched.jsonl \
  --out-dir .foundry/workspaces/<task-id>/post-authoring-finalize \
  --profile bafu \
  --queue-dir .foundry/workspaces/<task-id>/curation-queue \
  --schema-file .foundry/workspaces/<task-id>/context/process/outputs/schema.json \
  --yaml-file .foundry/workspaces/<task-id>/context/process/outputs/methodology.yaml \
  --ruleset-file .foundry/workspaces/<task-id>/context/process/outputs/runtime-ruleset.json \
  --patch-collect-report .foundry/workspaces/<task-id>/authoring-tasks/authoring-patch-collect-report.json \
  --require-patch-collect-report \
  --patch-apply-report .foundry/workspaces/<task-id>/patch-apply/outputs/dataset-patch-apply-report.json \
  --target-user-id <uuid> \
  --verify-remote
```

The finalize wrapper reruns SDK validation, deterministic QA, cleanup, post-authoring curation gate, dry-run, optional remote verification, and mutation manifest on one exact rows-file scope. Commit remains a later explicit CLI step after `dataset-post-authoring-finalize-report.json` and its mutation manifest are `ready_for_remote_write`, followed by post-commit remote verification.

For process rows, treat `tiangong-lca qa process` as deterministic QA output only. It classifies structural issues, reference-flow/exchange evidence, and material-balance observations; Foundry owns profile policy, AI authoring packages, curated patches/build plans, import-only trace cleanup, profile waivers, and the final prewrite decision.

Block publish preparation when source evidence, schema validation, required fields, duplicate checks, runtime rulesets, or Foundry process curation actions are unresolved.
