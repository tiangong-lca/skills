---
name: external-dataset-curated-import
description: Orchestrate structured external LCA packages such as BAFU or USLCI into TIDAS candidate rows through CLI conversion, queue state, child skills, deterministic gates, and readback-safe publish handoff.
---

# External Dataset Curated Import

Use this top-level skill when the input is a packaged or semi-structured LCA dataset that should enter TianGong through a repeatable import profile.

## Boundaries

- This skill is a workflow coordinator only.
- Do not parse source package formats in this skill. Use `tiangong-lca dataset import-lca convert` or `tidas-tools`.
- Do not implement queue state, schema validation, QA, remote writes, or database verification in this skill. Use the CLI.
- Do not make BAFU, USLCI, or any account a code path. They are profile inputs.
- Do not hand-edit row JSONL as the primary repair path. Use child skills to write structured decisions or patches, then apply them through deterministic CLI/Foundry commands.

## Required Inputs

- Foundry task directory under `.foundry/workspaces/<task-id>/`.
- Frozen source package manifest and checksum.
- Import profile id, such as `generic`, `bafu`, or `uslci`.
- Account/write guard when the task may eventually request remote write.
- Target entity scope: support, flow, process, lifecyclemodel, or a bounded subset.

## Workflow

1. Fetch SDK-backed contract context for every target TIDAS type:

```bash
tiangong-lca dataset context-pack \
  --type <process|flow|source|contact|unitgroup|flowproperty|lifecyclemodel> \
  --profile ai-import \
  --out-dir .foundry/workspaces/<task-id>/context/<type> \
  --json
```

2. Convert the source package:

```bash
tiangong-lca dataset import-lca convert \
  --input /abs/path/source-package \
  --output-dir .foundry/workspaces/<task-id>/conversion \
  --from-format auto \
  --target tidas \
  --json
```

3. Normalize converted rows into explicit support, flow, and process JSONL files.

4. Run schema validation and deterministic QA through the CLI.

5. Build the entity queue:

```bash
tiangong-lca dataset curation-queue build \
  --processes .foundry/workspaces/<task-id>/rows/processes.jsonl \
  --flows .foundry/workspaces/<task-id>/rows/flows.jsonl \
  --support .foundry/workspaces/<task-id>/rows/sources.jsonl \
  --out-dir .foundry/workspaces/<task-id>/curation-queue \
  --json
```

6. Drive the queue with CLI state, not ad hoc batches:

```bash
tiangong-lca dataset curation-queue next \
  --queue-dir .foundry/workspaces/<task-id>/curation-queue \
  --json
```

Execute only the returned task. Support tasks complete before flow tasks; flow tasks complete before dependent process tasks.

7. For semantic blockers, route to existing child skills:

- Support/source/contact repair: use Foundry authoring task packages or dedicated support child skills when they exist.
- Flow repair: use `flow-governance-review` or a future `flow-authoring` alias.
- Process repair: use `process-automated-builder` or a future `process-authoring` alias.
- Publish handoff: use `lca-publish-executor` only after deterministic prewrite gates are ready.

8. After each task finishes, write or update that task's `checkpoint.json`, then call `curation-queue next` again.

9. Before remote write, verify queue completion for the intended scope:

```bash
tiangong-lca dataset curation-queue verify \
  --queue-dir .foundry/workspaces/<task-id>/curation-queue \
  --type process \
  --json
```

The verify report must be `passed` for the committed scope. If verify is blocked while `next` still returns `ready`, continue the queue.

## Stop Rules

Stop and return blocker evidence when:

- source manifest or profile lock is missing;
- conversion reports blockers or unsupported source format;
- schema, QA, curation, or queue verify blocks with no runnable next task;
- a child skill would need to invent source evidence;
- remote write would require direct database access or bypass a CLI dry-run/verification gate.

## Done Criteria

- Source package, profile lock, contract context, converted rows, schema/QA reports, queue artifacts, task checkpoints, mapping/provenance, dry-run, commit handoff, and readback verification are all present for the final scope.
- `tiangong-lca dataset curation-queue verify` is `passed` for the committed scope.
- Foundry task completion report aggregates the exact closeout artifacts.
