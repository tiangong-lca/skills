---
name: source-evidence-dataset-development
description: Orchestrate evidence-driven TianGong TIDAS data creation or update from PDFs, Word files, web pages, URLs, APIs, databases, reports, or scientific literature.
---

# Source Evidence Dataset Development

Use this top-level skill when there is no import-ready LCA package and the task must develop or update TIDAS rows from evidence.

## Managed Foundry tasks

When the caller supplies a registered public Foundry task and its current result, use this skill as a domain helper. Retrieve and assess source evidence, or prepare the current requested evidence-backed decision or patch input. Keep the supplied workspace, task, actor and account intent. Return newly authored input files through the current public semantic/action contract and let Foundry register and validate them.

Advance the task only through its supplied structured `next_actions`, preserving executable, argv, CWD and binding. Do not run the standalone queue procedure below inside that task, overwrite registered rows or reports, write `checkpoint.json` yourself, or recreate a task after an uncertain write. Follow the original task's readback/recovery action. Task and account permission remain bound to the existing scope.

`foundry-tidas-import` is the ordinary entry for starting and continuing managed Foundry work. This helper can consume a supplied task result without requiring another skill directory to be installed. The remaining procedure applies to an independent CLI workflow whose state is outside the managed public task protocol.

## Boundaries

- This skill coordinates seed intake, evidence retrieval, entity planning, and child skill routing.
- Do not treat search results as final field evidence. Capture field-level evidence, limitations, and conflicts first.
- Do not copy external research skills into this repository. Resolve fast-moving evidence skills with `npx skills`.
- Do not write database rows. Remote write remains a later CLI/Foundry handoff after deterministic gates pass.

## Required Inputs

- Foundry task directory under `.foundry/workspaces/<task-id>/`.
- Execution seed or a clarification request if the seed is not executable.
- Target profile/account guard and intended use.
- Source starting points: files, URLs, reports, APIs, database ids, citations, or a bounded source-discovery request.

## Seed Gate

Before retrieval, freeze a `seed-manifest.json` with:

- object/process/product name;
- target entity plan;
- functional intent or declared unknown;
- geography and time scope or declared unknown;
- source starting points or source-discovery scope;
- intended use and quality target;
- profile/account guard.

If source starting points and source-discovery scope are both absent, stop and ask for clarification.

## Runtime Evidence Skills

For SCI paper or academic journal evidence, resolve the latest external skill:

```bash
npx skills use https://github.com/tiangong-ai/skills \
  --skill tiangong-kb-sci-search \
  --full-depth
```

`tiangong-kb-sci-search` searches only the `sci` channel. Keep report, patent, standard, official, web, and database evidence as separate channels.

Before using the skill, write a runtime resolution record in the Foundry workspace:

```text
.foundry/workspaces/<task-id>/runtime-skills/runtime-skill-resolution.json
```

Include the command, source repo, resolved `refs/heads/main` commit, skill name, evidence channel, timestamp, and output artifact paths.

## Workflow

1. Freeze seed and source starting points.
2. Build goal/scope and evidence questions by field.
3. Retrieve and capture evidence by channel.
4. Build an evidence dossier:
   - `evidence/sources.jsonl`
   - `evidence/chunks.jsonl`
   - `evidence/field-evidence.jsonl`
   - `evidence/conflicts.jsonl`
5. Fetch SDK-backed context packs for target TIDAS types.
6. Author candidate support, flow, process, and lifecyclemodel rows only from captured evidence or explicit proxy/waiver records.
7. Validate and QA rows through the CLI.
8. Build and drive the same CLI curation queue used by packaged imports.
9. Route semantic blockers to child skills such as `flow-governance-review`, `process-automated-builder`, or Foundry authoring tasks.
10. Verify queue scope, run dry-run/publish handoff, and require readback verification before completion.

Before any remote account read or write in steps 8-10, run `tiangong-lca auth status --json`; before commit, require `tiangong-lca auth doctor-auth --json`. A `login-required` result is a human handoff to `auth login` in a trusted terminal. Never request a username, password, credential, authorization code, token, or legacy API key. Keep a separate private session per account/project/client; headless tokens may come only from an approved orchestrator and never enter argv or artifacts.

## Stop Rules

Stop when:

- seed is not executable;
- critical field evidence is absent or conflicting and no scoped proxy/waiver is allowed;
- runtime evidence skill resolution is missing for SCI retrieval;
- a row would require invented units, amounts, locations, years, sources, or classifications;
- schema/QA/curation/queue verify blocks with no runnable next task;
- remote write would bypass CLI dry-run, commit handoff, or readback verification.

## Done Criteria

- Seed, source manifest, runtime skill resolution records, evidence dossier, entity plan, rows, schema/QA, curation queue, task checkpoints, mapping/provenance, dry-run, commit handoff, closeout, and readback verification are complete for the requested scope.
- Every authored field points to evidence, profile policy, remote reuse decision, or explicit waiver.
