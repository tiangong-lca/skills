---
title: skills Repo Architecture
docType: reference
scope: repo
status: active
authoritative: true
owner: skills
language: en
whenToUse:
  - when deciding whether a change belongs in the skills repository or the CLI repository
  - when changing checked-in skill instructions, wrappers, references, assets, or validation rules
whenToUpdate:
  - when skill package structure changes
  - when the CLI boundary changes
  - when skill validation or packaging ownership changes
checkPaths:
  - AGENTS.md
  - .docpact/config.yaml
  - "*/SKILL.md"
  - "*/agents/openai.yaml"
  - "*/scripts/**"
  - "*/references/**"
  - "*/assets/**"
  - .githooks/pre-push
  - scripts/docpact
  - scripts/docpact-gate.sh
  - scripts/install-git-hooks.sh
lastReviewedAt: 2026-06-04
lastReviewedCommit: 7c5039a212974a8e3c8392e31c18f72d0322dfe1
related:
  - AGENTS.md
  - .docpact/config.yaml
  - docs/agents/repo-validation.md
---

# skills Repo Architecture

`tiangong-lca-skills` owns checked-in skill packages and CLI-backed agent workflow wrappers for TianGong workflows.

Review note, 2026-06-02: dataset import curation queue changes keep this repository at the workflow-instruction layer; executable queue and curation gate behavior stays in CLI and Foundry.
Review note, 2026-06-04: Foundry now has two checked-in top-level scenario skills, `external-dataset-curated-import` and `source-evidence-dataset-development`. They coordinate existing CLI/child-skill surfaces and must not grow package parsing, database write, or evidence retrieval implementations.

## Owned Surfaces

- `*/SKILL.md` contains canonical skill instructions and trigger contracts.
- `*/agents/openai.yaml` contains wrapper contracts used by the skills CLI.
- `*/scripts/**`, `*/references/**`, and `*/assets/**` are skill-local support files intentionally shipped with a skill package.
- `scripts/validate-skills.mjs` and `test/**` define repo-level validation for wrappers and packaging rules.
- `README.md` and `README.zh-CN.md` explain installation and usage.

Top-level Foundry scenario skills are allowed in this repository when they only encode workflow order and routing:

- `external-dataset-curated-import`
- `source-evidence-dataset-development`

## Non-Owner Boundaries

- `tiangong-lca-cli` owns the native public command surface, low-level command semantics, REST clients, and auth behavior.
- External source-evidence research skill repositories, such as `tiangong-ai/skills`, own fast-moving Tiangong KB retrieval skills.
- Product/runtime repositories own business logic and API behavior.
- `lca-workspace` owns root integration state and submodule pointer updates.

If a skill needs a capability that does not exist in the CLI, add the capability to `tiangong-lca-cli` first and keep the skill as a thin wrapper over that CLI surface.

If a Foundry/source-evidence workflow needs an external Tiangong KB research skill, consume it with `npx skills` at runtime and record the resolved upstream ref in the task workspace. Do not copy the external skill package into this repository unless ownership is intentionally transferred.

Current-account dataset review is owned here only as a skill package and wrapper contract. Its durable runtime behavior belongs in public `tiangong-lca` CLI commands such as dataset validation, reference rewriting, lifecyclemodel save-draft, and lifecyclemodel graph export.

The shared wrapper launcher may prepare a local CLI checkout by running its build when source files are newer than `dist/src/main.js`. That is a developer-experience guard for stale local checkouts, not permission for skills to duplicate CLI implementation.

## Integration Semantics

A merged PR in this repository is repo-complete only. If the updated skill set must ship through the workspace, root integration must deliberately update the `tiangong-lca-skills` submodule pointer after merge.

## Local Docpact Push Gate

This repository has a versioned local `pre-push` hook under `.githooks/pre-push` that delegates to `scripts/docpact-gate.sh`, resolves the docpact CLI through `scripts/docpact`, builds the local sibling CLI when available, and runs `node scripts/validate-skills.mjs`. The hook is the local guard for docpact config validation, enforced doc-governance linting, and skill validation; the GitHub `validate-skills` workflow is manual-dispatch only.
