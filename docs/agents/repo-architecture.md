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
  - scripts/docpact-gate.sh
  - scripts/install-git-hooks.sh
lastReviewedAt: 2026-05-08
lastReviewedCommit: 813ad4c1cddcae0eb57116e7ae9bd2da2854c115
related:
  - AGENTS.md
  - .docpact/config.yaml
  - docs/agents/repo-validation.md
---

# skills Repo Architecture

`tiangong-lca-skills` owns checked-in skill packages and CLI-backed agent workflow wrappers for TianGong workflows.

## Owned Surfaces

- `*/SKILL.md` contains canonical skill instructions and trigger contracts.
- `*/agents/openai.yaml` contains wrapper contracts used by the skills CLI.
- `*/scripts/**`, `*/references/**`, and `*/assets/**` are skill-local support files intentionally shipped with a skill package.
- `scripts/validate-skills.mjs` and `test/**` define repo-level validation for wrappers and packaging rules.
- `README.md` and `README.zh-CN.md` explain installation and usage.

## Non-Owner Boundaries

- `tiangong-lca-cli` owns the native public command surface, low-level command semantics, REST clients, and auth behavior.
- Product/runtime repositories own business logic and API behavior.
- `lca-workspace` owns root integration state and submodule pointer updates.

If a skill needs a capability that does not exist in the CLI, add the capability to `tiangong-lca-cli` first and keep the skill as a thin wrapper over that CLI surface.

## Integration Semantics

A merged PR in this repository is repo-complete only. If the updated skill set must ship through the workspace, root integration must deliberately update the `tiangong-lca-skills` submodule pointer after merge.

## Local Docpact Push Gate

This repository has a versioned local `pre-push` hook under `.githooks/pre-push` that delegates to `scripts/docpact-gate.sh`. The hook is a local developer guard for docpact config validation and enforced doc-governance linting; CI remains the authoritative PR enforcement path.
