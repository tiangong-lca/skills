---
title: skills AI Working Guide
docType: contract
scope: repo
status: active
authoritative: true
owner: skills
language: en
whenToUse:
  - when a task may add, remove, rename, or restructure a checked-in TianGong skill
  - when deciding whether work belongs in this repository, in tiangong-lca-cli, or in a product/runtime repo
  - when routing from the workspace root into the skills repository
whenToUpdate:
  - when skill packaging rules or validation flow change
  - when repo ownership or CLI boundary rules change
  - when repo-local docpact governance or source docs change
checkPaths:
  - AGENTS.md
  - README.md
  - README.zh-CN.md
  - .docpact/config.yaml
  - docs/agents/**
  - */SKILL.md
  - */agents/openai.yaml
  - */scripts/**
  - */references/**
  - */assets/**
  - scripts/validate-skills.mjs
  - test/**
  - .github/workflows/**
  - .githooks/**
  - scripts/docpact
  - scripts/docpact-gate.sh
  - scripts/install-git-hooks.sh
lastReviewedAt: 2026-05-27
lastReviewedCommit: 44fc3f36632fa3defc7a53067b148df639ab855f
related:
  - .docpact/config.yaml
  - docs/agents/repo-architecture.md
  - docs/agents/repo-validation.md
  - README.md
  - README.zh-CN.md
  - scripts/validate-skills.mjs
---

# AGENTS.md — skills AI Working Guide

`tiangong-lca-skills` owns checked-in skill wrappers and skill packaging metadata for TianGong agent workflows. Start here when the task may change `SKILL.md`, `agents/openai.yaml`, validation rules, or the thin wrappers that connect skills to the unified CLI.

## AI Load Order

Load docs in this order:

1. `AGENTS.md`
2. `.docpact/config.yaml`
3. `docs/agents/repo-architecture.md`
4. `docs/agents/repo-validation.md`
5. `README.md` only when you need install or distribution context
6. the target skill's `SKILL.md`
7. `scripts/validate-skills.mjs` only when validation behavior itself is part of the task

Do not start by inferring behavior from chat history or one skill directory alone.

## Repo Ownership

This repo owns:

- `*/SKILL.md` for checked-in skill instructions
- `*/agents/openai.yaml` for the canonical CLI-backed wrapper contract
- skill-local `scripts/**`, `references/**`, and `assets/**` when they are part of one skill package
- `scripts/validate-skills.mjs` and repo validation tests
- `README.md` and `README.zh-CN.md` for install and usage guidance

This repo does not own:

- the public CLI command surface
- product runtime business logic
- workspace integration state after merge

Route those tasks to:

- `tiangong-lca-cli` for new native `tiangong-lca <noun> <verb>` commands
- the owning product/runtime repo for business logic or API changes
- `lca-workspace` for root integration after merge

## Runtime Facts

- Repo-local documentation governance is enforced through `.docpact/config.yaml` and `.github/workflows/ai-doc-lint.yml`.
- This repo is distribution-oriented; each skill should stay a thin wrapper over the unified `tiangong-lca` CLI
- If a capability is missing, add it to `tiangong-lca-cli` first, then update the skill wrapper here
- Current-account dataset review skills may orchestrate frozen local inputs through public CLI commands, but must not own direct database access, credential parsing, or private account runtime logic.
- Local CLI checkouts selected by wrappers may be rebuilt automatically when their source is newer than `dist/src/main.js`; wrappers should still keep the CLI command surface in `tiangong-lca-cli`.
- The canonical local validation command is `node scripts/validate-skills.mjs`
- You may pass one or more skill paths to validate only the touched skills
- For documentation-governance changes, run `scripts/docpact validate-config --root . --strict` and `scripts/docpact lint --root . --base origin/main --head HEAD --mode enforce`

## Hard Boundaries

- Do not add private business runtimes, MCP transports, or unrelated orchestration layers inside a skill when the behavior should live in the CLI or an owning repo
- Do not leave a changed `SKILL.md` without updating the paired `agents/openai.yaml` when the invocation contract changed
- Do not treat a merged repo PR here as workspace-delivery complete if the root repo still needs a submodule bump

## Workspace Integration

A merged PR in `tiangong-lca-skills` is repo-complete, not delivery-complete.

If the change must ship through the workspace:

1. merge the child PR into `tiangong-lca-skills`
2. update the `lca-workspace` submodule pointer deliberately
3. complete any later workspace-level validation that depends on the updated skill set

## Local Docpact Push Gate

Install the versioned local hook once per checkout:

```bash
./scripts/install-git-hooks.sh
```

The `pre-push` hook runs `scripts/docpact-gate.sh`, which delegates CLI lookup to `scripts/docpact` and performs strict config validation plus enforced lint before the push leaves the machine. It then builds the local sibling `tiangong-lca-cli` when available and runs `node scripts/validate-skills.mjs` as the local test gate. The wrapper checks `DOCPACT_BIN`, Cargo install locations, Homebrew install locations, and then `PATH`, so local agent shells should not fail only because bare `docpact` is unavailable. The default comparison base is `origin/main`. Override it for unusual stacks with `DOCPACT_BASE_REF=<ref>` or `scripts/docpact-gate.sh --base <ref>`. The gate writes its detailed report to a temporary file so normal pushes do not create `.docpact/runs/` artifacts. The GitHub `validate-skills` workflow is manual-dispatch only.
