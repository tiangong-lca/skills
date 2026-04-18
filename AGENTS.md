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
  - when the repo-local AI bootstrap docs under ai/ change
checkPaths:
  - AGENTS.md
  - README.md
  - README.zh-CN.md
  - ai/**/*.yaml
  - */SKILL.md
  - */agents/openai.yaml
  - */scripts/**
  - */references/**
  - */assets/**
  - scripts/validate-skills.mjs
  - test/**
  - .github/workflows/**
lastReviewedAt: 2026-04-18
lastReviewedCommit: 8a3f25207be167e315acc6fbbccb421a41077f79
related:
  - ai/repo.yaml
  - ai/doc-impact.yaml
  - README.md
  - README.zh-CN.md
  - scripts/validate-skills.mjs
---

# AGENTS.md — skills AI Working Guide

`tiangong-lca-skills` owns checked-in skill wrappers and skill packaging metadata for TianGong agent workflows. Start here when the task may change `SKILL.md`, `agents/openai.yaml`, validation rules, or the thin wrappers that connect skills to the unified CLI.

## AI Load Order

Load docs in this order:

1. `AGENTS.md`
2. `ai/repo.yaml`
3. `ai/doc-impact.yaml`
4. `README.md` only when you need install or distribution context
5. the target skill's `SKILL.md`
6. `scripts/validate-skills.mjs` only when validation behavior itself is part of the task

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

- `tiangong-lca-cli` for new native `tiangong <noun> <verb>` commands
- the owning product/runtime repo for business logic or API changes
- `lca-workspace` for root integration after merge

## Runtime Facts

- This repo is distribution-oriented; each skill should stay a thin wrapper over the unified `tiangong` CLI
- If a capability is missing, add it to `tiangong-lca-cli` first, then update the skill wrapper here
- The canonical local validation command is `node scripts/validate-skills.mjs`
- You may pass one or more skill paths to validate only the touched skills

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
