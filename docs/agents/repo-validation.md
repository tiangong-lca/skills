---
title: skills Validation Guide
docType: guide
scope: repo
status: active
authoritative: true
owner: skills
language: en
whenToUse:
  - when validating changed skills, wrappers, packaging rules, or documentation governance
  - when selecting proof for a skills repository PR
whenToUpdate:
  - when skill validation commands change
  - when wrapper or packaging proof expectations change
  - when docpact governance rules or CI behavior change
checkPaths:
  - AGENTS.md
  - .docpact/config.yaml
  - .github/workflows/ai-doc-lint.yml
  - scripts/validate-skills.mjs
  - test/**
  - "*/SKILL.md"
  - "*/agents/openai.yaml"
  - .githooks/pre-push
  - scripts/docpact-gate.sh
  - scripts/install-git-hooks.sh
lastReviewedAt: 2026-05-08
lastReviewedCommit: 813ad4c1cddcae0eb57116e7ae9bd2da2854c115
related:
  - AGENTS.md
  - .docpact/config.yaml
  - docs/agents/repo-architecture.md
---

# skills Validation Guide

The canonical local validation command is:

```bash
node scripts/validate-skills.mjs
```

You may pass one or more skill directories to validate only the touched skill packages.

## Required Validation Shape

- Skill instruction changes require validating the touched skill package.
- Wrapper contract changes require checking the paired `agents/openai.yaml` and `SKILL.md` together.
- Validation-script or test changes require running the full `node scripts/validate-skills.mjs` command when feasible.
- Documentation-governance changes require docpact validation.

## Docpact Validation

Run these commands for governance changes:

```bash
docpact validate-config --root . --strict
docpact lint --root . --base origin/main --head HEAD --mode enforce
```

The repository PR workflow runs the same docpact config validation and PR-shaped lint gate.

## Local Docpact Push Gate

Install the versioned local hook once per checkout:

```bash
./scripts/install-git-hooks.sh
```

The `pre-push` hook runs `scripts/docpact-gate.sh`, which performs strict config validation and `docpact lint --mode enforce` before the push leaves the machine. The default comparison base is `origin/main`. Override it for unusual stacks with `DOCPACT_BASE_REF=<ref>` or `scripts/docpact-gate.sh --base <ref>`. The gate writes its detailed report to a temporary file so normal pushes do not create `.docpact/runs/` artifacts.
