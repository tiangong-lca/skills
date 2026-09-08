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
  - .gitattributes
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
  - scripts/check-toolchain.mjs
  - scripts/lib/cli-launcher.mjs
  - package.json
  - pnpm-lock.yaml
  - test/**
  - .github/workflows/**
  - .githooks/**
  - scripts/docpact
  - scripts/docpact-gate.sh
  - scripts/install-git-hooks.sh
lastReviewedAt: 2026-09-08
lastReviewedCommit: 5fa2952fccf36d39e33fcb80fa1fbce7e88eaf37
lastReviewedNote: "Reviewed for Skills #94: the isolated legacy PowerShell startup remains unverified at30 seconds while the established C1 cached Windows test completes in about47 seconds overall. The diagnostic now permits a bounded90-second observation and records actual startup duration; noncompletion still fails with the host control. No interpreter, script, trust guard or final qualification requirement is changed."
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

Review note, 2026-06-02: dataset import curation queue guidance remains skill instruction only; CLI and Foundry own queue construction, curation package assembly, and deterministic gates.
Review note, 2026-08-29: CLI-backed package execution is pinned to Node 24.19.0, pnpm 11.24.0, and published CLI 0.1.3; external Vercel `npx skills` commands remain outside this package-manager migration.
Review note, 2026-09-01: Skills #89 advances the active wrapper contract to published OAuth-only CLI 0.1.7 and immutable release merge `cb5be8f1e209f69570f4c7ef4ef29d61af52eed7`; no wrapper floats through `latest`, discovers a sibling checkout, owns authentication logic, or provisions a password-encoded credential.

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
- `package.json`, `pnpm-lock.yaml`, and the shared CLI launcher/toolchain checks used by repo validation
- `README.md` and `README.zh-CN.md` for install and usage guidance

This repo does not own:

- the public CLI command surface
- external fast-moving source-evidence research skills such as `tiangong-kb-sci-search`
- product runtime business logic
- workspace integration state after merge

Route those tasks to:

- `tiangong-lca-cli` for new native `tiangong-lca <noun> <verb>` commands
- `tiangong-ai/skills` consumed through `npx skills` for external Tiangong KB research skills
- the owning product/runtime repo for business logic or API changes
- `lca-workspace` for root integration after merge

## Runtime Facts

- Repo-local documentation governance is encoded in `.docpact/config.yaml` and enforced locally by the pre-push docpact gate; `.github/workflows/ai-doc-lint.yml` is manual-dispatch fallback.
- This repo is distribution-oriented; each skill should stay a thin wrapper over the unified CLI or a data-only semantic role over current Foundry work items
- If a capability is missing, add it to `tiangong-lca-cli` first, then update the skill wrapper here
- Current-account dataset review skills may orchestrate frozen local inputs through public CLI commands, but must not own direct database access, credential parsing, or private account runtime logic.
- Active remote skills must check `tiangong-lca auth status --json`, hand `auth login` to a human-controlled trusted terminal when required, and use `auth doctor-auth` before account-sensitive commits. They never collect or emit usernames, passwords, authorization codes, tokens, or legacy API keys.
- Official Production public configuration belongs to the CLI and needs no Skills env setup. Only complete custom project URL/key/client/callback tuples override it; headless tokens require an explicit destination/key and must not implicitly select Production.
- The three independently installable hybrid-search packages carry byte-identical `scripts/lib/cli-launcher.mjs` bundles inside their own directories. The root launcher remains the only source authority; update the bundles together and retain the isolated copied-package plus byte-equality regression. Do not add authentication or transport logic to those bundles.
- Headless tokens are orchestrator-injected, short-lived, and absent from argv/prompts/logs/artifacts. Multi-account work uses a distinct private `TIANGONG_LCA_SESSION_FILE` per account/project/client and preserves expected identity evidence.
- Source-evidence import skills may instruct agents to resolve external research skills with `npx skills`, but this repository should not mirror or pin those external skill packages.
- `external-dataset-curated-import`, `source-evidence-dataset-development`, and `dataset-rls-maintenance` are top-level workflow skills only; executable conversion, queue state, validation, QA, write/delete/redo, and verify behavior stays in CLI/Foundry-owned commands.
- Dataset maintenance under user RLS must use CLI-owned maintenance plans and readback verification. Skills must not add direct Supabase CRUD, service-role paths, or broad delete filters.
- Node package execution is pinned to Node `24.19.0` and pnpm `11.24.0`; the default runtime is the exact published `@tiangong-lca/cli@0.1.11` and must never float through `@latest`.
- Never auto-discover or execute a sibling CLI checkout. Local execution is opt-in only through `--cli-dir` or `TIANGONG_LCA_CLI_DIR`; `--published-cli` explicitly overrides a local CLI environment.
- Local CLI checkouts selected by wrappers must match the pinned CLI package/engine/lockfile evidence. When their source is newer than `dist/src/main.js`, wrappers install with `pnpm install --frozen-lockfile` before `pnpm run build`; wrappers should still keep the CLI command surface in `tiangong-lca-cli`.
- CLI child processes use authoritative argv arrays with `shell: false` and preserve child exit/stdout/stderr.
- The canonical local validation command is `pnpm validate` after `pnpm install --frozen-lockfile`.
- You may pass one or more skill paths to validate only the touched skills
- For documentation-governance changes, run `scripts/docpact validate-config --root . --strict` and `scripts/docpact lint --root . --base origin/main --head HEAD --mode enforce`

## Hard Boundaries

- Do not add private business runtimes, MCP transports, or unrelated orchestration layers inside a skill when the behavior should live in the CLI or an owning repo
- Do not add legacy API-key flags, API-key assignment examples, or raw Authorization bearer examples to active skill instructions. The CLI has no legacy bootstrap or rollback mode.
- Do not vendor external runtime skills from `tiangong-ai/skills`; consuming projects should resolve them with `npx skills` and record the resolved upstream ref in task artifacts
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

The `pre-push` hook runs `scripts/docpact-gate.sh`, which delegates CLI lookup to `scripts/docpact` and performs strict config validation plus enforced lint before the push leaves the machine. It validates Node `24.19.0` / pnpm `11.24.0` and installs Skills from its frozen lockfile. The hook defaults to exact published CLI `0.1.11`; when `TIANGONG_LCA_CLI_DIR` is explicitly set, it validates that checkout's package/name/version/engine/lock evidence before any local install or build. It then runs `pnpm prepush:gate`. The wrapper checks `DOCPACT_BIN`, Cargo install locations, Homebrew install locations, and then `PATH`, so local agent shells should not fail only because bare `docpact` is unavailable. The default comparison base is `origin/main`. Override it for unusual stacks with `DOCPACT_BASE_REF=<ref>` or `scripts/docpact-gate.sh --base <ref>`. The gate writes its detailed report to a temporary file so normal pushes do not create `.docpact/runs/` artifacts. The GitHub `validate-skills` workflow is manual-dispatch only.

`foundry-tidas-authoring` is an internal on-demand semantic package. It reads supplied current task/context evidence and returns decision/patch files; it owns no runtime, credential parsing, deterministic apply or database operation. Its explicit-only policy is part of the approved Foundry entry migration.

`foundry-tidas-import` is the ordinary TianGong Foundry entry. Its instructions consume the public task protocol and current structured actions; the internal authoring role stays on demand. Source validation does not establish installability: final bundled bootstrap/lock, marketplace recommendation and four-platform public qualification require the actual F1 release.
