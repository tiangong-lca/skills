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
  - .gitattributes
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
  - scripts/check-toolchain.mjs
  - scripts/lib/cli-launcher.mjs
  - package.json
  - pnpm-lock.yaml
lastReviewedAt: 2026-09-08
lastReviewedCommit: 444bc9ca948361bb7e8a7b5da90be38a70663321
lastReviewedNote: "Reviewed for Skills #94: add a Windows-only early public-download probe that executes unchanged C1 PowerShell function bodies against the official Node24.19.0 checksum file and compares bytes with an independent fetch. This tests the real HTTP response before final F1 publication, not credentials or business data; local non-Windows skips are explicitly distinguished from required Windows proof."
related:
  - AGENTS.md
  - .docpact/config.yaml
  - docs/agents/repo-validation.md
---

# skills Repo Architecture

`tiangong-lca-skills` owns checked-in skill packages and CLI-backed agent workflow wrappers for TianGong workflows.

Review note, 2026-06-02: dataset import curation queue changes keep this repository at the workflow-instruction layer; executable queue and curation gate behavior stays in CLI and Foundry.
Review note, 2026-06-04: Foundry now has two checked-in top-level scenario skills, `external-dataset-curated-import` and `source-evidence-dataset-development`. They coordinate existing CLI/child-skill surfaces and must not grow package parsing, database write, or evidence retrieval implementations.
Review note, 2026-08-25: the repository adds only a pnpm validation package and shared JavaScript launcher contract; it does not add a first-party TypeScript compiler or move CLI business logic into Skills.

## Owned Surfaces

- `*/SKILL.md` contains canonical skill instructions and trigger contracts.
- `*/agents/openai.yaml` contains wrapper contracts used by the skills CLI.
- `*/scripts/**`, `*/references/**`, and `*/assets/**` are skill-local support files intentionally shipped with a skill package.
- `scripts/validate-skills.mjs` and `test/**` define repo-level validation for wrappers and packaging rules.
- `package.json` and `pnpm-lock.yaml` pin the validation-only Node `24.19.0` / pnpm `11.24.0` package contract; this does not turn the skill packages into a TypeScript runtime.
- `scripts/lib/cli-launcher.mjs` owns exact local/published CLI selection, package evidence checks, frozen local preparation, and argv-only process dispatch.
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

Remote authentication follows the same boundary. Skills may invoke `auth status`, instruct a human to run browser `auth login`, and require live redacted `doctor-auth`, but they never inspect session files or handle passwords, codes, or tokens. Explicit headless and multi-account configuration remain CLI/orchestrator responsibilities. The validator rejects password-equivalent invocation examples in active Markdown.

Official Production's public profile is owned and bundled by the CLI, not copied into Skills. A first install needs human browser login, not dashboard/client-ID setup. Complete custom environments and explicitly targeted headless tokens remain separate. The flow/process/lifecyclemodel hybrid-search packages bundle the root launcher byte-for-byte at their own `scripts/lib/cli-launcher.mjs` so copied individual installations do not import outside their package. The installed-consumer test checks bundle equality, a real published-CLI dry-run, no local checkout/session creation, and rejection of incomplete custom destinations.

If a Foundry/source-evidence workflow needs an external Tiangong KB research skill, consume it with `npx skills` at runtime and record the resolved upstream ref in the task workspace. Do not copy the external skill package into this repository unless ownership is intentionally transferred.

Current-account dataset review is owned here only as a skill package and wrapper contract. Its durable runtime behavior belongs in public `tiangong-lca` CLI commands such as dataset validation, reference rewriting, lifecyclemodel save-draft, and lifecyclemodel graph export.

The shared wrapper launcher defaults to pinned published `@tiangong-lca/cli@0.1.11` and never discovers sibling directories. CI binds immutable release merge `1f9f75fcae3c386e601b49a7da95df0d6a526f6f`. An explicit `--cli-dir` or `TIANGONG_LCA_CLI_DIR` may select an exact matching local checkout; only after package/engine/lock evidence passes may the launcher prepare it with `pnpm install --frozen-lockfile` and `pnpm run build` when source files are newer than `dist/src/main.js`. All execution stays argv-authoritative with `shell: false`, using native `pnpm.exe` on Windows. This is a developer-experience guard for stale local checkouts, not permission for skills to duplicate CLI implementation.

## Integration Semantics

A merged PR in this repository is repo-complete only. If the updated skill set must ship through the workspace, root integration must deliberately update the `tiangong-lca-skills` submodule pointer after merge.

## Local Docpact Push Gate

This repository has a versioned local `pre-push` hook under `.githooks/pre-push` that delegates to `scripts/docpact-gate.sh`, resolves the docpact CLI through `scripts/docpact`, installs Skills through its frozen pnpm lockfile, and defaults validation to the published CLI. It installs/builds a local CLI only when explicitly selected and only after launcher-owned package evidence validation. The hook then runs `pnpm prepush:gate`; it is the local guard for docpact config validation, enforced doc-governance linting, toolchain tests, and skill validation. The GitHub `validate-skills` workflow is manual-dispatch only.

The internal `foundry-tidas-authoring` package is data-only guidance over an existing Foundry work item. Its local semantic reference is self-contained; execution, artifact registration, current authorization and readback remain Foundry/CLI responsibilities. The original workflow skill purposes remain separate.

`foundry-tidas-import` owns ordinary task instructions for both packaged imports and source-evidence development. Its self-contained public workflow reference describes task/semantic selection and current-action consumption; it carries no state machine or database logic. A missing internal authoring skill can be handled from the runtime-generated work item and template without a sibling-directory dependency. Final bootstrap data and marketplace activation remain dependent on actual F1.

The Foundry entry distributes byte-identical POSIX/PowerShell bootstrap scripts from CLI `cli-v0.1.11` commit `1f9f75fcae3c386e601b49a7da95df0d6a526f6f`. Skills owns their packaging, not their implementation. Path-specific Git attributes prevent checkout newline conversion. The original scripts select only their adjacent lock; the final F1 lock remains absent until its public release is independently qualified.

The independently copied Foundry entry retains the original C1 copyright/license text at `assets/licenses/tiangong-cli-LICENSE` beside its bundled scripts. Its bytes are checked with the script pins; this adds no user-facing license command or confirmation flow.
