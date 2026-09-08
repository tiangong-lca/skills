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
  - .claude-plugin/marketplace.json
  - .gitattributes
  - AGENTS.md
  - .docpact/config.yaml
  - .github/workflows/ai-doc-lint.yml
  - scripts/validate-skills.mjs
  - scripts/check-toolchain.mjs
  - scripts/lib/cli-launcher.mjs
  - package.json
  - pnpm-lock.yaml
  - test/**
  - "*/SKILL.md"
  - "*/agents/openai.yaml"
  - .githooks/pre-push
  - scripts/docpact
  - scripts/docpact-gate.sh
  - scripts/install-git-hooks.sh
lastReviewedAt: 2026-09-08
lastReviewedCommit: 123dacb4c1167c8cff6c64fbf438969f77685669
lastReviewedNote: "Reviewed for Skills #94: list the ordinary Foundry entry first in the existing marketplace group, retain every previous membership, and keep semantic authoring internal/on-demand. Both README languages agree. Catalogue source validation passes; actual F1 lock and four-platform public installation remain pending."
related:
  - AGENTS.md
  - .docpact/config.yaml
  - docs/agents/repo-architecture.md
---

# skills Validation Guide

Install and validate with the exact repository toolchain:

```bash
pnpm install --frozen-lockfile
pnpm prepush:gate
```

Review note, 2026-06-04: external runtime source-evidence skill guidance remains documentation and instruction-layer work. The new top-level Foundry scenario skills are included in `scripts/validate-skills.mjs`; no new runtime validator path is required because the external Tiangong KB skill is not checked into this repository.
Review note, 2026-08-25: validation now covers the immutable pnpm/Node/CLI consumer contract, exact published and local CLI cases, argv-only dispatch, and live CLI help checks for every flow-governance wrapper manifest route.

The local `pre-push` hook runs docpact first, validates Node `24.19.0` / pnpm `11.24.0`, installs Skills from its frozen lockfile, and defaults to published CLI `0.1.12`. A local `tiangong-lca-cli` is installed/built only when explicitly selected and only after package/engine/lock evidence succeeds. The hook then runs the repository test/validation gate. The GitHub `validate-skills` workflow is manual-dispatch only, so ordinary pushes rely on the local gate.

You may pass one or more skill directories to validate only the touched skill packages.

## Required Validation Shape

- Skill instruction changes require validating the touched skill package.
- Wrapper contract changes require checking the paired `agents/openai.yaml` and `SKILL.md` together.
- Validation-script or test changes require running the full `pnpm prepush:gate` command when feasible.
- New CLI-backed skills must be added to the default validation list when they are intended to ship as part of the standard checked-in skill set.
- Wrapper-launcher changes require `pnpm test:launcher`, the pnpm consumer contract tests, an exact published `@tiangong-lca/cli@0.1.12` help case, and full skill validation against frozen, built CLI merge `6df087b0dda2544a0fd68f2a143559e81d20d60b`.
- Launcher filesystem fixtures and expected paths must use the host `node:path` implementation. A test that passes a synthetic `platform` may validate executable dispatch, but must not combine that target platform with host-resolved fake paths.
- Repo-wide Markdown guards inventory only root-repository Git-tracked `*.md` paths through argv-based `git -C <root> ls-files -z`. Fixture and validator Git children remove inherited repository-location `GIT_*` variables first, so hook context cannot redirect their index or worktree; untracked or nested CI checkouts are not part of the Skills documentation contract.
- Documentation-governance changes require docpact validation.
- Remote-auth instruction changes require `test/oauth-skill-contract.test.mjs`, the repository-wide password-equivalent doc guard, validation against the exact OAuth-capable local CLI, and the full `pnpm prepush:gate` once the published CLI pin is updated.
- First-install bootstrap changes require `test/installed-hybrid-bootstrap.test.mjs`: copy each of the three hybrid-search skill directories into a fresh isolated directory, clear public auth/CLI overrides, run the real pinned published wrapper without login, inspect its Production dry-run, reject an incomplete custom URL, and prove no session file or outside repository launcher is used. Bundled launchers must match the root authority byte-for-byte. A separate human-controlled fresh browser login plus live redacted doctor/read-only search remains release acceptance, never a repository test credential.

## Docpact Validation

Run these commands for governance changes:

```bash
scripts/docpact validate-config --root . --strict
scripts/docpact lint --root . --base origin/main --head HEAD --mode enforce
```

The manual `ai-doc-lint` workflow delegates to the same local docpact gate when remote reproduction is needed.

## Local Docpact Push Gate

Install the versioned local hook once per checkout:

```bash
./scripts/install-git-hooks.sh
```

The `pre-push` hook runs `scripts/docpact-gate.sh`, which delegates CLI lookup to `scripts/docpact` and performs strict config validation plus enforced lint before the push leaves the machine. It then runs `pnpm check:toolchain`, installs Skills with `pnpm install --frozen-lockfile`, and defaults to the exact published CLI. If `TIANGONG_LCA_CLI_DIR` is explicitly set, `scripts/check-toolchain.mjs --cli-dir` verifies package/name/version/engine/lock evidence before the hook permits frozen install/build. It finishes with `pnpm prepush:gate`. The wrapper checks `DOCPACT_BIN`, Cargo install locations, Homebrew install locations, and then `PATH`, so local agent shells should not fail only because bare `docpact` is unavailable. The default comparison base is `origin/main`. Override it for unusual stacks with `DOCPACT_BASE_REF=<ref>` or `scripts/docpact-gate.sh --base <ref>`. The gate writes its detailed report to a temporary file so normal pushes do not create `.docpact/runs/` artifacts.

The semantic-only Foundry authoring package is included in default validation. Validate a copied isolated package for its entry metadata and local reference closure; it intentionally has no wrapper script or bootstrap runtime. Runtime/F1 and ordinary entry bootstrap qualification remain separate requirements of the full migration.

Default validation now includes all 23 source skill directories: the original 21 plus the ordinary Foundry entry and internal authoring package. Validate the entry after an isolated copy and check its task/semantic field contracts against the selected Foundry source. This source check does not replace the pending original-script equality, final F1 lock binding or four-platform public cold/warm/bootstrap qualification.

`test/foundry-bootstrap-package.test.mjs` checks both script SHA-256 values against the immutable C1 source and executes an isolated host bootstrap without its lock to prove refusal before installation or application launch. POSIX syntax is checked with `sh -n`; Run the PowerShell execution case on Windows without execution-policy bypass. These negative/source checks do not count as final public bootstrap qualification.

Marketplace changes must resolve every listed skill to a real package, preserve existing group memberships, and list the ordinary Foundry entry before its internal semantic role. Check the two-language installation guidance together. Catalogue presence alone does not qualify a missing F1 lock or replace four-platform public bootstrap tests.
