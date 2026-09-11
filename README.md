---
docType: guide
scope: repo
status: active
authoritative: false
owner: skills
language: en
whenToUse:
  - when installing TianGong LCA skills
  - when checking wrapper execution expectations
whenToUpdate:
  - when skill installation guidance changes
  - when the unified CLI wrapper contract changes
checkPaths:
  - .claude-plugin/marketplace.json
  - README.md
  - README.zh-CN.md
  - scripts/lib/cli-launcher.mjs
  - scripts/validate-skills.mjs
  - "*/SKILL.md"
  - "*/scripts/**"
lastReviewedAt: 2026-09-11
lastReviewedCommit: ee3a4fd16160adfafa8e89cf9ddd8f1f9af1de05
lastReviewedNote: "Reviewed for Skills #98: exact official Foundry 0.1.8 manifest/lock bind release abac241 and qualified CLI 0.1.14. Original bootstrap bytes and current task/reference/native-contract/no-replay boundaries are preserved; actual copied consumer qualification is required on all four platforms."
---

# Tiangong LCA Skills

Repository: https://github.com/tiangong-lca/skills

Use the `skills` CLI from https://github.com/vercel-labs/skills to install, update, and manage these skills.

## Install the CLI

```bash
npm i skills@latest -g
```

## Install

- List available skills (no install):
  ```bash
  npx skills add https://github.com/tiangong-lca/skills --list
  ```
- Install all skills (project scope by default):
  ```bash
  npx skills add https://github.com/tiangong-lca/skills
  ```
- Install specific skills:
  ```bash
  npx skills add https://github.com/tiangong-lca/skills --skill flow-hybrid-search --skill process-hybrid-search
  ```

## Target agents and scope

- Target specific agents:
  ```bash
  npx skills add https://github.com/tiangong-lca/skills -a codex -a claude-code
  ```
- Install globally (user scope):
  ```bash
  npx skills add https://github.com/tiangong-lca/skills -g
  ```
- Scope notes:
  - Project scope installs into `./<agent>/skills/`.
  - Global scope installs into the per-agent user skills directory resolved by the `skills` CLI on the current platform. Use `npx skills list` to inspect the exact path on macOS, Linux, or Windows.

## Install method

- Interactive installs let you choose:
  - Symlink (recommended)
  - Copy

## Update and verify

- List installed skills:
  ```bash
  npx skills list
  ```
- Check for updates:
  ```bash
  npx skills check
  ```
- Update all skills:
  ```bash
  npx skills update
  ```

## External runtime skills

This repository owns checked-in TianGong LCA workflow skills. Fast-moving Tiangong KB research skills are consumed from their owning repositories at runtime instead of being mirrored here.

For source-evidence dataset development that needs SCI paper evidence, resolve the latest external skill from `tiangong-ai/skills`:

```bash
npx skills use https://github.com/tiangong-ai/skills --skill tiangong-kb-sci-search --full-depth
```

Optional local project install:

```bash
npx skills add https://github.com/tiangong-ai/skills --skill tiangong-kb-sci-search --agent '*' --yes --full-depth
npx skills update --project --yes
```

Consuming projects should record the resolved upstream ref and command in task artifacts. Do not copy `tiangong-kb-*` skill folders into this repository unless the ownership boundary changes deliberately.

## TianGong Foundry

Use `$foundry-tidas-import` as the ordinary entry for external dataset packages, source-evidence development and continuing an existing Foundry task. It selects the qualified runtime through its bundled bootstrap and keeps task outputs in a separate writable workspace.

```bash
npx skills add https://github.com/tiangong-lca/skills --skill foundry-tidas-import
```

The `lca-foundry-workflows` marketplace package lists this entry first. `foundry-tidas-authoring` is an internal role loaded only for a current semantic work item; it is not a second task entry. The ordinary entry can use the runtime's work-item instructions when that internal role is not installed.

The complete entry bundles the final release lock for [Foundry 0.1.8](https://github.com/tiangong-lca/data-foundry/releases/tag/foundry-runtime-v0.1.8), with CLI 0.1.14, Node 24.19.0 and TIDAS 0.3.0. Its public runtime is qualified for macOS arm64, Linux x64/arm64 and Windows x64. Keep the bundled scripts and adjacent lock together when installing or copying the entry. Installation and login do not grant permission to write data; continue the task's current authorization and recovery actions.

### Specialized workflows

The existing skills retain their independent uses:

- `$external-dataset-curated-import`: BAFU, USLCI, and other structured LCA package imports through CLI conversion, curation queue `next`/`verify`, child skills, and publish handoff gates.
- `$source-evidence-dataset-development`: evidence-driven data creation or update from PDFs, Word files, URLs, APIs, reports, database references, or scientific literature.
- `$dataset-rls-maintenance`: current-user RLS-scoped cleanup, delete/retire, reference repair, and redo planning for previously imported rows; orchestrates CLI maintenance plans and readback verification without private database access.

## Remote authentication

Remote skills use the CLI-owned Supabase OAuth session. Official Production requires no public environment setup or dashboard/client-ID handoff. The published CLI owns its public URL/key/client/callback profile; Skills do not copy it. Start with:

```bash
pnpm dlx --package=@tiangong-lca/cli@0.1.14 tiangong-lca auth status --json
```

If the result is `login-required`, stop the agent workflow and let the human user run `tiangong-lca auth login` in a trusted terminal. Skills and agents must never request a username, password, authorization code, access token, refresh token, or the deprecated encoded API key. Use `tiangong-lca auth doctor-auth --json` before account-sensitive reads or commits.

Only custom environments need a complete matching `TIANGONG_LCA_API_BASE_URL`, `TIANGONG_LCA_SUPABASE_PUBLISHABLE_KEY`, and registered `TIANGONG_LCA_OAUTH_CLIENT_ID` plus its exact callback. Partial custom settings must not inherit Production fields. Use a separate private `TIANGONG_LCA_SESSION_FILE` for each account/project/client context. Approved headless automation must explicitly set its destination/key and `TIANGONG_LCA_AUTH_MODE=access-token` before the orchestrator injects one short-lived `TIANGONG_LCA_ACCESS_TOKEN`; a token alone never selects Production and must never enter argv, prompts, logs, or artifacts. No legacy API-key mode exists.

The three hybrid-search skill folders are independently installable: each includes the same byte-checked launcher and its example input, without a Skills/CLI/Data Foundry checkout. The CLI remains the authentication authority.

## Validation

- Repository validation requires Node `24.19.0` and pnpm `11.24.0`; install the pinned validation package from `pnpm-lock.yaml` first:
  ```bash
  pnpm install --frozen-lockfile
  ```
- Validate the canonical CLI-backed wrappers and migration doc guards locally:
  ```bash
  pnpm validate
  ```
- Validate against an unpublished local CLI working tree:
  ```bash
  TIANGONG_LCA_CLI_DIR=/path/to/tiangong-lca-cli \
  pnpm validate
  ```
- Validate only the skills you changed:
  ```bash
  pnpm validate lifecycleinventory-qa process-hybrid-search
  ```
- CI runs the same validation in `.github/workflows/validate-skills.yml` after checking out immutable CLI `0.1.14` merge/tag commit `bcc5dbee5b909dbb912e09d99ca07e858d3d7cec`, installing both repositories with frozen pnpm lockfiles, and building the CLI.

## Execution note

Skills in this repository are expected to be thin wrappers over the unified `tiangong-lca` CLI.

Current rules:

- wrappers default to the exact published CLI through `pnpm dlx --package=@tiangong-lca/cli@0.1.14 tiangong-lca`; sibling directories are never auto-discovered
- local execution is opt-in only through `--cli-dir` or `TIANGONG_LCA_CLI_DIR`
- use `--published-cli` to override a local CLI environment for an explicit published-package case; nested wrappers propagate that selection
- local CLI overrides must identify `@tiangong-lca/cli@0.1.14` with its exact Node/pnpm engines and a v9 `pnpm-lock.yaml`; stale local builds are installed with `pnpm install --frozen-lockfile` before `pnpm run build`
- the local pre-push hook validates CLI package and lock evidence before it installs or builds an explicitly selected local checkout
- launcher execution uses argv arrays with `shell: false`, so paths containing spaces remain one argument and child exit/stdout/stderr are preserved
- for remote process QA snapshots, prefer `tiangong-lca process list --json` followed by `qa process --rows-file ...` instead of ad hoc bridge scripts
- use native cross-platform Node `.mjs` wrappers as the canonical entrypoint
- skill wrappers should not bundle business-specific Python runtimes, shell shims, MCP transports, or private env parsers
- remote skill instructions must use CLI OAuth status/login/doctor handoff and must not add API-key flags or bearer examples
- if a capability is missing, add a native `tiangong-lca <noun> <verb>` command first, then update the skill to call it

## Foundry semantic work

`foundry-tidas-authoring` is an internal on-demand role for a concrete Foundry work item. It reads the supplied full context and returns evidence-backed decisions or patch files to the invoking workflow. It does not install a runtime, manage authentication, apply rows or perform database operations.
