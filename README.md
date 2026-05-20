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
  - README.md
  - README.zh-CN.md
  - scripts/lib/cli-launcher.mjs
  - scripts/validate-skills.mjs
  - "*/SKILL.md"
  - "*/scripts/**"
lastReviewedAt: 2026-05-08
lastReviewedCommit: 83749eb1836f7d64a4cf59c21d46200baefbae7c
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

## Validation
- Validate the canonical CLI-backed wrappers and migration doc guards locally:
  ```bash
  node scripts/validate-skills.mjs
  ```
- Validate against an unpublished local CLI working tree:
  ```bash
  TIANGONG_LCA_CLI_DIR=/path/to/tiangong-lca-cli \
  node scripts/validate-skills.mjs
  ```
- Validate only the skills you changed:
  ```bash
  node scripts/validate-skills.mjs lifecycleinventory-review process-hybrid-search
  ```
- CI runs the same validation script in `.github/workflows/validate-skills.yml` after checking out and building `tiangong-lca-cli`.

## Execution note

Skills in this repository are expected to be thin wrappers over the unified `tiangong` CLI.

Current rules:

- wrappers auto-discover a local sibling CLI checkout first when `../tiangong-lca-cli` or `../tiangong-cli` exists
- otherwise wrappers fall back to the published CLI through `npm exec --yes --package=@tiangong-lca/cli@latest -- tiangong`
- use `--cli-dir` or `TIANGONG_LCA_CLI_DIR` to force a specific local CLI working tree during dev/CI
- for remote process review snapshots, prefer `tiangong process list --json` followed by `review process --rows-file ...` instead of ad hoc bridge scripts
- use native cross-platform Node `.mjs` wrappers as the canonical entrypoint
- skill wrappers should not bundle business-specific Python runtimes, shell shims, MCP transports, or private env parsers
- if a capability is missing, add a native `tiangong <noun> <verb>` command first, then update the skill to call it
