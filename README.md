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

- wrappers run the published CLI by default through `npx -y @tiangong-lca/cli@latest`
- use `--cli-dir` or `TIANGONG_LCA_CLI_DIR` only to force a local CLI working tree during dev/CI
- use native cross-platform Node `.mjs` wrappers as the canonical entrypoint
- do not keep business Python runtimes, shell shims, MCP transports, or private env parsers inside skills
- if a capability is missing, add a native `tiangong <noun> <verb>` command first, then update the skill to call it
