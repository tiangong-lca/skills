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
  - Global scope installs into `~/<agent>/skills/`.

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

## Execution note

Skills in this repository are expected to be thin wrappers over the unified `tiangong` CLI.

Current rules:

- keep `tiangong-lca-cli` available locally
- or set `TIANGONG_LCA_CLI_DIR` to point at that repo
- use native cross-platform Node `.mjs` wrappers as the canonical entrypoint
- do not keep business Python runtimes, shell shims, MCP transports, or private env parsers inside skills
- if a capability is missing, add a native `tiangong <noun> <verb>` command first, then update the skill to call it
