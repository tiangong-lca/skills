# 天工 LCA Skills

仓库地址: https://github.com/tiangong-lca/skills

请使用 https://github.com/vercel-labs/skills 提供的 `skills` CLI 来安装、更新和管理这些 skills。

## 安装 CLI
```bash
npm i skills@latest -g
```

## 安装
- 仅列出可用技能（不安装）:
  ```bash
  npx skills add https://github.com/tiangong-lca/skills --list
  ```
- 安装全部技能（默认项目级）:
  ```bash
  npx skills add https://github.com/tiangong-lca/skills
  ```
- 安装指定技能:
  ```bash
  npx skills add https://github.com/tiangong-lca/skills --skill flow-hybrid-search --skill process-hybrid-search
  ```

## 目标 agent 与作用域
- 指定 agent:
  ```bash
  npx skills add https://github.com/tiangong-lca/skills -a codex -a claude-code
  ```
- 全局安装（用户级）:
  ```bash
  npx skills add https://github.com/tiangong-lca/skills -g
  ```
- 作用域说明:
  - 项目级安装到 `./<agent>/skills/`.
  - 全局安装到 `skills` CLI 在当前平台解析出的 agent 用户目录。可通过 `npx skills list` 查看 macOS / Linux / Windows 上的实际路径。

## 安装方式
- 交互式安装可选:
  - Symlink (recommended)
  - Copy

## 更新与确认
- 列出已安装技能:
  ```bash
  npx skills list
  ```
- 检查更新:
  ```bash
  npx skills check
  ```
- 更新全部技能:
  ```bash
  npx skills update
  ```

## 校验
- 本地校验 CLI-backed wrapper 与迁移文档守卫:
  ```bash
  node scripts/validate-skills.mjs
  ```
- 若要联调未发布的本地 CLI working tree:
  ```bash
  TIANGONG_LCA_CLI_DIR=/path/to/tiangong-lca-cli \
  node scripts/validate-skills.mjs
  ```
- 只校验本次变更的 skill:
  ```bash
  node scripts/validate-skills.mjs lifecycleinventory-review process-hybrid-search
  ```
- CI 会在 `.github/workflows/validate-skills.yml` 中 checkout 并构建 `tiangong-lca-cli`，然后运行同一套校验脚本。

## 执行说明

本仓库中的 skills 已经收敛到统一的 `tiangong` CLI。

当前约定：

- skill wrapper 会优先自动发现本地 sibling CLI checkout：`../tiangong-lca-cli` 或 `../tiangong-cli`
- 如果没有可用的本地 sibling checkout，则回退到已发布 CLI：`npm exec --yes --package=@tiangong-lca/cli@latest -- tiangong`
- 在本地开发或 CI 联调时，也可以使用 `--cli-dir` / `TIANGONG_LCA_CLI_DIR` 强制指向特定的本地 CLI working tree
- 对远端 process review snapshot，优先使用 `tiangong process list --json` 再配合 `review process --rows-file ...`，不再鼓励 case-local bridge 脚本
- 对新迁移和后续重构的 skill，wrapper 入口优先直接使用原生 Node `.mjs`，不再新增 shell 兼容壳
- skills 内部不再保留业务 Python、MCP transport、私有 env parsing 或 shell shim
- 若能力缺失，先在 `tiangong-lca-cli` 中新增原生 `tiangong <noun> <verb>` 命令，再让 skill 调用它
