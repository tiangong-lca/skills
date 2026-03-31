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
  - 全局安装到 `~/<agent>/skills/`.

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
- 只校验本次变更的 skill:
  ```bash
  node scripts/validate-skills.mjs lifecycleinventory-review process-hybrid-search
  ```
- CI 会在 `.github/workflows/validate-skills.yml` 中 checkout 并构建 `tiangong-lca-cli`，然后运行同一套校验脚本。

## 执行说明

本仓库中的 skills 已经收敛到统一的 `tiangong` CLI。

当前约定：

- 本地保留 `tiangong-lca-cli` 仓库
- 或通过 `TIANGONG_LCA_CLI_DIR` 指向该仓库
- skill wrapper 统一委托 `bin/tiangong.js` 执行，而不是继续各自维护一套 `curl` 逻辑
- 对新迁移和后续重构的 skill，wrapper 入口优先直接使用原生 Node `.mjs`，不再新增 shell 兼容壳
- skills 内部不再保留业务 Python、MCP transport、私有 env parsing 或 shell shim
- 若能力缺失，先在 `tiangong-lca-cli` 中新增原生 `tiangong <noun> <verb>` 命令，再让 skill 调用它
