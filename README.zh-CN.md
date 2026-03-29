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

## 执行说明

轻量远程 skill 正在逐步收敛到统一的 `tiangong` CLI。

当前约定：

- 本地保留 `tiangong-lca-cli` 仓库
- 或通过 `TIANGONG_LCA_CLI_DIR` 指向该仓库
- skill wrapper 统一委托 `bin/tiangong.js` 执行，而不是继续各自维护一套 `curl` 逻辑

## 迁移矩阵（CLI-first）

| Skill | 当前执行面 | 对应 `tiangong` 子命令 | Python 依赖 | MCP 依赖 | 下一步迁移目标 |
| --- | --- | --- | --- | --- | --- |
| `flow-hybrid-search` | CLI 薄 wrapper | `tiangong search flow` | 否 | 否 | 仅维护文档与示例 |
| `process-hybrid-search` | CLI 薄 wrapper | `tiangong search process` | 否 | 否 | 仅维护文档与示例 |
| `lifecyclemodel-hybrid-search` | CLI 薄 wrapper | `tiangong search lifecyclemodel` | 否 | 否 | 仅维护文档与示例 |
| `embedding-ft` | CLI 薄 wrapper | `tiangong admin embedding-run` | 否 | 否 | 仅维护文档与示例 |
| `lifecyclemodel-resulting-process-builder` | CLI 主链 + skill 薄 wrapper | `tiangong lifecyclemodel build-resulting-process` / `publish-resulting-process` | 否 | 部分（待清理 lookup 遗留） | 去除遗留 lookup 分支并固定为 REST/CLI 语义 |
| `lca-publish-executor` | Node wrapper -> CLI | `tiangong publish run` | 否 | 否 | 继续清理历史兼容参数与文档 |
| `process-automated-builder` | Node wrapper -> CLI + legacy 并存 | `tiangong process auto-build` / `resume-build` / `publish-build` / `batch-build` | 是（legacy） | 是（legacy） | 将剩余 LangGraph/Python 阶段迁入 CLI 模块，并持续缩小 legacy 路径 |
| `lifecyclemodel-automated-builder` | legacy workflow | 规划中（`tiangong lifecyclemodel auto-build|validate-build|publish-build`） | 是 | 是 | 迁为 CLI 主链 |
| `lifecycleinventory-review` | legacy workflow | 规划中（`tiangong review process`） | 是 | 可选 | 迁为 CLI 主链 |
| `flow-governance-review` | legacy workflow | 规划中（`tiangong review flow`） | 是 | 可选 | 迁为 CLI 主链 |
| `lifecyclemodel-recursive-orchestrator` | legacy orchestrator | 规划中（CLI orchestrator） | 是 | 间接 | 子命令稳定后再迁 orchestrator |

说明：
- 目标状态是 `skill -> tiangong`，skills 仓库只保留文档、示例、薄 wrapper。
- MCP 不再是 CLI 默认传输层；优先直接调用 edge functions 或 Supabase 官方 JS SDK。
- 历史 Python/MCP 路径仅用于迁移期兼容，不是目标架构。
