---
docType: guide
scope: repo
status: active
authoritative: false
owner: skills
language: zh-CN
whenToUse:
  - when installing TianGong LCA skills with Chinese-language guidance
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
lastReviewedAt: 2026-06-04
lastReviewedCommit: 83749eb1836f7d64a4cf59c21d46200baefbae7c
---

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

## 外部运行时 skills

本仓库只维护 checked-in 的 TianGong LCA workflow skills。变化较快的 Tiangong KB research skills 应在使用项目中运行时解析，不在本仓库镜像。

source-evidence 数据集开发如果需要 SCI 论文证据，使用 `tiangong-ai/skills` 的最新外部 skill：

```bash
npx skills use https://github.com/tiangong-ai/skills --skill tiangong-kb-sci-search --full-depth
```

如确实需要本地项目级安装：

```bash
npx skills add https://github.com/tiangong-ai/skills --skill tiangong-kb-sci-search --agent '*' --yes --full-depth
npx skills update --project --yes
```

消费项目应在任务 artifact 中记录解析到的 upstream ref 和命令。除非所有权边界被明确调整，不要把 `tiangong-kb-*` skill 目录复制到本仓库。

## Foundry top-level workflows

- `$external-dataset-curated-import`：BAFU、USLCI 等结构化 LCA 数据包导入，走 CLI 转换、curation queue `next`/`verify`、子 skill 和发布 handoff gates。
- `$source-evidence-dataset-development`：从 PDF、Word、URL、API、报告、数据库引用或科学文献进行 evidence-driven 数据新增或更新。
- `$dataset-rls-maintenance`：在当前用户 RLS 可见范围内，对历史错误导入数据做清理、删除/退役、引用修复和 redo 计划；只编排 CLI maintenance plan 与 readback verification，不实现私有数据库访问。

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
  node scripts/validate-skills.mjs lifecycleinventory-qa process-hybrid-search
  ```
- CI 会在 `.github/workflows/validate-skills.yml` 中 checkout 并构建 `tiangong-lca-cli`，然后运行同一套校验脚本。

## 执行说明

本仓库中的 skills 已经收敛到统一的 `tiangong-lca` CLI。

当前约定：

- skill wrapper 会优先自动发现本地 sibling CLI checkout：`../tiangong-lca-cli` 或 `../tiangong-cli`
- 如果没有可用的本地 sibling checkout，则回退到已发布 CLI：`npm exec --yes --package=@tiangong-lca/cli@latest -- tiangong-lca`
- 在本地开发或 CI 联调时，也可以使用 `--cli-dir` / `TIANGONG_LCA_CLI_DIR` 强制指向特定的本地 CLI working tree
- 对远端 process QA snapshot，优先使用 `tiangong-lca process list --json` 再配合 `qa process --rows-file ...`，不再鼓励临时 bridge 脚本
- 对新迁移和后续重构的 skill，wrapper 入口优先直接使用原生 Node `.mjs`，不再新增 shell 兼容壳
- skill wrapper 不应再打包业务 Python、MCP transport、私有 env parsing 或 shell shim
- 若能力缺失，先在 `tiangong-lca-cli` 中新增原生 `tiangong-lca <noun> <verb>` 命令，再让 skill 调用它
