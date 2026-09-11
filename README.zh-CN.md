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

## TianGong Foundry

外部数据包导入、源证据数据开发和继续已有 Foundry 任务，统一使用 `$foundry-tidas-import` 作为日常入口。它通过随包 bootstrap 选择合格运行时，将任务输出放在独立可写工作区。

```bash
npx skills add https://github.com/tiangong-lca/skills --skill foundry-tidas-import
```

`lca-foundry-workflows` marketplace 包首先列出此入口。`foundry-tidas-authoring` 仅在当前语义工作项需要时加载，是内部角色；日常入口也能直接使用运行时提供的工作项说明，无须依赖另一个已安装技能目录。

完整入口随包提供 [Foundry 0.1.8](https://github.com/tiangong-lca/data-foundry/releases/tag/foundry-runtime-v0.1.8) 的最终发行锁，固定 CLI 0.1.14、Node 24.19.0 和 TIDAS 0.3.0。公开运行时已通过 macOS arm64、Linux x64/arm64 和 Windows x64 验证。安装或复制入口时，保持随包脚本与相邻 lock 完整。安装、登录不授予数据写入权限；继续执行任务当前的授权与恢复动作。

### 专项工作流

原有技能保留各自用途：

- `$external-dataset-curated-import`：BAFU、USLCI 等结构化 LCA 数据包导入，走 CLI 转换、curation queue `next`/`verify`、子 skill 和发布 handoff gates。
- `$source-evidence-dataset-development`：从 PDF、Word、URL、API、报告、数据库引用或科学文献进行 evidence-driven 数据新增或更新。
- `$dataset-rls-maintenance`：在当前用户 RLS 可见范围内，对历史错误导入数据做清理、删除/退役、引用修复和 redo 计划；只编排 CLI maintenance plan 与 readback verification，不实现私有数据库访问。

## 远程认证

远程 skill 统一使用 CLI 管理的 Supabase OAuth session。官方 Production 不需要 public 环境变量、Dashboard 或额外索取 client ID；公开 URL/key/client/callback profile 由 CLI 唯一维护，Skills 不复制。先运行：

```bash
pnpm dlx --package=@tiangong-lca/cli@0.1.14 tiangong-lca auth status --json
```

若结果是 `login-required`，停止 agent workflow，把可信终端交给人类运行 `tiangong-lca auth login`。skill/agent 不得索取用户名、密码、authorization code、access token、refresh token 或旧编码 API key。账号敏感读取和 commit 前运行 `tiangong-lca auth doctor-auth --json`。

只有自定义环境才需要完整匹配的 `TIANGONG_LCA_API_BASE_URL`、`TIANGONG_LCA_SUPABASE_PUBLISHABLE_KEY`、已注册的 `TIANGONG_LCA_OAUTH_CLIENT_ID` 及精确 callback；不完整配置不能混补 Production。每个 account/project/client 使用独立私有 `TIANGONG_LCA_SESSION_FILE`。批准的 headless 自动化必须显式配置目标 URL/key 和 `TIANGONG_LCA_AUTH_MODE=access-token`，再由 orchestrator 注入短期 `TIANGONG_LCA_ACCESS_TOKEN`；仅 token 不会默认指向 Production，也不得进入 argv、prompt、日志或产物。CLI 不再存在 legacy API-key 模式。

三个 hybrid-search 技能目录可以独立安装：各自携带经过字节一致性校验的启动器和示例输入，不需要 Skills/CLI/Data Foundry checkout；认证仍完全由 CLI 负责。

## 校验

- 仓库校验固定使用 Node `24.19.0` 与 pnpm `11.24.0`；先从 `pnpm-lock.yaml` 安装校验包：
  ```bash
  pnpm install --frozen-lockfile
  ```
- 本地校验 CLI-backed wrapper 与迁移文档守卫:
  ```bash
  pnpm validate
  ```
- 若要联调未发布的本地 CLI working tree:
  ```bash
  TIANGONG_LCA_CLI_DIR=/path/to/tiangong-lca-cli \
  pnpm validate
  ```
- 只校验本次变更的 skill:
  ```bash
  pnpm validate lifecycleinventory-qa process-hybrid-search
  ```
- CI 会在 `.github/workflows/validate-skills.yml` 中 checkout CLI `0.1.14` 的不可变 merge/tag commit `bcc5dbee5b909dbb912e09d99ca07e858d3d7cec`，用 frozen pnpm lockfile 安装两个仓库并构建 CLI，然后运行同一套校验。

## 执行说明

本仓库中的 skills 已经收敛到统一的 `tiangong-lca` CLI。

当前约定：

- skill wrapper 默认使用精确版本的已发布 CLI：`pnpm dlx --package=@tiangong-lca/cli@0.1.14 tiangong-lca`；不会自动发现任何 sibling 目录
- 本地执行只能通过 `--cli-dir` / `TIANGONG_LCA_CLI_DIR` 显式启用
- 使用 `--published-cli` 可覆盖本地 CLI 环境并显式执行 published-package case；嵌套 wrapper 会继续传播该选择
- 本地 CLI override 必须是带精确 Node/pnpm engines 和 v9 `pnpm-lock.yaml` 的 `@tiangong-lca/cli@0.1.14`；本地 build 过期时先执行 `pnpm install --frozen-lockfile`，再执行 `pnpm run build`
- 本地 pre-push hook 会先验证显式 local checkout 的 package/lock evidence，再允许 install 或 build
- launcher 只用 argv 数组并固定 `shell: false`，因此带空格路径保持为单个参数，并原样保留子进程 exit/stdout/stderr
- 对远端 process QA snapshot，优先使用 `tiangong-lca process list --json` 再配合 `qa process --rows-file ...`，不再鼓励临时 bridge 脚本
- 对新迁移和后续重构的 skill，wrapper 入口优先直接使用原生 Node `.mjs`，不再新增 shell 兼容壳
- skill wrapper 不应再打包业务 Python、MCP transport、私有 env parsing 或 shell shim
- 远程 skill 必须使用 CLI OAuth status/login/doctor handoff，不得新增 API-key flag 或 bearer 示例
- 若能力缺失，先在 `tiangong-lca-cli` 中新增原生 `tiangong-lca <noun> <verb>` 命令，再让 skill 调用它

## Foundry 语义工作

`foundry-tidas-authoring` 是面向具体 Foundry 工作项的内部按需角色。它读取已提供的完整上下文，将有证据的决定或 patch 文件交回调用流程；不安装运行时、不管理认证、不直接应用行数据或操作数据库。
