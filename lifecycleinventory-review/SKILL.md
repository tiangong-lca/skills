---
name: lifecycleinventory-review
description: "Review process-level or lifecyclemodel-level lifecycle inventory outputs from local TianGong build runs, plus frozen remote TianGong process snapshots. Use when auditing process_from_flow batches, remote process rows fetched through the canonical snapshot wrapper, or lifecyclemodel build artifacts through the unified CLI with reproducible review inputs and outputs."
---

# lifecycleinventory-review

当前保留 `process` 和 `lifecyclemodel` 两个 CLI-backed review profile；规则文档按 profile 收敛在各自的 `profiles/*/references/` 下。

## Profiles
- `process`（默认）：通过统一 CLI 执行 process_from_flow 产物复审。
- `lifecyclemodel`：通过统一 CLI 执行 lifecyclemodel build run 复审。

## 统一入口
- `node scripts/run-review.mjs`：本地 `process_from_flow` / `lifecyclemodel` review 入口，通过 `--profile` 选择子能力。
- `node scripts/run-remote-process-review.mjs`：远端 `process` snapshot freeze + review 的推荐 wrapper。

运行模型：

- canonical path 为 `skill -> Node .mjs wrapper -> tiangong review process | review lifecyclemodel`
- 两个 profile 都不再走 skill 私有 Python / OpenAI 入口
- 没有 shell 兼容壳

### 默认 profile
若未显式传入 `--profile`，默认使用 `process`。

## process profile
使用 `tiangong review process` 执行 process 维度复审：
- 统一规则与输出合同见 `profiles/process/references/process-review-rules.md`。
- 当前默认 process rubric 已移除 ILCD taxonomy 分类映射语义检查，但会检查 schema-required 结构完整性；重点放在 schema-required 字段存在性、命名、dataset type / flow 引用结构完整性、定量参考、单位与平衡、代表性、前景系统语境和工具生成语言清理。
- 命名检查默认按 `name.baseName`、`name.treatmentStandardsRoutes`、`name.mixAndLocationTypes`、`name.functionalUnitFlowProperties` 四字段拆分执行；其中 `baseName` / `treatmentStandardsRoutes` / `mixAndLocationTypes` 作为 schema-required 字段必须保留，禁止把整串 reference flow short description 直接塞进 `baseName`，也不要把 required 键省略掉。
- 当任务是复审当前认证可访问的远端 process（例如 `state_code=0/100`）时：
  - 先选择一个输出目录并冻结远端 snapshot。
  - canonical 路径是优先使用 `node scripts/run-remote-process-review.mjs --out-dir ... --list ... [--review ...]`。
  - 该 wrapper 会冻结 `tiangong process list --json` 的原始报告到 `inputs/process-list-report.json`，并额外落盘 `inputs/processes.snapshot.rows.jsonl` 后再调用 `node scripts/run-review.mjs --profile process --rows-file ...`。
  - 当 `tiangong process list` 的现有过滤维度不足以表达任务（例如缺少某类反向引用或更复杂筛选）时，可使用补充 bridge，并在 review 备注中记录该限制。
- 输入：`(--rows-file | --run-root) --out-dir [--run-id] [--start-ts] [--end-ts] [--logic-version] [--enable-llm] [--llm-model] [--llm-max-processes]`
- 输出：
  - `review-input-summary.json`
  - `review-input/materialization-summary.json`（仅 `--rows-file` 模式）
  - `one_flow_rerun_timing.md`
  - `one_flow_rerun_review_v2_1_zh.md`
  - `one_flow_rerun_review_v2_1_en.md`
  - `flow_unit_issue_log.md`
  - `review_summary_v2_1.json`
  - `process-review-report.json`

## lifecyclemodel profile
使用 `tiangong review lifecyclemodel` 执行 lifecyclemodel 维度复审：
- 输入：`--run-dir --out-dir [--start-ts] [--end-ts] [--logic-version]`
- 输出：
  - `model_summaries.jsonl`
  - `findings.jsonl`
  - `lifecyclemodel_review_summary.json`
  - `lifecyclemodel_review_zh.md`
  - `lifecyclemodel_review_en.md`
  - `lifecyclemodel_review_timing.md`
  - `lifecyclemodel_review_report.json`

## 运行示例
```bash
tiangong process list \
  --state-code 100 \
  --limit 20 \
  --json > /abs/path/process-list-report.json

node scripts/run-review.mjs \
  --profile process \
  --rows-file /abs/path/process-list-report.json \
  --out-dir /abs/path/review

node scripts/run-remote-process-review.mjs \
  --out-dir /abs/path/review-runs/process-review \
  --list --state-code 0 --state-code 100 --all

node scripts/run-remote-process-review.mjs \
  --out-dir /abs/path/review-runs/process-review \
  --list --user-id <owner> --state-code 0 --all \
  --review --logic-version 2026-04-14

node scripts/run-review.mjs \
  --profile process \
  --run-root /path/to/process_from_flow/<run_id> \
  --run-id <run_id> \
  --out-dir /abs/path/review \
  --start-ts 2026-02-22T16:01:51+00:00 \
  --end-ts 2026-02-22T16:21:40+00:00

node scripts/run-review.mjs \
  --profile lifecyclemodel \
  --run-dir /path/to/lifecyclemodel_auto_build/<run_id> \
  --out-dir /abs/path/lifecyclemodel-review
```

## 后续扩展
- flow review 已完全移出本 skill，由 `flow-governance-review` 单独承担。
- `profiles/process/references/process-review-rules.md`：沉淀 process 维度的默认 review rubric、输出合同和远端 snapshot 特殊约束。
- `profiles/lifecyclemodel`：沉淀 lifecycle model 维度复审规则与 CLI 输出约定。
