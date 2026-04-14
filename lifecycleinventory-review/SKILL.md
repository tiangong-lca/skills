---
name: lifecycleinventory-review
description: "Review process-level or lifecyclemodel-level lifecycle inventory outputs from local TianGong build runs. Use when auditing process_from_flow batches or lifecyclemodel build artifacts through the unified CLI."
---

# lifecycleinventory-review

当前保留 process 和 lifecyclemodel 两个 CLI-backed review profile。

## Profiles
- `process`（默认）：通过统一 CLI 执行 process_from_flow 产物复审。
- `lifecyclemodel`：通过统一 CLI 执行 lifecyclemodel build run 复审。

## 统一入口
使用 `node scripts/run-review.mjs`，通过 `--profile` 选择子能力。

运行模型：

- canonical path 为 `skill -> Node .mjs wrapper -> tiangong review process | review lifecyclemodel`
- 两个 profile 都不再走 skill 私有 Python / OpenAI 入口
- 没有 shell 兼容壳

### 默认 profile
若未显式传入 `--profile`，默认使用 `process`。

## process profile
使用 `tiangong review process` 执行 process 维度复审：
- 输入：`(--rows-file | --run-root) --out-dir [--run-id] [--start-ts] [--end-ts] [--logic-version] [--enable-llm] [--llm-model] [--llm-max-processes]`
- 远端 snapshot 路径：先用 `tiangong process list --json` 生成 process rows snapshot，再把该 JSON 报告或原始 rows JSON/JSONL 直接传给 `--rows-file`
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

node scripts/run-review.mjs \
  --profile process \
  --run-root /path/to/artifacts/process_from_flow/<run_id> \
  --run-id <run_id> \
  --out-dir /abs/path/review \
  --start-ts 2026-02-22T16:01:51+00:00 \
  --end-ts 2026-02-22T16:21:40+00:00

node scripts/run-review.mjs \
  --profile lifecyclemodel \
  --run-dir /path/to/artifacts/lifecyclemodel_auto_build/<run_id> \
  --out-dir /abs/path/lifecyclemodel-review
```

## 后续扩展
- flow review 已完全移出本 skill，由 `flow-governance-review` 单独承担。
- `profiles/lifecyclemodel`：沉淀 lifecycle model 维度复审规则与 CLI 输出约定。
