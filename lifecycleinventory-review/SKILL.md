---
name: lifecycleinventory-review
description: "Review process-level lifecycle inventory outputs from local process_from_flow runs. Use when auditing process dataset batches under a run root; `lifecyclemodel` remains reserved."
---

# lifecycleinventory-review

当前只保留 process dataset review。

## Profiles
- `process`（默认）：当前可用，通过统一 CLI 执行 process_from_flow 产物复审。
- `lifecyclemodel`：预留（not implemented yet）。

## 统一入口
使用 `node scripts/run-review.mjs`，通过 `--profile` 选择子能力。

运行模型：

- canonical path 为 `skill -> Node .mjs wrapper -> tiangong review process`
- process profile 不再走 skill 私有 Python/OpenAI 入口
- `lifecyclemodel` profile 继续明确保留为未实现
- 没有 shell 兼容壳

### 默认 profile
若未显式传入 `--profile`，默认使用 `process`。

## process profile
使用 `tiangong review process` 执行 process 维度复审：
- 输入：`--run-root --run-id --out-dir [--start-ts] [--end-ts] [--logic-version] [--enable-llm] [--llm-model] [--llm-max-processes]`
- 输出：
  - `one_flow_rerun_timing.md`
  - `one_flow_rerun_review_v2_1_zh.md`
  - `one_flow_rerun_review_v2_1_en.md`
  - `flow_unit_issue_log.md`
  - `review_summary_v2_1.json`
  - `process-review-report.json`

## 运行示例
```bash
node scripts/run-review.mjs \
  --profile process \
  --run-root /path/to/artifacts/process_from_flow/<run_id> \
  --run-id <run_id> \
  --out-dir /home/huimin/.openclaw/workspace/review \
  --start-ts 2026-02-22T16:01:51+00:00 \
  --end-ts 2026-02-22T16:21:40+00:00
```

## 后续扩展
- flow review 已完全移出本 skill，由 `flow-governance-review` 单独承担。
- `profiles/lifecyclemodel`：沉淀 lifecycle model 维度复审规则与 CLI 子命令。
当前 `lifecyclemodel` profile 调用会返回 “not implemented yet”。
