---
name: lifecycleinventory-review
description: "Review lifecycle inventory artifacts through the unified TianGong CLI. Use when auditing process-level outputs from local process_from_flow runs, while preserving one wrapper entrypoint for future lifecyclemodel review."
---

# lifecycleinventory-review

这个 skill 现在是统一 CLI 的薄包装层，不再自带 Python review runtime。

默认行为：

- `--profile process`（默认）委托给 `tiangong review process`
- `--profile lifecyclemodel` 委托给 `tiangong review lifecyclemodel`
- `scripts/run-lifecycleinventory-review.sh` 只是兼容壳，canonical 入口是 Node wrapper

## Profiles
- `process`（默认）：当前可用，执行 process_from_flow 产物复审。
- `lifecyclemodel`：已接到统一 CLI 命令树，但当前仍是 planned contract。

## 统一入口
使用 `scripts/run-lifecycleinventory-review.mjs`，通过 `--profile` 选择子能力。

兼容壳：

- `scripts/run-lifecycleinventory-review.sh`

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

可选语义审核说明：

- 只有显式传入 `--enable-llm` 时才启用
- 使用统一 CLI 的 `TIANGONG_LCA_LLM_*` 环境变量
- 不再使用 skill 私有的 `OPENAI_*`

## lifecyclemodel profile

- 当前只转发到 `tiangong review lifecyclemodel`
- 该 CLI 子命令目前仍是 planned；调用时会返回统一的 not implemented 提示
- 这样做的意义是先固定 skill 的统一入口，而不是继续保留 skill 私有 profile runtime

## 运行示例
```bash
node scripts/run-lifecycleinventory-review.mjs \
  --run-root /path/to/artifacts/process_from_flow/<run_id> \
  --run-id <run_id> \
  --out-dir /home/huimin/.openclaw/workspace/review \
  --start-ts 2026-02-22T16:01:51+00:00 \
  --end-ts 2026-02-22T16:21:40+00:00

node scripts/run-lifecycleinventory-review.mjs \
  --profile lifecyclemodel \
  --help
```

## 后续扩展
- flow review 已完全移出本 skill，由 `flow-governance-review` 和后续 `tiangong review flow` 承担。
- `profiles/lifecyclemodel` 目前只保留 planned 说明，用来描述未来 `tiangong review lifecyclemodel` 的目标范围。
- 该 skill 后续不应再新增 Python review workflow、独立 env parser 或直接 LLM transport。
