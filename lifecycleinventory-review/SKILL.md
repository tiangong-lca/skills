---
name: lifecycleinventory-review
description: "Review process-level lifecycle inventory outputs from local process_from_flow runs. Use when auditing process dataset batches under a run root; `lifecyclemodel` remains reserved."
---

# lifecycleinventory-review

当前只保留 process dataset review。

## Profiles
- `process`（默认）：当前可用，执行 process_from_flow 产物复审。
- `lifecyclemodel`：预留（not implemented yet）。

## 统一入口
使用 `scripts/run_review.py`，通过 `--profile` 选择子能力。

### 默认 profile
若未显式传入 `--profile`，默认使用 `process`。

## process profile
使用 `profiles/process/scripts/run_process_review.py` 执行 process 维度复审：
- 输入：`--run-root --run-id --out-dir [--start-ts] [--end-ts] [--logic-version] [--enable-llm] [--llm-model] [--llm-max-processes]`
- 输出：
  - `one_flow_rerun_timing.md`
  - `one_flow_rerun_review_v2_1_zh.md`
  - `one_flow_rerun_review_v2_1_en.md`
  - `flow_unit_issue_log.md`

## 运行示例
```bash
python scripts/run_review.py \
  --profile process \
  --run-root /path/to/artifacts/process_from_flow/<run_id> \
  --run-id <run_id> \
  --out-dir /home/huimin/.openclaw/workspace/review \
  --start-ts 2026-02-22T16:01:51+00:00 \
  --end-ts 2026-02-22T16:21:40+00:00
```

## 后续扩展
- flow review 已完全移出本 skill，由 `flow-governance-review` 单独承担。
- `profiles/lifecyclemodel`：沉淀 lifecycle model 维度复审规则与脚本。
当前 `lifecyclemodel` profile 调用会返回 “not implemented yet” 并提示下一步。
