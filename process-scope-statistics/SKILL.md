---
name: process-scope-statistics
description: "Delegate to the TianGong CLI to snapshot visible or owner-filtered process rows and compute repeatable coverage statistics for domains, crafts/routes, unit-process rows, and products. Use when you need stable process-scope metrics, especially for `state_code=0/100` maintenance planning."
---

# process-scope-statistics

Use this skill when the task is to answer questions like:
- “这些 process 覆盖多少领域/工艺/产品？”
- “当前可见的 `state_code=0,100` process 有多少单元过程、多少产品？”
- “给我一个以后可重复跑的 process scope 统计方案”

## Workflow

1. Create an output directory first.
2. Ensure `TIANGONG_LCA_*` is already exported or `.env` is available in the current working directory.
3. Run `node scripts/run-process-scope-statistics.mjs`.
4. Review the JSON summaries and Markdown reports in that output directory.
5. If you need a Chinese report, keep the `.zh-CN.md` output.

## Default Statistical Scope

Unless the user says otherwise, use:
- `--scope visible`
- `--state-codes 0,100`

`visible` means “all process rows visible to the current authenticated session”.
In practice this typically includes:
- draft rows owned by the current account, such as `state_code=0`
- publicly visible rows such as `state_code=100`

## Canonical Script

```bash
node scripts/run-process-scope-statistics.mjs \
  --out-dir /abs/path/process-scope-stats \
  --scope visible \
  --state-code 0 \
  --state-code 100
```

Useful options:
- `--scope current-user`
- `--state-code 0`
- `--state-code 100`
- `--page-size 200`
- `--reuse-snapshot`

The wrapper delegates to the canonical CLI command:

```bash
tiangong process scope-statistics --out-dir /abs/path/process-scope-stats --scope visible --state-code 0 --state-code 100
```

Compatibility note:
- `--state-codes 0,100` is still accepted as a wrapper/CLI alias, but repeatable `--state-code` flags are the canonical shape.

## Outputs

- `inputs/processes.snapshot.manifest.json`
- `inputs/processes.snapshot.rows.jsonl`
- `outputs/process-scope-summary.json`
- `outputs/domain-summary.json`
- `outputs/craft-summary.json`
- `outputs/product-summary.json`
- `outputs/type-of-dataset-summary.json`
- `reports/process-scope-statistics.md`
- `reports/process-scope-statistics.zh-CN.md`

## Metric Definitions

Read `references/metric-definitions.md` when you need the exact counting rules for:
- domain
- craft / route
- unit process
- product

## Notes

- This is a read-only statistics skill; it does not save remote edits.
- The skill is a thin wrapper over `tiangong process scope-statistics`; it does not carry a separate remote runtime or private `.env` parser.
- The metric layer is deterministic and string-based. It does not try to semantically merge cross-language variants unless they already share the same stable identifier.
