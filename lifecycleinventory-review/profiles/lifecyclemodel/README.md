# lifecyclemodel profile

Status: **implemented**.

Canonical path:
1. Run `node scripts/run-review.mjs --profile lifecyclemodel`.
2. The wrapper delegates directly to `tiangong review lifecyclemodel`.

Required inputs:
- `--run-dir <dir>`
- `--out-dir <dir>`

Optional inputs:
- `--start-ts <iso>`
- `--end-ts <iso>`
- `--logic-version <name>`

Review artifacts:
- `model_summaries.jsonl`
- `findings.jsonl`
- `lifecyclemodel_review_summary.json`
- `lifecyclemodel_review_zh.md`
- `lifecyclemodel_review_en.md`
- `lifecyclemodel_review_timing.md`
- `lifecyclemodel_review_report.json`
