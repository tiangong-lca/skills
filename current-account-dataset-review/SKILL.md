---
name: current-account-dataset-review
description: Review and repair current-account TianGong dataset snapshots across flow, process, and lifecyclemodel rows through public CLI commands only. Use when a task needs local schema validation, deterministic reference rewrites, lifecyclemodel save-draft handoff, lifecyclemodel graph audit, and artifact-first stop rules for account-owned dataset governance.
---

# Current Account Dataset Review

## Scope

- This skill is a thin orchestration layer over public `tiangong` CLI commands.
- Use it for current-account multi-type dataset governance when the input rows or manifests are already frozen locally.
- Do not import `tiangong-cli/dist/src/lib/**`, write direct Supabase REST glue, or parse raw credentials in this skill.
- If a capability is missing from the public CLI, record it as a CLI gap instead of adding private runtime logic here.

## Canonical Flow

1. Freeze or receive the account-scoped input rows.
2. Validate all local rows:

```bash
node current-account-dataset-review/scripts/run-current-account-dataset-review.mjs validate \
  --input /abs/path/rows.jsonl \
  --type auto \
  --out-dir /abs/path/dataset-validate
```

3. If an upstream flow version changes, rewrite downstream references locally first:

```bash
node current-account-dataset-review/scripts/run-current-account-dataset-review.mjs rewrite-references \
  --input /abs/path/rows.jsonl \
  --from flow:<old-id>@<old-version> \
  --to flow:<new-id>@<new-version> \
  --type process \
  --type lifecyclemodel \
  --out-dir /abs/path/reference-rewrite \
  --dry-run
```

4. Save lifecyclemodel drafts only after local validation passes:

```bash
node current-account-dataset-review/scripts/run-current-account-dataset-review.mjs save-lifecyclemodels \
  --input /abs/path/lifecyclemodels.jsonl \
  --out-dir /abs/path/lifecyclemodel-save-draft \
  --dry-run
```

5. Generate graph artifacts and connection findings for lifecyclemodels:

```bash
node current-account-dataset-review/scripts/run-current-account-dataset-review.mjs graph-lifecyclemodels \
  --input /abs/path/lifecyclemodels.jsonl \
  --out-dir /abs/path/lifecyclemodel-graph \
  --format all \
  --check-connections
```

## Runtime Contract

- Wrapper-local `--cli-dir` is the only supported override for choosing a local CLI checkout.
- Local dry-runs do not require remote credentials.
- Commit paths use the canonical CLI env only:
  - `TIANGONG_LCA_API_BASE_URL`
  - `TIANGONG_LCA_API_KEY`
  - `TIANGONG_LCA_SUPABASE_PUBLISHABLE_KEY`
- `TIANGONG_LCA_API_KEY` is sensitive and must come from env or the caller's secret store.

## Stop Rules

- Any commit must be preceded by a dry-run artifact set.
- If a publish or save-draft step partially fails, do not rerun the full scope blindly; use the generated failure artifacts to plan a targeted retry.
- After changing flow or process versions, re-validate downstream process and lifecyclemodel references.
- Lifecyclemodel review must include schema validation and graph connection checks.
- Final completion for remote write tasks requires a fresh remote verification pass. `dataset verify-remote` remains a CLI gap until implemented.

## Known CLI Gaps

- `dataset inventory --current-user`
- `dataset verify-remote`
- `flow publish-version` server-side next available version / conflict retry
- `dataset scope expand`
