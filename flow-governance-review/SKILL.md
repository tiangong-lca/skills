---
name: flow-governance-review
description: "Run the CLI-backed flow governance slices for review, remediation, deterministic process-flow repair, and reviewed-row publish handoff. Use `node scripts/run-flow-governance-review.mjs COMMAND ...` when you need the canonical `tiangong review flow` and `tiangong flow ...` surface from a skill wrapper."
---

# Flow Governance Review

Keep local JSON or JSONL payloads as the system of record. This skill is now a thin wrapper over CLI-backed governance commands only.

Do not use this skill for:

- arbitrary remote CRUD outside the explicit CLI commit commands
- hidden OpenClaw orchestration
- private Python remediation helpers
- remote scope export

## Runtime Model

- The canonical entrypoint is `node scripts/run-flow-governance-review.mjs <command> ...`.
- Supported commands are all CLI-backed:
  - `review-flows` -> `tiangong review flow`
  - `flow-get` -> `tiangong flow get`
  - `flow-list` -> `tiangong flow list`
  - `materialize-db-flows` -> `tiangong flow fetch-rows`
  - `materialize-approved-decisions` -> `tiangong flow materialize-decisions`
  - `remediate-flows` -> `tiangong flow remediate`
  - `publish-version` -> `tiangong flow publish-version`
  - `publish-reviewed-data` -> `tiangong flow publish-reviewed-data`
  - `build-flow-alias-map` -> `tiangong flow build-alias-map`
  - `scan-process-flow-refs` -> `tiangong flow scan-process-flow-refs`
  - `plan-process-flow-repairs` -> `tiangong flow plan-process-flow-repairs`
  - `apply-process-flow-repairs` -> `tiangong flow apply-process-flow-repairs`
  - `regen-product` -> `tiangong flow regen-product`
  - `validate-processes` -> `tiangong flow validate-processes`
- `publish-reviewed-data` is fully CLI-owned for both local preparation and commit-time process publish.
- There is no Python fallback path and no shell compatibility shim.

## Commands

- `flow-get`
- `flow-list`
- `materialize-db-flows`
- `materialize-approved-decisions`
- `remediate-flows`
- `publish-version`
- `publish-reviewed-data`
- `build-flow-alias-map`
- `scan-process-flow-refs`
- `plan-process-flow-repairs`
- `apply-process-flow-repairs`
- `regen-product`
- `validate-processes`
- `review-flows`

Run them through:

```bash
node scripts/run-flow-governance-review.mjs <command> ...
```

For CLI-backed deterministic governance slices, prefer:

```bash
node scripts/run-flow-governance-review.mjs materialize-db-flows \
  --refs-file /abs/path/flow-refs.json \
  --out-dir /abs/path/materialized \
  --fail-on-missing

node scripts/run-flow-governance-review.mjs materialize-approved-decisions \
  --decision-file /abs/path/approved-decisions.json \
  --flow-rows-file /abs/path/materialized/review-input-rows.jsonl \
  --out-dir /abs/path/decision-artifacts

node scripts/run-flow-governance-review.mjs review-flows \
  --rows-file /abs/path/flows.jsonl \
  --out-dir /abs/path/review

node scripts/run-flow-governance-review.mjs remediate-flows \
  --input-file /abs/path/invalid-flows.jsonl \
  --out-dir /abs/path/remediation

node scripts/run-flow-governance-review.mjs publish-version \
  --input-file /abs/path/ready-flows.jsonl \
  --out-dir /abs/path/publish \
  --dry-run

node scripts/run-flow-governance-review.mjs publish-reviewed-data \
  --flow-rows-file /abs/path/reviewed-flows.jsonl \
  --original-flow-rows-file /abs/path/original-flows.jsonl \
  --out-dir /abs/path/publish-reviewed

node scripts/run-flow-governance-review.mjs build-flow-alias-map \
  --old-flow-file /abs/path/old-flows.jsonl \
  --new-flow-file /abs/path/new-flows.jsonl \
  --out-dir /abs/path/alias-map

node scripts/run-flow-governance-review.mjs scan-process-flow-refs \
  --processes-file /abs/path/processes.jsonl \
  --scope-flow-file /abs/path/flows.jsonl \
  --out-dir /abs/path/scan

node scripts/run-flow-governance-review.mjs plan-process-flow-repairs \
  --processes-file /abs/path/processes.jsonl \
  --scope-flow-file /abs/path/flows.jsonl \
  --scan-findings /abs/path/scan/scan-findings.json \
  --out-dir /abs/path/repair-plan

node scripts/run-flow-governance-review.mjs apply-process-flow-repairs \
  --processes-file /abs/path/processes.jsonl \
  --scope-flow-file /abs/path/flows.jsonl \
  --scan-findings /abs/path/scan/scan-findings.json \
  --out-dir /abs/path/repair-apply

node scripts/run-flow-governance-review.mjs regen-product \
  --processes-file /abs/path/processes.jsonl \
  --scope-flow-file /abs/path/flows.jsonl \
  --out-dir /abs/path/regen \
  --apply

node scripts/run-flow-governance-review.mjs validate-processes \
  --original-processes-file /abs/path/before.jsonl \
  --patched-processes-file /abs/path/after.jsonl \
  --scope-flow-file /abs/path/flows.jsonl \
  --out-dir /abs/path/validate
```

## Removed Surface

The following legacy commands were intentionally removed with the Python runtime:

- `openclaw-entry`
- `openclaw-full-run`
- `run-governance`
- `flow-dedup-candidates`
- `export-openclaw-*`
- `apply-openclaw-*`
- `validate-openclaw-*`

If one of those workflows is still needed, reintroduce it first as a native `tiangong review ...` or `tiangong flow ...` command. Do not recreate it inside this skill.

## Preferred Usage

Use the supported commands as composable slices:

1. `materialize-db-flows` when the task must bind to real DB rows
2. `review-flows`
3. `materialize-approved-decisions` after merge decisions are approved
4. `remediate-flows`
5. `build-flow-alias-map` when version cleanup produced old/new scopes
6. `scan-process-flow-refs`
7. `plan-process-flow-repairs`
8. `apply-process-flow-repairs`
9. `validate-processes`
10. `publish-version` or `publish-reviewed-data`

## Standard Outputs

- `flow-alias-map.json` when alias building is applicable
- `scan-findings.json` and `repair-summary.json` when process snapshots are provided
- `publish-report.json` from `publish-reviewed-data`
- `prepared-flow-rows.json` and `flow-version-map.json` from `publish-reviewed-data`
- `skipped-unchanged-flow-rows.json` from `publish-reviewed-data` when `--original-flow-rows-file` is provided
- `resolved-flow-rows.jsonl`, `review-input-rows.jsonl`, and `fetch-summary.json` from `materialize-db-flows`
- `flow-dedup-canonical-map.json`, `flow-dedup-rewrite-plan.json`, `manual-semantic-merge-seed.current.json`, and `blocked-clusters.json` from `materialize-approved-decisions`

## Artifact Layout

Keep long-lived flow-processing bundles under `assets/artifacts/flow-processing/` instead of `docs/`.

Canonical retained bundles:

- `assets/artifacts/flow-processing/datasets/`: shared flow pool, invalid-input scope, resolved flow pool, reusable `process_pool.jsonl`
- `assets/artifacts/flow-processing/validation/`: grouped validation failures that still matter for remediation planning
- `assets/artifacts/flow-processing/naming/remaining-after-aggressive/`: post-aggressive completeness summaries and zero-process residuals
- `assets/artifacts/flow-processing/naming/zero-process-completion-pack/`: retained historical review materials
- `assets/artifacts/flow-processing/remediation/`: deterministic remediation and publish preparation artifacts
- `assets/artifacts/flow-remediation-batch-smoketest/`: historical smoke-test evidence for remediation-helper startup checks

Do not treat `docs/` as the canonical home for these machine artifacts anymore. New helper-script defaults in this repo now write to the artifact tree above.

## Load References On Demand

- `references/workflow.md`: command matrix, outputs, removed surface, and recommended sequencing.
- `references/env.md`: canonical CLI env expectations for read, review, and publish commands.
- `references/real-db-first-runbook.md`: real-DB-first execution guardrails, refs-file shape, and blocked-case handling.
- `references/decision-schema.md`: approved decision file schema, merge examples, and downstream artifact meanings.
