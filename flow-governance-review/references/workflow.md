# Workflow

## Purpose

This skill is a thin wrapper around the supported CLI-backed governance commands. It does not introduce a second orchestration runtime.

## Runtime Model

- Entry point: `node scripts/run-flow-governance-review.mjs <command> ...`
- Wrapper role:
  - launch `npm exec --yes --package=@tiangong-lca/cli@latest -- tiangong` by default
  - honor `TIANGONG_LCA_CLI_DIR` / `--cli-dir` only as a local dev/CI override
  - forward arguments to `tiangong`
  - expose no Python fallback path
- Command ownership:
  - review lives in `tiangong review flow`
  - read/repair/publish slices live in `tiangong flow ...`

Write outputs to an explicit directory such as `/abs/path/artifacts/<case_slug>/...`.

## Supported Commands

Run these through the wrapper:

```bash
node scripts/run-flow-governance-review.mjs <command> ...
```

Supported commands:

- `review-flows`
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

## Not Supported

The previous Python/OpenClaw orchestration layer is not part of the supported wrapper surface.

Not available anymore:

- `openclaw-entry`
- `openclaw-full-run`
- `run-governance`
- `flow-dedup-candidates`
- `export-openclaw-*`
- `apply-openclaw-*`
- `validate-openclaw-*`

If any of these workflows is required again, add a native `tiangong` command first and then reintroduce a thin wrapper.

## Recommended Sequences

### Review And Publish Flows

When the task must bind to real DB flow rows:

1. `materialize-db-flows`
2. `review-flows`
3. `materialize-approved-decisions`
4. `remediate-flows`
5. `publish-version` or `publish-reviewed-data`

When the task is already grounded on an existing local reviewed-row snapshot:

1. `review-flows`
2. `remediate-flows`
3. `publish-version` or `publish-reviewed-data`

### Repair Process Flow References

1. `scan-process-flow-refs`
2. `plan-process-flow-repairs`
3. `apply-process-flow-repairs`
4. `validate-processes`
5. `publish-reviewed-data` when local review decisions are complete

### Alias Map After Cleanup

1. `build-flow-alias-map`
2. `scan-process-flow-refs`
3. `plan-process-flow-repairs`

## Key Outputs

- `review-flows`
  - `rule_findings.jsonl`
  - `llm_findings.jsonl`
  - `findings.jsonl`
  - `flow_review_summary.json`
- `materialize-db-flows`
  - `resolved-flow-rows.jsonl`
  - `review-input-rows.jsonl`
  - `fetch-summary.json`
  - `missing-flow-refs.jsonl`
  - `ambiguous-flow-refs.jsonl`
- `materialize-approved-decisions`
  - `flow-dedup-canonical-map.json`
  - `flow-dedup-rewrite-plan.json`
  - `manual-semantic-merge-seed.current.json`
  - `decision-summary.json`
  - `blocked-clusters.json`
- `publish-version`
  - publish report emitted by the CLI
- `publish-reviewed-data`
  - `prepared-flow-rows.json`
  - `prepared-process-rows.json`
  - `flow-version-map.json`
  - `publish-report.json`
- `build-flow-alias-map`
  - `flow-alias-map.json`
  - `alias-summary.json`
- `scan-process-flow-refs`
  - `scan-findings.json`
  - `scan-summary.json`
- `plan-process-flow-repairs`
  - `repair-plan.json`
  - `repair-summary.json`
- `apply-process-flow-repairs`
  - `patched-processes.json`
  - `process-patches/<process-id__version>/...`
- `validate-processes`
  - `validation-report.json`
  - `validation-failures.jsonl`

## Example Output Layout

Generated machine outputs are typically organized under:

- `/abs/path/artifacts/<case_slug>/flow-processing/datasets/`
- `/abs/path/artifacts/<case_slug>/flow-processing/validation/`
- `/abs/path/artifacts/<case_slug>/flow-processing/naming/`
- `/abs/path/artifacts/<case_slug>/flow-processing/remediation/`

Do not reintroduce these artifacts under `docs/` or `assets/` in this repository.

## Guardrails

- Keep local JSON/JSONL payloads as the system of record.
- Use explicit CLI read/commit commands for remote interaction.
- If the task requires real DB binding, materialize DB rows first and do not substitute synthetic rows.
- If merge decisions were approved, materialize them through `materialize-approved-decisions` before alias or process-repair planning.
- Do not add helper scripts, private env parsing, or hidden transport logic back into this skill.
