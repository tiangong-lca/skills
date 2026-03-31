# Workflow

## Purpose

This skill exposes only the CLI-backed governance slices that still exist in the workspace. It is a local-first wrapper layer, not a second orchestration runtime.

## Runtime Model

- Entry point: `node scripts/run-flow-governance-review.mjs <command> ...`
- Wrapper role:
  - resolve `TIANGONG_LCA_CLI_DIR`
  - forward arguments to `tiangong`
  - expose no Python fallback path
- Command ownership:
  - review lives in `tiangong review flow`
  - read/repair/publish slices live in `tiangong flow ...`

## Supported Commands

Run these through the wrapper:

```bash
node scripts/run-flow-governance-review.mjs <command> ...
```

Supported commands:

- `review-flows`
- `flow-get`
- `flow-list`
- `remediate-flows`
- `publish-version`
- `publish-reviewed-data`
- `build-flow-alias-map`
- `scan-process-flow-refs`
- `plan-process-flow-repairs`
- `apply-process-flow-repairs`
- `regen-product`
- `validate-processes`

## Removed Surface

The previous Python/OpenClaw orchestration layer was intentionally removed.

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

## Artifact Layout

Persistent machine outputs belong under:

- `assets/artifacts/flow-processing/datasets/`
- `assets/artifacts/flow-processing/validation/`
- `assets/artifacts/flow-processing/naming/`
- `assets/artifacts/flow-processing/remediation/`

Do not reintroduce these artifacts under `docs/`.

## Guardrails

- Keep local JSON/JSONL payloads as the system of record.
- Use explicit CLI read/commit commands for remote interaction.
- Do not add helper scripts, private env parsing, or hidden transport logic back into this skill.
