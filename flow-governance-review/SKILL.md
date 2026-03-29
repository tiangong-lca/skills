---
name: flow-governance-review
description: "Run a fixed flow-governance workflow from a flow UUID or local flow/process snapshots and emit reusable artifacts: duplicate evidence, alias maps, deterministic process-flow repair plans, validation reports, OpenClaw review packs for naming/classification/reference decisions, reusable process-pool sync, and reviewed-row publish reports that can skip unchanged flows. Use when governing product/waste flows during LCA data authoring or when OpenClaw/other skills need one canonical governance entrypoint plus optional publish handoff."
---

# Flow Governance Review

Keep local JSON or JSONL payloads as the system of record. Start from a flow UUID or local snapshots of flows/processes, generate machine-readable artifacts, let OpenClaw or a human make the remaining semantic decisions, then apply, validate, and optionally publish only the approved reviewed rows. Repeated single-flow runs can share a local process pool so the skill reuses exact-version process rows instead of refetching the same full JSON every time.

Do not use this skill for:

- arbitrary remote CRUD outside the explicit reviewed-row publish command
- remote scope export

## Commands

- `openclaw-entry`
- `openclaw-full-run`
- `run-governance`
- `review-flows`
- `remediate-flows`
- `publish-version`
- `flow-dedup-candidates`
- `build-flow-alias-map`
- `scan-process-flow-refs`
- `plan-process-flow-repairs`
- `apply-process-flow-repairs`
- `validate-processes`
- `export-openclaw-dedup-review-pack`
- `export-openclaw-ref-review-pack`
- `apply-openclaw-ref-decisions`
- `export-openclaw-text-review-pack`
- `export-openclaw-classification-review-pack`
- `apply-openclaw-text-decisions`
- `apply-openclaw-classification-decisions`
- `validate-openclaw-text-decisions`
- `validate-openclaw-classification-decisions`
- `publish-reviewed-data`

Run them through:

```bash
scripts/run-flow-governance-review.sh <command> ...
```

`review-flows` is now a thin compatibility wrapper over the unified CLI:

```bash
tiangong review flow (--rows-file <file> | --flows-dir <dir> | --run-root <dir>) --out-dir <dir>
```

That means:

- the canonical runtime is `Node wrapper -> tiangong CLI`
- optional semantic review uses `TIANGONG_LCA_LLM_*`, not `OPENAI_*`
- `--with-reference-context` is not implemented in the CLI slice yet and should not be used through this skill

`remediate-flows` is also a thin compatibility wrapper over the unified CLI:

```bash
tiangong flow remediate --input-file <file> --out-dir <dir>
```

That means:

- the canonical runtime is `Node wrapper -> tiangong CLI`
- this wrapper owns only the round1 deterministic local remediation slice
- default local paths still match the historical invalid-flow pool and `assets/artifacts/flow-processing/remediation/round1/`
- round2 remote-validation recovery and later publish/sync steps remain separate follow-up stages

`publish-version` is now the canonical remediated-flow publish wrapper over the unified CLI:

```bash
tiangong flow publish-version --input-file <file> --out-dir <dir> --commit
```

That means:

- the canonical runtime is `Node wrapper -> tiangong CLI`
- the wrapper preserves the historical publish-stage behavior by adding `--commit` unless `--dry-run` is passed
- the wrapper keeps the historical `mcp-sync` artifact directory and legacy file names so downstream follow-up steps do not need a new contract
- remote writes now use `TIANGONG_LCA_API_BASE_URL` and `TIANGONG_LCA_API_KEY`, not skill-local MCP helpers or Supabase login env

For OpenClaw, prefer the unified entrypoint:

```bash
scripts/run-flow-governance-review.sh openclaw-entry \
  --flow-id 4b5eadd2-816c-420c-b382-5e0a975b53a6
```

If you already have a complete local flow snapshot, use the same entrypoint with `--subject-flows-file` or `--flows-file` instead of a UUID.

## Preferred Entry

Use `openclaw-entry` for OpenClaw-facing execution:

1. if only a flow UUID is available, fetch the subject flow through MCP CRUD when available and then run the standard workflow
2. if a local flow JSON or JSONL file is already available, run the standard workflow directly from that snapshot
3. if local subject flows are provided without a candidate pool file or live candidate-pool env, fall back to subject-only dedup instead of failing early
4. if `--process-pool-file` is provided, reuse exact-version process rows from that local pool, fetch only missing referencing rows, and sync deterministic or OpenClaw-applied process patches back into the pool
5. write `openclaw-handoff-summary.json` for downstream review/apply/publish handoff

Use `run-governance` for the standard local governance path:

1. review the local flow snapshot
2. build an explicit dedup snapshot and canonical evidence
3. build an alias map when old/new scopes are available
4. scan and deterministically repair process references
5. validate repaired processes
6. export residual OpenClaw review packs

The dedup stage is conservative: same-UUID version lineage is not treated as semantic dedup and is not auto-merged here. Cross-UUID high-similarity pairs are exported for OpenClaw semantic review instead of being merged by string similarity alone.

Use `publish-reviewed-data` only after explicit OpenClaw decisions have been applied locally and the patched rows passed local validation. Pass `--original-flow-rows-file` when you want unchanged flow rows skipped entirely instead of being version-bumped and published.

Use the lower-level commands only when another skill or an external orchestrator needs one specific artifact.

## Direct Helper Scripts

Some workflows in this skill are intentionally exposed as helper entrypoints under `scripts/`. The round1 remediation entrypoint is now CLI-backed; the remaining follow-up helpers below stay as direct Python utilities.

Canonical CLI-backed flow-governance entrypoints:

- `scripts/run-flow-governance-review.sh remediate-flows`
- `scripts/run-remediate-flows.mjs`
- `scripts/run-flow-governance-review.sh publish-version`
- `scripts/run-publish-version.mjs`

Remaining direct follow-up helpers:

- `scripts/mcp_sync_remediated_flows_batch.py`
- `scripts/remediate_remote_validation_failed_flows_round2.py`

The round1 remediation and first publish-version path are now CLI-backed. The remaining follow-up helper above still depends on the `process-automated-builder` runtime, `tiangong_lca_spec`, and `tidas_sdk`. `mcp_sync_remediated_flows_batch.py` is now retained only as a deprecated legacy helper, not the canonical entrypoint.

For the full command matrix, auxiliary naming-completion utilities, and recommended sequencing, read `references/workflow.md`.

## Standard Outputs

- `governance-run-manifest.json` from `run-governance`
- `flow-snapshot-manifest.json` from dedup
- `flow-dedup-canonical-map.json`
- `flow-dedup-rewrite-plan.json`
- `dedup-pack/review-pack.json` when semantic duplicate pairs need OpenClaw review
- `flow-alias-map.json` when alias building is applicable
- `scan-findings.json` and `repair-summary.json` when process snapshots are provided
- `review-pack.json` plus `review-pack-context.json` for OpenClaw text review or classification review
- `live-fetch-manifest.json` and `openclaw-handoff-summary.json` from `openclaw-full-run`
- `openclaw-handoff-summary.json` from `openclaw-entry`
- `publish-report.json` from `publish-reviewed-data`
- `skipped-unchanged-flow-rows.json` from `publish-reviewed-data` when `--original-flow-rows-file` is provided

For flow text review, the exported review pack can include:

- flow naming fields beyond `base_name`, such as treatment/routes, mix/location, flow property text, and synonyms
- linked process context so naming can use producer/consumer evidence instead of guessing in isolation
- condensed methodology constraints from the bundled `references/tidas_flows.yaml`

For flow classification review, the exported review pack can include:

- the current product or elementary classification path derived from the local flow row
- candidate classification paths with valid TIDAS class IDs for OpenClaw to choose from
- relevant review findings filtered from `review/findings.jsonl`
- condensed methodology constraints for `typeOfDataSet` and classification branches

## Artifact Layout

Keep long-lived flow-processing bundles under `assets/artifacts/flow-processing/` instead of `docs/`.

Canonical retained bundles:

- `assets/artifacts/flow-processing/datasets/`: shared flow pool, invalid-input scope, resolved flow pool, reusable `process_pool.jsonl`
- `assets/artifacts/flow-processing/validation/`: grouped validation failures that still matter for remediation planning
- `assets/artifacts/flow-processing/naming/remaining-after-aggressive/`: post-aggressive completeness summaries and zero-process residuals
- `assets/artifacts/flow-processing/naming/zero-process-completion-pack/`: retained OpenClaw review pack, final decisions, patched rows, and publish summaries for the zero-process completion run
- `assets/artifacts/flow-processing/remediation/`: deterministic remediation and MCP sync outputs written by related helper scripts
- `assets/artifacts/flow-remediation-batch-smoketest/`: historical smoke-test evidence for remediation-helper startup checks

Do not treat `docs/` as the canonical home for these machine artifacts anymore. New helper-script defaults in this repo now write to the artifact tree above.

## Load References On Demand

- `references/workflow.md`: command matrix, direct helper utilities, standard outputs, and recommended sequencing.
- `references/env.md`: optional live Supabase inputs for dedup analysis.
