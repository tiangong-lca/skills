---
name: flow-governance-review
description: "Run a fixed local flow-governance workflow from flow/process snapshots and emit reusable artifacts: duplicate evidence, alias maps, deterministic process-flow repair plans, validation reports, and OpenClaw review packs with flow naming context. Use when governing product/waste flows during LCA data authoring or when OpenClaw/other skills need one canonical local flow-governance entrypoint."
---

# Flow Governance Review

Keep the scope local and file-driven. Start from JSON or JSONL snapshots of flows/processes, generate machine-readable artifacts, let OpenClaw or a human make the remaining semantic decisions, then apply and validate only the approved local changes.

Do not use this skill for:

- append-only publish or direct CRUD writes
- remote scope export

## Commands

- `openclaw-full-run`
- `run-governance`
- `review-flows`
- `flow-dedup-candidates`
- `build-flow-alias-map`
- `scan-process-flow-refs`
- `plan-process-flow-repairs`
- `apply-process-flow-repairs`
- `validate-processes`
- `export-openclaw-ref-review-pack`
- `apply-openclaw-ref-decisions`
- `export-openclaw-text-review-pack`
- `apply-openclaw-text-decisions`
- `validate-openclaw-text-decisions`

Run them through:

```bash
scripts/run-flow-governance-review.sh <command> ...
```

For OpenClaw when only a flow UUID is available, prefer:

```bash
scripts/run-flow-governance-review.sh openclaw-full-run \
  --flow-id 4b5eadd2-816c-420c-b382-5e0a975b53a6
```

This wrapper materializes live subject-flow and referencing-process snapshots, runs the existing `run-governance` orchestrator unchanged, and writes `openclaw-handoff-summary.json` for downstream agent handoff.

## Preferred Entry

Use `run-governance` for the standard end-to-end governance path:

1. review the local flow snapshot
2. build an explicit dedup snapshot and canonical evidence
3. build an alias map when old/new scopes are available
4. scan and deterministically repair process references
5. validate repaired processes
6. export residual OpenClaw review packs

Use `openclaw-full-run` when OpenClaw starts from a flow UUID instead of prebuilt local snapshots and still needs the same standard outputs.

Use the lower-level commands only when another skill or an external orchestrator needs one specific artifact.

## Standard Outputs

- `governance-run-manifest.json` from `run-governance`
- `flow-snapshot-manifest.json` from dedup
- `flow-dedup-canonical-map.json`
- `flow-dedup-rewrite-plan.json`
- `flow-alias-map.json` when alias building is applicable
- `scan-findings.json` and `repair-summary.json` when process snapshots are provided
- `review-pack.json` plus `review-pack-context.json` for OpenClaw text review
- `live-fetch-manifest.json` and `openclaw-handoff-summary.json` from `openclaw-full-run`

For flow text review, the exported review pack can include:

- flow naming fields beyond `base_name`, such as treatment/routes, mix/location, flow property text, and synonyms
- linked process context so naming can use producer/consumer evidence instead of guessing in isolation
- condensed methodology constraints from the bundled `references/tidas_flows.yaml`

## Load References On Demand

- `references/workflow.md`: command matrix, standard outputs, and recommended sequencing.
- `references/env.md`: optional live Supabase inputs for dedup analysis.
