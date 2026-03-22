# Workflow

## Purpose

Use this skill to produce reusable flow-governance artifacts from local flow/process snapshots. The outputs are designed to be consumed by other skills or by OpenClaw, not just read as prose.

This skill intentionally keeps a narrow boundary:

- include duplicate evidence and canonical rewrite planning
- include alias-map generation after cleanup or version changes
- include process-flow reference scan, deterministic repair, and validation
- include bounded OpenClaw review-pack export/apply steps
- exclude remote publish and remote scope export

## Command Matrix

### `openclaw-full-run`

OpenClaw-friendly wrapper for the standard workflow when the caller starts with a flow UUID instead of local snapshots.

The wrapper:

1. fetches visible product/waste flow rows for `--flow-id`
2. fetches visible processes that reference that flow UUID
3. materializes those snapshots under `inputs/`
4. calls `run-governance` with the fetched snapshots
5. writes `live-fetch-manifest.json` and `openclaw-handoff-summary.json`

Recommended usage:

```bash
scripts/run-flow-governance-review.sh openclaw-full-run \
  --flow-id 4b5eadd2-816c-420c-b382-5e0a975b53a6
```

### `run-governance`

Preferred fixed orchestration entrypoint for this skill.

Use this when you want one standard local governance run instead of composing subcommands yourself. The orchestrator runs the applicable phases in this order:

1. `review-flows`
2. `flow-dedup-candidates`
3. `build-flow-alias-map` when old/new scopes are provided
4. `scan-process-flow-refs` when `--processes-file` is provided
5. `plan-process-flow-repairs`
6. `apply-process-flow-repairs`
7. `validate-processes`
8. `export-openclaw-ref-review-pack` when residual ref decisions remain
9. `export-openclaw-text-review-pack` for the target flow scope

Primary outputs:

- `governance-run-manifest.json`
- staged subdirectories under the run root (`review/`, `dedup/`, `alias/`, `scan/`, `repair/`, `repair-apply/`, `validate/`, `ref-pack/`, `flow-text-pack/`)

### `review-flows`

Run the bundled local flow review engine directly from this skill. This is the only supported flow-governance QA entrypoint in the repo.

Use this when you want:

- `findings.jsonl`
- `similarity_pairs.jsonl`
- `flow_review_summary.json`

It accepts either:

- `--rows-file` for a local JSON/JSONL flow snapshot
- `--flows-dir` for an existing per-flow JSON directory
- `--run-root` for an existing run with `cache/flows` or `exports/flows`

When `--rows-file` is used, the engine materializes `review-input/flows/*.json` plus `review-input/materialization-summary.json` so downstream evidence is tied to an explicit local snapshot.

### `flow-dedup-candidates`

Generate duplicate-family evidence, canonical maps, rewrite plans, and manual-review queues for product/waste flows.

Primary outputs:

- `flow-dedup-summary.json`
- `flow-dedup-candidate-clusters.json`
- `flow-dedup-canonical-map.json`
- `flow-dedup-rewrite-plan.json`
- `flow-dedup-manual-review.jsonl`
- `flow-snapshot-manifest.json`
- `merged-flow-snapshot.jsonl`

Recommended subject/candidate mode:

```bash
scripts/run-flow-governance-review.sh flow-dedup-candidates \
  --subject-flows-file /abs/path/local-draft-flows.jsonl \
  --candidate-flows-file /abs/path/candidate-pool.jsonl \
  --processes-file /abs/path/processes.jsonl \
  --out-dir /abs/path/out
```

If `--candidate-flows-file` is omitted, the command fetches the candidate pool live and writes it back out as `candidate-pool-flows.snapshot.jsonl` before analysis.

Live visible-process counts are optional and must be requested explicitly with `--live-ref-counts` when `--processes-file` is omitted.

### `build-flow-alias-map`

Build a deterministic alias map from a pre-clean flow snapshot and a post-clean flow snapshot.

Primary outputs:

- `alias-plan.json`
- `flow-alias-map.json`
- `manual-review-queue.jsonl`

### `scan-process-flow-refs`

Classify every process exchange reference against the current flow scope and optional alias map.

Primary outputs:

- `scan-summary.json`
- `scan-findings.json`
- `scan-findings.jsonl`

### `plan-process-flow-repairs`

Generate a deterministic repair plan from process rows, scope flows, alias map, and optional scan findings.

The auto-patch boundary is explicit:

- `--auto-patch-policy alias-only` is the default and recommended mode
- `--auto-patch-policy alias-or-unique-name` allows the older exact-name fallback
- `--auto-patch-policy disabled` turns every non-trivial case into manual review

Primary outputs:

- `repair-plan.json`
- `repair-plan.jsonl`
- `manual-review-queue.jsonl`
- `repair-summary.json`

### `apply-process-flow-repairs`

Apply only the deterministic subset of the repair plan and emit patch evidence per process.

Primary outputs:

- `patched-processes.json`
- `process-patches/<process-id__version>/before.json`
- `process-patches/<process-id__version>/after.json`
- `process-patches/<process-id__version>/diff.patch`
- `process-patches/<process-id__version>/evidence.json`

### `validate-processes`

Verify that only allowed flow-reference paths changed, that quantitative references stayed stable, and optionally run TIDAS validation.

Primary outputs:

- `validation-report.json`
- `validation-failures.jsonl`

### `export-openclaw-ref-review-pack`

Convert unresolved process-reference decisions into a bounded OpenClaw review pack.

Primary outputs:

- `review-pack.json`
- `review-pack.jsonl`
- `carryover-manual-review.jsonl`
- `review-pack-summary.json`
- `openclaw-instructions.md`

### `apply-openclaw-ref-decisions`

Apply explicit OpenClaw candidate choices back to local process rows.

Primary outputs:

- `patched-processes.json`
- `invalid-decisions.jsonl`
- `manual-review-unresolved.jsonl`
- `decision-summary.json`

### `export-openclaw-text-review-pack`

Export only whitelisted text fields for flow or process wording cleanup.

Allowed scope:

- flow naming/text fields supported by the bundled helper:
  - `base_name`
  - `treatment_standards_routes`
  - `mix_and_location_types`
  - `flow_properties_text`
  - `synonyms`
  - `short_description`
- process text fields already supported by the bundled helper

Additional flow-only context:

- optional linked process context derived from a local `--processes-file`
- optional condensed methodology guidance loaded from `tidas_flows.yaml` or an explicit `--methodology-file`
- `review-pack-context.json` as the stable context sidecar for OpenClaw

### `apply-openclaw-text-decisions`

Apply only whitelisted text patches and emit before/after evidence.

### `validate-openclaw-text-decisions`

Confirm that OpenClaw text edits stayed inside the whitelist and optionally re-run TIDAS validation.

## Recommended Sequence

1. Prefer `run-governance` unless you are integrating one substep into another skill.
2. If you need custom orchestration, run `review-flows` for QA findings and similarity evidence first.
3. Build an explicit subject/candidate snapshot with `flow-dedup-candidates`; inspect `flow-snapshot-manifest.json`.
4. Build `flow-alias-map.json` after canonical decisions or post-clean snapshots exist.
5. Scan process references with `scan-process-flow-refs`.
6. Generate and apply deterministic repairs with an explicit `--auto-patch-policy`.
7. Validate the patched subset.
8. Export only the residual manual-review queue to OpenClaw, plus the flow text review pack when naming/text cleanup is still needed.

## Guardrails

- Keep local JSON payloads as the system of record for this skill.
- Prefer explicit snapshots over hidden live reads; the dedup command now persists the exact snapshots it used.
- Use machine-readable artifacts as handoff boundaries; avoid burying decisions only in markdown.
- Treat `flow-alias-map.json` and explicit OpenClaw decision files as authoritative inputs for downstream patch steps.
- Treat `review-pack-context.json` as the authoritative context sidecar for flow naming/text review.
- Do not let this skill publish or mutate remote datasets.
