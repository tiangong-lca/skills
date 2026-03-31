# Workflow

## Goal

Turn existing local TianGong process-build runs into native lifecyclemodel `json_ordered` datasets through the unified TianGong CLI, without mutating remote state.

## Current Canonical Slice

The canonical entrypoint is:

- `node scripts/run-lifecyclemodel-automated-builder.mjs build --input <manifest> --out-dir <dir>`
- which delegates to `tiangong lifecyclemodel auto-build`

Follow-up commands are:

- `node scripts/run-lifecyclemodel-automated-builder.mjs validate --run-dir <dir>`
- `node scripts/run-lifecyclemodel-automated-builder.mjs publish --run-dir <dir>`

The implemented stage order is:

1. Intake
   - read one manifest from `--input`
   - require `local_runs[]`
2. Candidate normalization
   - load exported processes from the referenced local runs
   - reduce each process to the fields needed for graph inference and model assembly
3. Graph inference
   - infer links from shared flow UUIDs
   - score inferred connections using flow identity, classification, geography, token overlap, and amount plausibility
4. Model assembly
   - choose one reference process
   - compute `@multiplicationFactor`
   - produce native `json_ordered` only
5. Local artifact write
   - emit run-plan, manifest snapshot, selection note, model summary, connection summary, process catalog, and report

## Deferred Slices

The current canonical wrapper intentionally does not implement:

- remote process discovery
- reference-model discovery
- LLM / KB assisted selection
- any lifecyclemodel CRUD

Those remain future CLI slices, not work that should return to this skill repo.

Validation and publish handoff are already available as separate CLI-backed commands against one existing auto-build run.

## Local Run Mode

- If `manifest.local_runs` is present, treat each listed `process-automated-builder` run directory as a pre-scoped candidate graph.
- In this mode, selection is graph-first:
  - load exported processes in the run
  - optionally load reusable process datasets from `manifest.reuse.reusable_process_dirs`
  - infer links from shared flow UUIDs
  - score each inferred link with flow identity, classification, geography, token overlap, and amount plausibility
  - choose the reference process from the target flow and sink position
  - compute `@multiplicationFactor` backward from the reference process

If `manifest.discovery.reference_model_queries` is present, the current CLI slice records that discovery was requested and writes a deferred note in `discovery/reference-model-summary.json`; it does not execute discovery itself.

## Local Artifacts

- `run-plan.json`
- `resolved-manifest.json`
- `selection/selection-brief.md`
- `discovery/reference-model-summary.json`
- `models/**/tidas_bundle/lifecyclemodels/*.json`
- `models/**/summary.json`
- `models/**/connections.json`
- `models/**/process-catalog.json`
- `reports/lifecyclemodel-auto-build-report.json`

## Follow-up Artifacts

After `validate`:

- `reports/model-validations/*.json`
- `reports/lifecyclemodel-validate-build-report.json`

After `publish`:

- `stage_outputs/10_publish/publish-bundle.json`
- `stage_outputs/10_publish/publish-request.json`
- `stage_outputs/10_publish/publish-intent.json`
- `reports/lifecyclemodel-publish-build-report.json`

## Publish Note

This skill stops at native `json_ordered`. Platform-specific derivation such as `json_tg`, `rule_verification`, or publish-side graph materialization belongs to downstream publish layers, not to this skill.
