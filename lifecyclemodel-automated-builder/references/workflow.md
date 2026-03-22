# Workflow

## Goal

Turn existing TianGong process records into native lifecycle model `json_ordered` datasets without mutating remote state by default.

## Default Run Order

1. Discovery
   - Read account-accessible processes.
   - Read public records with `state_code=100`.
   - Optionally search lifecyclemodels via MCP and inspect a few comparable models as structural reference.
2. Candidate normalization
   - Reduce each process to the fields needed for graph inference and model assembly.
3. AI selection
   - Score candidate groups using shared exchange connectivity, quantitative reference completeness, geography, time, and classification coherence.
4. Model assembly
   - Build the process graph locally.
   - Choose one reference process.
   - Compute `@multiplicationFactor`.
   - Produce native `json_ordered` only.
5. Validation
   - Run strict `tidas-sdk` validation first.
   - Run `tidas-tools` classification validation second.
6. Publish gate
   - Disabled by default.
   - If enabled, call MCP `Database_CRUD_Tool insert lifecyclemodels` with `jsonOrdered` only.

## Local Run Mode

- If `manifest.local_runs` is present, treat each listed `process-automated-builder` run directory as a pre-scoped candidate graph.
- If `manifest.discovery.reference_model_queries` is present, treat MCP lifecyclemodel search results as read-only structural guidance.
- In this mode, selection is graph-first:
  - load exported processes in the run
  - optionally load reusable process datasets from `manifest.reuse.reusable_process_dirs`
  - infer links from shared flow UUIDs
  - score each inferred link with flow identity, classification, geography, token overlap, and amount plausibility
  - choose the reference process from the target flow and sink position
  - compute `@multiplicationFactor` backward from the reference process

## Local Artifacts

- `run-plan.json`
- `resolved-manifest.json`
- `selection/selection-brief.md`
- `discovery/reference-model-summary.json`
- `models/**/tidas_bundle/lifecyclemodels/*.json`
- `models/**/summary.json`
- `models/**/connections.json`
- `models/**/process-catalog.json`
- `reports/*-validation.json`

## Publish Note

This skill stops at native `json_ordered`. Platform-specific derivation such as `json_tg` or `rule_verification` belongs to downstream MCP or application-side publishing logic, not to this skill.
