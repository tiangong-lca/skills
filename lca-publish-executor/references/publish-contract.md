# Publish Contract

## Purpose

This skill is the stable publish boundary for OpenClaw-facing callers.

Upstream builder skills may each emit different local artifacts, but OpenClaw should only need one publish request shape:

- `inputs.bundle_paths[]` for upstream `publish-bundle.json`
- optional direct dataset arrays
- optional delegated `process_build_runs[]`

## Bundle Ingestion Rules

- Orchestrator bundle:
  - `lifecyclemodels[]`
  - `projected_processes[]`
  - `resulting_process_relations[]`
  - `process_build_runs[]`
- Resulting-process builder bundle:
  - `projected_processes[]`
  - `relations[]`
- Direct arrays in the request are appended after bundle ingestion.

## Publish Behavior

- `lifecyclemodels`:
  - normalized by `tiangong publish run`
  - executed only when the CLI publish layer is given an executor
- `processes`:
  - normalized by `tiangong publish run`
  - non-canonical projection payloads are reported as `deferred_projection_payload`
  - this prevents accidental inserts when an upstream resulting-process build output is still review-stage or incomplete, not a final `processDataSet`
- `sources`:
  - normalized by `tiangong publish run`
  - executed only when the CLI publish layer is given an executor
- `process_build_runs`:
  - kept inside the unified CLI publish request contract
  - commit-mode execution remains a CLI executor concern, not a skill-private subprocess concern
- `relations`:
  - persisted only to local `relation-manifest.json`
  - no remote relation table is assumed yet

If `publish.commit=true` is requested without a configured CLI executor, the CLI reports `deferred_no_executor` instead of reviving legacy MCP or Python publish code inside this skill.

## OpenClaw Boundary

Put this protocol in the skill, not in OpenClaw runtime config.

OpenClaw should know:

- when to call this skill
- how to populate the request JSON
- how to read `publish-report.json`

OpenClaw should not embed per-skill publish internals such as:

- how resulting-process builder bundles map to process payload arrays
- how orchestrator bundles expose delegated `process_build_runs`
- how CLI executor wiring is implemented
