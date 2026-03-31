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
  - normalized into the unified CLI publish request
  - downstream write semantics are owned by `tiangong publish run`
- `processes`:
  - normalized into the unified CLI publish request
  - non-canonical projection payloads can still be reported as deferred instead of being blindly committed
- `sources`:
  - normalized into the unified CLI publish request
- `process_build_runs`:
  - delegated through the CLI publish contract
  - callers should pass stable run ids or prepared publish bundles instead of expecting a skill-private publish path
- `relations`:
  - persisted only to local `relation-manifest.json`
  - no remote relation table is assumed yet

## OpenClaw Boundary

Put this protocol in the skill, not in OpenClaw runtime config.

OpenClaw should know:

- when to call this skill
- how to populate the request JSON
- how to read `publish-report.json`

OpenClaw should not embed per-skill publish internals such as:

- how resulting-process builder bundles map to process payload arrays
- how orchestrator bundles expose delegated `process_build_runs`
- how commit executors or relation manifests are materialized by the CLI
