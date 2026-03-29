# Publish Contract

## Purpose

This skill is the stable publish boundary for OpenClaw-facing callers.

The canonical execution path is now:

- `node scripts/run-lca-publish-executor.mjs`
- `-> tiangong publish run`

The skill no longer owns its own Python/MCP publish implementation.

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

- all request normalization, bundle ingestion, relation manifest generation, and report writing are delegated to `tiangong publish run`
- `lifecyclemodels`:
  - normalized into the unified CLI publish report
  - commit-time execution requires a CLI publish executor; otherwise the report marks them as deferred
- `processes`:
  - canonical `processDataSet` payloads are normalized into the unified CLI publish report
  - non-canonical projection payloads are reported as `deferred_projection_payload`
- `sources`:
  - normalized into the unified CLI publish report
- `process_build_runs`:
  - remain part of the stable request contract
  - commit-time execution requires a CLI publish executor; otherwise the report marks them as deferred
- `relations`:
  - persisted only to local `relation-manifest.json`
  - no remote relation table is assumed yet

## OpenClaw Boundary

Put this protocol in the skill, not in OpenClaw runtime config.

OpenClaw should know:

- when to call this skill
- how to populate the request JSON
- how to read `publish-report.json`
- that `commit=true` only performs real remote execution if the CLI runtime has explicit publish executors configured

OpenClaw should not embed per-skill publish internals such as:

- how resulting-process builder bundles map to process payload arrays
- how orchestrator bundles expose delegated `process_build_runs`
- how commit executors are implemented under `tiangong publish run`
