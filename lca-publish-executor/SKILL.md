---
name: lca-publish-executor
description: Prepare and execute unified local LCA publish requests through `tiangong publish run`, including lifecyclemodels, projected process datasets, source datasets, relation manifests, and delegated process-build publish entries. Use when OpenClaw or another skill already produced local publish artifacts and needs one stable CLI-first publish contract instead of custom per-skill glue.
---

# LCA Publish Executor

## Overview
- Accept one JSON request that points at `publish-bundle.json` files, direct dataset payloads, or `process_from_flow` run ids.
- Delegate request normalization, bundle ingestion, relation-manifest generation, and report writing to `tiangong publish run`.
- Keep resulting-process relation metadata as a local manifest until a dedicated remote relation table exists.
- If a projected process payload is not yet a canonical `processDataSet` wrapper, mark it as deferred in `publish-report.json` instead of attempting a blind commit.
- The skill no longer keeps a Python fallback entrypoint.

## When To Use
- Use after `lifecyclemodel-recursive-orchestrator publish`.
- Use after `lifecyclemodel-resulting-process-builder publish`.
- Use when a caller wants one standard publish manifest for multiple skills.
- Use when OpenClaw should call a single publish skill instead of branching on each upstream skill's publish details.

## Commands
```bash
node scripts/run-lca-publish-executor.mjs publish \
  --request assets/example-request.json \
  --json

node scripts/run-lca-publish-executor.mjs \
  --input assets/example-request.json \
  --out-dir ../../artifacts/lca-publish-executor/manual-run \
  --json

TIANGONG_LCA_CLI_DIR=/path/to/tiangong-lca-cli \
  node scripts/run-lca-publish-executor.mjs publish --request assets/example-request.json
```

## Request Contract
- Read `assets/request.schema.json` for the stable manifest shape.
- Read `references/publish-contract.md` for bundle ingestion rules and relation handling.
- `inputs.bundle_paths[]`: one or more upstream `publish-bundle.json` files.
- `inputs.lifecyclemodels[]` / `processes[]` / `sources[]`: direct dataset payloads or file references.
- `inputs.process_build_runs[]`: delegated `process_from_flow` publish targets; each item needs `run_id`.
- `publish.commit=false` means dry-run preparation only.
- `publish.relation_mode=local_manifest_only` is currently the only supported relation mode.
- `--request` on the Node wrapper is a compatibility alias for the CLI's `--input`.

## Outputs
- `normalized-request.json`
- `collected-inputs.json`
- `publish-report.json`
- `relation-manifest.json`
- No skill-local delegated publish logs are produced by the canonical path; commit-time execution details live in `publish-report.json`.

## Notes
- `tiangong publish run` is the only publish contract layer for this skill now.
- If `publish.commit=true` is requested without configured CLI publish executors, the publish report will mark those operations as deferred instead of performing direct remote writes.
- OpenClaw should know only how to call this request shape and read `publish-report.json`, not the per-skill publish internals.
