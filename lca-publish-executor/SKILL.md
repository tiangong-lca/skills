---
name: lca-publish-executor
description: Normalize and execute local LCA publish requests through the unified `tiangong publish run` contract. Use when OpenClaw or another skill already produced local publish artifacts and needs one stable publish surface instead of custom per-skill glue.
---

# LCA Publish Executor

## Overview
- Accept one JSON request that points at `publish-bundle.json` files, direct dataset payloads, or `process_from_flow` run ids.
- Delegate the entire request to `tiangong publish run` instead of keeping a skill-private publish runtime.
- Keep resulting-process relation metadata as a local manifest until a dedicated remote relation table exists.
- If a projected process payload is not yet a canonical `processDataSet` wrapper, let the CLI mark it as deferred in `publish-report.json` instead of attempting a blind remote insert.
- Treat `publish.commit=true` as a CLI executor request boundary; if no executor is configured in the CLI layer, commit-mode items stay deferred instead of reviving skill-local MCP logic.

## When To Use
- Use after `lifecyclemodel-recursive-orchestrator publish`.
- Use after `lifecyclemodel-resulting-process-builder publish`.
- Use when a caller wants one standard publish manifest for multiple skills.
- Use when OpenClaw should call a single publish skill instead of branching on each upstream skill's publish details.

## Commands
```bash
node scripts/run-lca-publish-executor.mjs publish \
  --request assets/example-request.json \
  --out-dir /tmp/lca-publish-executor-test
```

## Request Contract
- Read `assets/request.schema.json` for the stable manifest shape.
- Read `references/publish-contract.md` for bundle ingestion rules and relation handling.
- `inputs.bundle_paths[]`: one or more upstream `publish-bundle.json` files.
- `inputs.lifecyclemodels[]` / `processes[]` / `sources[]`: direct dataset payloads or file references.
- `inputs.process_build_runs[]`: delegated `process_from_flow` publish targets; each item needs `run_id`.
- `publish.commit=false` means dry-run preparation only.
- `publish.relation_mode=local_manifest_only` is currently the only supported relation mode.

## Outputs
- `normalized-request.json`
- `collected-inputs.json`
- `publish-report.json`
- `relation-manifest.json`

## Notes
- Keep TianGong LCA runtime keys in the caller env or pass standard CLI flags; do not hardcode secrets in requests.
- This skill is now a thin wrapper around `tiangong publish run`. OpenClaw should know only how to call this request shape, not the historical per-skill publish internals.
