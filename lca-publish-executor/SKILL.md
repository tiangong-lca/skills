---
name: lca-publish-executor
description: Publish local LCA artifact bundles through the approved MCP/local publish paths, including lifecyclemodels, projected process datasets, source datasets, and delegated `process_from_flow` publish-only runs. Use when OpenClaw or another skill already produced local publish artifacts and needs one stable publish contract instead of custom per-skill glue.
---

# LCA Publish Executor

## Overview
- Accept one JSON request that points at `publish-bundle.json` files, direct dataset payloads, or `process_from_flow` run ids.
- Publish lifecyclemodels/processes/sources through `Database_CRUD_Tool`, and keep resulting-process relation metadata as a local manifest until a dedicated remote relation table exists.
- Delegate complex `process_from_flow` publish work back to `process-automated-builder`'s `--publish-only` path instead of reimplementing its flow-auto-build/process-update logic.
- If a projected process payload is not yet a canonical `processDataSet` wrapper, mark it as deferred in `publish-report.json` instead of attempting a blind remote insert.

## When To Use
- Use after `lifecyclemodel-recursive-orchestrator publish`.
- Use after `lifecyclemodel-resulting-process-builder publish`.
- Use when a caller wants one standard publish manifest for multiple skills.
- Use when OpenClaw should call a single publish skill instead of branching on each upstream skill's publish details.

## Commands
```bash
python3 scripts/lca_publish_executor.py publish \
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
- `delegated-process-build-runs/*.log` for any delegated `process_from_flow` publish runs

## Notes
- Keep MCP keys in OpenClaw `.env`; do not hardcode them in requests.
- This skill is the publish contract layer. OpenClaw should know only how to call this request shape, not the per-skill publish internals.
