---
name: lca-publish-executor
description: Publish local LCA artifact bundles through the unified `tiangong-lca publish run` contract. Use when another skill already produced local publish artifacts and needs one stable publish request instead of per-skill glue.
---

# LCA Publish Executor

## Overview
- Accept one JSON request that points at `publish-bundle.json` files, direct dataset payloads, or `process_from_flow` run ids.
- Forward that request shape to `tiangong-lca publish run`.
- Keep relation metadata local when the publish mode says `local_manifest_only`.
- Reuse publish bundles prepared by upstream CLI-backed builders instead of inventing a second publish contract here.

## When To Use
- Use after `lifecyclemodel-recursive-orchestrator publish`.
- Use after `lifecyclemodel-resulting-process-builder publish`.
- Use when a caller wants one standard publish manifest for multiple skills.
- Use when another caller should know only one publish request shape.

## Commands
```bash
node scripts/run-lca-publish-executor.mjs publish \
  --request assets/example-request.json \
  --dry-run \
  --json
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
- whatever `tiangong-lca publish run` emits for the request shape
- at minimum expect `publish-report.json`
- expect `verification-report.json` next to the publish report; read it before treating a dry-run or commit as complete
- when relation mode stays local, also expect a local relation manifest in the publish output bundle

## Notes
- This wrapper is CLI-only; there is no Python or MCP fallback path.
- Keep this skill as a stable request façade only. Do not add publish internals here.
- Do not publish if upstream identity, build-plan, schema, bilingual, review, reference, matrix-readiness, or account-verification gates are missing or blocked.
- If `verification-report.json` contains blockers, stop and hand off the artifact path instead of retrying blindly.
- After a commit-mode publish, run the appropriate post-write verification: `tiangong-lca dataset verify-remote` for TIDAS references and `tiangong-lca process verify-rows` for persisted process row snapshots.
