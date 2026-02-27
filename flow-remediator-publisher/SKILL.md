---
name: flow-remediator-publisher
description: Review a curated list of existing flow datasets, generate safe remediation patches, and publish append-only new flow versions via MCP CRUD insert when batch flow governance/remediation is needed and direct database access is not allowed.
---

# Flow Remediator Publisher

## Overview

Use this skill for batch flow remediation on existing `flows` records when the input scope is a curated UUID list (for example an operator-exported `flow_list_100.json`) and the agent must not access the database directly.

This skill provides a staged pipeline to fetch flow JSON via MCP CRUD, run bootstrap review checks, generate safe fixes/proposals, validate, and append a new `uuid + version` record via `insert`.

## Scope Boundary (Avoid Overlap)

- `process-automated-builder` owns `process_from_flow` generation and publishing of process/source datasets.
- `lci-review` owns review logic/reporting (authoritative flow review should migrate there once `flow` profile is implemented).
- `flow-remediator-publisher` owns remediation and append-only publish orchestration for existing flow datasets.

Current bootstrap review in this skill is intentionally minimal and exists to support the remediation loop before `lci-review --profile flow` is available.

## Reuse Policy (Do Not Reimplement)

When classification/category or name changes require regenerating a product flow payload, reuse:

- `process-automated-builder/tiangong_lca_spec/product_flow_creation/service.py`

When fetching/publishing via MCP CRUD, reuse:

- `process-automated-builder/tiangong_lca_spec/publishing/crud.py`
- `process-automated-builder/tiangong_lca_spec/core/mcp_client.py`

Do not duplicate MCP CRUD client logic or product flow builder logic inside this skill unless blocked.

## Workflow

1. Operator exports UUID list (manual SQL outside the agent) to a local file such as `references/flow_list_100.json`.
2. Run `fetch` or `pipeline` to retrieve full flow JSON via MCP CRUD (`select`) and cache locally.
3. Run `review` to generate structured `findings.jsonl` (flow property, quantitative reference, unitgroup lookup context, duplicate similarity hints).
4. Run `propose-fix` to apply deterministic safe fixes and emit candidate fix proposals for manual/high-risk issues.
5. Run `validate` to re-review patched flows.
6. Run `publish --mode insert` to append a new flow version (version increment) via MCP CRUD `insert`.

## Commands

### One-shot pipeline (recommended for first run)

```bash
python3 scripts/run_flow_remediator.py pipeline \
  --uuid-list /abs/path/flow_list_100.json \
  --run-dir artifacts/flow-remediator/run-001 \
  --with-mcp-review-context \
  --publish-mode dry-run
```

### Staged execution

```bash
python3 scripts/run_flow_remediator.py fetch --uuid-list /abs/path/flow_list_100.json --run-dir artifacts/flow-remediator/run-001
python3 scripts/run_flow_remediator.py review --run-dir artifacts/flow-remediator/run-001 --with-mcp-context
python3 scripts/run_flow_remediator.py propose-fix --run-dir artifacts/flow-remediator/run-001
python3 scripts/run_flow_remediator.py validate --run-dir artifacts/flow-remediator/run-001 --with-mcp-context
python3 scripts/run_flow_remediator.py publish --run-dir artifacts/flow-remediator/run-001 --mode dry-run
```

## Product Flow Regeneration Helper

Use `regen-product-flow` when a remediation needs classification/name/category updates and in-place patching is not reliable. This subcommand rebuilds the flow payload by calling `ProductFlowCreationService` from `process-automated-builder`.

```bash
python3 scripts/run_flow_remediator.py regen-product-flow \
  --flow-file /abs/path/original_flow.json \
  --out-file /abs/path/regenerated_flow.json \
  --overrides-file /abs/path/request_overrides.json
```

`request_overrides.json` is a JSON object with `ProductFlowCreateRequest` fields (for example `classification`, `class_id`, `base_name_en`, `base_name_zh`).

## Required Runtime Configuration

This skill reuses `process-automated-builder` MCP configuration. Set the same environment variables before using `fetch`, `review --with-mcp-context`, or `publish`:

- `TIANGONG_LCA_REMOTE_TRANSPORT`
- `TIANGONG_LCA_REMOTE_SERVICE_NAME`
- `TIANGONG_LCA_REMOTE_URL`
- `TIANGONG_LCA_REMOTE_API_KEY`

## Load References On Demand

- `references/workflow.md`: input/output layout, staged commands, and publish safety gates
- `references/schemas.md`: bootstrap `findings`, fix proposals, and publish result schemas

