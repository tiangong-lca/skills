# Workflow

## Inputs

- Curated UUID list file (JSON/JSONL/TXT), typically manually exported via SQL by an operator.
- MCP CRUD access (read-only for `fetch`/review context, append-only `insert` for publish).
- Optional local overrides JSON for product flow regeneration.

## Run Directory Layout

Example: `artifacts/flow-remediator/run-001`

- `cache/flows/`: fetched flow JSON files (`<uuid>_<version>.json`)
- `fetch/fetch_log.jsonl`
- `fetch/fetch_summary.json`
- `review/findings.jsonl`
- `review/flow_summaries.jsonl`
- `review/similarity_pairs.jsonl`
- `review/review_summary.json`
- `fix/fix_proposals.jsonl`
- `fix/patch_manifest.jsonl`
- `fix/patched_flows/`
- `validate/` (same shape as `review/`)
- `publish/publish_results.jsonl`
- `publish/publish_summary.json`
- `pipeline_summary.json`

## Command Modes

## `fetch`

- Source of truth is the UUID list file, not arbitrary SQL.
- Uses MCP CRUD `select` on `flows`.
- Stores exact returned JSON locally for reproducibility.

## `review`

Bootstrap checks in initial version:

- `typeOfDataSet` presence and non-elementary gating
- `name` subtree recursive scan (including `Emergy`)
- quantitative reference vs selected flow property internal ID consistency
- optional flowproperty/unitgroup lookup via MCP CRUD for context
- same-category high-similarity hints (near-duplicate risk)

This is intentionally a bootstrap reviewer until `lci-review --profile flow` exists.

## `propose-fix`

Safe deterministic fixes only (initial version):

- fill missing dataset version
- align `referenceToReferenceFlowProperty` with selected `flowProperty` internal ID

High-risk or heuristic issues stay as candidate proposals and must be reviewed or regenerated.

## `validate`

- Re-runs bootstrap review on `fix/patched_flows`.
- Used as a regression gate before publish.

## `publish`

- Reads `fix/patch_manifest.jsonl`
- Re-checks latest version in DB via MCP CRUD `select`
- Increments version (`xx.xx.xxx` patch segment)
- Publishes via MCP CRUD `insert` (append-only)

Recommended default:

- `--mode dry-run`
- keep base-version check enabled (do not pass `--skip-base-check`)

## Product Flow Regeneration

Use `regen-product-flow` when patching is insufficient (for example classification/category or canonical name changes).

This subcommand rebuilds payloads by reusing:

- `process-automated-builder/tiangong_lca_spec/product_flow_creation/service.py`

## Safety Gates

- No direct database access in the skill.
- UUID scope is externally curated.
- Review and fix outputs are persisted before publish.
- Publish is append-only (`insert`) with version bump.
- Base version drift check is enabled by default.

