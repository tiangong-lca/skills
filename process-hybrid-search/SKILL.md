---
name: process-hybrid-search
description: Execute and troubleshoot Supabase edge function `process_hybrid_search`, which rewrites process descriptions and calls `hybrid_search_processes` with optional filters. Use when validating process query/filter behavior, tuning retrieval prompts, or debugging auth, embedding, and RPC failures.
---

# Process Hybrid Search

## Run Workflow
1. By default the wrapper runs the published CLI through `npm exec --yes --package=@tiangong-lca/cli@latest -- tiangong`. Use `TIANGONG_LCA_CLI_DIR` or `--cli-dir` only for local dev/CI overrides.
2. Set `TIANGONG_LCA_API_BASE_URL` and `TIANGONG_LCA_API_KEY`, or pass `--base-url` and `--api-key`.
3. Execute `node scripts/run-process-hybrid-search.mjs` with standard `tiangong search process` flags.
4. The wrapper delegates to `tiangong search process`.
5. Confirm response shape, then debug with focused references.

## Commands
```bash
TIANGONG_LCA_API_BASE_URL="https://example.supabase.co/functions/v1" \
TIANGONG_LCA_API_KEY="<your-api-key>" \
node scripts/run-process-hybrid-search.mjs --dry-run

TIANGONG_LCA_API_BASE_URL="https://example.supabase.co/functions/v1" \
TIANGONG_LCA_API_KEY="<your-api-key>" \
node scripts/run-process-hybrid-search.mjs

node scripts/run-process-hybrid-search.mjs \
  --input ./assets/example-request.json \
  --base-url "https://example.supabase.co/functions/v1" \
  --api-key "$TIANGONG_LCA_API_KEY"

# Force a local CLI working tree during dev/CI
TIANGONG_LCA_CLI_DIR=/path/to/tiangong-lca-cli \
node scripts/run-process-hybrid-search.mjs \
  --dry-run \
  --base-url "https://example.supabase.co/functions/v1" \
  --api-key "$TIANGONG_LCA_API_KEY"
```

## Fast Triage
- `400`: missing or invalid `query`.
- `500`: embedding provider or `hybrid_search_processes` RPC failure.
- Empty `data`: query/filter mismatch; inspect generated retrieval query and filter structure.

## Load References On Demand
- `references/env.md`: auth, region, and endpoint overrides.
- `references/request-response.md`: payload contract and RPC expectations.
- `references/prompts.md`: query-rewrite prompt constraints.
- `references/testing.md`: smoke test checklist.
