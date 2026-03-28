---
name: flow-hybrid-search
description: Execute and troubleshoot Supabase edge function `flow_hybrid_search`, which rewrites flow descriptions and calls `hybrid_search_flows` with optional filters. Use when validating flow query/filter behavior, tuning retrieval prompts, or debugging auth, embedding, and RPC failures.
---

# Flow Hybrid Search

## Run Workflow
1. Ensure `tiangong-lca-cli` is available locally, or set `TIANGONG_CLI_DIR`.
2. Set `TIANGONG_API_KEY` / `TIANGONG_LCA_APIKEY`, or pass `--token`.
3. Execute `scripts/run-flow-hybrid-search.sh` with a request JSON file.
4. The wrapper delegates to `tiangong search flow`.
5. Confirm response shape, then debug with focused references.

## Commands
```bash
scripts/run-flow-hybrid-search.sh --dry-run --token "$TIANGONG_LCA_APIKEY"
scripts/run-flow-hybrid-search.sh --token "$TIANGONG_LCA_APIKEY"
scripts/run-flow-hybrid-search.sh --data ./assets/example-request.json --token "$TIANGONG_LCA_APIKEY"
TIANGONG_CLI_DIR=/path/to/tiangong-lca-cli scripts/run-flow-hybrid-search.sh --dry-run --token "$TIANGONG_API_KEY"
```

## Fast Triage
- `400`: missing or invalid `query`.
- `500`: embedding provider or `hybrid_search_flows` RPC failure.
- Empty `data`: query/filter mismatch; inspect generated retrieval query and filter structure.

## Load References On Demand
- `references/env.md`: auth, region, and endpoint overrides.
- `references/request-response.md`: payload contract and RPC expectations.
- `references/prompts.md`: query-rewrite prompt constraints.
- `references/testing.md`: smoke test checklist.
