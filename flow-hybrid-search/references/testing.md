# Testing

## Preferred smoke test
```bash
TIANGONG_LCA_API_BASE_URL="https://example.supabase.co/functions/v1" \
TIANGONG_LCA_API_KEY="<your-api-key>" \
node scripts/run-flow-hybrid-search.mjs
```

## Dry run (request preview)
```bash
TIANGONG_LCA_API_BASE_URL="https://example.supabase.co/functions/v1" \
TIANGONG_LCA_API_KEY="<your-api-key>" \
node scripts/run-flow-hybrid-search.mjs --dry-run
```

## Direct CLI equivalent
```bash
npx -y @tiangong-lca/cli@latest \
  search flow \
  --input ./assets/example-request.json \
  --base-url "https://example.supabase.co/functions/v1" \
  --api-key "$TIANGONG_LCA_API_KEY" \
  --dry-run
```

Use `TIANGONG_LCA_CLI_DIR=/path/to/tiangong-lca-cli node scripts/run-flow-hybrid-search.mjs ...` only when validating an unpublished local CLI working tree.

## Checklist
- 200 response contains `data` (array, possibly empty).
- 400 appears only when `query` is missing/invalid.
- 500 indicates embedding provider or RPC failure (inspect Supabase logs).
