# Testing

## Preferred smoke test
```bash
TIANGONG_LCA_API_BASE_URL="https://example.supabase.co/functions/v1" \
TIANGONG_LCA_API_KEY="<your-api-key>" \
node scripts/run-lifecyclemodel-hybrid-search.mjs
```

## Dry run (request preview)
```bash
TIANGONG_LCA_API_BASE_URL="https://example.supabase.co/functions/v1" \
TIANGONG_LCA_API_KEY="<your-api-key>" \
node scripts/run-lifecyclemodel-hybrid-search.mjs --dry-run
```

## Direct CLI equivalent
```bash
node "${TIANGONG_LCA_CLI_DIR:-../tiangong-lca-cli}/bin/tiangong.js" \
  search lifecyclemodel \
  --input ./assets/example-request.json \
  --base-url "https://example.supabase.co/functions/v1" \
  --api-key "$TIANGONG_LCA_API_KEY" \
  --dry-run
```

## Checklist
- 200 response contains `data` (array, possibly empty).
- 400 appears only when `query` is missing/invalid.
- 500 indicates embedding provider or RPC failure (inspect Supabase logs).
