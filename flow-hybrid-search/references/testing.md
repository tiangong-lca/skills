# Testing

## Preferred smoke test
```bash
scripts/run-flow-hybrid-search.sh --token "$TIANGONG_LCA_APIKEY"
```

## Dry run (request preview)
```bash
scripts/run-flow-hybrid-search.sh --dry-run --token "$TIANGONG_LCA_APIKEY"
```

## Direct CLI equivalent
```bash
node "${TIANGONG_CLI_DIR:-../tiangong-lca-cli}/bin/tiangong.js" search flow --input ./assets/example-request.json --api-key "$TIANGONG_API_KEY" --dry-run
```

## Checklist
- 200 response contains `data` (array, possibly empty).
- 400 appears only when `query` is missing/invalid.
- 500 indicates embedding provider or RPC failure (inspect Supabase logs).
