# Testing & debugging

## Preferred smoke test
```bash
TIANGONG_LCA_API_BASE_URL="https://example.supabase.co/functions/v1" \
TIANGONG_LCA_API_KEY="<your-api-key>" \
node scripts/run-embedding-ft.mjs
```

## Dry run (request preview)
```bash
TIANGONG_LCA_API_BASE_URL="https://example.supabase.co/functions/v1" \
TIANGONG_LCA_API_KEY="<your-api-key>" \
node scripts/run-embedding-ft.mjs --dry-run
```

## Direct CLI equivalent
```bash
npm exec --yes --package=@tiangong-lca/cli@latest -- tiangong \
  admin embedding-run \
  --input ./assets/example-jobs.json \
  --base-url "https://example.supabase.co/functions/v1" \
  --api-key "$TIANGONG_LCA_API_KEY" \
  --dry-run
```

Use `TIANGONG_LCA_CLI_DIR=/path/to/tiangong-lca-cli node scripts/run-embedding-ft.mjs ...` only when validating an unpublished local CLI working tree.

## Checklist
- Response contains `completedJobs` and `failedJobs`.
- Logs include `processing embedding job` and successful update messages.
- Target row updates embedding vector and `embedding_ft_at`.

## Failure triage
- `400`: body is not a valid job array or field type mismatch.
- `500`: SageMaker request failure or unsupported embedding response shape.
- Missing row/content: check `id`, `version`, and `contentFunction`; job is acked to avoid retry loops.
