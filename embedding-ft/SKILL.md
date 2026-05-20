---
name: embedding-ft
description: Execute and troubleshoot Supabase edge function `embedding_ft` that consumes PGMQ embedding jobs, calls AWS SageMaker embeddings, and writes vectors back to Postgres. Use when validating job payload handling, investigating failed embeddings, tuning ack semantics, or adjusting worker auth/environment.
---

# Embedding FT

## Run Workflow
1. By default the wrapper runs the published CLI through `npm exec --yes --package=@tiangong-lca/cli@latest -- tiangong`. Use `TIANGONG_LCA_CLI_DIR` or `--cli-dir` only for local dev/CI overrides.
2. Set `TIANGONG_LCA_API_BASE_URL` and `TIANGONG_LCA_API_KEY`, or pass `--base-url` and `--api-key`.
3. Execute `node scripts/run-embedding-ft.mjs` with standard `tiangong admin embedding-run` flags.
4. The wrapper delegates to `tiangong admin embedding-run`.
5. Inspect `completedJobs` and `failedJobs`, then triage via references.

## Commands
```bash
TIANGONG_LCA_API_BASE_URL="https://example.supabase.co/functions/v1" \
TIANGONG_LCA_API_KEY="<your-api-key>" \
node scripts/run-embedding-ft.mjs --dry-run

TIANGONG_LCA_API_BASE_URL="https://example.supabase.co/functions/v1" \
TIANGONG_LCA_API_KEY="<your-api-key>" \
node scripts/run-embedding-ft.mjs

node scripts/run-embedding-ft.mjs \
  --input ./assets/example-jobs.json \
  --base-url "https://example.supabase.co/functions/v1" \
  --api-key "$TIANGONG_LCA_API_KEY"

# Force a local CLI working tree during dev/CI
TIANGONG_LCA_CLI_DIR=/path/to/tiangong-lca-cli \
node scripts/run-embedding-ft.mjs \
  --dry-run \
  --base-url "https://example.supabase.co/functions/v1" \
  --api-key "$TIANGONG_LCA_API_KEY"
```

## Fast Triage
- `400`: request body is not a valid job array.
- `500`: SageMaker request/response parsing failure.
- `completedJobs < submitted`: inspect queue payload, row version, and content function output.

## Load References On Demand
- `references/env.md`: auth and caller environment.
- `references/job-contract.md`: queue semantics and DB side effects.
- `references/testing.md`: smoke-test and debug checklist.
