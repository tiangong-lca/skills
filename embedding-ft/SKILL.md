---
name: embedding-ft
description: Execute and troubleshoot Supabase edge function `embedding_ft` that consumes PGMQ embedding jobs, calls AWS SageMaker embeddings, and writes vectors back to Postgres. Use when validating job payload handling, investigating failed embeddings, tuning ack semantics, or adjusting worker auth/environment.
---

# Embedding FT

## Run Workflow
1. Ensure `tiangong-lca-cli` is available locally, or set `TIANGONG_CLI_DIR`.
2. Set `TIANGONG_API_KEY`, `TOKEN`, or `EMBEDDING_FT_TOKEN`, or pass `--token`.
3. Execute `scripts/run-embedding-ft.sh` with a request file.
4. The wrapper delegates to `tiangong admin embedding-run`.
5. Inspect `completedJobs` and `failedJobs`, then triage via references.

## Commands
```bash
scripts/run-embedding-ft.sh --dry-run
scripts/run-embedding-ft.sh --token "$TOKEN"
scripts/run-embedding-ft.sh --data ./assets/example-jobs.json --token "$TOKEN"
TIANGONG_CLI_DIR=/path/to/tiangong-lca-cli scripts/run-embedding-ft.sh --dry-run --token "$TOKEN"
```

## Fast Triage
- `400`: request body is not a valid job array.
- `500`: SageMaker request/response parsing failure.
- `completedJobs < submitted`: inspect queue payload, row version, and content function output.

## Load References On Demand
- `references/env.md`: auth and caller environment.
- `references/job-contract.md`: queue semantics and DB side effects.
- `references/testing.md`: smoke-test and debug checklist.
