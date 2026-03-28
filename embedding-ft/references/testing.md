# Testing & debugging

## Preferred smoke test
```bash
scripts/run-embedding-ft.sh --token "$TOKEN"
```

## Dry run (request preview)
```bash
scripts/run-embedding-ft.sh --dry-run --token "$TOKEN"
```

## Direct CLI equivalent
```bash
node "${TIANGONG_CLI_DIR:-../tiangong-lca-cli}/bin/tiangong.js" admin embedding-run --input ./assets/example-jobs.json --api-key "$TIANGONG_API_KEY" --dry-run
```

## Checklist
- Response contains `completedJobs` and `failedJobs`.
- Logs include `processing embedding job` and successful update messages.
- Target row updates embedding vector and `embedding_ft_at`.

## Failure triage
- `400`: body is not a valid job array or field type mismatch.
- `500`: SageMaker request failure or unsupported embedding response shape.
- Missing row/content: check `id`, `version`, and `contentFunction`; job is acked to avoid retry loops.
