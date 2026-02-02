# Testing & debugging

```bash
curl -i --location --request POST "https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/embedding_ft" \
  --header "Authorization: Bearer $TOKEN" \
  --header 'Content-Type: application/json' \
  --data @assets/example-jobs.json
```

## Checklist
- Response lists completed vs failed jobs.
- Logs show `processing embedding job` and `embedding updated successfully` for each job.
- Ensure AWS creds valid; failures often surface as 500 `SageMaker endpoint request failed`.
- Verify `embedding_ft_at` updated and vector column populated.

## Failure triage
- 400: body not an array / schema mismatch.
- 500: SageMaker HTTP error or unexpected response shape; inspect decoded body.
- Missing rows: check `id/version` or replication lag; job is acked to avoid retry loop.
