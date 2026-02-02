# Job contract & queue semantics

## Queue
- Queue name: `embedding_jobs` in PGMQ.
- Each job must be acknowledged via `pgmq.delete(queue, jobId::bigint)`; the worker does this after processing or skip cases.
- `catchUnload()` races processing to catch edge shutdown; remaining pending jobs are marked failed in response payload only.

## Job payload
```json
{
  "jobId": 123,
  "id": "uuid",
  "version": "v1",
  "schema": "public",
  "table": "documents",
  "contentFunction": "content_for_embedding",
  "embeddingColumn": "embedding_ft"
}
```
- `contentFunction` is invoked as `${contentFunction}(t)` where `t` is the row; it must return string content.
- If row missing/version mismatch or content not string -> job is acked (deleted) and logged.
- Successful update also sets `embedding_ft_at = now()`.

## Expected DB side effects
- Update target row embedding column with JSON array of numbers.
- No upsert/insert: if `count === 0`, job is logged and acked, not retried.
