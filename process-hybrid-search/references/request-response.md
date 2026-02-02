# Request/response

## Input
```json
{
  "query": "Mechanical recycling process for PET bottles",
  "filter": {"region": "CN"}
}
```

## Output
- 200 `{ "data": [...] }` from `hybrid_search_processes` RPC or `[]` if none.
- 400 if `query` missing; 500 on model/embedding/RPC errors.

## RPC expectation
- Function `hybrid_search_processes(query_text text, query_embedding text, filter_condition jsonb|text)`.

## Auth
- Deployed with `--no-verify-jwt`; request does not require `Authorization` / `apikey`.
