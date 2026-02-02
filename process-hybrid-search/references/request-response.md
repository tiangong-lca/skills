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
- 需要 `Authorization: Bearer <TOKEN>`。
- `TOKEN` 可用 OAuth 登录获得的 JWT，或系统内生成的用户 apikey。
