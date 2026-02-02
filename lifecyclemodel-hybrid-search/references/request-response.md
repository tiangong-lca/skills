# Request/response

## Input
```json
{
  "query": "Cradle-to-gate model for recycled aluminum ingot",
  "filter": {"region": "EU"}
}
```

## Output
- 200 `{ "data": [...] }` from `hybrid_search_lifecyclemodels` RPC.
- Empty list -> `[]` with 200.
- 400 if `query` missing; 500 on SageMaker/OpenAI/RPC failures.

## RPC expectation
- Function `hybrid_search_lifecyclemodels(query_text text, query_embedding text, filter_condition jsonb|text)`.

## Auth
- 需要 `Authorization: Bearer <TOKEN>`。
- `TOKEN` 可用 OAuth 登录获得的 JWT，或系统内生成的用户 apikey。
