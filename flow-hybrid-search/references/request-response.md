# Request/response contract

## Input
```json
{
  "query": "steel scrap flow to furnace",
  "filter": {"country": "DE"}
}
```
- `filter` optional; if string, used directly; otherwise `JSON.stringify(filter)`.

## Output
- 200 `{ "data": [...] }` where rows come from `hybrid_search_flows` RPC; returns `[]` when no matches.
- 400 on missing `query`.
- 500 on RPC or embedding/model errors (`error.message` surfaced).

## Auth
- Requires `Authorization: Bearer <TOKEN>`.
- `TOKEN` is either an OAuth JWT or a user key generated in the system (derived from email + password).

## RPC expectation
- Expects Postgres function `hybrid_search_flows(query_text text, query_embedding text, filter_condition jsonb|text)`.
