# Request/response

## Endpoint
- POST `https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/process_hybrid_search`
- Headers: `Authorization: Bearer <TIANGONG_LCA_APIKEY>`, `x-region: us-east-1`

## Input
```json
{
  "query": "Open-loop mechanical recycling process for HDPE packaging",
  "filter": {
    "processInformation": {
      "geography": {
        "locationOfOperationSupplyOrProduction": {
          "@location": "CN"
        }
      },
      "time": {
        "common:referenceYear": 2021
      }
    }
  }
}
```
- `filter` optional; if string, used as-is; if object, passed as JSON.

## Output
- 200 `{ "data": [...] }` from `hybrid_search_processes`; returns `[]` when no matches.
- 400 when `query` is missing; 500 on embedding/model/RPC errors.

## RPC expectation
- Expects Postgres function `hybrid_search_processes(query_text text, query_embedding text, filter_condition jsonb|text)`.

## Auth
- `Authorization: Bearer <TIANGONG_LCA_APIKEY>` required; `TIANGONG_LCA_APIKEY` is a user key derived from email + password.
