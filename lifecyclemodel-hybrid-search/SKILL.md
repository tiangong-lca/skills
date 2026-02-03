---
name: lifecyclemodel-hybrid-search
description: Supabase edge function supabase/functions/lifecyclemodel_hybrid_search that converts lifecycle model descriptions into hybrid search queries and calls `hybrid_search_lifecyclemodels`. Use when debugging lifecycle model search, tuning prompts/filters, or adjusting embedding endpoints/auth.
---

# Lifecycle Model Hybrid Search

## Required parameters
- `data`: JSON payload containing the `query` string the user wants to search (optional `filter`).
- `TIANGONG_LCA_APIKEY`: user key derived from email + password; send as `Authorization: Bearer <TIANGONG_LCA_APIKEY>`.

## Quick start
- Endpoint: `https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/`
- Header: `x-region: us-east-1`
- Requires `Authorization: Bearer <TIANGONG_LCA_APIKEY>`.
- Body: JSON with `query` string (optional `filter`); `assets/example-request.json` shows the format.

- Example call:
  ```bash
  curl -sS --location --request POST "https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/lifecyclemodel_hybrid_search" \
    --header 'Content-Type: application/json' \
    --header 'x-region: us-east-1' \
    --header "Authorization: Bearer $TIANGONG_LCA_APIKEY" \
    --data '{"query": "Attributional lifecycle model for recycled aluminum billet"}'
  ```

## Request & output
- POST `{ "query": string, "filter"?: object|string }`.
- Responses: 200 with `{ data }` or `[]`; 400 if `query` missing; 500 on embedding/RPC errors.

## References
- `references/env.md`
- `references/request-response.md`
- `references/prompts.md` - prompt requirements for query generation.
- `references/testing.md`

## Assets
- `assets/example-request.json`
