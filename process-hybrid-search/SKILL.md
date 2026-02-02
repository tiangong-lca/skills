---
name: process-hybrid-search
description: Supabase edge function supabase/functions/process_hybrid_search that converts process descriptions into hybrid search queries and calls `hybrid_search_processes`. Use when debugging process search, tuning prompts/filters, or adjusting embedding endpoints/auth.
---

# Process Hybrid Search

## Quick start (remote only)
- Endpoint: `https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/`
- Header: `x-region: us-east-1`
- 需要 `Authorization: Bearer <TOKEN>`。
- `TOKEN` 可用 OAuth 登录获得的 JWT，或系统内生成的用户 key。
- 调用示例：
  ```bash
  curl -i --location --request POST "https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/process_hybrid_search" \
    --header 'Content-Type: application/json' \
    --header 'x-region: us-east-1' \
    --header "Authorization: Bearer $TOKEN" \
    --data @assets/example-request.json
  ```
- 模型/embedding 配置已在远端函数；调用方无需设置。

## Request & output
- POST `{ "query": string, "filter"?: object|string }`.
- Responses: 200 with `{ data }` or `[]`; 400 if `query` missing; 500 on embedding/RPC errors.

## Processing flow
1) Handle OPTIONS for CORS.
2) ChatOpenAI structured output (semantic_query_en + fulltext_query_en/zh arrays) with process-focused LCA system prompt.
3) Build OR-joined full-text string; embed `semantic_query_en` through SageMaker endpoint.
4) Call `supabase.rpc('hybrid_search_processes', { query_text, query_embedding, filter_condition })`; non-string filters get `JSON.stringify`.
5) Return results with CORS headers; errors logged and surfaced as JSON 500.

## Change points
- Update prompt/schema to capture additional process metadata (e.g., technology, geography).
- Swap models/endpoints via env vars and SageMaker constants.
- Adjust RPC name/params to target different index/table.

## References
- `references/env.md`
- `references/request-response.md`
- `references/prompts.md`
- `references/testing.md`

## Assets
- `assets/example-request.json`
