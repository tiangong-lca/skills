---
name: lifecyclemodel-hybrid-search
description: Supabase edge function supabase/functions/lifecyclemodel_hybrid_search that builds hybrid search queries for lifecycle models and calls `hybrid_search_lifecyclemodels`. Use when tuning prompts, embeddings, or auth/env for lifecycle model retrieval.
---

# Lifecycle Model Hybrid Search

## Quick start (remote only)
- Endpoint: `https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/`
- Header: `x-region: us-east-1`
- 需要 `Authorization: Bearer <TOKEN>`。
- `TOKEN` 可用 OAuth 登录获得的 JWT，或系统内生成的用户 apikey。
- 调用示例：
  ```bash
  curl -i --location --request POST "https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/lifecyclemodel_hybrid_search" \
    --header 'Content-Type: application/json' \
    --header 'x-region: us-east-1' \
    --header "Authorization: Bearer $TOKEN" \
    --data @assets/example-request.json
  ```
- 模型/SageMaker 已由远端函数配置，调用方无需额外 env。

## Request & output
- POST `{ "query": string, "filter"?: object|string }`.
- 200 with `{ data }` from RPC or `[]`; 400 if query missing; 500 on embedding/RPC failures.

## Processing flow
1) CORS OPTIONS handled.
2) ChatOpenAI structured output with schema (semantic_query_en + fulltext_query_en/zh arrays) and lifecycle-model specific system prompt.
3) Embed `semantic_query_en` via SageMaker; build full-text OR string from English+Chinese lists.
4) Call `supabase.rpc('hybrid_search_lifecyclemodels', { query_text, query_embedding, filter_condition })` (non-string filters JSON.stringify).
5) Return results with CORS headers; log errors on failure.

## Change points
- Adjust structured schema or prompt for domain nuance (e.g., LCIA, system boundaries).
- Swap model/embedding endpoint; update `OPENAI_CHAT_MODEL` or SageMaker constants.
- Filter handling: ensure caller format matches RPC expectations.

## References
- `references/env.md` — env 说明.
- `references/request-response.md` — payload, filter semantics, RPC details.
- `references/prompts.md` — lifecycle model prompt + schema.
- `references/testing.md` — curl & checklist.

## Assets
- `assets/example-request.json`
