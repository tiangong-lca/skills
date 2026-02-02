---
name: flow-hybrid-search
description: Supabase edge function supabase/functions/flow_hybrid_search that turns a flow description into LCA-aware full-text + embedding queries and calls `hybrid_search_flows`. Use when debugging hybrid search for flows, tuning prompts/models, or wiring auth/env.
---

# Flow Hybrid Search

## Quick start (remote only)
- Endpoint: `https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/`
- Header: `x-region: us-east-1`
- 需要 `Authorization: Bearer <TOKEN>`。
- `TOKEN` 可用 OAuth 登录获得的 JWT，或系统内生成的用户 apikey。
- 调用示例：
  ```bash
  curl -i --location --request POST "https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/flow_hybrid_search" \
    --header 'Content-Type: application/json' \
    --header 'x-region: us-east-1' \
    --header "Authorization: Bearer $TOKEN" \
    --data @assets/example-request.json
  ```
- 模型/SageMaker 已在远端配置，调用方无需设置。

## Request & output
- POST JSON: `{ "query": string, "filter"?: object|string }`.
- Returns 200 with `{ data }` array or `[]`; 400 if `query` missing; 500 on RPC/embedding errors.

## Processing flow
1) OPTIONS handled for CORS via `_shared/cors`.
2) Run ChatOpenAI (temperature 0, `OPENAI_CHAT_MODEL`) with structured output schema: `semantic_query_en` (string) + `fulltext_query_en[]` + `fulltext_query_zh[]`; prompt is flow-specific LCA instructions.
3) Combine full-text queries: `(q1) OR (q2)…`; generate embedding for `semantic_query_en` using SageMaker endpoint (JSON `{inputs: text}`) and extract first number array from response.
4) Call `supabase.rpc('hybrid_search_flows', { query_text, query_embedding: "[v1,...]", filter_condition })`.
5) Respond with data or empty array; errors logged and returned as JSON 500.

## Change points
- Prompt/fields: edit query schema or system prompt for different outputs.
- Model: switch `OPENAI_CHAT_MODEL` or SageMaker endpoint payload/region.
- Search backend: change RPC name/params to target different table/index.
- Filter handling: ensure callers send string vs object; function serializes non-string via `JSON.stringify`.

## References
- `references/env.md` — env 说明.
- `references/request-response.md` — payload, filters, and RPC expectations.
- `references/prompts.md` — current LCA prompt and structured schema.
- `references/testing.md` — curl and debugging checklist.

## Assets
- `assets/example-request.json` — sample query/filter body.
