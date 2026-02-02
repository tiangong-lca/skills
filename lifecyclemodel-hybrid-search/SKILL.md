---
name: lifecyclemodel-hybrid-search
description: Supabase edge function supabase/functions/lifecyclemodel_hybrid_search that builds hybrid search queries for lifecycle models and calls `hybrid_search_lifecyclemodels`. Use when tuning prompts, embeddings, or auth/env for lifecycle model retrieval.
---

# Lifecycle Model Hybrid Search

## Quick start (remote only)
- 固定配置：
  - `REMOTE_ENDPOINT=https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/`
  - `X_REGION=us-east-1`
- 已以 `--no-verify-jwt` 部署，调用方**不需要** JWT（`Authorization` / `apikey`）。
- 调用示例：
  ```bash
  curl -i --location --request POST "https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/lifecyclemodel_hybrid_search" \
    --header 'Content-Type: application/json' \
    --header 'x-region: us-east-1' \
    --data @assets/example-request.json
  ```
- JS/TS（supabase-js）调用示例：
  ```ts
  import { createClient } from "@supabase/supabase-js";

  const supabase = createClient(process.env.SUPABASE_URL!, process.env.SUPABASE_ANON_KEY!);
  const { data, error } = await supabase.functions.invoke("lifecyclemodel_hybrid_search", {
    body: { query: "steel scrap flow to furnace", filter: { country: "DE" } },
    headers: { "x-region": "us-east-1" },
  });
  ```
- supabase-js 会自动携带 anon key（`Authorization`），但函数不强制要求。
- 模型/SageMaker 已由远端函数配置，调用方无需额外 env。

## Request & output
- POST `{ "query": string, "filter"?: object|string }`.
- 200 with `{ data }` from RPC or `[]`; 400 if query missing; 500 on embedding/RPC failures.

## Processing flow
1) CORS OPTIONS handled; Edge runtime 未做 JWT 校验（部署使用 `--no-verify-jwt`）。
2) ChatOpenAI structured output with schema (semantic_query_en + fulltext_query_en/zh arrays) and lifecycle-model specific system prompt.
3) Embed `semantic_query_en` via SageMaker; build full-text OR string from English+Chinese lists.
4) Call `supabase.rpc('hybrid_search_lifecyclemodels', { query_text, query_embedding, filter_condition })` (non-string filters JSON.stringify).
5) Return results with CORS headers; log errors on failure.

## Change points
- Adjust structured schema or prompt for domain nuance (e.g., LCIA, system boundaries).
- Swap model/embedding endpoint; update `OPENAI_CHAT_MODEL` or SageMaker constants.
- Filter handling: ensure caller format matches RPC expectations.

## References
- `references/env.md` — env + serve command.
- `references/request-response.md` — payload, filter semantics, RPC details.
- `references/prompts.md` — lifecycle model prompt + schema.
- `references/testing.md` — curl & checklist.

## Assets & scripts
- `assets/example-request.json`
- `scripts/invoke_local.sh`
