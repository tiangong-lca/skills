# Testing

```bash
curl -i --location --request POST "https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/process_hybrid_search" \
  --header 'Content-Type: application/json' \
  --header 'x-region: us-east-1' \
  --header "Authorization: Bearer $TIANGONG_LCA_APIKEY" \
  --data @assets/example-request.json
```

Checklist:
- 200 response; `data` array present or empty.
- 400 only when `query` missing.
- 500 indicates backend model/RPC issue (check Supabase logs).
