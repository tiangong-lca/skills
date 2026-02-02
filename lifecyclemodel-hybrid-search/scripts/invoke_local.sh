#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/assets/example-request.json"
URL=${1:-${REMOTE_ENDPOINT:-http://localhost:54321/functions/v1}/lifecyclemodel_hybrid_search}
REGION=${X_REGION:-us-east-1}
API_KEY=${API_KEY:-${USER_API_KEY:-${SUPABASE_SERVICE_ROLE_KEY:-}}}

HEADERS=(-H "Content-Type: application/json" -H "x-region: $REGION")
if [[ -n "$API_KEY" ]]; then
  HEADERS+=(-H "Authorization: Bearer $API_KEY" -H "apikey: $API_KEY")
fi

curl -sS -X POST "${HEADERS[@]}" -d @"$DATA" "$URL"
