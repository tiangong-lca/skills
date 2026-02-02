# Env (caller side)
- `REMOTE_ENDPOINT`: fixed `https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/`
- `X_REGION`: fixed `us-east-1`
- 已使用 `--no-verify-jwt` 部署，调用方不需要 `Authorization` / `apikey`。

Model/embedding configuration is already set in the deployed function; caller does not provide these.
