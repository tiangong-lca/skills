# Env (caller side)
- Endpoint: `https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/`
- Header: `x-region: us-east-1`
- Auth: `Authorization: Bearer <TOKEN>`
- `TOKEN` 可用 OAuth 登录获得的 JWT，或系统内生成的用户 key。

Model/embedding configuration is already set in the deployed function; caller does not provide these.
