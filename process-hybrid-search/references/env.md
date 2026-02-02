# Env (caller side)
- Endpoint: `https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/`
- Header: `x-region: us-east-1`
- Auth: `Authorization: Bearer <TOKEN>`
- `TOKEN` 可用 OAuth 登录获得的 JWT，或系统内生成的用户 apikey。

Model/embedding settings are already configured on the deployed edge function; callers do not supply them.
