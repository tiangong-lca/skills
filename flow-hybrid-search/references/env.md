# Env (caller side)
- Endpoint: `https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/`
- Header: `x-region: us-east-1`
- Auth: `Authorization: Bearer <TIANGONG_LCA_APIKEY>`
- `TIANGONG_LCA_APIKEY` is a user key generated in the system (derived from email + password).

Model/embedding settings are already configured on the deployed edge function; callers do not supply them.
