# Env (caller side)
- Endpoint: `https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/`
- Header: `x-region: us-east-1`
- Auth: `Authorization: Bearer <TOKEN>`
- `TOKEN` is either an OAuth JWT or a user key generated in the system (derived from email + password).

Model/embedding configuration is already baked into the deployed function; callers do not set extra envs.
