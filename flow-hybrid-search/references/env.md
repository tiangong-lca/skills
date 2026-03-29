# Env (caller side)

- CLI path override: `TIANGONG_LCA_CLI_DIR`
- Auth variable: `TIANGONG_LCA_API_KEY`
- Base URL variable: `TIANGONG_LCA_API_BASE_URL`
- Region variable: `TIANGONG_LCA_REGION`
- Default endpoint remains `https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/flow_hybrid_search`

Wrapper behavior:

- the Node wrapper is the canonical entrypoint and only resolves the CLI path plus the default example `--input`
- the `.sh` file is a compatibility launcher that only execs the Node wrapper
- all other flags are the standard `tiangong search flow` flags
- internally it forwards to `tiangong search flow`

Model and embedding providers are configured in the deployed edge function.
