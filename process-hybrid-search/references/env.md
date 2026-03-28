# Env (caller side)

- CLI path override: `TIANGONG_CLI_DIR`
- Preferred auth variable: `TIANGONG_API_KEY`
- Compatible auth alias: `TIANGONG_LCA_APIKEY`
- Preferred base URL variable: `TIANGONG_API_BASE_URL`
- Compatible base URL alias: `SUPABASE_FUNCTIONS_URL`
- Preferred region variable: `TIANGONG_REGION`
- Compatible region alias: `SUPABASE_FUNCTION_REGION`
- Default endpoint remains `https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/process_hybrid_search`

Wrapper behavior:

- the shell script preserves `--token`, `--data`, `--base-url`, `--region`, and `--max-time`
- internally it forwards to `tiangong search process`

Model and embedding providers are configured in the deployed edge function.
