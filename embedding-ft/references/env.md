# Env (caller side)

- CLI path override: `TIANGONG_CLI_DIR`
- Preferred auth variable: `TIANGONG_API_KEY`
- Compatible auth aliases: `TIANGONG_LCA_APIKEY`, `TOKEN`, `EMBEDDING_FT_TOKEN`
- Preferred base URL variable: `TIANGONG_API_BASE_URL`
- Compatible base URL alias: `SUPABASE_FUNCTIONS_URL`
- Default endpoint remains `https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/embedding_ft`

Wrapper behavior:

- the shell script preserves `--token`, `--data`, `--base-url`, and `--max-time`
- internally it forwards to `tiangong admin embedding-run`
