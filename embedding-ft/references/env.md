# Env (caller side)

- CLI path override: `TIANGONG_LCA_CLI_DIR`
- Default CLI runtime: `npx -y @tiangong-lca/cli@latest`
- Auth variable: `TIANGONG_LCA_API_KEY`
- Base URL variable: `TIANGONG_LCA_API_BASE_URL`
- Default endpoint remains `https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/embedding_ft`

Wrapper behavior:

- the Node `.mjs` wrapper runs the published CLI by default and injects the example `--input` file when none is provided
- set `TIANGONG_LCA_CLI_DIR` or pass `--cli-dir` only when you need a local CLI working tree
- all other flags are the standard `tiangong admin embedding-run` flags
- internally it forwards to `tiangong admin embedding-run`
