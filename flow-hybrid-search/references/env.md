# Env (caller side)

- CLI path override: `TIANGONG_LCA_CLI_DIR`
- Default CLI runtime: `npm exec --yes --package=@tiangong-lca/cli@latest -- tiangong`
- Auth variable: `TIANGONG_LCA_API_KEY`
- Base URL variable: `TIANGONG_LCA_API_BASE_URL`
- Region variable: `TIANGONG_LCA_REGION`
- Default endpoint remains `https://qgzvkongdjqiiamzbbts.supabase.co/functions/v1/flow_hybrid_search`

Wrapper behavior:

- the Node `.mjs` wrapper runs the published CLI by default and injects the example `--input` file when none is provided
- set `TIANGONG_LCA_CLI_DIR` or pass `--cli-dir` only when you need a local CLI working tree
- all other flags are the standard `tiangong search flow` flags
- internally it forwards to `tiangong search flow`

Model and embedding providers are configured in the deployed edge function.
