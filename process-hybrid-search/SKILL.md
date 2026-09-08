---
name: process-hybrid-search
description: Execute and troubleshoot Supabase edge function `process_hybrid_search`, which rewrites process descriptions and calls `hybrid_search_processes` with optional filters. Use when validating process query/filter behavior, tuning retrieval prompts, or debugging auth, embedding, and RPC failures.
---

# Process Hybrid Search

## Run Workflow
1. The wrapper uses the repository-pinned published CLI. Use `TIANGONG_LCA_CLI_DIR` or `--cli-dir` only for an exact local dev/CI override.
2. Official Production needs no environment setup: the CLI owns its public profile. Only Dev/self-hosted/custom clients need the complete matching configuration in `references/env.md`.
3. Run `tiangong-lca auth status --json`. If it returns `login-required`, stop and ask the human user to run `tiangong-lca auth login` in a trusted terminal.
4. Execute `node scripts/run-process-hybrid-search.mjs` with standard `tiangong-lca search process` flags.
5. The wrapper delegates to `tiangong-lca search process`; the CLI resolves and refreshes its private OAuth session.
6. Confirm response shape, then debug with focused references.

Never ask an AI user for a username, password, authorization code, access token, refresh token, or the deprecated encoded API key. Headless runs require an explicit destination/publishable key and may use only an orchestrator-injected short-lived `TIANGONG_LCA_ACCESS_TOKEN`; do not pass it through argv, prompts, logs, or artifacts.

## Commands

```bash
pnpm dlx --package=@tiangong-lca/cli@0.1.11 tiangong-lca auth status --json

# Only the human user completes browser login when login-required is reported
pnpm dlx --package=@tiangong-lca/cli@0.1.11 tiangong-lca auth login

node scripts/run-process-hybrid-search.mjs --dry-run
node scripts/run-process-hybrid-search.mjs --input ./assets/example-request.json

# Explicit local CLI override for development only
node scripts/run-process-hybrid-search.mjs --cli-dir /path/to/tiangong-lca-cli --dry-run
```

## Fast Triage

- An independently installed skill includes its launcher and example input; no Skills/CLI/Data Foundry checkout is required. The launcher is a byte-checked bundle of the repository authority, not a separate auth implementation.
- Fresh `auth status` / `doctor-auth` returns `login-required` (exit 1), not a missing Production client. A custom `--base-url` requires matching complete custom configuration.
- `400`: missing or invalid `query`.
- `500`: embedding provider or `hybrid_search_processes` RPC failure.
- Empty `data`: query/filter mismatch; inspect generated retrieval query and filter structure.

## Load References On Demand
- `references/env.md`: auth, region, and endpoint overrides.
- `references/request-response.md`: payload contract and RPC expectations.
- `references/prompts.md`: query-rewrite prompt constraints.
- `references/testing.md`: smoke test checklist.
