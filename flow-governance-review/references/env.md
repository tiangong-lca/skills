# Env

## Local-First Mode

Prefer local JSON or JSONL inputs. In local mode, no remote credentials are required.

`--process-pool-file` is also local-first: it is just a JSON or JSONL working pool of exact-version process rows and does not require any remote credential by itself.

## Optional Live Inputs

`flow-dedup-candidates` can enrich its candidate analysis with live Supabase data.

Set:

- `SUPABASE_URL`
- `SUPABASE_PUBLISHABLE_KEY`

And one of:

- `SUPABASE_ACCESS_TOKEN`
- `SUPABASE_EMAIL` plus `SUPABASE_PASSWORD`

Optional overrides:

- `FLOW_GOVERNANCE_USER_ID`: default user id for live `state_code=0` flow fetches
- `FLOW_GOVERNANCE_ILCD_FLOW_DIR`: EF reference package flow directory used to distinguish ILCD public rows from TianGong public rows

## Optional MCP Inputs

`openclaw-entry --flow-id`, `openclaw-full-run`, and `publish-reviewed-data --commit` can use the TianGong LCA MCP endpoint.

Set as needed:

- `TIANGONG_LCA_REMOTE_URL`
- `TIANGONG_LCA_REMOTE_API_KEY`

Notes:

- `openclaw-entry --flow-id` prefers MCP CRUD to fetch the subject flow when the MCP credentials are valid.
- If MCP auth is missing or returns `401 Unauthorized`, UUID mode can only fall back to PostgREST when the Supabase env above is also available.
- `publish-reviewed-data` stays dry-run unless `--commit` is passed.
- `publish-reviewed-data --original-flow-rows-file` can use a local original snapshot to skip unchanged flow rows before planning any version bump or publish.

## Notes

- Flow-processing helper scripts in this repo now default to `flow-governance-review/assets/artifacts/flow-processing/`; old `docs/` paths are deprecated.
- `run-governance` stays local-first as well; if you pass explicit local flow/process snapshots, it will keep using those for review, dedup, repair, and OpenClaw pack export.
- If you pass both `--subject-flows-file` and `--candidate-flows-file`, dedup runs from explicit local snapshots and writes the merged snapshot it used.
- If you pass `--subject-flows-file` through `openclaw-entry` without `--candidate-flows-file` and no live candidate-pool env is available, the wrapper falls back to subject-only dedup.
- If you pass `--flows-file`, dedup treats that file as one explicit all-vs-all snapshot.
- If you also pass `--processes-file`, duplicate reference counts stay local as well.
- If you pass `--process-pool-file` to `openclaw-entry`, `openclaw-full-run`, `run-governance`, deterministic repair apply, or OpenClaw ref-apply, exact-version process rows can be reused locally and synced back after patching.
- If you omit `--processes-file`, live process reference counts are no longer implicit; you must opt in with `--live-ref-counts`.
- If you want only public live rows, pass `--no-user-0` instead of relying on a default user id.
- These scripts read the current process environment only; they do not load `.env` files themselves. If you run them through OpenClaw, make sure the runner has already sourced the intended env file, typically `~/.openclaw/.env` unless you explicitly want another env source.
- These commands do not require `OPENAI_API_KEY`; LLM work should happen through OpenClaw review packs and explicit decision files.
