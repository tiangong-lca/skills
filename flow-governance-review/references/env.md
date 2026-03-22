# Env

## Local-First Mode

Prefer local JSON or JSONL inputs. In local mode, no remote credentials are required.

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

## Notes

- `run-governance` stays local-first as well; if you pass explicit local flow/process snapshots, it will keep using those for review, dedup, repair, and OpenClaw pack export.
- If you pass both `--subject-flows-file` and `--candidate-flows-file`, dedup runs from explicit local snapshots and writes the merged snapshot it used.
- If you pass `--flows-file`, dedup treats that file as one explicit all-vs-all snapshot.
- If you also pass `--processes-file`, duplicate reference counts stay local as well.
- If you omit `--processes-file`, live process reference counts are no longer implicit; you must opt in with `--live-ref-counts`.
- If you want only public live rows, pass `--no-user-0` instead of relying on a default user id.
- These commands do not require `OPENAI_API_KEY`; LLM work should happen through OpenClaw review packs and explicit decision files.
