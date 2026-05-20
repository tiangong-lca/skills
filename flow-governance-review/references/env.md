# Env

## Local-First Mode

Prefer local JSON or JSONL inputs. In local mode, no remote credentials are required.

`--process-pool-file` is also local-only: it is just a JSON or JSONL working pool of exact-version process rows and does not require any remote credential by itself.

## Wrapper Resolution

Wrappers run the published CLI by default through `npm exec --yes --package=@tiangong-lca/cli@latest -- tiangong`.

Set `TIANGONG_LCA_CLI_DIR` or pass `--cli-dir` only when you need a local CLI working tree for dev/CI.

## Optional CLI Read / Commit Inputs

Commands that read or write through the TianGong LCA API use the CLI's canonical env contract:

- `TIANGONG_LCA_API_BASE_URL`
- `TIANGONG_LCA_API_KEY`
- `TIANGONG_LCA_SUPABASE_PUBLISHABLE_KEY`
- `TIANGONG_LCA_REGION` (optional)

Typical commands in this skill that may need those env values:

- `flow-get`
- `flow-list`
- `materialize-db-flows`
- `publish-version --commit`
- `publish-reviewed-data --commit`

Local-only commands that do not require remote credentials by themselves:

- `review-flows`
- `materialize-approved-decisions`
- `remediate-flows`
- `build-flow-alias-map`
- `scan-process-flow-refs`
- `plan-process-flow-repairs`
- `apply-process-flow-repairs`
- `regen-product`
- `validate-processes`

## Optional CLI LLM Inputs

Only `review-flows` can optionally enable the CLI LLM path. When using `--enable-llm`, set the CLI's canonical LLM env:

- `TIANGONG_LCA_REVIEW_LLM_BASE_URL`
- `TIANGONG_LCA_REVIEW_LLM_API_KEY`
- `TIANGONG_LCA_REVIEW_LLM_MODEL` (optional override also exists as a CLI flag)

## Notes

- Keep long-lived machine artifacts under an explicit output root such as `/abs/path/artifacts/<case_slug>/flow-processing/`.
- `publish-reviewed-data` stays dry-run unless `--commit` is passed.
- `publish-reviewed-data --original-flow-rows-file` can skip unchanged flow rows before version planning or commit.
- The wrapper does not load `.env` files. It forwards the current process environment to `tiangong`.
