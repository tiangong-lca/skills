# Operations Playbook

## Canonical Commands

```bash
node process-automated-builder/scripts/run-process-automated-builder.mjs auto-build --help
node process-automated-builder/scripts/run-process-automated-builder.mjs identity-preflight --help
node process-automated-builder/scripts/run-process-automated-builder.mjs evidence-search --help
node process-automated-builder/scripts/run-process-automated-builder.mjs build-plan --help
node process-automated-builder/scripts/run-process-automated-builder.mjs resume-build --help
node process-automated-builder/scripts/run-process-automated-builder.mjs publish-build --help
node process-automated-builder/scripts/run-process-automated-builder.mjs batch-build --help
```

## Run Evidence Retrieval

Use this before authoring values that depend on public facts, especially numeric/time/geography/technology/source fields.

```bash
node process-automated-builder/scripts/run-process-automated-builder.mjs evidence-search plan \
  --input /abs/path/evidence-search.request.json \
  --out-dir /abs/path/artifacts/<case_slug>/evidence/<field_slug> \
  --json

node process-automated-builder/scripts/run-process-automated-builder.mjs evidence-search run \
  --input /abs/path/evidence-search.request.json \
  --results /abs/path/search-results.json \
  --out-dir /abs/path/artifacts/<case_slug>/evidence/<field_slug> \
  --json
```

Use web/search tools to collect `search-results.json`; use Browser/Computer Use only when search snippets are insufficient, the page is JS-rendered, or login/session/UI evidence is required. Stop when `evidence-search-report.json` shows sufficient authoritative evidence, or when `outputs/evidence-search-declaration.json` makes the remaining gap explicit.

## Run Identity And Build Gates

```bash
node process-automated-builder/scripts/run-process-automated-builder.mjs identity-preflight \
  --input /abs/path/process-preflight.json \
  --out-dir /abs/path/artifacts/<case_slug>/identity \
  --json

node process-automated-builder/scripts/run-process-automated-builder.mjs build-plan validate \
  --input /abs/path/process-build-plan.json \
  --out-dir /abs/path/artifacts/<case_slug>/build-plan \
  --json

node process-automated-builder/scripts/run-process-automated-builder.mjs build-plan materialize \
  --input /abs/path/process-build-plan.json \
  --out-dir /abs/path/artifacts/<case_slug>/build-plan \
  --json
```

Stop on `block_duplicate`, `manual_review`, a failed build-plan gate, or any schema blocker. The skill should not override the CLI decision.
For factual fields, also stop on `completed_no_sufficient_evidence`. `completed_with_partial_evidence` may proceed only when the partial scope is explicit in the build plan and evidence manifest.

## Start One Run

```bash
node process-automated-builder/scripts/run-process-automated-builder.mjs auto-build \
  --flow-file /abs/path/reference-flow.json \
  --operation produce \
  --out-dir /abs/path/artifacts/<case_slug>/process_from_flow/<run_id> \
  --json
```

Equivalent request-file form:

```bash
node process-automated-builder/scripts/run-process-automated-builder.mjs auto-build \
  --input /abs/path/process-auto-build.request.json \
  --out-dir /abs/path/artifacts/<case_slug>/process_from_flow/<run_id> \
  --json
```

What this does today:

- normalizes the request
- creates one deterministic run root at the explicit `--out-dir`
- writes stage directories and manifests
- writes the initial state and handoff summary

## Resume One Existing Run

```bash
node process-automated-builder/scripts/run-process-automated-builder.mjs resume-build \
  --run-dir /abs/path/artifacts/<case_slug>/process_from_flow/<run_id> \
  --run-id <run_id> \
  --json
```

Use this when a caller wants:

- fresh resume metadata
- a stable resume history record
- a quick consistency check over state, handoff summary, and run manifest

## Prepare One Publish Bundle

```bash
node process-automated-builder/scripts/run-process-automated-builder.mjs publish-build \
  --run-dir /abs/path/artifacts/<case_slug>/process_from_flow/<run_id> \
  --run-id <run_id> \
  --json
```

Use this when:

- the run already contains local process/source datasets
- the next step should be unified publish handoff
- downstream publish should go through `tiangong-lca publish run`, not a skill-private path

Before this step, run `process complete-required-fields` or `process build-plan materialize` as appropriate. For annual supply or production volume, the CLI must use explicit evidence first, then reference-flow `meanAmount`, then reference-flow `resultingAmount`; the skill should only supply evidence and context.

## Prepare A Batch

```bash
node process-automated-builder/scripts/run-process-automated-builder.mjs batch-build \
  --input /abs/path/process-batch.request.json \
  --out-dir /abs/path/artifacts/<case_slug>/process_batch/<batch_id> \
  --json
```

Batch mode fans out deterministic local runs and records their reports in one batch ledger.

The wrapper intentionally requires `--out-dir` / `--run-dir` instead of letting the CLI fall back to `cwd/artifacts/...`.

## Required Env

- Wrappers use `npm exec --yes --package=@tiangong-lca/cli@latest -- tiangong-lca` by default.
- Set `TIANGONG_LCA_CLI_DIR` only when you need a local CLI working tree for dev/CI.

The canonical commands above do not require any legacy provider, transport, or OCR env stack.

## Failure Triage

- Local CLI override issues:
  - set `TIANGONG_LCA_CLI_DIR`
  - or pass `--cli-dir`
- Missing flow input:
  - provide `--input`
  - or one of `--flow-file`, `--flow-json`, `--flow-stdin`
- Reused run id:
  - choose a different `run_id`
  - or let the CLI generate one
- Publish handoff missing datasets:
  - inspect `exports/processes/`, `exports/sources/`, `cache/process_from_flow_state.json`
- Parallel writer conflict:
  - do not run two writers on the same `run_id`

## Explicit Non-Goals

- no hidden Python runtime
- no shell daemon or systemd layer
- no skill-private transport or publish implementation
- no reintroduction of LangGraph, MCP, or direct provider env parsing inside this skill
