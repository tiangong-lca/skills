# Operations Playbook

## Canonical Commands

```bash
node process-automated-builder/scripts/run-process-automated-builder.mjs auto-build --help
node process-automated-builder/scripts/run-process-automated-builder.mjs resume-build --help
node process-automated-builder/scripts/run-process-automated-builder.mjs publish-build --help
node process-automated-builder/scripts/run-process-automated-builder.mjs batch-build --help
```

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
- downstream publish should go through `tiangong publish run`, not a skill-private path

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

- Wrappers use `npm exec --yes --package=@tiangong-lca/cli@latest -- tiangong` by default.
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
