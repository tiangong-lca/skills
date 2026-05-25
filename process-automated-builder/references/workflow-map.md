# Workflow Map

## Goal

Keep `process-automated-builder` as a thin wrapper over the unified CLI surface:

- `tiangong process auto-build`
- `tiangong process resume-build`
- `tiangong process publish-build`
- `tiangong process batch-build`

This skill does not add a second runtime layer.

## Supported Input Paths

For a new run:

- `node scripts/run-process-automated-builder.mjs auto-build --input <request.json> --out-dir <dir>`
- or compatibility flow input flags:
  - `--flow-file`
  - `--flow-json`
  - `--flow-stdin`

For an existing run:

- `node scripts/run-process-automated-builder.mjs resume-build --run-dir <dir> [--run-id <run_id>]`
- `node scripts/run-process-automated-builder.mjs publish-build --run-dir <dir> [--run-id <run_id>]`

For a batch manifest:

- `node scripts/run-process-automated-builder.mjs batch-build --input <batch-request.json> --out-dir <dir>`

## Canonical Runtime Layers

1. Skill wrapper
   - native Node `.mjs`
   - launches `npm exec --yes --package=@tiangong-lca/cli@latest -- tiangong` by default
   - resolves `TIANGONG_LCA_CLI_DIR` only when a local override is requested
   - forwards arguments to `tiangong`
2. CLI implementation
   - owns request normalization
   - owns run-id and artifact layout
   - owns state locking
   - owns publish handoff contract
3. Downstream modules
   - live inside `tiangong-lca-cli`
   - if new behavior is needed, add it there first

There is no Python, shell, MCP, LangGraph, OpenAI, KB, or TianGong unstructured runtime left in this skill.

The wrapper does not infer output directories from `cwd`; pass them explicitly.
For repeatable runs, use explicit paths such as `/abs/path/artifacts/<case_slug>/...`.

## Current Output Layout

`auto-build` prepares one run root under the explicit `--out-dir` and writes:

- `request/pff-request.json`
- `request/request.normalized.json`
- `request/source-policy.json`
- `input/input_manifest.json`
- `manifests/flow-summary.json`
- `manifests/assembly-plan.json`
- `manifests/lineage-manifest.json`
- `manifests/invocation-index.json`
- `manifests/run-manifest.json`
- `cache/process_from_flow_state.json`
- `cache/agent_handoff_summary.json`
- `reports/process-auto-build-report.json`
- staged directories under `stage_outputs/01_*` through `stage_outputs/10_publish/`

`resume-build` reopens the explicit `--run-dir` and writes:

- `manifests/resume-metadata.json`
- `manifests/resume-history.jsonl`
- `reports/process-resume-build-report.json`

`publish-build` prepares one local publish handoff bundle under the explicit `--run-dir` and writes:

- `stage_outputs/10_publish/publish-bundle.json`
- `stage_outputs/10_publish/publish-request.json`
- `stage_outputs/10_publish/publish-intent.json`
- `reports/process-publish-build-report.json`

`batch-build` prepares multiple local runs under the explicit `--out-dir` and writes:

- `request/batch-request.json`
- `request/request.normalized.json`
- `manifests/invocation-index.json`
- `manifests/run-manifest.json`
- `reports/process-batch-build-report.json`

## Parallel Rules

- Run-level parallel is allowed when each item uses a distinct `run_id`.
- One `run_id` must have exactly one active writer.
- The state file lock in the CLI is authoritative. Do not build a second coordination layer in this skill.

## Policy

- Do not reintroduce old env names or skill-private HTTP clients.
- Do not add new business scripts here.
- If a future step needs LLM, KB search, unstructured parsing, validation, or publish execution, expose it as a native `tiangong process ...` capability first.
