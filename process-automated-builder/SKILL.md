---
name: process-automated-builder
description: Execute the canonical CLI-owned `process_from_flow` local run workflow. Use `node scripts/run-process-automated-builder.mjs auto-build|resume-build|publish-build|batch-build` when you need the unified `tiangong process ...` surface from a skill wrapper.
---

# Process Automated Builder

## Scope
- Prepare one local `process_from_flow` run from a request or reference-flow input.
- Reopen an existing run to write deterministic resume metadata.
- Prepare one local publish bundle from an existing run.
- Prepare a deterministic batch of local process-build runs.

This skill is now CLI-only. It no longer exposes Python, LangGraph, MCP, OpenAI, KB, or TianGong unstructured fallback paths.

## Canonical Runtime
1. Read `references/workflow-map.md` and `references/operations-playbook.md`.
2. Use `node scripts/run-process-automated-builder.mjs auto-build ...` to create one local run root.
3. Continue with `resume-build`, `publish-build`, or `batch-build` as needed.
4. If a missing capability is discovered, add a native `tiangong process ...` command in `tiangong-lca-cli` first. Do not add new business runtime inside this skill.

## Parallel Execution Contract
- `Run-level parallel`: multiple flow inputs can run concurrently, but each run must use a distinct `run_id`.
- `In-run parallel`: do not run multiple writers against the same `run_id`.
- Single-writer rule:
  - never let multiple agents write the same `artifacts/process_from_flow/<run_id>/cache/process_from_flow_state.json`
  - within one `run_id`, only one active writer process is allowed at a time
  - enforcement is code-level through the CLI state lock

## Canonical Node Wrapper Commands
```bash
node scripts/run-process-automated-builder.mjs auto-build --help
node scripts/run-process-automated-builder.mjs resume-build --help
node scripts/run-process-automated-builder.mjs publish-build --help
node scripts/run-process-automated-builder.mjs batch-build --help

node scripts/run-process-automated-builder.mjs auto-build \
  --flow-file /abs/path/reference-flow.json \
  --operation produce \
  --json

node scripts/run-process-automated-builder.mjs resume-build --run-id <run_id> --json
node scripts/run-process-automated-builder.mjs publish-build --run-id <run_id> --json
node scripts/run-process-automated-builder.mjs batch-build --input /abs/path/batch-request.json --json
```

## Runtime Requirements
- The wrapper runs the published CLI by default through `npx -y @tiangong-lca/cli@latest`.
- Set `TIANGONG_LCA_CLI_DIR` or pass `--cli-dir` only when you need a local CLI working tree for dev/CI.
- The current canonical commands are local artifact commands. They do not require any legacy provider, transport, or OCR env stack.
- If a future native CLI command needs additional env, document it in `tiangong-lca-cli` first and keep this skill as a thin caller only.

## Fast Troubleshooting
- Local CLI override issues: set `TIANGONG_LCA_CLI_DIR` or pass `--cli-dir` only when you intentionally need an unpublished working tree.
- Missing `--input` / `--flow-file`: new runs need one explicit request or reference-flow input.
- Run-level conflicts: do not reuse the same `run_id` across concurrent writers.
- Publish handoff issues: inspect `stage_outputs/10_publish/` and `cache/agent_handoff_summary.json` before touching downstream publish flow.
- If someone asks for the deleted legacy end-to-end workflow, the correct fix is a new native `tiangong process ...` command, not reintroducing Python here.

## Load References On Demand
- `references/workflow-map.md`: current CLI-only execution map and artifact contract.
- `references/operations-playbook.md`: concise command examples and troubleshooting.
