---
name: process-automated-builder
description: Execute the supported `process_from_flow` CLI workflow. Use `node scripts/run-process-automated-builder.mjs auto-build|resume-build|publish-build|batch-build` when you need the unified `tiangong process ...` surface from a skill wrapper.
---

# Process Automated Builder

## Scope
- Prepare one local `process_from_flow` run from a request or reference-flow input.
- Reopen an existing run to write deterministic resume metadata.
- Prepare one local publish bundle from an existing run.
- Prepare a deterministic batch of local process-build runs.

This skill uses the CLI only. Legacy alternate runtimes are not part of the supported path.

## Canonical Runtime
1. Read `references/workflow-map.md` and `references/operations-playbook.md`.
2. Choose an explicit output directory, for example `/abs/path/artifacts/<case_slug>/...`.
3. Use `node scripts/run-process-automated-builder.mjs auto-build ... --out-dir <dir>` to create one local run root.
4. Continue with `resume-build`, `publish-build`, or `batch-build` as needed, passing `--run-dir` or `--out-dir` explicitly.
5. If a missing capability is discovered, add a native `tiangong process ...` command in `tiangong-lca-cli` first. Do not add new business runtime inside this skill.

## Parallel Execution Contract
- `Run-level parallel`: multiple flow inputs can run concurrently, but each run must use a distinct `run_id`.
- `In-run parallel`: do not run multiple writers against the same `run_id`.
- Single-writer rule:
  - never let multiple agents write the same `<run_dir>/cache/process_from_flow_state.json`
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
  --out-dir /abs/path/artifacts/<case_slug>/process_from_flow/<run_id> \
  --json

node scripts/run-process-automated-builder.mjs resume-build --run-dir /abs/path/artifacts/<case_slug>/process_from_flow/<run_id> --run-id <run_id> --json
node scripts/run-process-automated-builder.mjs publish-build --run-dir /abs/path/artifacts/<case_slug>/process_from_flow/<run_id> --run-id <run_id> --json
node scripts/run-process-automated-builder.mjs batch-build --input /abs/path/batch-request.json --out-dir /abs/path/artifacts/<case_slug>/process_batch/<batch_id> --json
```

## Runtime Requirements
- The wrapper runs the published CLI by default through `npm exec --yes --package=@tiangong-lca/cli@latest -- tiangong`.
- Set `TIANGONG_LCA_CLI_DIR` or pass `--cli-dir` only when you need a local CLI working tree for dev/CI.
- The wrapper requires explicit output paths instead of relying on `cwd/artifacts/...` defaults.
- For repeatable runs, use an explicit output root such as `/abs/path/artifacts/<case_slug>/...`.
- The current canonical commands prepare local run outputs and do not depend on legacy private runtimes.
- If a future native CLI command needs additional env, document it in `tiangong-lca-cli` first and keep this skill as a thin caller only.

## Process Name Contract
- Any generated process payload must preserve the four-part process name object:
  `name.baseName`, `name.treatmentStandardsRoutes`, `name.mixAndLocationTypes`, `name.functionalUnitFlowProperties`.
- `baseName`, `treatmentStandardsRoutes`, and `mixAndLocationTypes` are schema-required in current TianGong process payloads. Keep the keys even when one field is semantically empty; do not collapse the whole reference-flow short description back into `baseName`.
- When name splitting is ambiguous, align with `../lifecycleinventory-review/profiles/process/references/process-review-rules.md` instead of inventing a one-off local convention.

## Fast Troubleshooting
- Local CLI override issues: set `TIANGONG_LCA_CLI_DIR` or pass `--cli-dir` only when you intentionally need an unpublished working tree.
- Missing `--out-dir` or `--run-dir`: the wrapper requires an explicit output path such as `/abs/path/artifacts/<case_slug>/...`.
- Missing `--input` / `--flow-file`: new runs need one explicit request or reference-flow input.
- Run-level conflicts: do not reuse the same `run_id` across concurrent writers.
- Publish preparation issues: inspect `stage_outputs/10_publish/` and `cache/agent_handoff_summary.json` before touching downstream publish flow.
- If a required step is missing, add it as a native `tiangong process ...` command instead of reintroducing a legacy runtime here.

## Load References On Demand
- `references/workflow-map.md`: current CLI-only execution map and output layout.
- `references/operations-playbook.md`: concise command examples and troubleshooting.
