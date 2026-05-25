---
name: process-automated-builder
description: Execute the supported `process_from_flow` CLI workflow. Use `node scripts/run-process-automated-builder.mjs auto-build|resume-build|publish-build|batch-build` when you need the unified `tiangong-lca process ...` surface from a skill wrapper.
---

# Process Automated Builder

## Scope
- Prepare one local `process_from_flow` run from a request or reference-flow input.
- Reopen an existing run to write deterministic resume metadata.
- Prepare one local publish bundle from an existing run.
- Prepare a deterministic batch of local process-build runs.
- Run the process identity, build-plan, required-field, and local verification gates that must surround any automated process generation.

This skill uses the CLI only. Legacy alternate runtimes are not part of the supported path.

## Canonical Runtime
1. Read `references/workflow-map.md` and `references/operations-playbook.md`.
2. Choose an explicit output directory, for example `/abs/path/artifacts/<case_slug>/...`.
3. Use `node scripts/run-process-automated-builder.mjs auto-build ... --out-dir <dir>` to create one local run root.
4. Continue with `resume-build`, `publish-build`, or `batch-build` as needed, passing `--run-dir` or `--out-dir` explicitly.
5. If a missing capability is discovered, add a native `tiangong-lca process ...` command in `tiangong-lca-cli` first. Do not add new business runtime inside this skill.

## Production Gate Sequence

Use this sequence for target-quality process production:

1. Decide `unit_of_analysis` before generation.

The skill/Codex workflow must author a `unit_of_analysis` object in `process-build-plan.json` before `build-plan validate` or `materialize` can pass. This is a semantic decision, not a CLI industry-judgement step. Use source evidence, PCR/PEF/ILCD/TIDAS context, flow governance findings, and the target data purpose to decide:

- `target_kind`
- functional unit or declared unit
- reference flow identity, reference unit, reference amount, and flow property
- scaling evidence or explicit scaling evidence status
- decision: `ready_for_materialization`, `declared_unit_dataset`, `blocked_until_scaling_evidence`, or `manual_review`

Only `ready_for_materialization` and `declared_unit_dataset` may proceed to materialization. `blocked_until_scaling_evidence` and `manual_review` are autonomous stop states. The CLI only checks presence/completeness and blocks non-automatic decisions; it does not choose the unit basis for the agent.

Minimal artifact shape:

```json
{
  "unit_of_analysis": {
    "target_kind": "countable_durable_product",
    "decision": "ready_for_materialization",
    "functional_unit": {
      "what": "provide visual display",
      "how_much": "1 LCD monitor",
      "how_well": "representative LCD technology",
      "how_long": "documented service lifetime or display-hours"
    },
    "reference_flow": {
      "flow_identity": "LCD monitor, finished product",
      "reference_unit": "item",
      "reference_amount": 1,
      "flow_property": "Number of items"
    },
    "scaling_evidence_status": "source_available"
  }
}
```

2. Run identity preflight before generation:

```bash
node scripts/run-process-automated-builder.mjs identity-preflight \
  --input /abs/path/process-preflight.json \
  --out-dir /abs/path/artifacts/<case_slug>/identity \
  --json
```

Input artifact: `process-preflight.json` with the target identity, candidate rows, and optional remote candidate search settings.
Output artifacts: `outputs/identity-decision.json`, `outputs/identity-candidates.jsonl`, and `outputs/identity-candidate-sources.json`.
Blockers: `block_duplicate` means do not create a new process; `manual_review` means stop autonomous execution and hand off the candidate evidence.

3. Materialize or complete only through CLI-owned gates:

```bash
node scripts/run-process-automated-builder.mjs build-plan validate \
  --input /abs/path/process-build-plan.json \
  --out-dir /abs/path/artifacts/<case_slug>/build-plan \
  --json

node scripts/run-process-automated-builder.mjs build-plan materialize \
  --input /abs/path/process-build-plan.json \
  --out-dir /abs/path/artifacts/<case_slug>/build-plan \
  --json
```

Input artifact: `process-build-plan.json`.
Output artifacts: `outputs/build-plan-gate-report.json` and, for `materialize`, `outputs/materialized-process.json`.
Blockers: any failed gate, unresolved identity decision, missing `unit_of_analysis`, non-automatic unit decision, missing evidence, schema failure, duplicate finding, or required-field gap stops publish preparation.

4. Complete required authoring fields for row snapshots when a full build plan is not the input:

```bash
node scripts/run-process-automated-builder.mjs complete-required-fields \
  --input /abs/path/processes.jsonl \
  --out /abs/path/processes.completed.jsonl \
  --default-unit MJ
```

The CLI owns the deterministic writer. For `annualSupplyOrProductionVolume`, use an explicit evidence value when present. If there is no evidence value, derive the value from the quantitative reference flow's `meanAmount` first, then `resultingAmount`, and write the field with the reference unit per year. Do not invent production-volume figures in the skill.

5. Use `tidas-bilingual-transcreation` after materialization when bilingual fields are weak or newly generated. Apply and validate translations through `tiangong-lca dataset bilingual extract/apply/validate`.

6. Prepare publish handoff only after local schema, process review, bilingual validation, reference verification, and matrix/compute readiness gates pass.

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
node scripts/run-process-automated-builder.mjs identity-preflight --help
node scripts/run-process-automated-builder.mjs build-plan --help
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
node scripts/run-process-automated-builder.mjs verify-rows --rows-file /abs/path/process-list-report.json --out-dir /abs/path/artifacts/<case_slug>/process-verify
```

## Runtime Requirements
- The wrapper runs the published CLI by default through `npm exec --yes --package=@tiangong-lca/cli@latest -- tiangong-lca`.
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
- Identity blockers: `block_duplicate` and `manual_review` are stop states, not warnings.
- Unit-of-analysis blockers: missing `unit_of_analysis`, `manual_review`, and `blocked_until_scaling_evidence` are stop states. Fix the skill-authored decision artifact; do not patch around the CLI gate.
- Build-plan blockers: inspect `outputs/build-plan-gate-report.json`; do not patch around a failed CLI gate inside the skill.
- If a required step is missing, add it as a native `tiangong-lca process ...` command instead of reintroducing a legacy runtime here.

## Load References On Demand
- `references/workflow-map.md`: current CLI-only execution map and output layout.
- `references/operations-playbook.md`: concise command examples and troubleshooting.
