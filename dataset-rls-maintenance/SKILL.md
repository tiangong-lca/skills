---
name: dataset-rls-maintenance
description: Orchestrate current-user RLS-scoped TianGong dataset cleanup, delete, retirement, and redo workflows through CLI maintenance plans, explicit approvals, and readback verification.
---

# Dataset RLS Maintenance

Use this skill when a Foundry or account-governance task needs to remove, retire, repair, or redo rows that were previously imported incorrectly under the current user's database access.

## Boundaries

- This skill is a workflow coordinator only.
- Do not call Supabase directly, parse raw credentials, bypass RLS, or write private database delete logic inside the skill.
- Do not delete rows from a broad filter such as only `state_code=0`.
- Do not operate on rows that are not visible to the current authenticated user.
- Do not delete public/shared/canonical support rows. Treat them as protected unless the maintenance plan explicitly proves current-user ownership and policy allows the operation.
- If the CLI maintenance command is missing, record it as a CLI gap and stop before any database mutation.

## Required Inputs

- A frozen maintenance scope manifest with task id, source import run id when available, dataset ids, versions, table/type, intended operation, and reason.
- Current account guard: user id or account identity expected by the task.
- Existing import artifacts when the redo starts from a Foundry workspace.
- The replacement source manifest or seed manifest when the operation includes redo.
- Explicit operator approval for any commit path.

## CLI Surface

For the narrow "clear my current account so I can re-import" workflow, use the implemented command:

```bash
tiangong-lca dataset maintenance clear-account \
  --out-dir ./dataset-maintenance/account-clear \
  --json

tiangong-lca dataset maintenance clear-account \
  --out-dir ./dataset-maintenance/account-clear \
  --commit \
  --confirm <current-account-email> \
  --json
```

This command deletes only rows visible to the current authenticated user and filtered by that user's `user_id`
through RLS. It covers authorable dataset tables in reference-safe delete order:
`lifecyclemodels`, `processes`, `flows`, `sources`, and `contacts`.
It does not delete `unitgroups` or `flowproperties`, because those are treated as protected support data by default.

Use `--state-code <n>` only when the operator asks for a narrower cleanup. Omit it when the operator explicitly
asks to clear all current-account data for re-import.

For row-level delete, retirement, repair, or redo workflows, the owning command family remains:

```bash
tiangong-lca dataset maintenance plan \
  --scope ./maintenance-scope.json \
  --operation <delete|retire|redo-import|repair-references> \
  --out-dir ./dataset-maintenance \
  --json

tiangong-lca dataset maintenance apply \
  --plan ./dataset-maintenance/maintenance-plan.json \
  --commit \
  --json

tiangong-lca dataset maintenance verify \
  --plan ./dataset-maintenance/maintenance-plan.json \
  --out-dir ./dataset-maintenance/verify \
  --json
```

Until the row-level plan/apply/verify commands are implemented, the skill may plan and document the required artifacts,
but must not substitute ad hoc SQL, REST deletes, or Foundry-local database code.

## Workflow

1. Freeze the maintenance scope manifest before any remote action. Include dataset type/table, id, version, expected owner, state code, import run id, source package or evidence task id, and operation reason for every target row.
2. Capture the current-user visible remote snapshot through the CLI. The snapshot must prove which rows are visible under RLS and which requested rows are missing, protected, public/shared, or owned by someone else.
3. Build a reference impact report before delete or redo. Processes, lifecyclemodels, flows, sources, contacts, and support rows must be checked for inbound and outbound references at the same scope.
4. Classify every target row into one of: `delete_candidate`, `retire_candidate`, `rewrite_reference`, `redo_candidate`, `skip_missing`, `protected`, or `blocked`.
5. Produce `maintenance-plan.json` and `dry-run-report.json`. The plan must include row-level actions, order of operations, dependency blockers, expected post-state, and rollback notes.
6. Require explicit approval before `apply --commit`. Approval must name the plan file, operation, current user/account, row counts, and whether redo rows are ready.
7. Execute only through the CLI maintenance command or existing public CLI save-draft/publish paths named by the approved plan.
8. Run readback verification after any commit. Verification must re-fetch affected rows and references and compare them against the plan.
9. For redo, start a fresh Foundry import or source-evidence task from the corrected source manifest after cleanup is verified. Do not reuse stale `.foundry` artifacts as proof of the new run.

## Required Artifacts

- `maintenance-scope.json`
- `rls-visible-snapshot.json`
- `protected-rows.jsonl`
- `reference-impact-report.json`
- `maintenance-plan.json`
- `dry-run-report.json`
- `approval-record.json`
- `commit-report.json`
- `readback-verify-report.json`
- `redo-source-manifest.json` when redo is part of the task

## Stop Conditions

- The current account guard does not match the remote session.
- Any target row is not visible under current-user RLS.
- A target row is public/shared/canonical support data without an explicit policy exception.
- Inbound references would dangle after deletion.
- Replacement rows are not validated when the operation is redo.
- The task asks for direct SQL, REST delete, service-role access, or any RLS bypass.
